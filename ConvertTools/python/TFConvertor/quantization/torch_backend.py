"""Optional Torch CPU/CUDA and tfllm observers over recorded SDK Op calls."""

import time
from contextlib import contextmanager, nullcontext
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .graph import Ref, refs, tree, sha256


def pair(v):
    return (v, v) if isinstance(v, int) else tuple(v)


def pad4(v):
    if isinstance(v, int):
        return (v,) * 4
    return (v[0], v[0], v[1], v[1]) if len(v) == 2 else tuple(v)


class TorchGraph(nn.Module):
    def __init__(self, graph):
        super().__init__()
        self.graph = graph
        self.ops = nn.ModuleDict()
        self.external_padding = {}
        # SDK parameter names can contain dots/slashes; use private buffer names.
        # Registered buffers follow Module.to(), including non-weight constants.
        self.param_buffers = {}
        for index, (name, value) in enumerate(graph.params.items()):
            buffer = f"_constant_{index}"
            self.register_buffer(buffer, torch.from_numpy(value.copy()))
            self.param_buffers[name] = buffer
        supported = {
            "Placeholder",
            "Placeholder2",
            "Convolution2",
            "DeConvolution2",
            "InnerProduct2",
            "MatMul",
            "MatMul2",
            "ReLU",
            "Sigmoid",
            "Swish",
            "HardSigmoid",
            "HardSigmoid2",
            "Tanh",
            "Softmax",
            "GeLU",
            "Scale2",
            "GlobalAvePooling",
            "MaxPooling",
            "AvePooling",
            "Flatten",
            "Flatten2Matrix",
            "Reshape",
            "Reshape2",
            "Squeeze",
            "Expand",
            "ExpandDims",
            "BroadCast",
            "BroadCast2",
            "Concat",
            "Split",
            "Slice",
            "Gather",
            "Gather2",
            "Range",
            "ConstantOfShape",
            "Crop3",
            "Crop2",
            "NearestReSize1",
            "NearestReSize2",
            "NearestReSize3",
            "Add",
            "Sub",
            "Mul",
            "Div",
            "Max",
            "Min",
            "ReduceMean",
            "Mean",
            "ReduceSum",
            "ReduceMax",
            "ReduceMin",
            "Transpose",
            "Cast",
            "Shape",
        }
        for index, c in enumerate(graph.calls):
            name = c["op"]
            p = c["args"]
            if name not in supported and not name.startswith("_symbol_"):
                raise ValueError("Torch observer does not support SDK Op " + name)
            if name == "Convolution2":
                w = self.params[p["weight"].name]
                b = self.params[p["bias"].name] if isinstance(p["bias"], Ref) else None
                top, bottom, left, right = pad4(p["pad"])
                symmetric = top == bottom and left == right
                self.external_padding[str(index)] = (
                    None if symmetric else (left, right, top, bottom)
                )
                m = nn.Conv2d(
                    w.shape[1] * p["group"],
                    w.shape[0],
                    pair(p["kernel"]),
                    stride=pair(p["stride"]),
                    padding=(top, left) if symmetric else (0, 0),
                    dilation=p["dilation"],
                    groups=p["group"],
                    bias=b is not None,
                    dtype=w.dtype,
                )
                with torch.no_grad():
                    m.weight.copy_(w)
                    if b is not None:
                        m.bias.copy_(b.reshape(-1))
                self.ops[str(index)] = m
            elif name == "DeConvolution2":
                if p["group"] != 1:
                    raise ValueError(
                        "Torch observer does not support grouped SDK DeConvolution2"
                    )
                top, bottom, left, right = pad4(p["pad"])
                if (
                    top != bottom
                    or left != right
                    or p.get("outPadH", 0)
                    or p.get("outPadW", 0)
                ):
                    raise ValueError(
                        "Torch observer requires symmetric DeConvolution2 padding and no output padding"
                    )
                # SDK stores the spatially flipped [out,in,H,W] filter.
                w = self.params[p["weight"].name]
                bias = (
                    self.params[p["bias"].name] if isinstance(p["bias"], Ref) else None
                )
                m = nn.ConvTranspose2d(
                    w.shape[1],
                    w.shape[0],
                    pair(p["kernel"]),
                    stride=pair(p["stride"]),
                    dilation=p["dilation"],
                    output_padding=(p.get("outPadH", 0), p.get("outPadW", 0)),
                    bias=bias is not None,
                    dtype=w.dtype,
                )
                with torch.no_grad():
                    m.weight.copy_(w.flip(-2, -1).permute(1, 0, 2, 3))
                    if bias is not None:
                        m.bias.copy_(bias.reshape(-1))
                self.ops[str(index)] = m
            elif name in ("InnerProduct2", "MatMul", "MatMul2"):
                right = p["weight" if name == "InnerProduct2" else "InputB"]
                # Attention uses activation x activation, often with batches.
                # Only a constant rank-2 RHS is a Linear module for tfllm.
                if name != "InnerProduct2" and (
                    not isinstance(right, Ref)
                    or right.name not in self.params
                    or self.params[right.name].ndim != 2
                    or p.get("transA")
                    or p.get("bias2Row")
                ):
                    continue
                w = self.params[right.name]
                w = (
                    w.reshape(p["outchannels"], -1)
                    if name == "InnerProduct2"
                    else (w.T if not p.get("transB") else w)
                )
                if w.ndim != 2:
                    raise ValueError(
                        "Torch observer requires constant rank2 MatMul weight"
                    )
                bias = p.get("bias", p.get("Bias"))
                b = self.params[bias.name] if isinstance(bias, Ref) else None
                m = nn.Linear(w.shape[1], w.shape[0], bias=b is not None, dtype=w.dtype)
                with torch.no_grad():
                    m.weight.copy_(w)
                    if b is not None:
                        m.bias.copy_(b.reshape(-1))
                self.ops[str(index)] = m
        self.eval()
        # Observations retain only scalar bounds. Release an activation once
        # its final consumer has finished, including branched residual graphs.
        last_use = {}
        for index, call in enumerate(graph.calls):
            for name in refs(call["args"]):
                last_use[name] = index
        self.release_after = [[] for _ in graph.calls]
        for name, index in last_use.items():
            if name not in graph.params and name not in graph.outputs:
                self.release_after[index].append(name)

    @property
    def params(self):
        return {
            name: getattr(self, buffer) for name, buffer in self.param_buffers.items()
        }

    def forward(self, x, observe=None, precision=None):
        values = dict(self.params)
        for index, c in enumerate(self.graph.calls):
            name = c["op"]
            outs = c["outputs"]
            shape = self.graph.meta[outs[0]]["shape"]
            p = {
                key: tree(
                    value,
                    lambda v, key=key: (
                        precision.resolve(c, key, v, values)
                        if precision is not None
                        else values.get(v.name)
                    ) if isinstance(v, Ref) else v,
                )
                for key, value in c["args"].items()
            }
            a = p.get("Input", p.get("InputA"))
            if name.startswith("_symbol_"):
                y = getattr(a, name[len("_symbol_") :])(p["other"])
                if y is NotImplemented:
                    raise ValueError("unsupported Torch symbolic overload")
            elif name in ("Placeholder", "Placeholder2"):
                y = x
            elif name == "Convolution2":
                padding = self.external_padding[str(index)]
                y = self.ops[str(index)](
                    F.pad(a, padding) if padding is not None else a
                )
            elif name == "DeConvolution2":
                y = self.ops[str(index)](a)
                t, b, l, r = pad4(p["pad"])
                y = y[..., t : y.shape[-2] - b, l : y.shape[-1] - r]
            elif name in ("InnerProduct2", "MatMul", "MatMul2"):
                if str(index) in self.ops:
                    m = self.ops[str(index)]
                    y = m(a.reshape(-1, m.in_features)).reshape(shape)
                else:
                    left = a.transpose(-2, -1) if p.get("transA") else a
                    right = (
                        p["InputB"].transpose(-2, -1)
                        if p.get("transB")
                        else p["InputB"]
                    )
                    y = torch.matmul(left, right)
                    bias = p.get("Bias")
                    if bias is not None:
                        if p.get("bias2Row"):
                            bias = (
                                bias.reshape(-1, 1)
                                if bias.numel() == y.shape[-2]
                                else bias.reshape(*y.shape[:-1], 1)
                            )
                        y = y + bias
            elif name == "Scale2":
                w = p["weight"] if p["weight"] is not None else p["bias"]
                s = [1] * a.ndim
                if w.numel() == a.shape[1]:
                    s[1] = w.numel()
                elif w.numel() != 1:
                    s = [1] + list(a.shape[1:])
                y = a + w.reshape(s) if p["weight"] is None else a * w.reshape(s)
                if p["weight"] is not None and p["bias"] is not None:
                    y = y + p["bias"].reshape(s)
            elif name == "GlobalAvePooling":
                y = a.mean((-2, -1), keepdim=True)
            elif name in ("MaxPooling", "AvePooling"):
                t, b, l, r = pad4(p["pad"])
                if t != b or l != r:
                    raise ValueError("Torch observer requires symmetric pool padding")
                y = (F.max_pool2d if name == "MaxPooling" else F.avg_pool2d)(
                    a,
                    pair(p["kernel"]),
                    (pair(p["stride"])),
                    (t, l),
                    ceil_mode=p["ceilmode"],
                )
            elif name in (
                "Flatten",
                "Flatten2Matrix",
                "Reshape",
                "Reshape2",
                "Squeeze",
                "ExpandDims",
            ):
                y = a.reshape(shape)
            elif name in ("Expand", "BroadCast", "BroadCast2"):
                y = a.expand(shape).contiguous()
            elif name == "Concat":
                # SDK Concat extends shorter ranks with trailing singleton dims.
                inputs = [
                    v.reshape(*v.shape, *([1] * (len(shape) - v.ndim)))
                    for v in p["Inputs"]
                ]
                y = torch.cat(inputs, p["axis"])
            elif name in ("Split", "Slice"):
                y = list(
                    torch.split(
                        a,
                        [self.graph.meta[o]["shape"][p["axis"]] for o in outs],
                        p["axis"],
                    )
                )
            elif name in ("NearestReSize1", "NearestReSize2", "NearestReSize3"):
                y = F.interpolate(a, size=shape[-2:], mode="nearest")
            elif name in ("Gather", "Gather2"):
                indices = torch.as_tensor(
                    p["indices"], device=a.device, dtype=torch.int64
                )
                axis = p["axis"] % a.ndim
                indices = torch.where(indices < 0, indices + a.shape[axis], indices)
                y = torch.index_select(a, axis, indices.reshape(-1)).reshape(shape)
            elif name == "Range":
                start, limit, step = (p[k].item() for k in ("start", "limit", "step"))
                y = torch.arange(
                    start, limit, step, dtype=p["start"].dtype, device=p["start"].device
                ).reshape(shape)
            elif name == "ConstantOfShape":
                from TFDL2 import TFDataType as D

                dtype = {
                    D.TFDL_FLOAT: torch.float32,
                    D.TFDL_FLOAT16: torch.float16,
                    D.TFDL_INT32: torch.int32,
                    D.TFDL_INT64: torch.int64,
                }[p["dtype"]]
                y = torch.full(
                    shape, np.asarray(p["value"]).item(), dtype=dtype, device=a.device
                )
            elif name == "Crop2":
                starts = p["start"].reshape(-1).tolist()
                ends = p["end"].reshape(-1).tolist()
                axes = (
                    list(range(len(starts)))
                    if p["axis"] is None
                    else p["axis"].reshape(-1).tolist()
                )
                steps = (
                    [1] * len(starts)
                    if p["step"] is None
                    else p["step"].reshape(-1).tolist()
                )
                slices = [slice(None)] * a.ndim
                for start, end, axis, step in zip(starts, ends, axes, steps):
                    axis = int(axis) % a.ndim
                    if step <= 0:
                        raise ValueError("Torch observer requires positive Crop2 steps")
                    # Crop2's SDK negative ends are inclusive; positive ends
                    # are exclusive, and INT64_MAX means the entire tail.
                    stop = (
                        None
                        if end == np.iinfo(np.int64).max
                        else (int(end) if end >= 0 else a.shape[axis] + int(end) + 1)
                    )
                    slices[axis] = slice(int(start), stop, int(step))
                y = a[tuple(slices)]
            elif name == "Crop3":
                slices = [slice(None)] * a.ndim
                for i, (start, end, step) in enumerate(
                    p["crops"], p["startAxis"] % a.ndim
                ):
                    if step <= 0:
                        raise ValueError("Torch observer requires positive Crop3 steps")
                    # SDK Crop3 uses an inclusive end; negative ends count from the tail.
                    stop = end + a.shape[i] if end < 0 else end
                    slices[i] = slice(start, stop + 1, step)
                y = a[tuple(slices)]
            elif name in ("Add", "Sub", "Mul", "Div", "Max", "Min"):
                b = p["InputB"]
                f = {
                    "Add": torch.add,
                    "Sub": torch.sub,
                    "Mul": torch.mul,
                    "Div": torch.div,
                    "Max": torch.maximum,
                    "Min": torch.minimum,
                }[name]
                if name in ("Max", "Min"):
                    b = torch.as_tensor(b, dtype=a.dtype, device=a.device)
                y = f(p["InputA"], b)
            elif name in ("ReduceMean", "Mean", "ReduceSum", "ReduceMax", "ReduceMin"):
                f = {
                    "ReduceMean": torch.mean,
                    "Mean": torch.mean,
                    "ReduceSum": torch.sum,
                    "ReduceMax": torch.amax,
                    "ReduceMin": torch.amin,
                }[name]
                y = f(a, tuple(p["dims"]), keepdim=p["keep_dims"])
            elif name == "Transpose":
                y = a.permute(p["dims"])
            elif name == "Cast":
                from TFDL2 import TFDataType as D

                y = a.to(
                    {
                        D.TFDL_FLOAT: torch.float32,
                        D.TFDL_FLOAT16: torch.float16,
                        D.TFDL_INT32: torch.int32,
                        D.TFDL_INT64: torch.int64,
                    }[p["dstType"]]
                )
            elif name == "Shape":
                from TFDL2 import TFDataType as D

                y = torch.tensor(
                    a.shape,
                    device=a.device,
                    dtype=(
                        torch.int64
                        if self.graph.meta[outs[0]]["dtype"] == D.TFDL_INT64
                        else torch.int32
                    ),
                ).reshape(shape)
            elif name == "ReLU":
                y = F.relu(a)
            elif name == "Sigmoid":
                y = torch.sigmoid(a)
            elif name == "Swish":
                y = F.silu(a)
            elif name == "Tanh":
                y = torch.tanh(a)
            elif name == "Softmax":
                y = F.softmax(a, p["axis"])
            elif name == "GeLU":
                y = F.gelu(a)
            elif name in ("HardSigmoid", "HardSigmoid2"):
                y = torch.clamp(
                    a * p.get("alpha", 1.0 / 6.0) + p.get("beta", 0.5), 0, 1
                )
            results = y if isinstance(y, list) else [y]
            for out, value in zip(outs, results):
                if precision is not None:
                    value = precision.output(out, value)
                if tuple(value.shape) != tuple(self.graph.meta[out]["shape"]):
                    raise ValueError("Torch/SDK shape mismatch at " + out)
                values[out] = value
                if observe:
                    observe(out, value)
            for expired in self.release_after[index]:
                values.pop(expired, None)
        return [values[n] for n in self.graph.outputs]


@contextmanager
def cuda_fp32():
    """Avoid ambient AMP/TF32 during calibration, then restore caller settings."""
    matmul = torch.backends.cuda.matmul
    cudnn = torch.backends.cudnn
    if hasattr(matmul, "fp32_precision") and hasattr(cudnn, "conv"):
        settings = [
            (matmul, "fp32_precision", "ieee"),
            (cudnn.conv, "fp32_precision", "ieee"),
        ]
    else:
        # PyTorch before 2.9; do not mix legacy and new precision controls.
        settings = [(matmul, "allow_tf32", False), (cudnn, "allow_tf32", False)]
    previous = []
    try:
        for owner, attribute, value in settings:
            previous.append((owner, attribute, getattr(owner, attribute)))
            setattr(owner, attribute, value)
        with torch.autocast(device_type="cuda", enabled=False):
            yield
    finally:
        for owner, attribute, value in reversed(previous):
            setattr(owner, attribute, value)


def collect(
    graph,
    images,
    backend,
    threads=8,
    spatial="native",
    chip=0,
    cuda_device=0,
    *,
    workers=1,
    workdir=None,
    images_path=None,
    executor=None,
    compile_model=False,
):
    if executor is None:
        executor = "threads" if backend == "tfllm-npu" else "processes"
    if type(workers) is not int or not 1 <= workers <= 16:
        raise ValueError("calibration workers must be in [1,16]")
    if workers != 1 and backend != "tfllm-npu":
        raise ValueError("multiple calibration workers currently require tfllm-npu")
    if executor not in ("processes", "threads"):
        raise ValueError("calibration executor must be processes or threads")
    if compile_model and (backend != "tfllm-npu" or executor != "threads"):
        raise ValueError("compiled calibration requires tfllm-npu with the threads executor")
    if executor == "threads" and backend != "tfllm-npu":
        raise ValueError("the threads calibration executor requires tfllm-npu")
    started = time.perf_counter()
    if executor == "threads" and backend == "tfllm-npu":
        from .threaded_collect import collect_threaded

        ranges, report = collect_threaded(
            graph, images, threads, spatial, chip, workers, workdir, compile_model
        )
    elif workers > 1:
        from .parallel_collect import collect_parallel

        ranges, report = collect_parallel(
            graph, images, threads, spatial, chip, workers, workdir, images_path
        )
    else:
        ranges, report = _collect_single(
            graph, images, backend, threads, spatial, chip, cuda_device
        )
        report.update(workers=1, workers_requested=1, threads=threads)
        report["execution_seconds"] = report["seconds"]
    # Include model setup, subprocess startup/merge and runtime cleanup. The
    # parent measures elapsed wall time, not the sum of overlapping workers.
    report["seconds"] = time.perf_counter() - started
    report["timing_scope"] = (
        "whole collector including setup, worker startup/merge and cleanup; excludes image preprocessing and SDK Quantize"
    )
    return ranges, report


def _collect_single(graph, images, backend, threads, spatial, chip, cuda_device):
    if backend not in ("torch-cpu", "torch-cuda", "tfllm-npu"):
        raise ValueError("unknown Torch calibration backend: " + backend)
    device = torch.device("cpu")
    if backend == "torch-cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "torch-cuda requires CUDA-enabled PyTorch and an available NVIDIA GPU"
            )
        if not 0 <= cuda_device < torch.cuda.device_count():
            raise ValueError("cuda-device is outside the visible CUDA device count")
        device = torch.device("cuda", cuda_device)
    previous_threads = torch.get_num_threads()
    runtime = None
    model = None
    observations = {}

    def observe(name, value):
        # Scalars stay on the execution device. One host transfer at the end
        # avoids synchronizing CUDA at every observed layer.
        bounds = torch.stack(
            (value.amin(), value.amax())
            if value.dtype == torch.bool
            else torch.aminmax(value)
        ).to(torch.float64)
        if name in observations:
            previous = observations[name]
            bounds = torch.stack(
                (
                    torch.minimum(previous[0], bounds[0]),
                    torch.maximum(previous[1], bounds[1]),
                )
            )
        observations[name] = bounds

    try:
        torch.set_num_threads(threads)
        model = TorchGraph(graph)
        started = time.perf_counter()
        if backend == "tfllm-npu":
            from tfllm.torch import NpuRuntime, optimize

            runtime = NpuRuntime(
                chip=chip,
                backend="npu",
                memory_fraction=0.125,
                spatial_convolution=spatial,
            )
            optimize(
                model,
                runtime=runtime,
                strict=False,
                operators=("conv2d", "linear"),
                precision="auto",
            )
        with (
            torch.cuda.device(device) if device.type == "cuda" else nullcontext(),
            cuda_fp32() if device.type == "cuda" else nullcontext(),
            torch.inference_mode(),
        ):
            model.to(device)
            for i, image in enumerate(images):
                model(torch.from_numpy(image[None].copy()).to(device), observe)
                print(f"{backend}: calibrated {i+1}/{len(images)}", flush=True)
            if not observations:
                raise ValueError("no calibration observations")
            names = list(observations)
            bounds = torch.stack([observations[n] for n in names]).cpu().numpy()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
        invalid = ~np.isfinite(bounds).all(axis=1)
        if invalid.any():
            raise ValueError(
                "nonfinite observation: " + names[int(np.flatnonzero(invalid)[0])]
            )
        ranges = {n: (float(lo), float(hi)) for n, (lo, hi) in zip(names, bounds)}
        report = dict(
            backend=backend,
            device=str(device),
            torch_version=torch.__version__,
            approximate=backend == "tfllm-npu",
            samples=len(images),
            tensors=len(ranges),
            seconds=time.perf_counter() - started,
        )
        if device.type == "cuda":
            report.update(
                device_name=torch.cuda.get_device_name(device),
                cuda_version=torch.version.cuda,
                precision="fp32",
                autocast=False,
                tf32=False,
            )
        if runtime is not None:
            from tfllm.torch import optimization_report

            report["coverage"] = optimization_report(model).to_dict()
            library = Path(runtime._lib._name)
            report["runtime_library_sha256"] = sha256(library)
            if not any(
                c["status"] == "npu" and c["count"] > 0
                for c in report["coverage"]["calls"]
            ):
                raise RuntimeError("no actual NPU calibration calls")
        return ranges, report
    finally:
        try:
            if runtime is not None:
                from tfllm.torch import restore

                try:
                    restore(model)
                finally:
                    runtime.close()
        finally:
            torch.set_num_threads(previous_threads)
