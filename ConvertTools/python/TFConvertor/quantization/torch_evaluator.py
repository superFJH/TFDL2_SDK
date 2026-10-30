"""Reusable backend execution with SDK weight and activation Q/DQ simulation.

This estimates mixed-precision candidates, not SDK integer kernel bit patterns
or target FB latency. The exported FB must still pass the SDK export audit.
"""

import time
from contextlib import nullcontext

import numpy as np
import torch
from TFDL2 import TFDataType as DT

from .graph import Ref
from .torch_backend import TorchGraph, cuda_fp32


class MixedPrecision:
    def __init__(self, graph, model, device):
        self.graph = graph
        self.model = model
        self.quantized = set()
        self.quantized_inputs = set()
        self.qparams = {}
        for name, (lo, hi) in graph.ranges.items():
            lo, hi = min(lo, 0.0), max(hi, 0.0)
            if lo == hi:
                hi = lo + 1e-10
            scale = (hi - lo) / 255.0
            if not np.isfinite(scale) or scale <= 0:
                raise ValueError("invalid evaluation range: " + name)
            zero = np.clip(np.floor(-lo / scale + 0.5), 0, 255)
            self.qparams[name] = (
                torch.tensor(scale, dtype=torch.float32, device=device),
                torch.tensor(zero, dtype=torch.float32, device=device),
            )
        self.weights = {}
        for name, (data, _, _) in graph.weights.items():
            scales, zeros = graph.weight_qparams[name]
            scales = np.asarray(scales, np.float32)
            zeros = np.asarray(zeros, np.float32)
            if scales.size != zeros.size or scales.size not in (1, data.shape[0]):
                raise ValueError("unsupported evaluation weight quantization: " + name)
            shape = (scales.size,) + (1,) * (data.ndim - 1)
            value = (data.astype(np.float32) - zeros.reshape(shape)) * scales.reshape(shape)
            self.weights[name] = torch.from_numpy(value).to(device)
        self.module_weights = {}
        for key, module in model.ops.items():
            c = graph.calls[int(key)]
            arg = "weight" if c["op"] in ("Convolution2", "DeConvolution2", "InnerProduct2") else "InputB"
            name = c["args"][arg].name
            if name not in self.weights:
                continue

            def layout(value):
                if c["op"] == "DeConvolution2":
                    return value.flip(-2, -1).permute(1, 0, 2, 3)
                if c["op"] == "InnerProduct2":
                    return value.reshape(c["args"]["outchannels"], -1)
                if c["op"] in ("MatMul", "MatMul2") and not c["args"].get("transB"):
                    return value.T
                return value

            self.module_weights[key] = (layout(self.weights[name]), layout(model.params[name]))
        self.module_modes = {}

    def configure(self, selected, quantized, input_type):
        selected = set(selected) | self.graph.forced_float
        if selected - set(self.graph.units):
            raise ValueError("unknown precision unit")
        self.quantized = set()
        self.quantized_inputs = set()
        for c in self.graph.calls:
            if c in self.graph.placeholders:
                enabled = quantized and input_type == DT.TFDL_UINT8
            else:
                enabled = (quantized and c["outputs"][0] not in selected
                           and any(self.graph.meta[o]["dtype"] == DT.TFDL_FLOAT for o in c["outputs"]))
            if enabled:
                self.quantized.update(o for o in c["outputs"]
                                      if self.graph.meta[o]["dtype"] == DT.TFDL_FLOAT)
                # A non-scalar binary constant uses a float kernel in Graph.build.
                method = {"Add": "__add__", "Sub": "__sub__", "Mul": "__mul__",
                          "Div": "__truediv__"}.get(c["op"], c["op"].removeprefix("_symbol_"))
                right = c["args"].get("other", c["args"].get("InputB"))
                left = c["args"].get("Input", c["args"].get("InputA"))

                def constant(value):
                    return self.graph.params.get(value.name) if isinstance(value, Ref) else (
                        np.asarray(value) if value is not None and np.isscalar(value) else None)

                rc, lc = constant(right), constant(left)
                affine = rc is not None and rc.size == 1 and method in (
                    "__mul__", "__rmul__", "__add__", "__radd__", "__sub__", "__rsub__", "__truediv__")
                binary = method in ("__add__", "__sub__", "__mul__", "__truediv__",
                                    "__radd__", "__rsub__", "__rmul__", "__rtruediv__")
                if not (binary and not affine and (rc is not None or lc is not None)):
                    self.quantized_inputs.add(c["outputs"][0])
        # Update only changed modules. tfllm's versioned weight cache repacks
        # changed weights on the next warmup instead of retaining stale packs.
        with torch.no_grad():
            for key, (packed, original) in self.module_weights.items():
                enabled = self.graph.calls[int(key)]["outputs"][0] in self.quantized
                if self.module_modes.get(key, False) != enabled:
                    self.model.ops[key].weight.copy_(packed if enabled else original)
                self.module_modes[key] = enabled

    def qdq(self, name, value):
        scale, zero = self.qparams[name]
        return (torch.clamp(torch.floor(value / scale + zero + 0.5), 0, 255) - zero) * scale

    def resolve(self, call, key, ref, values):
        value = values.get(ref.name)
        if not ref.name:
            return value
        if ref.name in self.graph.params:
            weight_keys = {"Convolution2": "weight", "DeConvolution2": "weight",
                           "InnerProduct2": "weight", "MatMul": "InputB", "MatMul2": "InputB"}
            if call["outputs"][0] in self.quantized and key == weight_keys.get(call["op"]):
                return self.weights.get(ref.name, value)
        elif (call["outputs"][0] in self.quantized_inputs
              and ref.name not in self.quantized
              and self.graph.meta[ref.name]["dtype"] == DT.TFDL_FLOAT):
            return self.qdq(ref.name, value)
        return value

    def output(self, name, value):
        return self.qdq(name, value) if name in self.quantized else value


class TorchEvaluator:
    def __init__(self, graph, backend, options):
        self.backend = backend
        self.device = torch.device("cpu")
        if backend == "torch-cuda":
            index = int(options.get("cuda_device", 0))
            if not torch.cuda.is_available():
                raise RuntimeError("torch-cuda evaluation requires CUDA-enabled PyTorch and an available NVIDIA GPU")
            if not 0 <= index < torch.cuda.device_count():
                raise ValueError("cuda-device is outside the visible CUDA device count")
            self.device = torch.device("cuda", index)
        self.previous_threads = torch.get_num_threads()
        self.runtime = self.model = None
        self.closed = False
        self.report = dict(
            backend=backend, device=str(self.device), approximate=True,
            precision="SDK activation Q/DQ and dequantized SDK weights; backend floating kernels",
            timing_scope="backend execution including input/output transfers; excludes setup and warmup; not SDK FB latency",
        )
        try:
            torch.set_num_threads(int(options.get("threads", 8)))
            self.model = TorchGraph(graph).to(self.device)
            self.precision = MixedPrecision(graph, self.model, self.device)
            if backend == "tfllm-npu":
                from tfllm.torch import NpuRuntime, optimize

                self.runtime = NpuRuntime(
                    chip=int(options.get("chip", 0)), backend="npu", memory_fraction=0.125,
                    spatial_convolution=options.get("spatial", "native"),
                )
                optimize(self.model, runtime=self.runtime, strict=False,
                         operators=("conv2d", "linear"), precision="auto")
        except BaseException:
            self.close()
            raise

    def evaluate(self, images, selected=(), quantized=True, input_type=DT.TFDL_UINT8, warmup=2):
        if self.closed:
            raise RuntimeError("evaluation backend is closed")
        self.precision.configure(selected, quantized, input_type)

        def run(image):
            tensors = self.model(torch.from_numpy(image[None]).to(self.device), precision=self.precision)
            result = [t.detach().cpu().numpy().copy() for t in tensors]
            if any(np.issubdtype(a.dtype, np.floating) and not np.isfinite(a).all() for a in result):
                raise FloatingPointError("nonfinite output")
            return result

        with (
            torch.cuda.device(self.device) if self.device.type == "cuda" else nullcontext(),
            cuda_fp32() if self.device.type == "cuda" else torch.autocast("cpu", enabled=False),
            torch.inference_mode(),
        ):
            for i in range(warmup):
                run(images[i % len(images)])
            outputs = []
            elapsed = 0.0
            for image in images:
                started = time.perf_counter()
                result = run(image)
                elapsed += time.perf_counter() - started
                if not outputs:
                    outputs = [[] for _ in result]
                for rows, value in zip(outputs, result):
                    rows.append(value)
        if self.runtime is not None:
            from tfllm.torch import optimization_report

            self.report["coverage"] = optimization_report(self.model).to_dict()
            if not any(c["status"] == "npu" and c["count"] > 0 for c in self.report["coverage"]["calls"]):
                raise RuntimeError("no actual NPU evaluation calls")
        return [np.stack(rows) for rows in outputs], elapsed * 1000 / len(images)

    def close(self):
        if self.closed:
            return
        self.closed = True
        try:
            if self.runtime is not None:
                from tfllm.torch import restore

                try:
                    if self.model is not None:
                        restore(self.model)
                finally:
                    self.runtime.close()
        finally:
            torch.set_num_threads(self.previous_threads)
