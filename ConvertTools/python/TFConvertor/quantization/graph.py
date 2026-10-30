"""Replay the SDK's own Op calls; the SDK owns weights, Q/DQ and FB encoding."""

from dataclasses import dataclass
from pathlib import Path
import hashlib
import time
import numpy as np
from TFDL2 import (
    TFContext,
    TFCalibration,
    CalibrationMode,
    TFDataType as DT,
    TFSymbol,
    TFExecutor,
    Op,
)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@dataclass(frozen=True)
class Ref:
    name: str


@dataclass(frozen=True)
class ContextRef:
    pass


def tree(value, function):
    if isinstance(value, tuple):
        return tuple(tree(v, function) for v in value)
    if isinstance(value, list):
        return [tree(v, function) for v in value]
    if isinstance(value, dict):
        return {k: tree(v, function) for k, v in value.items()}
    return function(value)


def refs(value):
    result = []
    tree(value, lambda v: result.append(v.name) if isinstance(v, Ref) else None)
    return result


CPU = dict(UseHardware=False, FrugalMode=False, cpuLimit=8, optimize={})


class Graph:
    def __init__(self, converter, threads=8):
        if not hasattr(converter._context, "_GetTensorByName") or not hasattr(
            Op, "capture_operations"
        ):
            raise RuntimeError(
                "Rebuild/install this SDK Python package: cd Python && pip install ."
            )
        if not getattr(converter, "_op_calls", None):
            raise ValueError("buildTFmodel must run with SDK Op recording enabled")
        self.context = converter._context
        self.outputs = [str(v) for v in self.context.GetOutSymbols()]
        self.calls = []

        def encode(v):
            if isinstance(v, TFSymbol):
                return Ref(str(v))
            if isinstance(v, TFContext):
                return ContextRef()
            return v

        for name, arguments, result in converter._op_calls:
            if name in {"Convolution", "InnerProduct", "Scale", "MatMulWeight", "LSTM"}:
                raise ValueError(
                    "automatic quantization requires explicit-weight SDK Op: " + name
                )
            self.calls.append(
                dict(
                    op=name,
                    args=tree(arguments, encode),
                    outputs=refs(tree(result, encode)),
                )
            )
        # Copy original weights before any SDK quantization modifies its storage.
        used = set(refs([c["args"] for c in self.calls])) & set(converter.params)
        self.params = {n: self.context.GetParam(n).toNumpy().copy() for n in used}
        self.cpu = dict(CPU, cpuLimit=threads)
        self.calibration = TFCalibration(self.context, CalibrationMode.Naive, self.cpu)
        self.meta = {
            n: dict(
                shape=list(self.context.GetTensorByName(n).shape),
                dtype=self.context.GetTensorByName(n).dtype,
            )
            for n in self.context.GetAllTensorNames()
        }
        covered = {n for c in self.calls for n in c["outputs"]}
        if set(self.meta) - covered:
            raise ValueError(
                "SDK preparation changed recorded graph; unsupported hidden tensors: "
                + str(sorted(set(self.meta) - covered)[:5])
            )
        self.placeholders = [
            c for c in self.calls if c["op"] in ("Placeholder", "Placeholder2")
        ]
        if len(self.placeholders) != 1:
            raise ValueError("image calibration requires one input")
        c = self.placeholders[0]
        self.input_name = c["outputs"][0]
        p = c["args"]
        shape = self.meta[self.input_name]["shape"]
        if (
            len(shape) != 4
            or shape[0] != 1
            or shape[1] not in (1, 3)
            or min(shape) <= 0
            or self.meta[self.input_name]["dtype"] != DT.TFDL_FLOAT
        ):
            raise ValueError("requires fixed FP32 NCHW batch1 input, C=1 or 3")

        def vector(key, default):
            a = np.asarray(p.get(key) or [default], np.float32)
            if a.size == 1:
                a = np.repeat(a, shape[1])
            if a.size != shape[1] or not np.isfinite(a).all():
                raise ValueError("invalid input " + key)
            return a.reshape(1, -1, 1, 1)

        self.shape = shape
        self.mean = vector("mean", 0)
        self.scale = vector("scale", 1)
        if np.any(self.scale == 0):
            raise ValueError("zero input scale")
        self.units = [
            c["outputs"][0]
            for c in self.calls
            if c not in self.placeholders
            and any(self.meta[o]["dtype"] == DT.TFDL_FLOAT for o in c["outputs"])
        ]
        safe_uint8 = {
            "Convolution2",
            "DeConvolution2",
            "InnerProduct2",
            "MatMul",
            "MatMul2",
            "ReLU",
            "ReLUX",
            "LeakyReLU",
            "PReLU",
            "Sigmoid",
            "HardSigmoid",
            "HardSigmoid2",
            "Tanh",
            "Softmax",
            "GeLU",
            "Swish",
            "HardSwish",
            "Scale2",
            "GlobalAvePooling",
            "MaxPooling",
            "AvePooling",
            "TFMaxPooling",
            "TFAvePooling",
            "Flatten",
            "Flatten2Matrix",
            "Reshape",
            "Reshape2",
            "Squeeze",
            "Expand",
            "ExpandDims",
            "BroadCast",
            "BroadCast2",
            "Transpose",
            "Pad",
            "Pad2",
            "Concat",
            "Split",
            "Add",
            "Sub",
            "Mul",
            "Div",
            "Max",
            "Min",
            "ReduceMean",
            "ReduceMax",
            "ReduceMin",
            "ReduceSum",
            "Mean",
            "Cast",
            "Crop",
            "Crop2",
            "Crop3",
        }
        self.forced_float = {
            c["outputs"][0]
            for c in self.calls
            if c["outputs"][0] in self.units
            and c["op"] not in safe_uint8
            and not c["op"].startswith("_symbol_")
        }
        self.ranges = {}
        self.weights = {}
        self.weight_qparams = {}

    def raw(self, image):
        return np.ascontiguousarray(
            image[None] / self.scale + self.mean, dtype=np.float32
        )

    def collect_cpu(self, images):
        input_tensor = self.calibration.GetInputs()[0]
        for image in images:
            input_tensor.fromNumpy(self.raw(image))
            self.calibration()
        ranges = {}
        for name, meta in self.meta.items():
            tensor = self.context.GetTensorByName(name)
            if tensor.qmin and tensor.qmax:
                ranges[name] = (float(tensor.qmin[0]), float(tensor.qmax[0]))
            elif meta["dtype"] != DT.TFDL_FLOAT:
                # Shape-forward integer tensors are prepared by the SDK but
                # are not scheduled calibration commands.
                value = tensor.toNumpy()
                ranges[name] = (float(value.min()), float(value.max()))
            else:
                raise ValueError("SDK observer did not collect " + name)
        return ranges

    def register_ranges(self, ranges):
        """Supply observer statistics; leave normal graph quantization to the SDK."""
        if set(ranges) != set(self.meta):
            raise ValueError("observer tensor coverage differs from SDK graph")
        self.ranges = {}
        for name, (lo, hi) in ranges.items():
            lo, hi = float(lo), float(hi)
            if not np.isfinite([lo, hi]).all() or lo > hi:
                raise ValueError("invalid observer range: " + name)
            self.ranges[name] = (lo, hi)
            self.context.AddInt8Config(name, hi, lo)

    def set_ranges(self, ranges, merge_concat=True):
        self.ranges = {n: (float(lo), float(hi)) for n, (lo, hi) in ranges.items()}
        if set(self.ranges) != set(self.meta):
            raise ValueError("observer tensor coverage differs from SDK graph")
        parent = {n: n for n in self.ranges}

        def find(n):
            while parent[n] != n:
                parent[n] = parent[parent[n]]
                n = parent[n]
            return n

        views = {
            "GlobalAvePooling",
            "MaxPooling",
            "AvePooling",
            "TFMaxPooling",
            "TFAvePooling",
            "Flatten",
            "Flatten2Matrix",
            "Reshape",
            "Reshape2",
            "Squeeze",
            "Expand",
            "ExpandDims",
            "BroadCast",
            "BroadCast2",
            "Transpose",
            "Pad",
            "Pad2",
            "Crop",
            "Crop2",
            "Crop3",
            "Cast",
            "ReduceMax",
            "ReduceMin",
            "ReduceMean",
            "Mean",
            "_symbol___getitem__",
        }
        for c in self.calls:
            if c["op"] in ("Sigmoid", "HardSigmoid", "HardSigmoid2", "Softmax"):
                for o in c["outputs"]:
                    self.ranges[o] = (0.0, 1.0)
            elif c["op"] == "Tanh":
                for o in c["outputs"]:
                    self.ranges[o] = (-1.0, 1.0)
            inputs = [n for n in refs(c["args"]) if n in parent]
            if inputs and (
                c["op"] in views or (merge_concat and c["op"] in ("Concat", "Split"))
            ):
                for n in (inputs if c["op"] == "Concat" else inputs[:1]) + c["outputs"]:
                    parent[find(n)] = find(inputs[0])
        groups = {}
        for n in parent:
            groups.setdefault(find(n), []).append(n)
        for group in groups.values():
            lo = min(self.ranges[n][0] for n in group)
            hi = max(self.ranges[n][1] for n in group)
            if not np.isfinite([lo, hi]).all() or lo > hi:
                raise ValueError("invalid observer range")
            if lo == hi:
                hi = lo + max(abs(lo) * 1e-6, 1e-10)
            for n in group:
                self.ranges[n] = (lo, hi)
        for name, (lo, hi) in self.ranges.items():
            self.context.AddInt8Config(name, hi, lo)
        self.calibration.QuantizeLite(
            {self.input_name: DT.TFDL_UINT8}, MergeConcate=merge_concat
        )
        # Read the SDK's final scalar activation scales and per-channel weights.
        for n in self.meta:
            t = self.context.GetTensorByName(n)
            if t.qscale and t.qzeropoint:
                scale = float(t.qscale[0])
                zero = int(t.qzeropoint[0])
                self.ranges[n] = (-zero * scale, (255 - zero) * scale)
        for group in groups.values():
            lo = min(self.ranges[n][0] for n in group)
            hi = max(self.ranges[n][1] for n in group)
            for n in group:
                self.ranges[n] = (lo, hi)
        for name in self.params:
            t = self.context.GetParam(name)
            if t.dtype == DT.TFDL_UINT8:
                self.weights[name] = (t.toNumpy().copy(), list(t.qmax), list(t.qmin))
                self.weight_qparams[name] = (list(t.qscale), list(t.qzeropoint))

    def build(self, selected=(), quantized=True, input_type=DT.TFDL_UINT8):
        selected = set(selected) | self.forced_float
        if selected - set(self.units):
            raise ValueError("unknown precision unit")
        context = TFContext("automatic")
        symbols = {}
        types = {}
        conversions = {}
        layers = []
        registered = {}
        configs = {}
        names = set(self.params)

        def private_name(prefix):
            name = prefix
            index = 0
            while name in names:
                index += 1
                name = prefix + "_" + str(index)
            names.add(name)
            return name

        def parameter(name, floating):
            key = (name, not floating and name in self.weights)
            if key not in registered:
                dest = private_name(name + "__sdk_uint8") if key[1] else name
                if key[1]:
                    data, hi, lo = self.weights[name]
                    context.RegisterQuantizedParamToContext(dest, data, hi, lo)
                else:
                    context.RegisterParamToContext(**{dest: self.params[name]})
                registered[key] = context.GetParamSymbol(dest)
            return registered[key]

        def range_for(symbol, original):
            configs[str(symbol)] = self.ranges[original]
            lo, hi = self.ranges[original]
            # Op constructors register their Data immediately. Make ranges
            # available to CloseContext's preparation as well as final Modify.
            context.AddInt8Config(str(symbol), hi, lo)

        with context:
            for index, c in enumerate(self.calls):
                floating = (
                    not quantized
                    or c["outputs"][0] in selected
                    or not any(
                        self.meta[o]["dtype"] == DT.TFDL_FLOAT for o in c["outputs"]
                    )
                )
                desired = DT.TFDL_FLOAT if floating else DT.TFDL_UINT8
                method = {
                    "Add": "__add__",
                    "Sub": "__sub__",
                    "Mul": "__mul__",
                    "Div": "__truediv__",
                }.get(c["op"], c["op"].removeprefix("_symbol_"))
                right = c["args"].get("other", c["args"].get("InputB"))
                left = c["args"].get("Input", c["args"].get("InputA"))

                def constant(value):
                    if isinstance(value, Ref):
                        return self.params.get(value.name)
                    return (
                        np.asarray(value)
                        if value is not None and np.isscalar(value)
                        else None
                    )

                right_constant = constant(right)
                left_constant = constant(left)
                affine = (
                    right_constant is not None
                    and right_constant.size == 1
                    and method
                    in (
                        "__mul__",
                        "__rmul__",
                        "__add__",
                        "__radd__",
                        "__sub__",
                        "__rsub__",
                        "__truediv__",
                    )
                )
                binary = method in (
                    "__add__",
                    "__sub__",
                    "__mul__",
                    "__truediv__",
                    "__radd__",
                    "__rsub__",
                    "__rmul__",
                    "__rtruediv__",
                )
                float_kernel = (
                    not floating
                    and binary
                    and not affine
                    and (left_constant is not None or right_constant is not None)
                )
                kernel_type = DT.TFDL_FLOAT if float_kernel else desired

                def resolve(v, target_type=kernel_type, weight=False):
                    if isinstance(v, ContextRef):
                        return context
                    if not isinstance(v, Ref):
                        return v
                    n = v.name
                    if not n:
                        return TFSymbol()
                    if n in self.params:
                        return parameter(n, floating or float_kernel or not weight)
                    symbol = symbols[n]
                    if (
                        quantized
                        and self.meta[n]["dtype"] == DT.TFDL_FLOAT
                        and types[n] != target_type
                    ):
                        key = (n, target_type)
                        if key not in conversions:
                            out = (
                                Op.DeQuantize(symbol)
                                if target_type == DT.TFDL_FLOAT
                                else Op.Quantize(symbol)
                            )
                            range_for(out, n)
                            conversions[key] = out
                        return conversions[key]
                    return symbol

                weight_keys = {
                    "Convolution2": {"weight"},
                    "DeConvolution2": {"weight"},
                    "InnerProduct2": {"weight"},
                    "MatMul": {"InputB"},
                    "MatMul2": {"InputB"},
                }.get(c["op"], set())
                float_keys = {
                    "RoiPool": {"Rois"},
                    "CropAndResize": {"AlignMask"},
                    "WarpWithOpFlow": {"Flow"},
                    "WarpWithAffine": {"transMat"},
                    "WarpWithPespect": {"transMat"},
                    "WarpWithGridFace": {"transMat"},
                }.get(c["op"], set())
                arguments = {
                    k: tree(
                        v,
                        lambda ref, k=k: resolve(
                            ref,
                            DT.TFDL_FLOAT if k in float_keys else kernel_type,
                            k in weight_keys,
                        ),
                    )
                    for k, v in c["args"].items()
                }
                if c in self.placeholders:
                    arguments["outDatatype"] = (
                        input_type if quantized else DT.TFDL_FLOAT
                    )
                scalar_method = method
                scalar = right_constant.item() if affine else None
                if (
                    quantized
                    and not floating
                    and np.isscalar(scalar)
                    and scalar_method
                    in (
                        "__mul__",
                        "__rmul__",
                        "__add__",
                        "__radd__",
                        "__sub__",
                        "__rsub__",
                        "__truediv__",
                    )
                ):
                    # SDK Eltwise requires matching input types. Scalar affine
                    # arithmetic is the SDK Scale operator with float constants.
                    w, b = 1.0, 0.0
                    if scalar_method in ("__mul__", "__rmul__"):
                        w = float(scalar)
                    elif scalar_method in ("__add__", "__radd__"):
                        b = float(scalar)
                    elif scalar_method == "__sub__":
                        b = -float(scalar)
                    elif scalar_method == "__rsub__":
                        w, b = -1.0, float(scalar)
                    else:
                        if scalar == 0:
                            raise ValueError("division by zero")
                        w = 1.0 / float(scalar)
                    wn = private_name(f"__sdk_scalar_w_{index}")
                    bn = private_name(f"__sdk_scalar_b_{index}")
                    context.RegisterParamToContext(
                        **{wn: np.array([w], np.float32), bn: np.array([b], np.float32)}
                    )
                    result = Op.Scale2(
                        arguments.get("Input", arguments.get("InputA")),
                        context.GetParamSymbol(wn),
                        context.GetParamSymbol(bn),
                    )
                else:
                    result = (
                        Op._symbol_methods[c["op"]](
                            arguments["Input"], arguments["other"]
                        )
                        if c["op"].startswith("_symbol_")
                        else getattr(Op, c["op"])(**arguments)
                    )
                results = (
                    list(result) if isinstance(result, (tuple, list)) else [result]
                )
                if len(results) != len(c["outputs"]):
                    raise ValueError("Op replay output count changed")
                for original, symbol in zip(c["outputs"], results):
                    if float_kernel and self.meta[original]["dtype"] == DT.TFDL_FLOAT:
                        layers.append(
                            dict(layerName=str(symbol), outputDataType="TFDtypeFp32")
                        )
                        range_for(symbol, original)
                        symbol = Op.Quantize(symbol)
                    actual = (
                        (input_type if quantized else DT.TFDL_FLOAT)
                        if c in self.placeholders
                        else (
                            desired
                            if self.meta[original]["dtype"] == DT.TFDL_FLOAT
                            else self.meta[original]["dtype"]
                        )
                    )
                    symbols[original] = symbol
                    types[original] = actual
                    if original in self.ranges:
                        range_for(symbol, original)
                    if actual in (DT.TFDL_FLOAT, DT.TFDL_UINT8):
                        layers.append(
                            dict(
                                layerName=str(symbol),
                                outputDataType=(
                                    "TFDtypeUint8"
                                    if actual == DT.TFDL_UINT8
                                    else "TFDtypeFp32"
                                ),
                            )
                        )
        context.SetOutputs([str(symbols[n]) for n in self.outputs])
        for name, (lo, hi) in configs.items():
            context.AddInt8Config(name, hi, lo)
        context.Modify(dict(Layer=layers))
        return context

    def evaluate(self, context, images, config, warmup=2):
        executor = TFExecutor(context, dict(config))
        input_tensor = executor.GetInputs()[0]
        if config.get("UseHardware"):
            from TFDL2 import TFDL2 as native

            if not hasattr(native, "GetDeviceNum") or native.GetDeviceNum() <= 0:
                raise RuntimeError(
                    "SDK did not initialize NPU hardware; check SDK Python installation/runtime library paths"
                )

        def run(image):
            if input_tensor.dtype == DT.TFDL_UINT8:
                scale = np.asarray(input_tensor.qscale, np.float32)
                zero = np.asarray(input_tensor.qzeropoint, np.float32)
                if scale.size != 1 or zero.size != 1:
                    raise ValueError("input requires per-tensor quantization")
                value = np.clip(
                    np.floor(image[None] / scale[0] + zero[0] + 0.5), 0, 255
                ).astype(np.uint8)
            else:
                value = self.raw(image)
            input_tensor.fromNumpy(value)
            tensors = executor()
            result = []
            for t in tensors:
                a = t.toNumpy()
                if t.dtype == DT.TFDL_UINT8:
                    scales = np.asarray(t.qscale, np.float32)
                    zeros = np.asarray(t.qzeropoint, np.float32)
                    if scales.size > 1:
                        if a.ndim < 2 or a.shape[1] != scales.size:
                            raise ValueError("unsupported per-channel output layout")
                        shape = (1, scales.size) + (1,) * (a.ndim - 2)
                        scales = scales.reshape(shape)
                        zeros = zeros.reshape(shape)
                    a = (a.astype(np.float32) - zeros) * scales
                if np.issubdtype(a.dtype, np.floating) and not np.isfinite(a).all():
                    raise FloatingPointError("nonfinite output")
                result.append(a)
            return result

        for i in range(warmup):
            run(images[i % len(images)])
        outputs = []
        elapsed = 0
        for image in images:
            start = time.perf_counter()
            result = run(image)
            elapsed += time.perf_counter() - start
            if not outputs:
                outputs = [[] for _ in result]
            for rows, value in zip(outputs, result):
                rows.append(value)
        return [np.stack(rows) for rows in outputs], elapsed * 1000 / len(images)
