"""Fixed-shape, immutable inference snapshots with compiled CPU operators."""
import time
import threading

import torch
from torch.fx.experimental.proxy_tensor import make_fx


def _metadata(inputs):
    if not inputs or any(not isinstance(x, torch.Tensor) for x in inputs):
        raise TypeError("snapshot inputs must be positional Tensors")
    return tuple((tuple(x.shape), tuple(x.stride()), x.dtype, x.device) for x in inputs)


def _weights(model):
    try:
        return tuple(
            (name, id(value), value.data_ptr(), value._version, tuple(value.shape), value.dtype, value.device)
            for name, value in (*model.named_parameters(), *model.named_buffers())
        )
    except RuntimeError as error:
        raise ValueError("load snapshot weights outside inference_mode so mutations can be detected") from error


def _configuration(model):
    def value(v):
        if v is None or type(v) in (bool, int, float, str):
            return v
        if isinstance(v, tuple):
            return tuple(value(item) for item in v)
        return (type(v), id(v))
    return tuple(
        (name, id(module), tuple((key, value(v)) for key, v in module.__dict__.items()
                                if not key.startswith("_")))
        for name, module in model.named_modules()
    )


def _check_data_dependencies(graph):
    dependencies = {}
    native = []
    for node in graph.nodes:
        sources = []
        torch.fx.node.map_arg((node.args, node.kwargs), lambda v: sources.append(v))
        depends = node.op == "placeholder" or any(dependencies.get(v, False) for v in sources)
        dependencies[node] = depends
        if "_local_scalar_dense" in str(node.target) and depends:
            raise ValueError("snapshot cannot freeze input-dependent scalar values or Python branches")
        if node.op == "call_function" and str(node.target).startswith("tfllm."):
            native.append(str(node.target))
    if not native:
        raise ValueError("snapshot captured no TFLLM native operators")
    return native


class CompiledSnapshot:
    """Owns a read-only inference capture, not the caller's runtime.

    Shape/dtype/stride changes, weight reloads, restore(), and training mode
    invalidate this capture. Rebuild explicitly; never run stale packed weights.
    Python configuration/state is frozen, so this API is for pure eval models.
    """
    def __init__(self, model, example_inputs, *, options=None):
        if torch.is_grad_enabled():
            raise RuntimeError("create snapshots inside no_grad() or inference_mode()")
        if any(module.training for module in model.modules()):
            raise ValueError("snapshot requires eval() on the complete model")
        self.model = model
        self.inputs = _metadata(example_inputs)
        self.weights = _weights(model)
        self.configuration = _configuration(model)
        self.states = tuple(
            (module, module._tfllm_optimization)
            for module in model.modules() if hasattr(module, "_tfllm_optimization")
        )
        if not self.states:
            raise ValueError("optimize the model with a TFLLM runtime before creating a snapshot")
        self.closed = False
        self.calls = 0
        self._condition = threading.Condition()
        self._active = 0
        self._closing = False
        start = time.perf_counter()
        # Populate the adapter's immutable packed-weight cache before tracing;
        # packing/cloning is setup work and must not enter the captured graph.
        model(*example_inputs)
        warmup_seconds = time.perf_counter() - start
        start = time.perf_counter()
        # The SDK's shape subgraphs may extract constants with item()/tolist().
        # Inspect the unoptimized capture and reject every extraction that
        # depends on input data, before an immutable graph can be published.
        graph = make_fx(model, _error_on_data_dependent_ops=False)(*example_inputs)
        native = _check_data_dependencies(graph.graph)
        capture_seconds = time.perf_counter() - start
        start = time.perf_counter()
        self.compiled = torch.compile(
            graph, backend="inductor", fullgraph=True, dynamic=False, options=options
        )
        self.compiled(*example_inputs)
        if _configuration(model) != self.configuration or _weights(model) != self.weights:
            raise ValueError("snapshot capture requires a pure eval model with immutable configuration and weights")
        self.report = dict(
            torch_version=torch.__version__, warmup_seconds=warmup_seconds,
            capture_seconds=capture_seconds,
            compile_and_first_call_seconds=time.perf_counter()-start,
            captured_native_operators=len(native), native_targets=sorted(set(native)),
            precision="preserves explicit input/weight dtypes; compiler fusion may change floating rounding",
            scope="immutable eval model; fixed input metadata; CPU Inductor plus TFLLM custom operators",
        )

    def __call__(self, *inputs):
        with self._condition:
            if self.closed or self._closing:
                raise RuntimeError("snapshot is closed")
            self._active += 1
        try:
            return self._run(inputs)
        finally:
            with self._condition:
                self._active -= 1
                self._condition.notify_all()

    def _run(self, inputs):
        if torch.is_grad_enabled():
            raise RuntimeError("snapshots are inference-only; use no_grad() or inference_mode()")
        if _metadata(inputs) != self.inputs:
            raise ValueError("snapshot input shape, stride, dtype or device changed; rebuild the snapshot")
        if any(module.training for module in self.model.modules()):
            raise RuntimeError("snapshot model entered training mode")
        if any(getattr(module, "_tfllm_optimization", None) is not state for module, state in self.states):
            raise RuntimeError("TFLLM model adaptation changed; rebuild the snapshot")
        if _configuration(self.model) != self.configuration:
            raise RuntimeError("snapshot module configuration changed; rebuild before inference")
        if _weights(self.model) != self.weights:
            raise RuntimeError("snapshot weights changed; rebuild before inference")
        result = self.compiled(*inputs)
        with self._condition:
            self.calls += 1
        return result

    def close(self):
        # The user still owns optimize()/restore() and the shared NpuRuntime.
        with self._condition:
            self._closing = True
            self._condition.wait_for(lambda: self._active == 0)
            self.closed = True
            self.compiled = None
            self.model = None
            self.states = ()


def compile_snapshot(model, example_inputs, *, options=None):
    """Compile a fixed-shape, immutable TFLLM eval model (tested on Torch 2.14).

    One-time capture/compilation is charged separately from steady inference.
    Use the original eager model for unsupported/data-dependent graphs. This
    function raises on unsupported captures or compiler/device errors.
    """
    if isinstance(example_inputs, torch.Tensor):
        example_inputs = (example_inputs,)
    return CompiledSnapshot(model, tuple(example_inputs), options=options)
