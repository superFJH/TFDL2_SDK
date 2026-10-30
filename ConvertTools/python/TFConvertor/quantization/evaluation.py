"""Dispatch search evaluation independently of the SDK FB export executor."""


def sdk_config(backend, options):
    threads = int(options.get("threads", 8))
    if backend == "sdk-cpu":
        return dict(UseHardware=False, FrugalMode=False, cpuLimit=threads, optimize={})
    return dict(
        UseHardware=True,
        FrugalMode=True,
        Core=[int(options.get("chip", 0)) * 8],
        cpuLimit=threads,
        useCache=True,
        optimize=dict(
            MakeAlign=True,
            RuntimeQuantConv=True,
            RuntimeQuantMatMul=True,
            RuntimeQuantPostProcess=True,
        ),
    )


class SdkEvaluator:
    def __init__(self, graph, options):
        self.graph = graph
        self.config = sdk_config("sdk-cpu", options)
        self.report = dict(backend="sdk-cpu", device="cpu", approximate=False,
                           timing_scope="SDK CPU execution including input/output transfers; excludes build and warmup")

    def evaluate(self, images, selected=(), quantized=True, input_type=None):
        kwargs = {} if input_type is None else dict(input_type=input_type)
        context = self.graph.build(selected, quantized=quantized, **kwargs)
        return self.graph.evaluate(context, images, self.config)

    def close(self):
        pass


def create_evaluator(graph, backend, options):
    if backend == "sdk-cpu":
        return SdkEvaluator(graph, options)
    if backend not in ("torch-cpu", "torch-cuda", "tfllm-npu"):
        raise ValueError("unknown evaluation backend: " + backend)
    from .torch_evaluator import TorchEvaluator

    return TorchEvaluator(graph, backend, options)
