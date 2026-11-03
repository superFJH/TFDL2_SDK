"""Extract explicit floating islands from an ONNX Q/DQ graph.

This is a conservative precision hint for the SDK graph, not a promise that
the SDK will reproduce another runtime's quantized kernels or qparams.  A
weighted operation needs quantized activation and weight inputs to form an
integer kernel.  When either is missing, keep the connected unquantized
island floating up to its Q/DQ boundaries.
"""

from collections import defaultdict


_WEIGHTED = {"Conv", "ConvTranspose", "Gemm", "MatMul"}
_BOUNDARIES = {"QuantizeLinear", "DequantizeLinear"}


def float_islands_from_qdq(qdq_model, source_model):
    """Return a JSON-ready precision policy for a matching source ONNX graph.

    NNCF and ORT keep the source operator outputs while adding Q/DQ nodes.  A
    mismatch is rejected so that a renamed or otherwise rewritten graph cannot
    silently apply a precision decision to the wrong SDK tensor.
    """
    source_outputs = {
        name for node in source_model.graph.node for name in node.output if name
    }
    producers = {
        name: node for node in qdq_model.graph.node
        for name in node.output if name
    }
    consumers = defaultdict(list)
    for node in qdq_model.graph.node:
        for name in node.input:
            if name:
                consumers[name].append(node)

    seeds = []
    for node in qdq_model.graph.node:
        if node.op_type not in _WEIGHTED or len(node.input) < 2:
            continue
        data_quantized = (
            producers.get(node.input[0]) is not None
            and producers[node.input[0]].op_type == "DequantizeLinear"
        )
        weight_quantized = (
            producers.get(node.input[1]) is not None
            and producers[node.input[1]].op_type == "DequantizeLinear"
        )
        if not (data_quantized and weight_quantized):
            seeds.append(node)

    floating = set()
    backward_seen = set()
    forward_seen = set()

    def backward(name):
        if name in backward_seen:
            return
        backward_seen.add(name)
        producer = producers.get(name)
        if producer is None or producer.op_type in _BOUNDARIES:
            return
        floating.add(name)
        for input_name in producer.input:
            backward(input_name)

    def forward(name):
        if name in forward_seen:
            return
        forward_seen.add(name)
        for consumer in consumers[name]:
            if consumer.op_type in _BOUNDARIES:
                continue
            for output_name in consumer.output:
                if output_name:
                    floating.add(output_name)
                    forward(output_name)

    for node in seeds:
        backward(node.input[0])
        for output_name in node.output:
            if output_name:
                floating.add(output_name)
                forward(output_name)

    missing = sorted(floating - source_outputs)
    if missing:
        raise ValueError(
            "Q/DQ float island outputs are absent from source ONNX: "
            + ", ".join(missing[:5])
        )
    return {
        "format": "onnx-qdq-float-islands-v1",
        "selection": "weighted inputs without matching activation/weight DQ; trace to Q/DQ boundaries",
        "seed_nodes": [
            {"name": node.name, "op": node.op_type, "outputs": list(node.output)}
            for node in seeds
        ],
        "float_onnx_outputs": sorted(floating),
    }
