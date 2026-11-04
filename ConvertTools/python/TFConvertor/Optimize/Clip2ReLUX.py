from ..TFConvertor import TFConvertor
import numpy as np


def Clip2RuLUXOnnx(Model:TFConvertor):
    """Normalize Clip(0, positive constant) in both ONNX bound encodings."""
    count = 0
    for node in Model.nodes.values():
        if node.op != "Clip":
            continue
        bounds = []
        for index, key in [(1, "min"), (2, "max")]:
            if len(node.inputs) > index and node.inputs[index]:
                value = Model.params.get(node.inputs[index])
            else:
                value = node.attr.get(key)
            if value is None or np.asarray(value).size != 1:
                break
            bounds.append(float(np.asarray(value).reshape(-1)[0]))
        if len(bounds) != 2 or bounds[0] != 0 or not np.isfinite(bounds[1]) or bounds[1] <= 0:
            continue
        node.op = "ReluX"
        node.inputs = node.inputs[:1]
        node.attr = {"max": bounds[1]}
        count += 1
    return count
