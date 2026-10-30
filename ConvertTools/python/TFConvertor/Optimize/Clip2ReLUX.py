from ..TFConvertor import NodeConvertor,TFConvertor
import numpy as np


def Clip2RuLUXOnnx(Model:TFConvertor):
    nodes = list(Model.nodes.values())

    for index,node in enumerate(nodes):
        if node.op == "Clip":
            # opset >= 11 的 Clip 把 min/max 作为输入，而不是属性；原来这里直接
            # node.attr["min"] 会 KeyError（README 却推荐导 opset 17）。取不到就跳过。
            if "min" not in node.attr or "max" not in node.attr:
                continue
            if node.attr["min"] == 0 and node.attr["max"] > 0:
                node.op = "ReluX"




