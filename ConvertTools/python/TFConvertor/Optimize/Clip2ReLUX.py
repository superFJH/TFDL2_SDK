from ..TFConvertor import NodeConvertor,TFConvertor
import numpy as np


def Clip2RuLUXOnnx(Model:TFConvertor):
    nodes = list(Model.nodes.values())

    for index,node in enumerate(nodes):
        if node.op == "Clip":
            if node.attr["min"] == 0 and node.attr["max"] > 0:
                node.op = "ReluX"




