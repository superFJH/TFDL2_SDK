from ..TFConvertor import NodeConvertor,TFConvertor
import numpy as np

def dilate_weights(weight, dilation):
    """
    根据 dilation 参数扩展卷积核权重。

    参数:
    weight (torch.Tensor): 形状为 (out_channels, in_channels, kernel_height, kernel_width) 的卷积核权重。
    dilation (int): 空洞卷积的 dilation 参数。

    返回:
    torch.Tensor: 扩展后的卷积核权重，形状为 (out_channels, in_channels, dilated_kernel_height, dilated_kernel_width)。
    """
    if dilation == 1:
        return weight  # 如果 dilation 为 1，直接返回原权重

    out_channels, in_channels, kernel_height, kernel_width = weight.shape

    # 计算扩展后的卷积核大小
    dilated_kernel_height = (kernel_height - 1) * dilation + 1
    dilated_kernel_width = (kernel_width - 1) * dilation + 1

    # 初始化扩展后的权重张量，填充为 0
    dilated_weight = np.zeros((out_channels, in_channels, dilated_kernel_height, dilated_kernel_width),dtype=weight.dtype)

    # 将原始权重插入到扩展后的权重张量中
    dilated_weight[:, :, ::dilation, ::dilation] = weight

    return dilated_weight
def ReplaceDilationConv(Model:TFConvertor):
    nodes = list(Model.nodes.values())

    for index,node in enumerate(nodes):
        if node.op == "Conv":
            if "dilations" not in node.attr:
                continue
            dilation = node.attr["dilations"]
            if type(dilation) is list or type(dilation) is tuple:
                if dilation[0] == 1:
                    continue
            dilation = dilation[0]
            convnode = node

            if convnode.inputs[1] not in Model.params.keys():
                raise RuntimeError("Merge BatchNorm Error Can't find W")
            W = Model.params[convnode.inputs[1]]
            newW = dilate_weights(W,dilation)
            node.attr["dilations"] = [1,1]
            node.attr["kernel_shape"] = (newW.shape[-2],newW.shape[-1])


            Model.params[convnode.inputs[1]] = np.ascontiguousarray(newW)





