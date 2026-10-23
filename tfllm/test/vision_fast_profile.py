"""Board-measured, inference-only SAM adapters for vision_fast_inference.py."""
import types

import torch
from torch import nn
from torch.nn import functional as F


class PatchProjection(nn.Module):
    """Express a large patch convolution as unfold plus native pointwise Conv."""
    def __init__(self, conv):
        super().__init__()
        self.kernel_size, self.stride = conv.kernel_size, conv.stride
        self.padding, self.dilation = conv.padding, conv.dilation
        channels = conv.in_channels * conv.kernel_size[0] * conv.kernel_size[1]
        self.pointwise = nn.Conv2d(channels, conv.out_channels, 1,
                                   bias=conv.bias is not None,
                                   dtype=conv.weight.dtype, device=conv.weight.device)
        # Keep the loaded values; never rewrite a checkpoint on disk.
        with torch.no_grad():
            self.pointwise.weight.copy_(conv.weight.reshape(conv.out_channels, channels, 1, 1))
            if conv.bias is not None:
                self.pointwise.bias.copy_(conv.bias)
        self.eval()

    def forward(self, x):
        h = (x.shape[-2] + 2*self.padding[0] - self.dilation[0]*(self.kernel_size[0]-1)-1)//self.stride[0]+1
        w = (x.shape[-1] + 2*self.padding[1] - self.dilation[1]*(self.kernel_size[1]-1)-1)//self.stride[1]+1
        patches = F.unfold(x, self.kernel_size, dilation=self.dilation,
                           padding=self.padding, stride=self.stride)
        return self.pointwise(patches.reshape(x.shape[0], patches.shape[1], h, w))


def prepare(model):
    """Equivalent official-model rewrites; return undo callbacks for the caller."""
    undo = []
    for parent in list(model.modules()):
        for name, child in list(parent.named_children()):
            if type(child) is nn.modules.linear.NonDynamicallyQuantizableLinear:
                plain = nn.Linear(child.in_features, child.out_features,
                                  bias=child.bias is not None,
                                  dtype=child.weight.dtype, device=child.weight.device)
                plain.weight, plain.bias = child.weight, child.bias
                plain.train(child.training)
                setattr(parent, name, plain)
                undo.append(lambda p=parent, n=name, c=child: setattr(p, n, c))
            elif (type(child) is nn.Conv2d and max(child.kernel_size) > 7
                  and child.groups == 1 and child.padding_mode == "zeros"
                  and not isinstance(child.padding, str)):
                setattr(parent, name, PatchProjection(child))
                undo.append(lambda p=parent, n=name, c=child: setattr(p, n, c))
    for module in model.modules():
        if (type(module).__module__, type(module).__name__) != ("sam3.model.vitdet", "Mlp"):
            continue
        had_forward, previous = "forward" in module.__dict__, module.forward
        def forward(this, x):
            x = this.drop1(this.act(this.fc1(x)))
            return this.drop2(this.fc2(this.norm(x)))
        module.forward = types.MethodType(forward, module)
        def restore(m=module, prev=previous, had=had_forward):
            if had:
                m.forward = prev
            else:
                m.__dict__.pop("forward", None)
        undo.append(restore)
    return undo


def guard_short_linears(model, originals, state, min_rows):
    """Keep short FP32 projections in Torch; leave actual NPU errors visible."""
    for path, module in model.named_modules():
        if type(module) is not nn.Linear or id(module) not in originals or "forward" not in module.__dict__:
            continue
        native, original = module.forward, originals[id(module)]
        def forward(x, m=module, npu=native, cpu=original, name=path):
            if (x.dtype == torch.float32 and not torch.is_autocast_enabled("cpu")
                    and x.numel()//x.shape[-1] < min_rows):
                state.report.record(name, "cpu_linear", "short FP32 projection")
                return cpu(x)
            return npu(x)
        module.forward = forward
