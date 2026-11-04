"""Optimize floating subgraphs while preserving NNCF quantization contracts."""
import hashlib

import numpy as np

from .Clip2ReLUX import Clip2RuLUXOnnx
from .MergeGeLU import MergeGeLUOnnx
from .MergeBatchnorm import ReplaceBatchNorm
from .PrepareParam import ReshapGemmWeightOnnx


def _boundaries(model):
    return [(n.name, n.op, tuple(n.inputs), tuple(n.outputs), repr(sorted(n.attr.items())))
            for n in model.nodes.values() if n.op in ('QuantizeLinear', 'DequantizeLinear')]


def _parameter(model, name):
    value = np.asarray(model.params[name])
    return value.dtype.str, value.shape, hashlib.sha256(value.tobytes()).hexdigest()


def OptimizeQuantizedOnnx(model, fold_static, protected_parameters=()):
    boundaries = _boundaries(model)
    protected = set(protected_parameters)
    for node in model.nodes.values():
        if node.op in ('QuantizeLinear', 'DequantizeLinear'):
            protected.update(n for n in node.inputs[1:] if n)
    parameters = {name: _parameter(model, name) for name in protected}
    for node in model.nodes.values():
        if node.op == 'Gemm' and (node.attr.get('alpha', 1.) != 1. or
                                 node.attr.get('beta', 1.) != 1. or node.attr.get('transA', 0)):
            raise ValueError('strict QDQ Gemm currently requires alpha=beta=1 and transA=0')
    # No Conv/BN folding, attention restructuring or weight reparameterization:
    # those would require changing the quantization chosen by NNCF.
    folded = fold_static(model)
    counts = dict(Clip2ReLUX=Clip2RuLUXOnnx(model),
                  MergeGeLU=MergeGeLUOnnx(model),
                  FoldStaticShape=folded)
    ReplaceBatchNorm(model)  # float BN -> private Scale; never changes Conv weights
    ReshapGemmWeightOnnx(model)  # existing native floating Gemm parameter layout
    if _boundaries(model) != boundaries:
        raise RuntimeError('Optimize changed an NNCF Q/DQ boundary')
    if any(name not in model.params or _parameter(model, name) != value
           for name, value in parameters.items()):
        raise RuntimeError('Optimize changed an NNCF quantization parameter or integer Conv weight')
    return dict(mode='preserve_quantization', passes=counts,
                skipped={'MergeSwish': 'retain Sigmoid/Mul rounding before quantized consumers'},
                qdq_boundaries_preserved=len(boundaries),
                protected_parameters_preserved=len(parameters),
                integer_conv_weights_preserved=len(set(protected_parameters)))
