"""Import an ONNX QDQ graph without calibration or reselecting its qparams.

Safe Conv/Relu/Q regions use native quantized Conv with UINT8 output and fused
ReLU. Float consumers bypass direct Quantize/Dequantize roundtrips and use the
pre-Quant FP32 values. Integer consumers retain their quantized inputs and
source qparams. Floating regions use native operators with the dequantized
original weights and corrected biases.
This intentionally does not invoke Quantize/QuantizeLite graph calibration.
"""

import copy
import hashlib
import json
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


def scalar_qparams(node, initializers):
    """Return ONNX scalar parameters and their equivalent UINT8 storage."""
    if len(node.input) < 2 or node.input[1] not in initializers:
        raise ValueError(f"QDQ requires constant scale: {node.name}")
    scale = np.asarray(initializers[node.input[1]])
    zero = (np.asarray(initializers[node.input[2]]) if len(node.input) > 2
            else np.array(0, np.uint8))
    if scale.dtype != np.float32 or scale.size != 1 or zero.size != 1:
        raise ValueError(f"strict QDQ import currently requires scalar FP32 scale: {node.name}")
    if zero.dtype not in (np.dtype('uint8'), np.dtype('int8')):
        raise ValueError(f"QDQ requires 8-bit integer zero-point: {node.name}")
    value = float(scale.reshape(-1)[0])
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"invalid QDQ scale: {node.name}")
    offset = 128 if zero.dtype == np.int8 else 0
    return value, int(zero.reshape(-1)[0]) + offset, offset


def import_qdq(path, name, input_shapes=None, mean=None, std=None):
    from TFDL2 import TFContext, TFDataType, Op
    from ..onnxConvertor import OnnxConvertor
    from ..Optimize.PrepareParam import ReshapGemmWeightOnnx
    from ..Optimize.MergeBatchnorm import ReplaceBatchNorm

    if not hasattr(TFContext, 'SetQuantization'):
        raise RuntimeError("strict QDQ import requires the SDK Python SetQuantization binding")
    original = onnx.load(path)
    onnx.checker.check_model(original)
    initializers = {item.name: numpy_helper.to_array(item)
                    for item in original.graph.initializer}
    original_nodes = {output: node for node in original.graph.node for output in node.output}
    consumers = {}
    public_outputs = {o.name for o in original.graph.output}
    for node in original.graph.node:
        for tensor_name in node.input:
            consumers.setdefault(tensor_name, []).append(node)
    conv_regions = {}
    for node in original.graph.node:
        if node.op_type != 'Conv' or node.output[0] in public_outputs:
            continue
        users = consumers.get(node.output[0], [])
        if len(users) != 1 or users[0].op_type != 'Relu':
            continue
        relu = users[0]
        if relu.output[0] in public_outputs:
            continue
        users = consumers.get(relu.output[0], [])
        if len(users) == 1 and users[0].op_type == 'QuantizeLinear':
            conv_regions[node.output[0]] = (relu, users[0])
    quantized_weights = {}
    constant_dq = set()
    prepared = copy.deepcopy(original)
    for node in original.graph.node:
        if node.op_type not in ('QuantizeLinear', 'DequantizeLinear'):
            continue
        scale, zero, offset = scalar_qparams(node, initializers)
        if node.op_type == 'DequantizeLinear' and node.input[0] in initializers:
            codes = initializers[node.input[0]]
            if codes.dtype not in (np.dtype('uint8'), np.dtype('int8')):
                raise ValueError("unsupported QDQ weight dtype: " + node.name)
            physical = (codes.astype(np.int16) + offset).astype(np.uint8)
            value = (physical.astype(np.float32) - np.float32(zero)) * np.float32(scale)
            quantized_weights[node.output[0]] = (physical, scale, zero)
            prepared.graph.initializer.append(numpy_helper.from_array(value, node.output[0]))
            constant_dq.add(node.output[0])
    kept = [node for node in prepared.graph.node if node.output[0] not in constant_dq]
    del prepared.graph.node[:]
    prepared.graph.node.extend(kept)
    if not any(node.op_type == 'QuantizeLinear' for node in kept):
        raise ValueError("NNCF QDQ import requires activation QuantizeLinear nodes")

    # Plan the effective dependencies before creating SDK nodes. A quantized
    # Conv consumes the UINT8 tensor before DQ; creating every original DQ
    # first leaves orphan float tensors (and costly dead SDK operations).
    integer_convs = {}
    aliases = {}
    dependencies = {output:list(n.input) for n in original.graph.node for output in n.output}
    for output, region in conv_regions.items():
        conv = original_nodes[output]
        dq = original_nodes.get(conv.input[0])
        if (dq is not None
                and dq.op_type == 'DequantizeLinear' and conv.input[1] in quantized_weights):
            integer_convs[output] = region
            dependencies[output] = [dq.input[0]] + list(conv.input[1:])
            relu,quantizer = region
            aliases[relu.output[0]] = output
            aliases[quantizer.output[0]] = output
    float_bypasses = {}
    for node in kept:
        if node.op_type != 'DequantizeLinear':
            continue
        q = original_nodes.get(node.input[0])
        # Native integer Conv outputs have no pre-quant FP32 value. For a
        # float producer, bypass Q/DQ on every float edge; keep Q if another
        # consumer still needs its UINT8 codes.
        if (q is not None and q.op_type == 'QuantizeLinear'
                and q.output[0] not in aliases):
            if scalar_qparams(q,initializers) != scalar_qparams(node,initializers):
                raise ValueError('Q/DQ parameters differ: '+node.name)
            float_bypasses[node.output[0]] = q.input[0]
            aliases[node.output[0]] = q.input[0]

    def canonical(tensor_name):
        while tensor_name in aliases:
            tensor_name = aliases[tensor_name]
        return tensor_name

    needed = set()
    pending = list(public_outputs)
    while pending:
        tensor_name = canonical(pending.pop())
        if tensor_name in needed:
            continue
        needed.add(tensor_name)
        pending.extend(dependencies.get(tensor_name,()))

    # Use the existing float parsers without running graph calibration or
    # ordinary fusion passes across the remaining integer boundaries.
    with tempfile.TemporaryDirectory(prefix='tfdl_qdq_') as temporary:
        source = Path(temporary) / 'prepared.onnx'
        onnx.save(prepared, source)
        converter = OnnxConvertor(name)
        converter.load(str(source))
    ReplaceBatchNorm(converter)
    for node in converter.nodes.values():
        if node.op == 'Gemm' and (node.attr.get('alpha',1.)!=1. or node.attr.get('beta',1.)!=1.
                                  or node.attr.get('transA',0)):
            raise ValueError('strict QDQ Gemm currently requires alpha=beta=1 and transA=0')
    ReshapGemmWeightOnnx(converter)

    context = TFContext(name)
    symbols = {}
    fused_quantizers = {}
    fused_relus = {}
    report = {'format': 'tfdl-onnx-qdq-import-v1', 'source': str(Path(path).resolve()),
              'source_sha256': hashlib.sha256(Path(path).read_bytes()).hexdigest(),
              'integer_kernels': True, 'native_only': True,
              'bypass_float_roundtrips':True,
              'rounding_semantics_preserved':False,
              'removed_source_nodes':[], 'bypassed_float_roundtrips':[],
              'sdk_expand_connections':[],
              'operators': {}, 'quantizers': [], 'native_conv_regions': [],
              'integer_weights': [], 'mapping': {}, 'float_reduction_backend': 'sdk'}
    counts = Counter()

    def parameter(param_name):
        if param_name not in symbols:
            array = np.ascontiguousarray(converter.params[param_name])
            context.RegisterParamToContext(**{param_name: array})
            symbols[param_name] = context.GetParamSymbol(param_name)
        return symbols[param_name]

    def resolve(tensor_name):
        if not tensor_name or tensor_name == 'ISNULL':
            return None
        return parameter(tensor_name) if tensor_name in converter.params else symbols[tensor_name]

    def integer_weight(node, index):
        weight_name = node.inputs[1]
        codes, scale, zero = quantized_weights[weight_name]
        codes = np.ascontiguousarray(codes)
        private_name = f'__onnx_integer_weight_{index}'
        if not context.RegisterQuantizedParamToContext(private_name, codes,
                [(255-zero)*scale], [-zero*scale]):
            raise RuntimeError('could not register integer weight: ' + weight_name)
        if not context.SetQuantization(private_name, scale, zero):
            raise RuntimeError('could not preserve weight qparams: ' + weight_name)
        report['integer_weights'].append({'onnx': weight_name, 'sdk': private_name,
                                         'scale': scale, 'source_scale': scale,
                                         'zero_point': zero,
                                         'shape': list(codes.shape),
                                         'codes_sha256': hashlib.sha256(codes.tobytes()).hexdigest()})
        return context.GetParamSymbol(private_name)

    with context:
        for input_name, attrs in converter.inputs.items():
            if attrs['dtype'] != TFDataType.TFDL_FLOAT:
                raise ValueError('strict QDQ public inputs currently require FP32')
            shape = tuple((input_shapes or {}).get(input_name, attrs['shape']))
            preprocessing = {}
            if mean is not None or std is not None:
                if mean is None or std is None:
                    raise ValueError('internal preprocessing requires both mean and std')
                preprocessing = dict(mean=tuple(mean),scale=tuple(1.0/np.asarray(std)))
            symbols[input_name] = Op.Placeholder2(context, shape, attrs['dtype'], **preprocessing)
        for index, node in enumerate(converter.nodes.values()):
            if not any(canonical(output) in needed for output in node.outputs):
                report['removed_source_nodes'].append({'name':node.name,'op':node.op,
                    'outputs':node.outputs,'reason':'unreachable after quantized consumer lowering'})
                continue
            source_node = original_nodes.get(node.outputs[0])
            if node.op == 'QuantizeLinear':
                scale, zero, offset = scalar_qparams(source_node, initializers)
                if node.outputs[0] in fused_quantizers:
                    output = fused_quantizers[node.outputs[0]]
                else:
                    output = Op.Quantize(resolve(node.inputs[0]))
                if not context.SetQuantization(str(output), scale, zero):
                    raise RuntimeError('could not preserve activation qparams: ' + node.name)
                report['quantizers'].append({'onnx': node.outputs[0], 'sdk': str(output),
                                             'scale': scale, 'zero_point': zero, 'storage_offset': offset})
            elif node.op == 'Relu' and node.outputs[0] in fused_relus:
                output = fused_relus[node.outputs[0]]
            elif node.op == 'DequantizeLinear':
                if node.outputs[0] in float_bypasses:
                    target=float_bypasses[node.outputs[0]]
                    output=resolve(target)
                    report['bypassed_float_roundtrips'].append({
                        'onnx':node.outputs[0],'float_source':target,'sdk':str(output)})
                    symbols[node.outputs[0]]=output
                    report['mapping'][node.outputs[0]]=str(output)
                    continue
                scale, zero, offset = scalar_qparams(source_node, initializers)
                input_symbol = resolve(node.inputs[0])
                q_record = next((r for r in report['quantizers'] if r['onnx'] == node.inputs[0]), None)
                if q_record is None or q_record['scale'] != scale or q_record['zero_point'] != zero:
                    raise ValueError('Q/DQ with different parameters is not supported: ' + node.name)
                output = Op.DeQuantize(input_symbol)
            else:
                input_dq = original_nodes.get(node.inputs[0]) if node.inputs else None
                native_integer = node.outputs[0] in integer_convs
                if native_integer:
                    activation = resolve(input_dq.input[0])
                    weight = integer_weight(node, index)
                    attr = node.attr
                    pads = attr.get('pads', [0, 0, 0, 0])
                    if attr.get('auto_pad', 'NOTSET') != 'NOTSET':
                        raise ValueError('strict integer Conv requires explicit pads')
                    dilations = attr.get('dilations', [1, 1])
                    if dilations[0] != dilations[1]:
                        raise ValueError('strict integer Conv requires equal dilation')
                    if max(pads) >= 8:
                        pad_name = f'__onnx_explicit_pad_{index}'
                        pad_array = np.array([0,0,pads[0],pads[1],0,0,pads[2],pads[3]],np.int64)
                        context.RegisterParamToContext(**{pad_name: pad_array})
                        activation = Op.Pad2(activation,context.GetParamSymbol(pad_name))
                        sx_pad,zx_pad,_ = scalar_qparams(input_dq,initializers)
                        context.SetQuantization(str(activation),sx_pad,zx_pad)
                        pads = [0,0,0,0]
                    bias = resolve(node.inputs[2]) if len(node.inputs)>2 else None
                    output = Op.Convolution2(activation, weight, bias,
                        attr.get('kernel_shape', quantized_weights[node.inputs[1]][0].shape[2:]),
                        [pads[0], pads[2], pads[1], pads[3]], attr.get('strides', [1, 1]),
                        dilations[0], quantized_weights[node.inputs[1]][0].shape[0], attr.get('group', 1))
                    relu_node, q_node = conv_regions[node.outputs[0]]
                    sy, zy, _ = scalar_qparams(q_node, initializers)
                    context.SetQuantization(str(output),sy,zy)
                    context.Modify({'Layer':[{'layerName':str(output),
                        'outputDataType':'TFDtypeUint8','ActivationType':'ReLU'}]})
                    context.SetQuantization(str(output),sy,zy)
                    fused_quantizers[q_node.output[0]] = output
                    fused_relus[relu_node.output[0]] = output
                    report['native_conv_regions'].append({
                        'conv':node.outputs[0],'relu':relu_node.output[0],
                        'quantizer':q_node.output[0],'sdk_quantized':str(output),
                        'note':'native quantized Conv with FP32 source bias and fused ReLU'})
                    counts['integer_' + node.op] += 1
                else:
                    arguments = [resolve(tensor_name) for tensor_name in node.inputs]
                    output = node(*arguments)
                    if node.op in ('Conv', 'Gemm'):
                        counts['float_' + node.op] += 1
            outputs = list(output) if isinstance(output, (tuple, list)) else [output]
            if len(outputs) != len(node.outputs):
                raise RuntimeError('QDQ op output count mismatch: ' + node.name)
            for onnx_name, sdk in zip(node.outputs, outputs):
                symbols[onnx_name] = sdk
                report['mapping'][onnx_name] = str(sdk)
            if node.op=='Expand':
                report['sdk_expand_connections'].append({'layerName':str(outputs[0]),
                    'layerType':'Expand','input':[str(resolve(n)) for n in node.inputs],
                    'output':[str(o) for o in outputs]})
            counts[node.op] += 1
    # Select the source public outputs after CloseContext's terminal discovery.
    context.SetOutputs([str(resolve(output.name)) for output in original.graph.output])
    report['public_outputs']=[{'onnx':o.name,'sdk':str(resolve(o.name))}
                              for o in original.graph.output]
    actual_outputs=[str(s) for s in context.GetOutSymbols()]
    if actual_outputs != [o['sdk'] for o in report['public_outputs']]:
        raise RuntimeError('SDK public output selection differs from the source graph')
    report['operators'] = dict(counts)
    layers = {}
    # This SDK's JSON accessor crashes for Expand, even though the
    # native node executes and serializes correctly. Record that gap.
    report['sdk_attr_audit_skipped'] = []
    for tensor_name in context.GetAllTensorNames():
        if tensor_name.startswith('TFDL_Expand_'):
            report['sdk_attr_audit_skipped'].append(tensor_name)
            continue
        raw = context._GetAttr(tensor_name)
        if raw:
            layer = json.loads(raw)
            if 'layerName' in layer:
                layers[layer['layerName']] = layer
    report['sdk_layers'] = list(layers.values())
    report['sdk_layer_types'] = dict(Counter(l['layerType'] for l in layers.values()))
    if any(l['layerType']=='Custom' for l in layers.values()):
        raise RuntimeError('native-only import unexpectedly created a Custom node')
    connections=list(layers.values())+report['sdk_expand_connections']
    consumed={n for layer in connections for n in layer.get('input',[])}
    terminal=[n for layer in connections for n in layer['output'] if n not in consumed]
    report['graph_audit']={'layer_count':len(connections),'terminal_tensors':terminal,
        'declared_outputs':actual_outputs,
        'unexpected_terminal_tensors':sorted(set(terminal)-set(actual_outputs)),
        'dangling_dequantize':[layer['layerName'] for layer in connections
            if layer['layerType']=='DeQuantize' and all(n not in consumed for n in layer['output'])]}
    producers={n:layer for layer in connections for n in layer['output']}
    report['graph_audit']['quantize_to_dequantize_edges']=[
        {'quantize':producers[layer['input'][0]]['layerName'],'dequantize':layer['layerName']}
        for layer in connections if layer['layerType']=='DeQuantize'
        and producers.get(layer['input'][0],{}).get('layerType')=='Quantize']
    if report['graph_audit']['quantize_to_dequantize_edges']:
        raise RuntimeError('native graph still contains a direct Quantize/DeQuantize roundtrip')
    if report['graph_audit']['unexpected_terminal_tensors']:
        raise RuntimeError('native graph contains unexpected terminal tensors')
    converter._context = context
    converter.renames = symbols
    converter.qdq_import_report = report
    return converter
