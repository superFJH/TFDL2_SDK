"""Import an ONNX QDQ graph without calibration or reselecting its qparams.

Safe Conv/Relu/Q regions use native quantized Conv with UINT8 output and fused
ReLU. Other QDQ Conv regions use native integer inputs/weights and FP32 output
without adding an uncalibrated output quantizer. Float consumers bypass direct
Quantize/Dequantize roundtrips and use the
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
    zero = (np.asarray(initializers[node.input[2]]) if len(node.input) > 2 and node.input[2]
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


def weight_qparams(node, initializers):
    """Read constant weight Q/DQ, retaining the ONNX per-channel axis."""
    if len(node.input) < 2 or node.input[1] not in initializers:
        raise ValueError(f"QDQ requires constant weight scale: {node.name}")
    codes = np.asarray(initializers[node.input[0]])
    if codes.dtype not in (np.dtype('uint8'), np.dtype('int8')):
        raise ValueError("unsupported QDQ weight dtype: " + node.name)
    scale = np.asarray(initializers[node.input[1]])
    zero = (np.asarray(initializers[node.input[2]]) if len(node.input) > 2 and node.input[2]
            else np.zeros(scale.shape, codes.dtype))
    if zero.dtype != codes.dtype:
        raise ValueError("weight zero-point dtype differs from integer weights: " + node.name)
    if scale.size == 1 and zero.size == 1:
        value = float(scale.reshape(-1)[0])
        if scale.dtype != np.float32 or not np.isfinite(value) or value <= 0:
            raise ValueError("weight QDQ requires positive FP32 scale: " + node.name)
        offset = 128 if codes.dtype == np.int8 else 0
        return value, int(zero.reshape(-1)[0]) + offset, offset, None
    if (scale.dtype != np.float32 or scale.ndim != 1 or not scale.size
            or zero.shape != scale.shape or not np.isfinite(scale).all() or np.any(scale <= 0)):
        raise ValueError("per-channel weight QDQ requires positive FP32 scale and matching zero-point vectors: " + node.name)
    axis = next((int(attr.i) for attr in node.attribute if attr.name == 'axis'), 1)
    if not -codes.ndim <= axis < codes.ndim:
        raise ValueError("per-channel weight axis is out of range: " + node.name)
    axis %= codes.ndim
    if scale.size != codes.shape[axis]:
        raise ValueError("per-channel weight scale count differs from the axis dimension: " + node.name)
    offset = 128 if codes.dtype == np.int8 else 0
    return scale.copy(), zero.astype(np.int32) + offset, offset, axis


def import_qdq(path, name, input_shapes=None, mean=None, std=None, input_quantization=None):
    from TFDL2 import TFContext, TFDataType, Op
    from ..onnxConvertor import OnnxConvertor

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
        if node.op_type == 'DequantizeLinear' and node.input[0] in initializers:
            codes = initializers[node.input[0]]
            scale, zero, offset, axis = weight_qparams(node, initializers)
            physical = (codes.astype(np.int16) + offset).astype(np.uint8)
            broadcast = [1] * codes.ndim
            if axis is not None:
                broadcast[axis] = len(scale)
            value = ((physical.astype(np.float32) - np.asarray(zero, np.float32).reshape(broadcast))
                     * np.asarray(scale, np.float32).reshape(broadcast))
            quantized_weights[node.output[0]] = (physical, scale, zero, axis)
            prepared.graph.initializer.append(numpy_helper.from_array(value, node.output[0]))
            constant_dq.add(node.output[0])
        else:
            # Activation quantization remains per-tensor, independently of weights.
            scalar_qparams(node, initializers)
    kept = [node for node in prepared.graph.node if node.output[0] not in constant_dq]
    del prepared.graph.node[:]
    prepared.graph.node.extend(kept)
    if not any(node.op_type == 'QuantizeLinear' for node in kept):
        raise ValueError("NNCF QDQ import requires activation QuantizeLinear nodes")

    # Optimize before planning dependencies: a fused GeLU/Swish or a folded
    # shape expression must no longer depend on the removed source nodes.
    with tempfile.TemporaryDirectory(prefix='tfdl_qdq_') as temporary:
        source = Path(temporary) / 'prepared.onnx'
        onnx.save(prepared, source)
        converter = OnnxConvertor(name)
        converter.load(str(source))
    for input_name, shape in (input_shapes or {}).items():
        converter.inputs[input_name]['shape'] = list(shape)
        converter.tensor_shapes[input_name] = list(shape)
    protected_weights = {n.input[1] for n in original.graph.node if n.op_type == 'Conv'
                         and n.input[1] in quantized_weights}
    optimization = converter.optmize(preserve_quantization=True,
                                    protected_parameters=protected_weights)

    # Plan the effective dependencies before creating SDK nodes. A quantized
    # Conv consumes the UINT8 tensor before DQ; creating every original DQ
    # first leaves orphan float tensors (and costly dead SDK operations).
    # NNCF propagates a shared quantization range across layout operations.
    # A Conv after Concat/Split/Reshape still consumes quantized values even
    # though there is no DQ immediately adjacent to it. Track an integer view
    # separately from the original float view used by genuine float branches.
    integer_views = {}
    def integer_view(tensor_name):
        if tensor_name in integer_views:
            return integer_views[tensor_name]
        integer_views[tensor_name] = None
        node = original_nodes.get(tensor_name)
        if node is None:
            return None
        if node.op_type == 'DequantizeLinear' and node.input[0] not in initializers:
            result = dict(kind='quantized', source=node.input[0],
                          qparams=scalar_qparams(node, initializers)[:2])
        elif node.op_type in ('Concat','Split','Reshape','Transpose','MaxPool','Resize','Slice'):
            if node.op_type == 'Resize':
                mode = next((onnx.helper.get_attribute_value(a) for a in node.attribute if a.name=='mode'),b'nearest')
                if mode != b'nearest':
                    return None
            inputs = list(node.input) if node.op_type=='Concat' else [node.input[0]]
            parents = [integer_view(n) for n in inputs]
            if any(p is None for p in parents) or len({p['qparams'] for p in parents}) != 1:
                return None
            result = dict(kind='layout', source=tensor_name, node=node,
                          inputs=inputs, qparams=parents[0]['qparams'])
        else:
            return None
        integer_views[tensor_name] = result
        return result

    def integer_dependencies(tensor_name):
        view = integer_view(tensor_name)
        if view['kind']=='quantized':
            return [view['source']]
        return [n for parent in view['inputs'] for n in integer_dependencies(parent)]

    integer_convs = {}
    aliases = {}
    dependencies = {output:list(n.inputs) for n in converter.nodes.values() for output in n.outputs}
    for conv in original.graph.node:
        if conv.op_type != 'Conv':
            continue
        output = conv.output[0]
        view = integer_view(conv.input[0])
        if view is not None and conv.input[1] in quantized_weights:
            if quantized_weights[conv.input[1]][3] not in (None, 0):
                raise ValueError('native integer Conv requires per-output-channel weight axis 0: ' + conv.name)
            region = conv_regions.get(output)
            integer_convs[output] = region
            dependencies[output] = integer_dependencies(conv.input[0]) + list(conv.input[1:])
            if region is not None:
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

    # SetInput already holds normalized UINT8 codes. A source input Q with
    # those same parameters is the input tensor itself, not DQ(input) -> Q.
    # Cut its float dependency before liveness analysis so a decoded input is
    # only created when a surviving float branch actually consumes it.
    input_quantizers = {}
    for node in kept:
        if node.op_type != 'QuantizeLinear':
            continue
        qparams = (input_quantization or {}).get(node.input[0])
        if qparams is None:
            continue
        scale, zero, _ = scalar_qparams(node, initializers)
        if scale == qparams['scale'] and zero == qparams['zero_point']:
            # The current SDK's direct input Conv specialization does not
            # support normalized codes with nonuniform image preprocessing.
            # The NNCF frontend uses a separate input range for that case.
            direct_conv = any(integer_view(original_nodes[output].input[0])['kind']=='quantized'
                and integer_view(original_nodes[output].input[0])['source']==node.output[0]
                for output in integer_convs)
            if (direct_conv and mean is not None and std is not None
                    and (np.any(np.asarray(mean) != 0)
                         or np.any(np.asarray(std) != np.asarray(std)[0]))):
                raise ValueError('SDK direct UINT8 input Conv does not support folded '
                    'input Q with nonzero means or unequal channel stds; use a distinct '
                    'input quantization range to retain the preprocessing boundary')
            input_quantizers[node.output[0]] = node.input[0]
            dependencies[node.output[0]] = []

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

    context = TFContext(name)
    symbols = {}
    fused_quantizers = {}
    fused_relus = {}
    report = {'format': 'tfdl-onnx-qdq-import-v1', 'source': str(Path(path).resolve()),
              'source_sha256': hashlib.sha256(Path(path).read_bytes()).hexdigest(),
              'integer_kernels': True, 'native_only': True,
              'bypass_float_roundtrips':True,
              'optimization':optimization,
              'rounding_semantics_preserved':False,
              'removed_source_nodes':[], 'bypassed_float_roundtrips':[],
              'sdk_expand_connections':[], 'sdk_opaque_connections':[],
              'folded_input_quantizers':[], 'integer_layout_views':[],
              'operators': {}, 'inputs': [], 'quantizers': [], 'native_conv_regions': [],
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

    converter_nodes = {o:n for n in converter.nodes.values() for o in n.outputs}
    resolved_integer_views = {}
    def resolve_integer(tensor_name):
        if tensor_name in resolved_integer_views:
            return resolved_integer_views[tensor_name]
        view = integer_view(tensor_name)
        if view['kind']=='quantized':
            return resolve(view['source'])
        node = converter_nodes[tensor_name]
        integer_inputs = [resolve_integer(n) for n in view['inputs']]
        arguments = (integer_inputs if node.op=='Concat' else
                     integer_inputs + [resolve(n) for n in node.inputs[1:]])
        if node.op=='Split':
            # Only create the requested integer slice. A full multi-output
            # Split can leave unused terminal tensors when another branch
            # needs only the original float view.
            lengths = node.attr.get('split')
            if lengths is None:
                lengths = np.asarray(converter.params[node.inputs[1]]).reshape(-1).tolist()
            index = node.outputs.index(tensor_name)
            start = sum(lengths[:index])
            value = Op.Crop3(integer_inputs[0],((start,start+lengths[index]-1,1),),node.attr['axis'])
            scale,zero = view['qparams']
            context.SetQuantization(str(value),scale,zero)
            resolved_integer_views[tensor_name] = value
            report['integer_layout_views'].append({'onnx':tensor_name,'sdk':str(value),
                'op':'Split','scale':scale,'zero_point':zero})
            counts['integer_Split'] += 1
            return value
        value = node(*arguments)
        outputs = list(value) if isinstance(value,(list,tuple)) else [value]
        scale,zero = view['qparams']
        for name,sdk in zip(node.outputs,outputs):
            context.SetQuantization(str(sdk),scale,zero)
            resolved_integer_views[name] = sdk
            report['integer_layout_views'].append({'onnx':name,'sdk':str(sdk),
                'op':node.op,'scale':scale,'zero_point':zero})
        counts['integer_'+node.op] += 1
        return resolved_integer_views[tensor_name]

    def integer_weight(node, index):
        weight_name = node.inputs[1]
        codes, scale, zero, axis = quantized_weights[weight_name]
        codes = np.ascontiguousarray(codes)
        scales = np.asarray(scale, np.float32).reshape(-1)
        zeros = np.asarray(zero, np.float32).reshape(-1)
        private_name = f'__onnx_integer_weight_{index}'
        if not context.RegisterQuantizedParamToContext(private_name, codes,
                ((np.float32(255)-zeros)*scales).tolist(), (-zeros*scales).tolist()):
            raise RuntimeError('could not register integer weight: ' + weight_name)
        # Parameter tensors serialize per-channel ranges, separately from the
        # activation q-info registry. The SDK rejects vector SetQuantization
        # for parameters; read back the registered metadata instead.
        if axis is None and not context.SetQuantization(private_name, scale, zero):
            raise RuntimeError('could not preserve weight qparams: ' + weight_name)
        actual = context.GetParam(private_name)
        actual_scales = np.asarray(actual.qscale, np.float32)
        actual_zeros = np.asarray(actual.qzeropoint)
        if (not np.allclose(actual_scales, scales, rtol=2e-7, atol=0)
                or not np.all(actual_zeros == np.asarray(zero))):
            raise RuntimeError('SDK changed per-channel weight qparams: ' + weight_name)
        report['integer_weights'].append({'onnx': weight_name, 'sdk': private_name,
                                         'scale': actual_scales.tolist() if axis is not None else scale,
                                         'source_scale': scales.tolist() if axis is not None else scale,
                                         'zero_point': np.asarray(zero).tolist(),
                                         'axis': axis,
                                         'granularity': 'per_channel' if axis is not None else 'per_tensor',
                                         'shape': list(codes.shape),
                                         'codes_sha256': hashlib.sha256(codes.tobytes()).hexdigest()})
        return context.GetParamSymbol(private_name)

    with context:
        for input_name, attrs in converter.inputs.items():
            if attrs['dtype'] != TFDataType.TFDL_FLOAT:
                raise ValueError('QDQ source ONNX inputs currently require FP32')
            shape = tuple((input_shapes or {}).get(input_name, attrs['shape']))
            preprocessing = {}
            if mean is not None or std is not None:
                if mean is None or std is None:
                    raise ValueError('internal preprocessing requires both mean and std')
                preprocessing = dict(mean=tuple(mean),scale=tuple(1.0/np.asarray(std)))
            qparams = (input_quantization or {}).get(input_name)
            dtype = TFDataType.TFDL_UINT8 if qparams is not None else attrs['dtype']
            placeholder = Op.Placeholder2(context, shape, dtype, **preprocessing)
            if qparams is not None:
                if not context.SetQuantization(str(placeholder),qparams['scale'],qparams['zero_point']):
                    raise RuntimeError('could not register UINT8 input parameters: '+input_name)
                symbols[input_name] = (Op.DeQuantize(placeholder)
                    if input_name in needed else placeholder)
                for quantizer, source_input in input_quantizers.items():
                    if source_input == input_name and quantizer in needed:
                        fused_quantizers[quantizer] = placeholder
                        report['folded_input_quantizers'].append({
                            'onnx':quantizer,'input':input_name,'sdk':str(placeholder),
                            'scale':qparams['scale'],'zero_point':qparams['zero_point']})
            else:
                symbols[input_name] = placeholder
            report['inputs'].append({'onnx':input_name,'sdk':str(placeholder),
                'dtype':'uint8' if qparams is not None else 'float32',
                'shape':list(shape),'mean':list(mean) if mean is not None else None,
                'scale':preprocessing.get('scale'), 'quantization':qparams})
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
            elif node.op == 'Split':
                lengths = node.attr.get('split')
                if lengths is None:
                    lengths = np.asarray(converter.params[node.inputs[1]]).reshape(-1).tolist()
                activation = resolve(node.inputs[0])
                for part, onnx_name in enumerate(node.outputs):
                    if canonical(onnx_name) not in needed:
                        continue
                    start = sum(lengths[:part])
                    value = Op.Crop3(activation,((start,start+lengths[part]-1,1),),node.attr['axis'])
                    symbols[onnx_name] = value
                    report['mapping'][onnx_name] = str(value)
                counts['Split'] += 1
                continue
            else:
                native_integer = node.outputs[0] in integer_convs
                if native_integer:
                    activation = resolve_integer(node.inputs[0])
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
                        sx_pad,zx_pad = integer_view(node.inputs[0])['qparams']
                        context.SetQuantization(str(activation),sx_pad,zx_pad)
                        pads = [0,0,0,0]
                    bias = resolve(node.inputs[2]) if len(node.inputs)>2 else None
                    output = Op.Convolution2(activation, weight, bias,
                        attr.get('kernel_shape', quantized_weights[node.inputs[1]][0].shape[2:]),
                        [pads[0], pads[2], pads[1], pads[3]], attr.get('strides', [1, 1]),
                        dilations[0], quantized_weights[node.inputs[1]][0].shape[0], attr.get('group', 1))
                    region = integer_convs[node.outputs[0]]
                    if region is not None:
                        relu_node, q_node = region
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
                            'output_dtype':'uint8',
                            'note':'native quantized Conv with FP32 source bias and fused ReLU'})
                    else:
                        # SiLU and prediction heads often have no calibrated
                        # pre-activation output Q. Native integer Conv supports
                        # direct FP32 postprocessing; keep that boundary exact.
                        context.Modify({'Layer':[{'layerName':str(output),
                            'outputDataType':'TFDtypeFp32'}]})
                        report['native_conv_regions'].append({
                            'conv':node.outputs[0],'relu':None,'quantizer':None,
                            'sdk_quantized':str(output),'output_dtype':'float32',
                            'note':'native UINT8 inputs and weights with FP32 postprocessing'})
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
            elif node.op=='Split':
                # _GetAttr segfaults on multi-output Slice tensor names in
                # this SDK. Record actual native connectivity without calling
                # that accessor on the individual outputs.
                report['sdk_opaque_connections'].append({
                    'layerName':str(outputs[0]).rsplit('_',1)[0],
                    'layerType':'Slice','input':[str(resolve(node.inputs[0]))],
                    'output':[str(o) for o in outputs],
                    'audit_source':'native Split parser outputs'})
            counts[node.op] += 1
    # Select the source public outputs after CloseContext's terminal discovery.
    context.SetOutputs([str(resolve(output.name)) for output in original.graph.output])
    report['public_outputs']=[{'onnx':o.name,'sdk':str(resolve(o.name))}
                              for o in original.graph.output]
    actual_outputs=[str(s) for s in context.GetOutSymbols()]
    if actual_outputs != [o['sdk'] for o in report['public_outputs']]:
        raise RuntimeError('SDK public output selection differs from the source graph')
    report['operators'] = dict(counts)
    report['integer_kernels'] = bool(report['native_conv_regions'])
    report['native_conv_output_dtypes'] = dict(Counter(
        row['output_dtype'] for row in report['native_conv_regions']))
    layers = {}
    # Some native multi-output tensors and Expand cannot be queried through
    # this SDK's JSON accessor. Their connectivity is recorded at creation.
    report['sdk_attr_audit_skipped'] = []
    opaque_outputs={o for row in report['sdk_opaque_connections'] for o in row['output']}
    for tensor_name in context.GetAllTensorNames():
        if tensor_name.startswith('TFDL_Expand_') or tensor_name in opaque_outputs:
            report['sdk_attr_audit_skipped'].append(tensor_name)
            continue
        raw = context._GetAttr(tensor_name)
        if raw:
            layer = json.loads(raw)
            if 'layerName' in layer:
                layers[layer['layerName']] = layer
    connections=(list(layers.values())+report['sdk_expand_connections']
                 +report['sdk_opaque_connections'])
    report['sdk_layers'] = connections
    report['sdk_layer_types'] = dict(Counter(l['layerType'] for l in connections))
    if any(l['layerType']=='Custom' for l in layers.values()):
        raise RuntimeError('native-only import unexpectedly created a Custom node')
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
    report['graph_audit']['dequantize_to_quantize_edges']=[
        {'dequantize':producers[layer['input'][0]]['layerName'],'quantize':layer['layerName']}
        for layer in connections if layer['layerType']=='Quantize'
        and producers.get(layer['input'][0],{}).get('layerType')=='DeQuantize']
    if report['graph_audit']['quantize_to_dequantize_edges']:
        raise RuntimeError('native graph still contains a direct Quantize/DeQuantize roundtrip')
    if report['graph_audit']['unexpected_terminal_tensors']:
        raise RuntimeError('native graph contains unexpected terminal tensors: '
                           +str(report['graph_audit']['unexpected_terminal_tensors']))
    converter._context = context
    converter.renames = symbols
    converter.qdq_import_report = report
    return converter
