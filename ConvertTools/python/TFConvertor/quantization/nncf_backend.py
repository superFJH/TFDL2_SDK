"""Optional NNCF ONNX PTQ followed by native TFDL export.

NNCF stays lazy: ordinary conversion never imports or installs this dependency.
QDQ is an internal artifact; customers supply the original floating ONNX.
"""

import hashlib
import importlib
import json
import shlex
import sys
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import onnx


def load_nncf():
    try:
        return importlib.import_module('nncf')
    except ImportError as error:
        requirements = Path(__file__).resolve().parents[2] / 'requirements-nncf.txt'
        command = (f'{shlex.quote(sys.executable)} -m pip install -r '
                   f'{shlex.quote(str(requirements))}')
        raise ImportError(
            '已开启 --nncf，但未安装或无法加载可选依赖 NNCF。\n'
            f'请自行安装到当前 Python 环境：\n  {command}\n'
            f'原始错误：{error}\n未自动安装依赖，也未回退到其它量化算法。'
        ) from error


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _input_spec(model, overrides):
    params = {p.name for p in model.graph.initializer}
    inputs = [p for p in model.graph.input if p.name not in params]
    if len(inputs) != 1 or inputs[0].type.tensor_type.elem_type != onnx.TensorProto.FLOAT:
        raise ValueError('--nncf currently requires one FP32 image input')
    info = inputs[0]
    if set(overrides or {}) - {info.name}:
        raise ValueError('--input_shapes contains an unknown model input')
    shape = list((overrides or {}).get(info.name, [d.dim_value for d in info.type.tensor_type.shape.dim]))
    if len(shape) != 4 or shape[0] != 1 or shape[1] not in (1, 3) or min(shape) <= 0:
        raise ValueError('--nncf requires fixed NCHW batch=1, C=1 or 3; set --input_shapes for dynamic inputs')
    del info.type.tensor_type.shape.dim[:]
    for value in shape:
        info.type.tensor_type.shape.dim.add().dim_value = value
    return info.name, shape


def _preprocessing(mean, std, channels):
    def vector(value, default):
        result = np.asarray([default] if value is None else value, np.float32).reshape(-1)
        if result.size == 1:
            result = np.repeat(result, channels)
        if result.size != channels or not np.isfinite(result).all():
            raise ValueError('--nncf mean/std must have one finite value or one per input channel')
        return result
    mean = vector(mean, 0.)
    std = vector(std, 1.)
    if np.any(std <= 0):
        raise ValueError('--nncf std must be positive')
    return mean, std


def _ort_outputs(model, images, threads):
    import onnxruntime as ort
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(model.SerializeToString(), options, providers=['CPUExecutionProvider'])
    rows = [[] for _ in session.get_outputs()]
    for image in images:
        for values, output in zip(rows, session.run(None, {session.get_inputs()[0].name: image[None]})):
            values.append(output)
    return [np.concatenate(values, axis=0) for values in rows]


def _fb_outputs(path, images, mean, std, threads):
    from TFDL2 import TFContext, TFExecutor
    context = TFContext(path=str(path))
    executor = TFExecutor(context, dict(UseHardware=False, FrugalMode=False, cpuLimit=threads,
        optimize=dict(RuntimeQuantMatMul=False, RuntimeQuantConv=False,
                      RuntimeFastQuantConv=False, RuntimeQuantPostProcess=False)))
    rows = [[] for _ in context.GetOutSymbols()]
    for image in images:
        raw = image[None] * std.reshape(1, -1, 1, 1) + mean.reshape(1, -1, 1, 1)
        executor.GetInputs()[0].fromNumpy(np.ascontiguousarray(raw, dtype=np.float32))
        outputs = executor()
        if len(outputs) != len(rows):
            raise RuntimeError('NNCF export check: output count changed')
        for values, output in zip(rows, outputs):
            values.append(output.toNumpy().copy())
    return [np.concatenate(values, axis=0) for values in rows]


def convert_with_nncf(args, destination, select_images):
    """Run packaged NNCF PTQ, preserve its integer weights and float islands."""
    from . import read_images
    from .metrics import compare, passes, classification
    from .qdq_import import import_qdq

    incompatible = []
    for option in ('fp16', 'auto_fallback', 'external_qparams_json', 'stoptensor', 'fuse_se_block',
                   'calibration_compile'):
        if getattr(args, option, None):
            incompatible.append('--' + option.replace('_', '-'))
    if args.calibration_backend != 'sdk-cpu' or args.calibration_workers != 1:
        incompatible.append('--calibration-backend/--calibration-workers')
    if incompatible:
        raise ValueError('--nncf uses its own ONNX PTQ; incompatible options: ' + ', '.join(incompatible))
    nncf = load_nncf()
    from nncf.quantization.advanced_parameters import QuantizationParameters
    source = Path(args.onnxpath).expanduser().resolve()
    model = onnx.load(str(source))
    onnx.checker.check_model(model)
    if any(n.op_type in ('QuantizeLinear', 'DequantizeLinear', 'FakeQuantize') for n in model.graph.node):
        raise ValueError('--nncf expects the original floating ONNX, not an already quantized model')
    name, shape = _input_spec(model, args.input_shapes)
    mean, std = _preprocessing(args.mean, args.std, shape[1])
    spec = SimpleNamespace(shape=shape, mean=mean.reshape(1, -1, 1, 1),
                           scale=(1. / std).reshape(1, -1, 1, 1))
    paths = select_images(args)
    if not paths and not args.calibration_bin:
        raise ValueError('--nncf requires --quantimgDir, --calibration-list or --calibration-bin; random calibration is disabled')
    images = read_images(spec, paths, args.cvtype, args.calibration_bin)
    work = Path(args.quant_workdir or (str(destination) + '.nncf')).expanduser().resolve()
    qdq_path = work / 'model.nncf.onnx'
    if qdq_path == source:
        raise ValueError('NNCF work directory would overwrite the source ONNX')
    work.mkdir(parents=True, exist_ok=True)
    fb_path = Path(str(destination) + '.quant.fb').resolve()
    report = dict(format='tfdl-nncf-ptq-v1', status='quantizing', source_onnx=str(source),
        source_onnx_sha256=_sha256(source), nncf_version=nncf.__version__,
        calibration=dict(samples=len(images), input_name=name, input_shape=shape,
                         normalized_sha256=hashlib.sha256(images.tobytes()).hexdigest(),
                         image_paths=paths, binary=args.calibration_bin,
                         binary_sha256=_sha256(args.calibration_bin) if args.calibration_bin else None,
                         cvtype=args.cvtype, mean=mean.tolist(), std=std.tolist(),
                         backend='NNCF ONNX / ORT CPU', threads='NNCF/ORT defaults'),
        quantization=dict(per_channel=False, fast_bias_correction=True,
                          target_device='ANY', sdk_recalibration=False,
                          remove_float_qdq_roundtrips=True),
        input_contract='FB embeds (input-mean)/std; calibration binary is already normalized',
        qdq_model=str(qdq_path), exported_model=str(fb_path))
    started = time.perf_counter()

    def save():
        report['seconds'] = time.perf_counter() - started
        (work / 'report.json').write_text(json.dumps(report, indent=2))

    save()
    try:
        print(f'NNCF {nncf.__version__}: {len(images)} calibration images, ONNX/ORT CPU PTQ', flush=True)
        # TFDL's native importer currently supports scalar scale/zero-point.
        # Explicit per-tensor settings also cover ONNX opset >= 13.
        advanced = nncf.AdvancedQuantizationParameters(
            activations_quantization_params=QuantizationParameters(per_channel=False),
            weights_quantization_params=QuantizationParameters(per_channel=False))
        dataset = nncf.Dataset(images, lambda image: {name: np.ascontiguousarray(image[None])})
        quantized = nncf.quantize(model, dataset, subset_size=len(images),
            target_device=nncf.TargetDevice.ANY, fast_bias_correction=True, advanced_parameters=advanced)
        onnx.checker.check_model(quantized)
        onnx.save(quantized, str(qdq_path))
        report['qdq_sha256'] = _sha256(qdq_path)
        report['nncf_operators'] = dict(Counter(n.op_type for n in quantized.graph.node))
        preprocessing = dict(mean=mean.tolist(), std=std.tolist()) if args.mean is not None or args.std is not None else {}
        converter = import_qdq(str(qdq_path), destination.name, **preprocessing)
        converter.dump(str(destination) + '.quant')
        imported = converter.qdq_import_report
        (work / 'import.json').write_text(json.dumps(imported, indent=2))
        report['fb_sha256'] = _sha256(fb_path)
        report['native_conv_count'] = len(imported['native_conv_regions'])
        report['sdk_layer_types'] = imported['sdk_layer_types']
        report['graph_audit'] = imported['graph_audit']
        if report['native_conv_count'] == 0:
            print('Warning: this graph has no native UINT8 Conv regions; inspect the report before assuming INT8 acceleration.', flush=True)

        reference_model = onnx.load(args.testonnxpath) if args.testonnxpath else model
        if args.check:
            reference = _ort_outputs(reference_model, images, args.calibration_threads)
            qdq_outputs = _ort_outputs(quantized, images, args.calibration_threads)
            actual = _fb_outputs(fb_path, images, mean, std, args.calibration_threads)
            report['calibration_check'] = dict(
                samples=len(images), backend='TFDL CPU and ORT CPU',
                float_vs_nncf=compare(reference, qdq_outputs),
                float_vs_fb=compare(reference, actual), nncf_vs_fb=compare(qdq_outputs, actual))
            if not all(m['valid'] and m['integer_equal'] for k,m in report['calibration_check'].items()
                       if isinstance(m,dict)):
                raise RuntimeError('NNCF export check produced nonfinite or incompatible outputs')

        if args.validation_bin:
            validation = read_images(spec, None, args.cvtype, args.validation_bin)
            reference = _ort_outputs(reference_model, validation, args.calibration_threads)
            actual = _fb_outputs(fb_path, validation, mean, std, args.calibration_threads)
            metrics = compare(reference, actual)
            hashes = {hashlib.sha256(image.tobytes()).digest() for image in images}
            overlap = sum(hashlib.sha256(image.tobytes()).digest() in hashes for image in validation)
            metrics.update(samples=len(validation), overlap_samples=overlap,
                           input_sha256=_sha256(args.validation_bin),
                           satisfied=passes(metrics, args.cosine_distance, args.min_cosine))
            if args.validation_labels:
                labels = json.loads(Path(args.validation_labels).read_text())
                if isinstance(labels, dict):
                    labels = [row['label_alphabetical'] for row in labels['samples']]
                metrics.update(classification(reference, actual, labels))
            if args.max_accuracy_loss is not None:
                metrics['meets_accuracy_target'] = metrics['accuracy_loss_pp'] <= args.max_accuracy_loss + 1e-12
            report['validation'] = metrics
            np.savez_compressed(work / 'validation-reference.npz', *reference)
            np.savez_compressed(work / 'validation-actual.npz', *actual)
            if not metrics['valid'] or not metrics['integer_equal']:
                raise RuntimeError('NNCF validation produced nonfinite or incompatible outputs')
            if ((args.require_validation_target and (not metrics['satisfied'] or overlap))
                    or (args.max_accuracy_loss is not None and (not metrics['meets_accuracy_target'] or overlap))):
                raise RuntimeError('NNCF exported FB failed the held-out cosine/accuracy/independence gate')
        report['status'] = 'exported'
        save()
        converter._nncf_workdir = str(work)
        print('Quantized model:', fb_path, flush=True)
        print('NNCF report:', work / 'report.json', flush=True)
        return converter
    except Exception as error:
        report.update(status='failed', error=str(error))
        save()
        raise
