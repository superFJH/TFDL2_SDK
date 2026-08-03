#!/usr/bin/env python3
"""One-click timm -> ONNX -> onnxslim -> TFDL2 conversion.

Examples:

  # Random timm weights, export both float and quantized models.
  python ConvertTools/python/example/timm2tfdl.py --model vit_tiny_patch16_224 --pretrain=false --input-shape 1,3,224,224 --output-dir /tmp/vit_tfdl

  # Download timm pretrained weights and calibrate quantization with images.
  python ConvertTools/python/example/timm2tfdl.py --model swin_tiny_patch4_window7_224 --pretrain --input-shape 1 3 224 224 --calib-dir ./calibration_images
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import types
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import timm
import torch


SUPPORTED_MODELS = (
    "vit_tiny_patch16_224",
    "deit_tiny_distilled_patch16_224",
    "swin_tiny_patch4_window7_224",
    "pit_ti_224",
    "cait_xxs24_224",
    "levit_128s",
    "mobilevit_xxs",
)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_bool(value: str | bool | None) -> bool:
    if value is None or value is True:
        return True
    if value is False:
        return False
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"expected true/false, got {value!r}")


def parse_input_shape(values: list[str] | None) -> tuple[int, int, int, int] | None:
    if values is None:
        return None
    flattened: list[str] = []
    for value in values:
        flattened.extend(part for part in value.split(",") if part)
    try:
        shape = tuple(int(part) for part in flattened)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--input-shape must contain integers") from exc
    if len(shape) == 3:
        shape = (1, *shape)
    if len(shape) != 4 or any(dim <= 0 for dim in shape):
        raise argparse.ArgumentTypeError(
            "--input-shape must be N,C,H,W or C,H,W, for example 1,3,224,224"
        )
    return shape


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")


def inspect_onnx(path: Path) -> dict[str, object]:
    model = onnx.load(str(path), load_external_data=False)
    onnx.checker.check_model(model)
    return {
        "nodes": len(model.graph.node),
        "size_mb": round(path.stat().st_size / (1024 * 1024), 2),
    }


def comparison_metrics(
    reference: np.ndarray,
    actual: np.ndarray,
    *,
    min_cosine: float,
    atol: float,
    rtol: float,
    policy: str = "strict",
) -> dict[str, object]:
    reference = np.asarray(reference, dtype=np.float32)
    actual = np.asarray(actual, dtype=np.float32)
    if reference.shape != actual.shape:
        return {
            "passed": False,
            "reference_shape": list(reference.shape),
            "actual_shape": list(actual.shape),
            "error": "shape mismatch",
        }
    reference_flat = reference.reshape(-1).astype(np.float64)
    actual_flat = actual.reshape(-1).astype(np.float64)
    finite = bool(np.isfinite(reference_flat).all() and np.isfinite(actual_flat).all())
    if not finite:
        return {
            "passed": False,
            "reference_shape": list(reference.shape),
            "actual_shape": list(actual.shape),
            "error": "NaN or Inf found",
        }
    difference = actual_flat - reference_flat
    denominator = float(np.linalg.norm(reference_flat) * np.linalg.norm(actual_flat))
    if denominator == 0.0:
        cosine = 1.0 if np.array_equal(reference_flat, actual_flat) else 0.0
    else:
        cosine = float(np.dot(reference_flat, actual_flat) / denominator)
    max_abs = float(np.max(np.abs(difference))) if difference.size else 0.0
    mean_abs = float(np.mean(np.abs(difference))) if difference.size else 0.0
    rmse = float(np.sqrt(np.mean(difference * difference))) if difference.size else 0.0
    reference_abs_max = float(np.max(np.abs(reference_flat))) if reference_flat.size else 0.0
    abs_limit = float(atol + rtol * reference_abs_max)
    near_zero_reference = reference_abs_max <= 1e-3
    if policy == "quant":
        passed = bool(cosine >= min_cosine or (near_zero_reference and max_abs <= 1e-3))
        gate = "cosine" if cosine >= min_cosine else ("near_zero_absolute" if passed else "failed")
    else:
        passed = bool(cosine >= min_cosine and max_abs <= abs_limit)
        gate = "cosine_and_error" if passed else "failed"
    return {
        "passed": passed,
        "shape": list(reference.shape),
        "cosine_similarity": cosine,
        "cosine_distance": float(1.0 - cosine),
        "max_abs": max_abs,
        "mean_abs": mean_abs,
        "rmse": rmse,
        "reference_abs_max": reference_abs_max,
        "max_abs_limit": abs_limit,
        "gate": gate,
        "near_zero_reference": near_zero_reference,
    }


def compare_sample_outputs(
    reference_outputs: list[np.ndarray],
    actual_outputs: list[np.ndarray],
    *,
    min_cosine: float,
    atol: float,
    rtol: float,
    policy: str = "strict",
) -> dict[str, object]:
    if len(reference_outputs) != len(actual_outputs):
        return {
            "passed": False,
            "error": f"output count mismatch: reference={len(reference_outputs)}, actual={len(actual_outputs)}",
        }
    outputs = [
        comparison_metrics(
            reference,
            actual,
            min_cosine=min_cosine,
            atol=atol,
            rtol=rtol,
            policy=policy,
        )
        for reference, actual in zip(reference_outputs, actual_outputs)
    ]
    return {"passed": all(bool(item["passed"]) for item in outputs), "outputs": outputs}


def summarize_stage(samples: list[dict[str, object]]) -> dict[str, object]:
    output_metrics = [
        output
        for sample in samples
        for output in sample.get("outputs", [])
        if isinstance(output, dict) and "cosine_similarity" in output
    ]
    result: dict[str, object] = {
        "passed": bool(samples) and all(bool(sample.get("passed")) for sample in samples),
        "samples": samples,
    }
    if output_metrics:
        result.update(
            min_cosine_similarity=min(float(item["cosine_similarity"]) for item in output_metrics),
            max_cosine_distance=max(float(item["cosine_distance"]) for item in output_metrics),
            max_abs=max(float(item["max_abs"]) for item in output_metrics),
            max_rmse=max(float(item["rmse"]) for item in output_metrics),
        )
    return result


def compare_output_batches(
    reference: list[list[np.ndarray]],
    actual: list[list[np.ndarray]],
    *,
    min_cosine: float,
    atol: float,
    rtol: float,
    policy: str = "strict",
) -> dict[str, object]:
    if len(reference) != len(actual):
        return {
            "passed": False,
            "error": f"sample count mismatch: reference={len(reference)}, actual={len(actual)}",
            "samples": [],
        }
    return summarize_stage(
        [
            compare_sample_outputs(
                reference_outputs,
                actual_outputs,
                min_cosine=min_cosine,
                atol=atol,
                rtol=rtol,
                policy=policy,
            )
            for reference_outputs, actual_outputs in zip(reference, actual)
        ]
    )


def torch_output_list(output: Any) -> list[np.ndarray]:
    if isinstance(output, torch.Tensor):
        tensors = [output]
    elif isinstance(output, (tuple, list)) and all(isinstance(item, torch.Tensor) for item in output):
        tensors = list(output)
    else:
        raise TypeError(f"unsupported timm output type: {type(output)!r}")
    return [tensor.detach().cpu().float().numpy() for tensor in tensors]


def make_verification_samples(
    model: torch.nn.Module,
    input_shape: tuple[int, int, int, int],
    *,
    count: int,
    seed: int,
    embed_preprocess: bool,
    normalized_mean: list[float],
    normalized_std: list[float],
    raw_input_max: float,
) -> tuple[list[np.ndarray], list[np.ndarray], list[list[np.ndarray]]]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + 1009)
    mean = torch.tensor(normalized_mean, dtype=torch.float32).reshape(1, -1, 1, 1)
    std = torch.tensor(normalized_std, dtype=torch.float32).reshape(1, -1, 1, 1)
    onnx_inputs: list[np.ndarray] = []
    tfdl_inputs: list[np.ndarray] = []
    torch_outputs: list[list[np.ndarray]] = []
    with torch.inference_mode():
        for _ in range(max(1, count)):
            if embed_preprocess:
                raw = torch.rand(input_shape, generator=generator, dtype=torch.float32) * raw_input_max
                model_input = (raw / raw_input_max - mean) / std
                tfdl_input = raw
            else:
                model_input = torch.randn(input_shape, generator=generator, dtype=torch.float32)
                tfdl_input = model_input
            onnx_inputs.append(np.ascontiguousarray(model_input.numpy().astype(np.float32)))
            tfdl_inputs.append(np.ascontiguousarray(tfdl_input.numpy().astype(np.float32)))
            torch_outputs.append(torch_output_list(model(model_input)))
    return onnx_inputs, tfdl_inputs, torch_outputs


def run_onnx_reference(path: Path, samples: list[np.ndarray]) -> list[list[np.ndarray]]:
    from onnx.reference import ReferenceEvaluator

    model = onnx.load(str(path))
    initializer_names = {item.name for item in model.graph.initializer}
    input_names = [item.name for item in model.graph.input if item.name not in initializer_names]
    if len(input_names) != 1:
        raise ValueError(f"verification expects one ONNX input, got {input_names}")
    evaluator = ReferenceEvaluator(model)
    return [
        [np.asarray(output) for output in evaluator.run(None, {input_names[0]: sample})]
        for sample in samples
    ]


def save_verification_npz(
    path: Path,
    onnx_inputs: list[np.ndarray],
    tfdl_inputs: list[np.ndarray],
    reference_outputs: list[list[np.ndarray]],
    calibration_inputs: list[np.ndarray],
) -> None:
    if not reference_outputs or len(reference_outputs[0]) != 1:
        raise ValueError("current timm verifier expects exactly one model output")
    np.savez_compressed(
        path,
        onnx_inputs=np.stack(onnx_inputs),
        tfdl_inputs=np.stack(tfdl_inputs),
        reference_outputs=np.stack([sample[0] for sample in reference_outputs]),
        calibration_inputs=np.stack(calibration_inputs),
    )


def make_random_calibration_inputs(
    input_shape: tuple[int, int, int, int],
    *,
    count: int,
    seed: int,
    embed_preprocess: bool,
    raw_input_max: float,
) -> list[np.ndarray]:
    rng = np.random.default_rng(seed + 2027)
    if embed_preprocess:
        values = rng.uniform(0.0, raw_input_max, size=(max(1, count), *input_shape)).astype(np.float32)
    else:
        values = rng.normal(0.0, 1.0, size=(max(1, count), *input_shape)).astype(np.float32)
    return [np.ascontiguousarray(item) for item in values]


def load_verification_npz(
    path: str | Path,
    *,
    input_kind: str = "tfdl",
) -> tuple[list[np.ndarray], list[list[np.ndarray]]]:
    with np.load(path) as data:
        if input_kind == "onnx":
            input_key = "onnx_inputs"
        elif input_kind == "calibration":
            input_key = "calibration_inputs"
        else:
            input_key = "tfdl_inputs"
        inputs = [np.ascontiguousarray(item) for item in data[input_key]]
        outputs = [[np.ascontiguousarray(item)] for item in data["reference_outputs"]]
    return inputs, outputs


def slim_onnx(raw_path: Path, slim_path: Path) -> None:
    """Run onnxslim without importing an ABI-incompatible onnxruntime wheel."""

    from onnx.reference import ReferenceEvaluator

    class ReferenceInferenceSession:
        def __init__(self, model: str | bytes, providers=None, **kwargs):
            del providers, kwargs
            if isinstance(model, (bytes, bytearray)):
                model = onnx.load_model_from_string(model)
            elif isinstance(model, str):
                model = onnx.load(model)
            self.evaluator = ReferenceEvaluator(model)

        def run(self, output_names, feeds):
            return self.evaluator.run(output_names or None, feeds)

    runtime_stub = types.ModuleType("onnxruntime")
    runtime_stub.InferenceSession = ReferenceInferenceSession
    sys.modules["onnxruntime"] = runtime_stub

    import onnxslim.utils
    from onnxslim import slim

    onnxslim.utils.is_onnxruntime_available = lambda: False
    slim(str(raw_path), str(slim_path))


def calibration_images(directory: Path | None, limit: int) -> list[str]:
    if directory is None:
        return []
    if not directory.is_dir():
        raise FileNotFoundError(f"calibration directory does not exist: {directory}")
    images = sorted(
        str(path.resolve())
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if limit > 0:
        images = images[:limit]
    if not images:
        raise FileNotFoundError(f"no calibration images found under {directory}")
    return images


def _broadcast_quant_parameter(value: Any, shape: tuple[int, ...]) -> np.ndarray:
    parameter = np.asarray(value, dtype=np.float32).reshape(-1)
    if parameter.size == 1:
        return parameter.reshape((1,) * len(shape))
    if len(shape) >= 2 and parameter.size == shape[1]:
        return parameter.reshape((1, parameter.size, *((1,) * (len(shape) - 2))))
    if parameter.size == shape[-1]:
        return parameter.reshape((*((1,) * (len(shape) - 1)), parameter.size))
    raise ValueError(f"cannot broadcast quant parameter of size {parameter.size} to {shape}")


def _input_for_tfdl_tensor(tensor: Any, value: np.ndarray) -> np.ndarray:
    from TFDL2.Common import TFDataType

    value = np.asarray(value, dtype=np.float32)
    if tensor.dtype == TFDataType.TFDL_UINT8:
        scale = _broadcast_quant_parameter(tensor.qscale, value.shape)
        zero_point = _broadcast_quant_parameter(tensor.qzeropoint, value.shape)
        return np.ascontiguousarray(np.clip(np.round(value / scale + zero_point), 0, 255).astype(np.uint8))
    if tensor.dtype == TFDataType.TFDL_FLOAT16:
        return np.ascontiguousarray(value.astype(np.float16))
    return np.ascontiguousarray(value.astype(np.float32))


def _output_from_tfdl_tensor(tensor: Any) -> np.ndarray:
    from TFDL2.Common import TFDataType

    value = tensor.toNumpy()
    if tensor.dtype == TFDataType.TFDL_UINT8:
        scale = _broadcast_quant_parameter(tensor.qscale, value.shape)
        zero_point = _broadcast_quant_parameter(tensor.qzeropoint, value.shape)
        return np.ascontiguousarray((value.astype(np.float32) - zero_point) * scale)
    return np.ascontiguousarray(value.astype(np.float32))


def execute_tfdl_context(context: Any, samples: list[np.ndarray]) -> list[list[np.ndarray]]:
    from TFDL2 import TFExecutor

    executor = TFExecutor(context, {"UseHardware": False, "FrugalMode": True})
    inputs = executor.GetInputs()
    if len(inputs) != 1:
        raise ValueError(f"verification expects one TFDL input, got {len(inputs)}")
    results: list[list[np.ndarray]] = []
    for sample in samples:
        inputs[0].fromNumpy(_input_for_tfdl_tensor(inputs[0], sample))
        results.append([_output_from_tfdl_tensor(output) for output in executor()])
    return results


def quantize_context_with_samples(
    context: Any,
    samples: list[np.ndarray],
    *,
    calibration_mode: str = "mean",
    merge_concate: bool = False,
    stopquanttensors: tuple[str, ...] = (),
    avoidtensors: tuple[str, ...] = (),
) -> None:
    from TFDL2 import CalibrationMode, TFCalibration
    from TFDL2.Common import TFDataType

    modes = {
        "naive": CalibrationMode.Naive,
        "mean": CalibrationMode.MEAN,
        "kld": CalibrationMode.KLD,
        "coverage": CalibrationMode.COVERAGE,
    }
    calibration = TFCalibration(
        context,
        modes[calibration_mode],
        {"UseHardware": False, "FrugalMode": True},
    )
    inputs = calibration.GetInputs()
    if len(inputs) != 1:
        raise ValueError(f"random calibration expects one TFDL input, got {len(inputs)}")
    for sample in samples:
        inputs[0].fromNumpy(np.ascontiguousarray(sample.astype(np.float32)))
        calibration()
    calibration.Quantize(
        {inputs[0].name: TFDataType.TFDL_UINT8},
        stopquanttensors=stopquanttensors,
        avoidtensors=avoidtensors,
        MergeEltwise=False,
        MergeConcate=merge_concate,
        Perchannel=True,
    )


def child_convert(args: argparse.Namespace) -> int:
    """Build/dump in a child process so native fatal output is observable."""

    python_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(python_root))
    from TFDL2 import DECODER_FLAGS
    from TFConvertor import OnnxConvertor

    convertor = OnnxConvertor(args.context_name)
    convertor.load(args.onnx)
    convertor.optmize()
    mean = json.loads(args.mean_json) if args.mean_json else None
    std = json.loads(args.std_json) if args.std_json else None
    convertor.buildTFmodel(mean=mean, std=std)

    verification_inputs: list[np.ndarray] = []
    if args.verification_npz:
        verification_inputs, _ = load_verification_npz(args.verification_npz, input_kind="calibration")

    if args.float_base:
        convertor.dump(args.float_base)

    if args.quant_base:
        images = json.loads(args.calibration_json)
        if images:
            convertor.quantContext(
                calibration_list=images,
                decoderflags=DECODER_FLAGS.TFCV_RGB,
                MergeConcate=False,
            )
        elif verification_inputs:
            quantize_context_with_samples(
                convertor._context,
                verification_inputs,
                calibration_mode=args.random_calib_mode,
                merge_concate=args.merge_concate,
            )
        else:
            convertor.quantContext(
                calibration_list=None,
                decoderflags=DECODER_FLAGS.TFCV_RGB,
                MergeConcate=False,
            )
        convertor.dump(args.quant_base)
    return 0


def child_verify_fb(args: argparse.Namespace) -> int:
    from TFDL2 import TFContext

    verification_inputs, verification_outputs = load_verification_npz(
        args.verification_npz,
        input_kind="onnx" if args.verification_kind == "int8" else "tfdl",
    )
    context = TFContext(path=args.fb_path)
    actual_outputs = execute_tfdl_context(context, verification_inputs)
    if args.verification_kind == "fp32":
        key = "slim_onnx_vs_tfdl_fp32"
        min_cosine, atol, rtol = args.min_cosine, args.verify_atol, args.verify_rtol
    else:
        key = "slim_onnx_vs_tfdl_int8"
        min_cosine, atol, rtol = args.min_quant_cosine, args.quant_verify_atol, args.quant_verify_rtol
    result = compare_output_batches(
        verification_outputs,
        actual_outputs,
        min_cosine=min_cosine,
        atol=atol,
        rtol=rtol,
        policy="quant" if args.verification_kind == "int8" else "strict",
    )
    payload = {key: result}
    Path(args.verification_json).write_text(json.dumps(payload, indent=2) + "\n")
    print(
        f"[VERIFY] {key}: passed={result['passed']} "
        f"min_cos={result.get('min_cosine_similarity')} max_abs={result.get('max_abs')}"
    )
    return 0 if result["passed"] else 2


def run_fb_verifier(
    fb_path: Path,
    verification_npz: Path,
    verification_json: Path,
    verification_kind: str,
    timeout: int,
    min_cosine: float,
    verify_atol: float,
    verify_rtol: float,
    min_quant_cosine: float,
    quant_verify_atol: float,
    quant_verify_rtol: float,
) -> dict[str, object]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--_verify-fb",
        "--fb-path",
        str(fb_path.resolve()),
        "--verification-npz",
        str(verification_npz.resolve()),
        "--verification-json",
        str(verification_json.resolve()),
        "--verification-kind",
        verification_kind,
        "--min-cosine",
        str(min_cosine),
        "--verify-atol",
        str(verify_atol),
        "--verify-rtol",
        str(verify_rtol),
        "--min-quant-cosine",
        str(min_quant_cosine),
        "--quant-verify-atol",
        str(quant_verify_atol),
        "--quant-verify-rtol",
        str(quant_verify_rtol),
    ]
    env = os.environ.copy()
    python_root = str(Path(__file__).resolve().parents[1])
    env["PYTHONPATH"] = python_root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    completed = subprocess.run(
        command,
        cwd=Path(__file__).resolve().parents[3],
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
        check=False,
    )
    verify_log = verification_json.with_suffix(".log")
    verify_log.write_text(completed.stdout, encoding="utf-8")
    print(completed.stdout, end="")
    fatal = next((line.strip() for line in completed.stdout.splitlines() if "<Fatal>" in line), None)
    if fatal or not verification_json.exists() or completed.returncode not in (0, 2):
        detail = fatal or f"{verification_kind} verifier exited with {completed.returncode}"
        raise RuntimeError(detail)
    return json.loads(verification_json.read_text())


def run_converter_child(
    slim_path: Path,
    output_dir: Path,
    output_name: str,
    output_format: str,
    images: list[str],
    mean: list[float] | None,
    std: list[float] | None,
    timeout: int,
    verification_npz: Path | None,
    min_cosine: float,
    verify_atol: float,
    verify_rtol: float,
    min_quant_cosine: float,
    quant_verify_atol: float,
    quant_verify_rtol: float,
    random_calib_mode: str,
    merge_concate: bool,
) -> tuple[Path | None, Path | None, Path, dict[str, object]]:
    float_base = output_dir / output_name if output_format in {"fb", "both"} else None
    quant_base = output_dir / f"{output_name}.quant" if output_format in {"quant", "both"} else None
    float_fb = Path(f"{float_base}.fb") if float_base else None
    quant_fb = Path(f"{quant_base}.fb") if quant_base else None
    concat_suffix = ".concat" if merge_concate else ".no_concat"
    attempt_suffix = f".{random_calib_mode}{concat_suffix}" if output_format == "quant" and not images else ""
    log_path = output_dir / f"onnx2tfdl.{output_format}{attempt_suffix}.log"
    verification_json = output_dir / f"{output_name}.{output_format}{attempt_suffix}.verification.json"

    for artifact in (float_fb, quant_fb):
        if artifact and artifact.exists():
            artifact.unlink()
    if verification_json.exists():
        verification_json.unlink()

    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--_convert",
        "--onnx",
        str(slim_path.resolve()),
        "--context-name",
        output_name,
        "--calibration-json",
        json.dumps(images),
        "--random-calib-mode",
        random_calib_mode,
        "--merge-concate",
        str(merge_concate).lower(),
    ]
    if float_base:
        command.extend(["--float-base", str(float_base.resolve())])
    if quant_base:
        command.extend(["--quant-base", str(quant_base.resolve())])
    if mean is not None and std is not None:
        command.extend(["--mean-json", json.dumps(mean), "--std-json", json.dumps(std)])
    if verification_npz is not None:
        command.extend(
            [
                "--verification-npz",
                str(verification_npz.resolve()),
                "--verification-json",
                str(verification_json.resolve()),
                "--min-cosine",
                str(min_cosine),
                "--verify-atol",
                str(verify_atol),
                "--verify-rtol",
                str(verify_rtol),
                "--min-quant-cosine",
                str(min_quant_cosine),
                "--quant-verify-atol",
                str(quant_verify_atol),
                "--quant-verify-rtol",
                str(quant_verify_rtol),
            ]
        )
    env = os.environ.copy()
    python_root = str(Path(__file__).resolve().parents[1])
    env["PYTHONPATH"] = python_root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    completed = subprocess.run(
        command,
        cwd=Path(__file__).resolve().parents[3],
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
        check=False,
    )
    log_path.write_text(completed.stdout, encoding="utf-8")
    print(completed.stdout, end="")

    missing = [str(path) for path in (float_fb, quant_fb) if path and not path.is_file()]
    fatal = next((line.strip() for line in completed.stdout.splitlines() if "<Fatal>" in line), None)
    if completed.returncode != 0 or fatal or missing:
        detail = fatal or f"converter exited with {completed.returncode}"
        if missing:
            detail += f"; missing artifacts: {', '.join(missing)}"
        raise RuntimeError(detail)
    verification: dict[str, object] = {}
    if verification_npz is not None:
        artifact = float_fb if output_format == "fb" else quant_fb
        assert artifact is not None
        verification = run_fb_verifier(
            artifact,
            verification_npz,
            verification_json,
            "fp32" if output_format == "fb" else "int8",
            timeout,
            min_cosine,
            verify_atol,
            verify_rtol,
            min_quant_cosine,
            quant_verify_atol,
            quant_verify_rtol,
        )
    return float_fb, quant_fb, log_path, verification


def export_timm(args: argparse.Namespace) -> int:
    if args.list_models:
        print("\n".join(SUPPORTED_MODELS))
        return 0
    if not args.model:
        raise ValueError("--model is required (or use --list-models)")

    explicit_shape = parse_input_shape(args.input_shape)
    fused_attention = args.fused_attn == "true"
    from timm.layers import set_fused_attn

    set_fused_attn(fused_attention)
    torch.manual_seed(args.seed)
    create_kwargs: dict[str, object] = {}
    if explicit_shape is not None:
        _, channels, height, width = explicit_shape
        create_kwargs.update(img_size=(height, width), in_chans=channels)

    print(f"[1/5] create timm model: {args.model} (pretrained={args.pretrained})", flush=True)
    model = timm.create_model(args.model, pretrained=args.pretrained, **create_kwargs).eval()
    data_config = timm.data.resolve_model_data_config(model)
    if explicit_shape is None:
        channels, height, width = (int(value) for value in data_config["input_size"])
        input_shape = (1, channels, height, width)
    else:
        input_shape = explicit_shape

    normalized_mean = list(args.mean or data_config.get("mean", (0.485, 0.456, 0.406)))
    normalized_std = list(args.std or data_config.get("std", (0.229, 0.224, 0.225)))
    if len(normalized_mean) != input_shape[1] or len(normalized_std) != input_shape[1]:
        raise ValueError("mean/std length must equal the input channel count")
    mean = std = None
    if args.embed_preprocess:
        # TFConvertor embeds (raw - mean) / std in the input Placeholder.
        mean = [float(value) * args.raw_input_max for value in normalized_mean]
        std = [float(value) * args.raw_input_max for value in normalized_std]

    verification_count = args.verify_samples if args.verify else 1
    onnx_inputs, tfdl_inputs, torch_outputs = make_verification_samples(
        model,
        input_shape,
        count=verification_count,
        seed=args.seed,
        embed_preprocess=args.embed_preprocess,
        normalized_mean=normalized_mean,
        normalized_std=normalized_std,
        raw_input_max=args.raw_input_max,
    )
    random_calibration_inputs = make_random_calibration_inputs(
        input_shape,
        count=args.random_calib_count,
        seed=args.seed,
        embed_preprocess=args.embed_preprocess,
        raw_input_max=args.raw_input_max,
    )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_name = args.output_name or f"{safe_name(args.model)}_{input_shape[2]}x{input_shape[3]}"
    raw_path = output_dir / f"{output_name}.raw.onnx"
    slim_path = output_dir / f"{output_name}.slim.onnx"
    verification_npz = output_dir / f"{output_name}.verification.npz"
    report_path = output_dir / f"{output_name}.report.json"
    if report_path.exists():
        report_path.unlink()

    sample = torch.from_numpy(onnx_inputs[0])
    print(f"[2/5] export ONNX: input={input_shape}, opset={args.opset}", flush=True)
    with torch.inference_mode():
        # Run once first so unsupported custom resolutions fail before writing
        # a misleading ONNX artifact.
        model(sample)
        torch.onnx.export(
            model,
            (sample,),
            str(raw_path),
            input_names=["input"],
            output_names=["output"],
            opset_version=args.opset,
            dynamo=False,
            do_constant_folding=True,
        )

    print("[3/5] run onnxslim", flush=True)
    slim_onnx(raw_path, slim_path)
    raw_info = inspect_onnx(raw_path)
    slim_info = inspect_onnx(slim_path)

    verification: dict[str, object] = {
        "enabled": bool(args.verify),
        "samples": verification_count,
        "fp32_thresholds": {
            "min_cosine": args.min_cosine,
            "atol": args.verify_atol,
            "rtol": args.verify_rtol,
        },
        "int8_thresholds": {
            "min_cosine": args.min_quant_cosine,
            "atol": args.quant_verify_atol,
            "rtol": args.quant_verify_rtol,
        },
    }
    if args.verify:
        print("[4/5] verify timm, raw ONNX and slim ONNX", flush=True)
        raw_outputs = run_onnx_reference(raw_path, onnx_inputs)
        slim_outputs = run_onnx_reference(slim_path, onnx_inputs)
        verification["timm_vs_raw_onnx"] = compare_output_batches(
            torch_outputs,
            raw_outputs,
            min_cosine=args.min_cosine,
            atol=args.verify_atol,
            rtol=args.verify_rtol,
        )
        verification["timm_vs_slim_onnx"] = compare_output_batches(
            torch_outputs,
            slim_outputs,
            min_cosine=args.min_cosine,
            atol=args.verify_atol,
            rtol=args.verify_rtol,
        )
        verification["raw_onnx_vs_slim_onnx"] = compare_output_batches(
            raw_outputs,
            slim_outputs,
            min_cosine=args.min_cosine,
            atol=args.verify_atol,
            rtol=args.verify_rtol,
        )
        for stage in ("timm_vs_raw_onnx", "timm_vs_slim_onnx", "raw_onnx_vs_slim_onnx"):
            result = verification[stage]
            assert isinstance(result, dict)
            print(
                f"[VERIFY] {stage}: passed={result['passed']} "
                f"min_cos={result.get('min_cosine_similarity')} max_abs={result.get('max_abs')}"
            )
            if not result["passed"]:
                report_path.write_text(json.dumps({"verification": verification}, indent=2) + "\n")
                raise RuntimeError(f"{stage} verification failed")
        save_verification_npz(
            verification_npz,
            onnx_inputs,
            tfdl_inputs,
            slim_outputs,
            random_calibration_inputs,
        )

    images = calibration_images(args.calib_dir, args.calib_count)

    calibration_mode = (
        f"{len(images)} images"
        if images
        else (f"{len(random_calibration_inputs)} deterministic random samples" if args.verify else "SDK random calibration")
    )
    print(f"[5/5] build and verify TFDL: format={args.format}, calibration={calibration_mode}", flush=True)
    converter_kwargs = {
        "slim_path": slim_path,
        "output_dir": output_dir,
        "output_name": output_name,
        "images": images,
        "mean": mean,
        "std": std,
        "timeout": args.timeout,
        "verification_npz": verification_npz if args.verify else None,
        "min_cosine": args.min_cosine,
        "verify_atol": args.verify_atol,
        "verify_rtol": args.verify_rtol,
        "min_quant_cosine": args.min_quant_cosine,
        "quant_verify_atol": args.quant_verify_atol,
        "quant_verify_rtol": args.quant_verify_rtol,
        "random_calib_mode": args.random_calib_mode,
        "merge_concate": args.merge_concate,
    }
    float_fb = quant_fb = None
    log_paths: list[Path] = []
    run_float_stage = args.format in {"fb", "both"} or (args.format == "quant" and args.verify)
    if run_float_stage:
        float_fb, _, float_log, float_verification = run_converter_child(
            output_format="fb",
            **converter_kwargs,
        )
        log_paths.append(float_log)
        verification.update(float_verification)
    selected_calibration_mode: str | None = None
    selected_merge_concate: bool | None = None
    if args.format in {"quant", "both"}:
        if images:
            calibration_configs = [("naive", False)]
        elif args.random_calib_mode == "auto":
            calibration_configs = [("mean", True), ("naive", False)]
        else:
            calibration_configs = [(args.random_calib_mode, args.merge_concate)]
        calibration_attempts: list[dict[str, object]] = []
        for calibration_mode, merge_concate in calibration_configs:
            attempt_kwargs = dict(converter_kwargs)
            attempt_kwargs["random_calib_mode"] = calibration_mode
            attempt_kwargs["merge_concate"] = merge_concate
            _, quant_fb, quant_log, quant_verification = run_converter_child(
                output_format="quant",
                **attempt_kwargs,
            )
            log_paths.append(quant_log)
            quant_result = quant_verification.get("slim_onnx_vs_tfdl_int8", {})
            calibration_attempts.append(
                {"mode": calibration_mode, "merge_concate": merge_concate, "result": quant_result}
            )
            verification.update(quant_verification)
            if not args.verify or bool(quant_result.get("passed")):
                selected_calibration_mode = calibration_mode
                selected_merge_concate = merge_concate
                break
            print(f"[VERIFY] INT8 calibration mode {calibration_mode} failed; trying fallback")
        verification["int8_calibration_attempts"] = calibration_attempts
    if args.format == "quant" and float_fb is not None:
        float_fb.unlink(missing_ok=True)
        float_fb = None

    report = {
        "model": args.model,
        "pretrained": args.pretrained,
        "fused_attention": fused_attention,
        "input_shape": list(input_shape),
        "embedded_preprocess": args.embed_preprocess,
        "raw_onnx": str(raw_path),
        "raw_onnx_info": raw_info,
        "slim_onnx": str(slim_path),
        "slim_onnx_info": slim_info,
        "float_fb": str(float_fb) if float_fb else None,
        "quant_fb": str(quant_fb) if quant_fb else None,
        "calibration_images": len(images),
        "random_calibration_mode": selected_calibration_mode if not images else None,
        "random_calibration_merge_concate": selected_merge_concate if not images else None,
        "logs": [str(path) for path in log_paths],
        "verification": verification,
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("Conversion complete:")
    for path in (float_fb, quant_fb, report_path):
        if path:
            print(f"  {path}")
    failed_stages = [
        name
        for name, result in verification.items()
        if isinstance(result, dict) and "passed" in result and not result["passed"]
    ]
    if failed_stages:
        print(f"Numerical verification failed: {', '.join(failed_stages)}")
        return 2
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", help="timm model name")
    parser.add_argument("--list-models", action="store_true", help="list the models covered by the regression suite")
    parser.add_argument(
        "--pretrained",
        "--pretrain",
        nargs="?",
        const=True,
        default=False,
        type=parse_bool,
        metavar="BOOL",
        help="use timm pretrained weights; accepts --pretrained or --pretrained true/false",
    )
    parser.add_argument("--no-pretrained", "--no-pretrain", action="store_false", dest="pretrained")
    parser.add_argument(
        "--input-shape",
        nargs="+",
        help="fixed N,C,H,W shape; accepts '1,3,224,224' or '1 3 224 224'",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("timm_tfdl_output"))
    parser.add_argument("--output-name", help="artifact basename; default includes model and HxW")
    parser.add_argument("--format", choices=("fb", "quant", "both"), default="both")
    parser.add_argument("--calib-dir", type=Path, help="recursive image directory for quant calibration")
    parser.add_argument("--calib-count", type=int, default=100, help="maximum calibration images; <=0 means all")
    parser.add_argument("--random-calib-count", type=int, default=8, help="deterministic samples when no calibration images are supplied")
    parser.add_argument(
        "--random-calib-mode",
        choices=("auto", "naive", "mean", "kld", "coverage"),
        default="auto",
        help="SDK calibration mode for deterministic random inputs; auto verifies fallbacks",
    )
    parser.add_argument(
        "--merge-concate",
        nargs="?",
        const=True,
        default=False,
        type=parse_bool,
        metavar="BOOL",
        help="merge adjacent concat quantization regions for an explicit calibration mode",
    )
    parser.add_argument("--fused-attn", choices=("true", "false"), default="false")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--timeout", type=int, default=600, help="TFDL child-process timeout in seconds")
    parser.add_argument(
        "--verify",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="numerically compare timm, raw/slim ONNX and TFDL outputs; enabled by default",
    )
    parser.add_argument("--verify-samples", type=int, default=2, help="deterministic samples used for numerical verification")
    parser.add_argument("--min-cosine", type=float, default=0.9999, help="minimum FP32 cosine similarity")
    parser.add_argument("--verify-atol", type=float, default=1e-3, help="FP32 absolute-error tolerance")
    parser.add_argument("--verify-rtol", type=float, default=2e-2, help="FP32 relative-error tolerance")
    parser.add_argument("--min-quant-cosine", type=float, default=0.9, help="minimum INT8 cosine similarity")
    parser.add_argument("--quant-verify-atol", type=float, default=0.1, help="INT8 absolute-error tolerance")
    parser.add_argument("--quant-verify-rtol", type=float, default=0.2, help="INT8 relative-error tolerance")
    parser.add_argument(
        "--embed-preprocess",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="embed timm mean/std for raw [0, raw-input-max] image input",
    )
    parser.add_argument("--raw-input-max", type=float, default=255.0)
    parser.add_argument("--mean", type=float, nargs="+", help="override timm normalized channel means")
    parser.add_argument("--std", type=float, nargs="+", help="override timm normalized channel standard deviations")

    # Private child-process interface.
    parser.add_argument("--_convert", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--_verify-fb", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--onnx", help=argparse.SUPPRESS)
    parser.add_argument("--context-name", help=argparse.SUPPRESS)
    parser.add_argument("--float-base", help=argparse.SUPPRESS)
    parser.add_argument("--quant-base", help=argparse.SUPPRESS)
    parser.add_argument("--calibration-json", default="[]", help=argparse.SUPPRESS)
    parser.add_argument("--mean-json", help=argparse.SUPPRESS)
    parser.add_argument("--std-json", help=argparse.SUPPRESS)
    parser.add_argument("--verification-npz", help=argparse.SUPPRESS)
    parser.add_argument("--verification-json", help=argparse.SUPPRESS)
    parser.add_argument("--fb-path", help=argparse.SUPPRESS)
    parser.add_argument("--verification-kind", choices=("fp32", "int8"), help=argparse.SUPPRESS)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.verify_samples <= 0:
        raise ValueError("--verify-samples must be positive")
    started = time.monotonic()
    try:
        if args._convert:
            return child_convert(args)
        if args._verify_fb:
            return child_verify_fb(args)
        return export_timm(args)
    finally:
        if not args._convert and not args._verify_fb:
            print(f"elapsed: {time.monotonic() - started:.2f}s")


if __name__ == "__main__":
    raise SystemExit(main())
