#!/usr/bin/env python3
"""Export the Mage-VL Qwen3 text decoder as one ONNX Runtime W8A8 model.

To keep peak disk usage bounded, each decoder layer is exported to temporary
FP32 ONNX, dynamically quantized, and the FP32 file is removed before moving
to the next layer.  The small quantized graph files are then composed into one
decoder graph while their external INT8 weight files stay separate.
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import time
from pathlib import Path

import onnx
import torch
from onnx import compose, external_data_helper, helper
from onnxruntime.quantization import QuantType, quantize_dynamic

import ort_qwen_decoder as decoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--layer", type=int, action="append")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--dummy-past", type=int, default=4)
    parser.add_argument("--keep-fp32", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--compact-only",
        action="store_true",
        help="rewrite existing external-data files without stale appended weights",
    )
    parser.add_argument(
        "--skip-merge",
        action="store_true",
        help="export requested layers only; useful for a one-layer smoke test",
    )
    parser.add_argument(
        "--extract-final-head-only",
        action="store_true",
        help=(
            "extract the already-quantized final RMSNorm/LM-head subgraph "
            "from decoder.w8a8.onnx without rebuilding decoder weights"
        ),
    )
    return parser.parse_args()


def _export_layer(
    module: torch.nn.Module,
    path: Path,
    config: decoder.prefill.QwenPrefillConfig,
    dummy_past: int,
    opset: int,
) -> None:
    if dummy_past <= 0:
        raise ValueError("--dummy-past must be positive")
    args = (
        torch.zeros((1, 1, config.hidden_size), dtype=torch.float32),
        torch.zeros(
            (1, dummy_past, config.num_key_value_heads, config.head_dim),
            dtype=torch.float16,
        ),
        torch.zeros(
            (1, dummy_past, config.num_key_value_heads, config.head_dim),
            dtype=torch.float16,
        ),
        torch.zeros((1, 1, 1, config.head_dim), dtype=torch.float32),
        torch.ones((1, 1, 1, config.head_dim), dtype=torch.float32),
    )
    torch.onnx.export(
        module.eval(),
        args,
        str(path),
        input_names=(
            "hidden",
            "past_key",
            "past_value",
            "position_sin",
            "position_cos",
        ),
        output_names=("hidden_out", "new_key", "new_value"),
        dynamic_axes={
            "past_key": {1: "past_sequence"},
            "past_value": {1: "past_sequence"},
        },
        opset_version=opset,
        dynamo=False,
        do_constant_folding=True,
        external_data=False,
    )


def _export_final_head(
    module: torch.nn.Module,
    path: Path,
    config: decoder.prefill.QwenPrefillConfig,
    opset: int,
) -> None:
    torch.onnx.export(
        module.eval(),
        (torch.zeros((1, 1, config.hidden_size), dtype=torch.float32),),
        str(path),
        input_names=("hidden",),
        output_names=("logits",),
        opset_version=opset,
        dynamo=False,
        do_constant_folding=True,
        external_data=False,
    )


def _quantize(fp32_path: Path, quantized_path: Path) -> None:
    # ORT's external-data writer appends when the target data file already
    # exists. Remove only this builder's exact output files before rebuilding,
    # otherwise repeated --force builds silently multiply artifact size.
    quantized_path.unlink(missing_ok=True)
    quantized_path.with_suffix(quantized_path.suffix + ".data").unlink(missing_ok=True)
    quantize_dynamic(
        fp32_path,
        quantized_path,
        op_types_to_quantize=("MatMul",),
        per_channel=True,
        reduce_range=False,
        weight_type=QuantType.QInt8,
        use_external_data_format=True,
        extra_options={
            # Only constant-weight projections/MLP/LM-head are W8A8. QK and AV
            # keep floating-point attention semantics in this first decoder.
            "MatMulConstBOnly": True,
        },
    )


def _external_locations(path: Path) -> list[Path]:
    model = onnx.load_model(path, load_external_data=False)
    locations = set()
    for initializer in model.graph.initializer:
        if not external_data_helper.uses_external_data(initializer):
            continue
        for entry in initializer.external_data:
            if entry.key == "location":
                locations.add(path.parent / entry.value)
    return sorted(locations)


def _compact_external_model(path: Path) -> dict[str, int]:
    locations = _external_locations(path)
    if len(locations) != 1:
        raise ValueError(
            f"{path} must use exactly one external-data file, got {locations}"
        )
    data_path = locations[0]
    before = data_path.stat().st_size
    model = onnx.load_model(path, load_external_data=True)
    backup = data_path.with_name(data_path.name + ".precompact")
    temporary = path.with_name(path.name + ".compact")
    if backup.exists() or temporary.exists():
        raise FileExistsError("stale compaction temporary/backup file")
    data_path.rename(backup)
    try:
        onnx.save_model(
            model,
            temporary,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=data_path.name,
            size_threshold=1024,
        )
        onnx.checker.check_model(str(temporary), full_check=False)
        path.unlink()
        temporary.rename(path)
        backup.unlink()
    except BaseException:
        temporary.unlink(missing_ok=True)
        data_path.unlink(missing_ok=True)
        backup.rename(data_path)
        raise
    return {"before_bytes": before, "after_bytes": data_path.stat().st_size}


def _replace_edge(model: onnx.ModelProto, old: str, new: str) -> None:
    if old == new:
        return
    graph = model.graph
    for node in graph.node:
        for index, value in enumerate(node.input):
            if value == old:
                node.input[index] = new
        for index, value in enumerate(node.output):
            if value == old:
                node.output[index] = new
    for collection in (graph.input, graph.output, graph.value_info):
        for item in collection:
            if item.name == old:
                item.name = new


def _fix_external_locations(model: onnx.ModelProto, directory: str) -> None:
    for initializer in model.graph.initializer:
        if not external_data_helper.uses_external_data(initializer):
            continue
        for entry in initializer.external_data:
            if entry.key != "location":
                continue
            location = Path(entry.value)
            if location.is_absolute():
                raise ValueError(
                    f"absolute ONNX external-data path is not relocatable: {location}"
                )
            if location.parts[:1] != (directory,):
                entry.value = str(Path(directory) / location)


def _append_unique_value_info(
    destination: list,
    seen: set[str],
    values,
) -> None:
    for value in values:
        if value.name in seen:
            continue
        seen.add(value.name)
        destination.append(value)


def compose_decoder(
    layer_paths: list[Path],
    final_path: Path,
    output_path: Path,
) -> dict[str, int]:
    nodes = []
    initializers = []
    sparse_initializers = []
    inputs = []
    outputs = []
    value_infos = []
    input_names: set[str] = set()
    output_names: set[str] = set()
    value_info_names: set[str] = set()
    functions = []
    opsets: dict[str, int] = {}
    ir_version = 0
    previous_hidden = decoder.HIDDEN_INPUT

    for layer, path in enumerate(layer_paths):
        model = onnx.load_model(path, load_external_data=False)
        _fix_external_locations(model, "layers")
        prefix = f"layer_{layer:02d}/"
        model = compose.add_prefix(model, prefix)
        replacements = {
            prefix + "hidden": previous_hidden,
            prefix + "past_key": decoder.past_key_name(layer),
            prefix + "past_value": decoder.past_value_name(layer),
            prefix + "position_sin": decoder.SIN_INPUT,
            prefix + "position_cos": decoder.COS_INPUT,
            prefix + "hidden_out": f"layer_{layer:02d}.hidden_out",
            prefix + "new_key": decoder.present_key_name(layer),
            prefix + "new_value": decoder.present_value_name(layer),
        }
        for old, new in replacements.items():
            _replace_edge(model, old, new)
        graph = model.graph
        nodes.extend(graph.node)
        initializers.extend(graph.initializer)
        sparse_initializers.extend(graph.sparse_initializer)
        functions.extend(model.functions)
        for imported in model.opset_import:
            opsets[imported.domain] = max(
                opsets.get(imported.domain, 0), int(imported.version)
            )
        ir_version = max(ir_version, int(model.ir_version))
        for value in graph.input:
            if layer > 0 and value.name == previous_hidden:
                continue
            if value.name not in input_names:
                input_names.add(value.name)
                inputs.append(value)
        for value in graph.output:
            if value.name == f"layer_{layer:02d}.hidden_out":
                continue
            if value.name not in output_names:
                output_names.add(value.name)
                outputs.append(value)
        _append_unique_value_info(value_infos, value_info_names, graph.value_info)
        previous_hidden = f"layer_{layer:02d}.hidden_out"

    model = onnx.load_model(final_path, load_external_data=False)
    _fix_external_locations(model, "layers")
    prefix = "final/"
    model = compose.add_prefix(model, prefix)
    _replace_edge(model, prefix + "hidden", previous_hidden)
    _replace_edge(model, prefix + "logits", decoder.LOGITS_OUTPUT)
    graph = model.graph
    nodes.extend(graph.node)
    initializers.extend(graph.initializer)
    sparse_initializers.extend(graph.sparse_initializer)
    functions.extend(model.functions)
    for imported in model.opset_import:
        opsets[imported.domain] = max(
            opsets.get(imported.domain, 0), int(imported.version)
        )
    ir_version = max(ir_version, int(model.ir_version))
    _append_unique_value_info(value_infos, value_info_names, graph.value_info)
    for value in graph.output:
        if value.name not in output_names:
            output_names.add(value.name)
            outputs.append(value)

    combined_graph = helper.make_graph(
        nodes,
        "MageQwen3W8A8ExternalKvDecoder",
        inputs,
        outputs,
        initializer=initializers,
        value_info=value_infos,
        sparse_initializer=sparse_initializers,
    )
    combined = helper.make_model(
        combined_graph,
        opset_imports=[
            helper.make_opsetid(domain, version)
            for domain, version in sorted(opsets.items())
        ],
        functions=functions,
        producer_name="Mage-Vit Qwen-decode-ort",
    )
    combined.ir_version = ir_version
    onnx.save_model(combined, output_path)
    onnx.checker.check_model(str(output_path), full_check=False)
    return {
        "nodes": len(nodes),
        "initializers": len(initializers),
        "inputs": len(inputs),
        "outputs": len(outputs),
    }


def extract_final_head_model(
    decoder_path: Path,
    output_path: Path,
    config: decoder.prefill.QwenPrefillConfig,
) -> dict[str, int]:
    """Extract the W8A8 final head while reusing its existing data shard."""
    model = onnx.load_model(decoder_path, load_external_data=False)
    nodes = [
        copy.deepcopy(node)
        for node in model.graph.node
        if node.name.startswith("final/")
    ]
    initializers = [
        copy.deepcopy(value)
        for value in model.graph.initializer
        if value.name.startswith("final/")
    ]
    if not nodes or not initializers:
        raise RuntimeError(f"{decoder_path} does not contain a final/ subgraph")
    produced = {name for node in nodes for name in node.output}
    initializer_names = {value.name for value in initializers}
    external_inputs = {
        name
        for node in nodes
        for name in node.input
        if name and name not in produced and name not in initializer_names
    }
    hidden_edge = f"layer_{config.num_hidden_layers - 1:02d}.hidden_out"
    if external_inputs != {hidden_edge}:
        raise RuntimeError(
            "unexpected final-head external inputs: "
            f"{sorted(external_inputs)}"
        )
    for node in nodes:
        for index, name in enumerate(node.input):
            if name == hidden_edge:
                node.input[index] = "hidden"
    logits = next(
        (copy.deepcopy(value) for value in model.graph.output if value.name == "logits"),
        None,
    )
    if logits is None:
        raise RuntimeError(f"{decoder_path} does not expose logits")
    graph = helper.make_graph(
        nodes,
        "MageQwen3W8A8FinalHead",
        [
            helper.make_tensor_value_info(
                "hidden",
                onnx.TensorProto.FLOAT,
                [1, 1, config.hidden_size],
            )
        ],
        [logits],
        initializer=initializers,
    )
    extracted = helper.make_model(
        graph,
        opset_imports=[copy.deepcopy(value) for value in model.opset_import],
        functions=[copy.deepcopy(value) for value in model.functions],
        producer_name="Mage-Vit Qwen-decode-ort final head extractor",
    )
    extracted.ir_version = model.ir_version
    onnx.save_model(extracted, output_path)
    onnx.checker.check_model(str(output_path), full_check=False)
    return {
        "nodes": len(nodes),
        "initializers": len(initializers),
        "inputs": 1,
        "outputs": 1,
    }


def _audit_quantized(path: Path) -> dict[str, object]:
    model = onnx.load_model(path, load_external_data=False)
    counts: dict[str, int] = {}
    for node in model.graph.node:
        counts[node.op_type] = counts.get(node.op_type, 0) + 1
    quantized = counts.get("MatMulInteger", 0) + counts.get("MatMulIntegerToFloat", 0)
    if quantized == 0:
        raise RuntimeError(f"{path} contains no integer MatMul after quantization")
    return {
        "file": str(path),
        "node_types": dict(sorted(counts.items())),
        "integer_matmuls": quantized,
    }


def main() -> None:
    args = parse_args()
    if args.opset < 17:
        raise ValueError("Qwen decoder export requires ONNX opset >=17")
    root = Path(args.model_path)
    output = Path(args.output_dir)
    layer_dir = output / "layers"
    layer_dir.mkdir(parents=True, exist_ok=True)
    config = decoder.prefill.QwenPrefillConfig.from_model(root)
    if args.extract_final_head_only:
        decoder_path = output / "decoder.w8a8.onnx"
        head_path = output / "final_head.w8a8.onnx"
        result = extract_final_head_model(decoder_path, head_path, config)
        manifest_path = output / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["final_head_model"] = head_path.name
        manifest["final_head_graph"] = result
        manifest_path.write_text(json.dumps(manifest, indent=2))
        print(json.dumps({"model": str(head_path), "graph": result}, indent=2))
        return
    layers = decoder.validate_layers(
        args.layer or range(config.num_hidden_layers), config.num_hidden_layers
    )
    index = decoder.prefill.SafeTensorIndex(root)
    report: dict[str, object] = {
        "model_path": str(root),
        "output_dir": str(output),
        "layers": [],
        "started_unix": time.time(),
    }

    if args.compact_only:
        for layer in layers:
            path = layer_dir / f"layer_{layer:02d}.w8a8.onnx"
            result = _compact_external_model(path)
            print(
                f"compact layer {layer:02d}: "
                f"{result['before_bytes']} -> {result['after_bytes']}",
                flush=True,
            )
        if not args.skip_merge:
            if layers != list(range(config.num_hidden_layers)):
                raise ValueError("full compaction/merge requires every layer")
            final_path = layer_dir / "final_head.w8a8.onnx"
            result = _compact_external_model(final_path)
            print(
                "compact final head: "
                f"{result['before_bytes']} -> {result['after_bytes']}",
                flush=True,
            )

    for layer in layers:
        fp32_path = layer_dir / f"layer_{layer:02d}.fp32.onnx"
        quantized_path = layer_dir / f"layer_{layer:02d}.w8a8.onnx"
        started = time.perf_counter()
        if args.force or not quantized_path.exists():
            weights = decoder.prefill.load_layer_weights(root, layer, index)
            module = decoder.Qwen3DecodeLayer(config, layer, weights)
            _export_layer(module, fp32_path, config, args.dummy_past, args.opset)
            del module, weights
            gc.collect()
            _quantize(fp32_path, quantized_path)
            if not args.keep_fp32:
                fp32_path.unlink(missing_ok=True)
        audit = _audit_quantized(quantized_path)
        item = {
            "layer": layer,
            "seconds": time.perf_counter() - started,
            **audit,
        }
        report["layers"].append(item)
        print(
            f"layer {layer:02d}: {item['seconds']:.2f}s, "
            f"integer_matmuls={audit['integer_matmuls']}",
            flush=True,
        )

    if args.skip_merge:
        report["total_seconds"] = time.time() - float(report["started_unix"])
        (output / "build.report.json").write_text(json.dumps(report, indent=2))
        return
    expected = list(range(config.num_hidden_layers))
    if layers != expected:
        raise ValueError("a merged decoder requires every layer in order")

    final_fp32 = layer_dir / "final_head.fp32.onnx"
    final_quantized = layer_dir / "final_head.w8a8.onnx"
    if args.force or not final_quantized.exists():
        norm_name, head_name = decoder.prefill.final_weight_names(index)
        final_weights = index.read((norm_name, head_name))
        final_module = decoder.Qwen3FinalHead(
            config, final_weights[norm_name], final_weights[head_name]
        )
        _export_final_head(final_module, final_fp32, config, args.opset)
        del final_module, final_weights
        gc.collect()
        _quantize(final_fp32, final_quantized)
        if not args.keep_fp32:
            final_fp32.unlink(missing_ok=True)
    report["final_head"] = _audit_quantized(final_quantized)

    model_path = output / "decoder.w8a8.onnx"
    layer_paths = [layer_dir / f"layer_{layer:02d}.w8a8.onnx" for layer in expected]
    report["combined"] = compose_decoder(layer_paths, final_quantized, model_path)
    final_head_path = output / "final_head.w8a8.onnx"
    report["standalone_final_head"] = extract_final_head_model(
        model_path, final_head_path, config
    )
    quantization = {
        "profile": "dynamic-u8s8-w8a8",
        "weight_type": "QInt8",
        "activation_type": "QUInt8-dynamic",
        "per_channel_weights": True,
        "quantized_ops": ["constant-weight MatMul"],
        "floating_ops": ["RMSNorm", "RoPE", "QK", "Softmax", "AV", "residual"],
        "onnxruntime_version": __import__("onnxruntime").__version__,
    }
    manifest = decoder.decoder_manifest(
        root, config, model_file=model_path.name, quantization=quantization
    )
    manifest["final_head_model"] = final_head_path.name
    manifest["final_head_graph"] = report["standalone_final_head"]
    manifest["graph"] = report["combined"]
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    report["total_seconds"] = time.time() - float(report["started_unix"])
    (output / "build.report.json").write_text(json.dumps(report, indent=2))
    print(
        json.dumps(
            {
                "model": str(model_path),
                "manifest": str(output / "manifest.json"),
                "graph": report["combined"],
                "total_seconds": report["total_seconds"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
