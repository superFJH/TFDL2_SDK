#!/usr/bin/env python3
"""Export one fixed-grid Qwen3-VL vision graph with a BSC FP16 MatMul trunk.

This is deliberately a separate exporter from ``build_vision_single.py``.  The
existing exporter is the Conv1x1/BCS implementation used by the deployment;
this one is its A@W counterpart for SDK and NPU evaluation:

* residual stream and all parameterized linears use BSC ``[B,S,C]``;
* every checkpoint Linear is stored offline as ``W[K,N]`` and executed as
  ``MatMul(A[B,S,K], W[K,N], transA=false, transB=false)``;
* Q@K, Softmax and A@V retain their FP16 attention layout/semantics.

It intentionally exports FP16 only.  It must not silently replace the
Conv-native deployment profile until its topology and accuracy are reviewed.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from checkpoint import ModelContract, SafeTensorIndex, sha256
from contract import (
    interpolated_vision_position_embedding,
    validate_grid,
    vision_rope,
)


# Keep the BSC exporter fully local to GELab.  These helpers deliberately do
# not import the similarly named Qwen3-VL example exporters: deploy projects
# must remain portable on their own.
_VISION_PREFIX = "model.visual."


def _params(values: dict[str, np.ndarray], precision: str) -> dict[str, np.ndarray]:
    if precision != "fp16":
        raise ValueError(f"BSC MatMul vision only supports fp16, got {precision!r}")
    return {
        name: np.ascontiguousarray(value, dtype=np.float16)
        for name, value in values.items()
    }


def _conv1x1(weight: np.ndarray) -> np.ndarray:
    value = np.asarray(weight, dtype=np.float32)
    if value.ndim != 2:
        raise ValueError(f"expected Linear [out,in], got {value.shape}")
    return np.ascontiguousarray(value[:, :, None, None])


def _load_patch_weights(
    index: SafeTensorIndex,
    contract: ModelContract,
    grid_h: int,
    grid_w: int,
) -> dict[str, np.ndarray]:
    names = (
        _VISION_PREFIX + "patch_embed.proj.weight",
        _VISION_PREFIX + "patch_embed.proj.bias",
        _VISION_PREFIX + "pos_embed.weight",
    )
    source = index.read(names)
    patch = source[names[0]]
    expected = (
        contract.vision_hidden_size,
        3,
        contract.temporal_patch_size,
        contract.vision_patch_size,
        contract.vision_patch_size,
    )
    if tuple(patch.shape) != expected:
        raise ValueError(f"unexpected patch Conv3d shape {patch.shape}, expected {expected}")
    return {
        "patch.weight": np.ascontiguousarray(
            patch.reshape(contract.vision_hidden_size, -1, 1, 1), dtype=np.float32
        ),
        "patch.bias": np.ascontiguousarray(source[names[1]], dtype=np.float32),
        "patch.position": interpolated_vision_position_embedding(
            source[names[2]], grid_h, grid_w, contract.spatial_merge_size
        ),
    }


def _load_block_weights(
    index: SafeTensorIndex, contract: ModelContract, layer: int
) -> dict[str, np.ndarray]:
    if not 0 <= layer < contract.vision_depth:
        raise IndexError(f"vision layer {layer} outside [0,{contract.vision_depth})")
    prefix = _VISION_PREFIX + f"blocks.{layer}."
    suffixes = (
        "norm1.weight",
        "norm1.bias",
        "attn.qkv.weight",
        "attn.qkv.bias",
        "attn.proj.weight",
        "attn.proj.bias",
        "norm2.weight",
        "norm2.bias",
        "mlp.linear_fc1.weight",
        "mlp.linear_fc1.bias",
        "mlp.linear_fc2.weight",
        "mlp.linear_fc2.bias",
    )
    source = index.read(prefix + suffix for suffix in suffixes)

    def get(suffix: str) -> np.ndarray:
        return np.ascontiguousarray(source[prefix + suffix], dtype=np.float32)

    return {
        "norm1.weight": get("norm1.weight"),
        "norm1.bias": get("norm1.bias"),
        "qkv.weight": _conv1x1(get("attn.qkv.weight")),
        "qkv.bias": get("attn.qkv.bias"),
        "proj.weight": _conv1x1(get("attn.proj.weight")),
        "proj.bias": get("attn.proj.bias"),
        "norm2.weight": get("norm2.weight"),
        "norm2.bias": get("norm2.bias"),
        "fc1.weight": _conv1x1(get("mlp.linear_fc1.weight")),
        "fc1.bias": get("mlp.linear_fc1.bias"),
        "fc2.weight": _conv1x1(get("mlp.linear_fc2.weight")),
        "fc2.bias": get("mlp.linear_fc2.bias"),
        "attention.scale": np.asarray(
            [contract.vision_head_dim**-0.5], dtype=np.float32
        ),
    }


def _load_merger_weights(
    index: SafeTensorIndex, contract: ModelContract, kind: str
) -> dict[str, np.ndarray]:
    if kind == "main":
        prefix = _VISION_PREFIX + "merger."
    elif kind.startswith("deepstack_"):
        merger_index = int(kind.split("_", 1)[1])
        if not 0 <= merger_index < len(contract.deepstack_visual_indexes):
            raise IndexError(f"invalid DeepStack merger {kind}")
        prefix = _VISION_PREFIX + f"deepstack_merger_list.{merger_index}."
    else:
        raise ValueError(f"unknown merger kind {kind!r}")
    suffixes = (
        "norm.weight",
        "norm.bias",
        "linear_fc1.weight",
        "linear_fc1.bias",
        "linear_fc2.weight",
        "linear_fc2.bias",
    )
    source = index.read(prefix + suffix for suffix in suffixes)

    def get(suffix: str) -> np.ndarray:
        return np.ascontiguousarray(source[prefix + suffix], dtype=np.float32)

    return {
        "norm.weight": get("norm.weight"),
        "norm.bias": get("norm.bias"),
        "fc1.weight": _conv1x1(get("linear_fc1.weight")),
        "fc1.bias": get("linear_fc1.bias"),
        "fc2.weight": _conv1x1(get("linear_fc2.weight")),
        "fc2.bias": get("linear_fc2.bias"),
    }


def _namespace(
    destination: dict[str, np.ndarray], prefix: str, values: dict[str, np.ndarray]
) -> None:
    for name, value in values.items():
        destination[f"{prefix}.{name}"] = value


def _load_all_weights(
    index: SafeTensorIndex, contract: ModelContract, grid_h: int, grid_w: int
) -> dict[str, np.ndarray]:
    weights = _load_patch_weights(index, contract, grid_h, grid_w)
    for layer in range(contract.vision_depth):
        _namespace(
            weights,
            f"layers.{layer}",
            _load_block_weights(index, contract, layer),
        )
    for kind in ("deepstack_0", "deepstack_1", "deepstack_2", "main"):
        _namespace(weights, f"merger.{kind}", _load_merger_weights(index, contract, kind))
    return weights


def _split_qkv_parameters(
    weights: dict[str, np.ndarray], depth: int
) -> dict[str, np.ndarray]:
    split = dict(weights)
    for layer in range(depth):
        prefix = f"layers.{layer}"
        fused_weight = split.pop(prefix + ".qkv.weight")
        fused_bias = split.pop(prefix + ".qkv.bias")
        for name, weight, bias in zip(
            ("q", "k", "v"),
            np.split(fused_weight, 3, axis=0),
            np.split(fused_bias, 3, axis=0),
        ):
            split[prefix + f".{name}.weight"] = np.ascontiguousarray(weight)
            split[prefix + f".{name}.bias"] = np.ascontiguousarray(bias)
    return split


def _bsc_matmul_weights(
    weights: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, list[int]]]:
    """Convert Conv1x1 [N,K,1,1] checkpoint tensors into MatMul [K,N]."""
    converted: dict[str, np.ndarray] = {}
    source_shapes: dict[str, list[int]] = {}
    for name, value in weights.items():
        array = np.asarray(value)
        if name.endswith(".weight") and array.ndim == 4:
            if tuple(array.shape[2:]) != (1, 1):
                raise ValueError(f"{name}: expected Conv1x1, got {array.shape}")
            source_shapes[name] = [int(array.shape[0]), int(array.shape[1])]
            converted[name] = np.ascontiguousarray(
                array[:, :, 0, 0].transpose(1, 0), dtype=np.float32
            )
        else:
            converted[name] = np.ascontiguousarray(array, dtype=np.float32)
    return converted, source_shapes


def _projection_names(contract: ModelContract) -> list[str]:
    names = ["patch"]
    for layer in range(contract.vision_depth):
        names.extend(
            f"layers.{layer}.{part}"
            for part in ("q", "k", "v", "proj", "fc1", "fc2")
        )
    for kind in ("deepstack_0", "deepstack_1", "deepstack_2", "main"):
        names.extend((f"merger.{kind}.fc1", f"merger.{kind}.fc2"))
    return names


def _audit_bsc_matmuls(
    context: Any,
    symbols: dict[str, str],
    contract: ModelContract,
    source_shapes: dict[str, list[int]],
) -> dict[str, Any]:
    """Prove that all former Conv1x1 projections are FP16 A@W MatMuls."""
    rows: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []
    expected = _projection_names(contract)
    for logical in expected:
        node = symbols.get(logical + ".matmul")
        activation = symbols.get(logical + ".matmul_input")
        weight = symbols.get(logical + ".matmul_weight")
        reasons: list[str] = []
        attribute: dict[str, Any] = {}
        tensor: Any | None = None
        try:
            if node is None or activation is None or weight is None:
                raise KeyError("missing MatMul symbols")
            attribute = json.loads(context._GetAttr(node))
            inputs = [str(value) for value in attribute.get("input", [])]
            parameters = attribute.get("param", {})
            tensor = context.GetParam(weight)
            expected_source_shape = source_shapes.get(logical + ".weight")
            expected_weight_shape = (
                [expected_source_shape[1], expected_source_shape[0]]
                if expected_source_shape is not None
                else None
            )
            if attribute.get("layerType") != "MatMul":
                reasons.append(
                    f"expected MatMul, got {attribute.get('layerType')}"
                )
            if inputs != [activation, weight]:
                reasons.append(
                    "expected [activation, weight] inputs "
                    f"[{activation}, {weight}], got {inputs}"
                )
            if bool(parameters.get("transA")):
                reasons.append("transA must be false")
            if bool(parameters.get("transB")):
                reasons.append("transB must be false")
            if "FLOAT16" not in str(tensor.dtype):
                reasons.append(f"weight must be FLOAT16, got {tensor.dtype}")
            if expected_weight_shape is None:
                reasons.append("source Conv1x1 shape is unavailable")
            elif list(tensor.shape) != expected_weight_shape:
                reasons.append(
                    "weight must be pre-transposed [K,N]="
                    f"{expected_weight_shape}, got {list(tensor.shape)}"
                )
            row: dict[str, Any] = {
                "projection": logical,
                "node": node,
                "operator": attribute.get("layerType"),
                "inputs": inputs,
                "activation_is_left_input": bool(inputs and inputs[0] == activation),
                "weight_is_right_input": bool(
                    len(inputs) == 2 and inputs[1] == weight
                ),
                "trans_a": bool(parameters.get("transA")),
                "trans_b": bool(parameters.get("transB")),
                "weight": weight,
                "weight_dtype": str(tensor.dtype),
                "weight_shape": list(tensor.shape),
                "expected_weight_shape": expected_weight_shape,
                "valid": not reasons,
            }
        except Exception as error:
            reasons.append(f"graph attribute inspection failed: {error}")
            row = {"projection": logical, "node": node, "valid": False}
        if reasons:
            row["reason"] = "; ".join(reasons)
            invalid.append(dict(row))
        rows.append(row)

    patch = next(row for row in rows if row["projection"] == "patch")
    transformer = [row for row in rows if str(row["projection"]).startswith("layers.")]
    merger = [row for row in rows if str(row["projection"]).startswith("merger.")]
    return {
        "format": "qwen3-vl-vision-bsc-matmul-audit-v1",
        "main_trunk_layout": "BSC",
        "projection_operator": "MatMul",
        "projection_layout": "A[B,S,K] @ W[K,N]",
        "weight_pretransposed": True,
        "projection_trans_a": False,
        "projection_trans_b": False,
        "qkv_projection_layout": "split-q-k-v",
        "expected_transformer_logical_projections": contract.vision_depth * 6,
        "covered_transformer_logical_projections": len(transformer),
        "former_conv1x1_projection_count": len(expected),
        "projection_matmul_nodes": len(rows),
        "patch_matmul": patch,
        "transformer_matmul_nodes": len(transformer),
        "merger_matmul_nodes": len(merger),
        "convolution_projection_nodes": 0,
        "invalid_projection_count": len(invalid),
        "ok": not invalid,
        "invalid_projections": invalid,
        "projections": rows,
    }


def build_bsc_matmul_graph(
    contract: ModelContract,
    grid_h: int,
    grid_w: int,
    weights: dict[str, np.ndarray],
    *,
    softmax_fp32: bool,
):
    from TFDL2 import Op, TFContext
    from TFDL2.Common import TFDataType

    sequence = grid_h * grid_w
    merged_sequence = sequence // contract.spatial_merge_size**2
    hidden_size = contract.vision_hidden_size
    heads = contract.vision_heads
    head_dim = contract.vision_head_dim

    split_weights = _split_qkv_parameters(weights, contract.vision_depth)
    graph_weights, source_shapes = _bsc_matmul_weights(split_weights)
    context = TFContext(f"qwen3vl_visual_g{grid_h}x{grid_w}_bsc_matmul")
    context.RegisterParamToContext(**_params(graph_weights, "fp16"))
    rope_sin, rope_cos = vision_rope(
        grid_h, grid_w, head_dim, contract.spatial_merge_size
    )
    # ApplyRope's existing BHDN ABI retains FP32 RoPE tables.  Activations and
    # every parameterized projection remain FP16.
    context.RegisterParamToContext(
        __vision_rope_sin=np.ascontiguousarray(
            np.transpose(rope_sin, (0, 1, 3, 2)), dtype=np.float32
        ),
        __vision_rope_cos=np.ascontiguousarray(
            np.transpose(rope_cos, (0, 1, 3, 2)), dtype=np.float32
        ),
    )
    symbols: dict[str, str] = {}

    def linear(value: Any, logical: str) -> Any:
        weight = context.GetParamSymbol(logical + ".weight")
        bias = context.GetParamSymbol(logical + ".bias")
        symbols[logical + ".matmul_input"] = str(value)
        symbols[logical + ".matmul_weight"] = str(weight)
        product = Op.MatMul(value, weight, transA=False, transB=False)
        symbols[logical + ".matmul"] = str(product)
        return Op.Add(product, bias)

    def merger(value: Any, kind: str, postshuffle_norm: bool) -> Any:
        prefix = f"merger.{kind}"
        merged_width = hidden_size * contract.spatial_merge_size**2
        # Token ordering deliberately mirrors the official checkpoint and the
        # existing Conv exporter: adjacent spatial-merge patches are already
        # contiguous in the processor's patch sequence.
        if postshuffle_norm:
            value = Op.Reshape(value, (1, merged_sequence, merged_width))
            value = Op.LayerNorm2(
                value,
                context.GetParamSymbol(prefix + ".norm.weight"),
                context.GetParamSymbol(prefix + ".norm.bias"),
                axis=2,
                eps=1e-6,
            )
        else:
            value = Op.LayerNorm2(
                value,
                context.GetParamSymbol(prefix + ".norm.weight"),
                context.GetParamSymbol(prefix + ".norm.bias"),
                axis=2,
                eps=1e-6,
            )
            value = Op.Reshape(value, (1, merged_sequence, merged_width))
        value = linear(value, prefix + ".fc1")
        value = Op.GeLU(value)
        return linear(value, prefix + ".fc2")

    with context:
        pixels = Op.Placeholder2(
            context,
            (1, sequence, contract.patch_vector_size),
            TFDataType.TFDL_FLOAT16,
        )
        rope_sin = context.GetParamSymbol("__vision_rope_sin")
        rope_cos = context.GetParamSymbol("__vision_rope_cos")
        hidden = linear(pixels, "patch")
        hidden = Op.Add(hidden, context.GetParamSymbol("patch.position"))

        deep_outputs: list[Any] = []
        for layer in range(contract.vision_depth):
            prefix = f"layers.{layer}"
            residual = hidden
            norm1 = Op.LayerNorm2(
                hidden,
                context.GetParamSymbol(prefix + ".norm1.weight"),
                context.GetParamSymbol(prefix + ".norm1.bias"),
                axis=2,
                eps=1e-6,
            )
            query = linear(norm1, prefix + ".q")
            key = linear(norm1, prefix + ".k")
            value = linear(norm1, prefix + ".v")

            # BSC -> BHDN for ApplyRope, then use the same FP16 QK/Softmax/AV
            # layout as the Conv-native implementation.
            query = Op.Transpose(
                Op.Reshape(query, (1, sequence, heads, head_dim)), (0, 2, 3, 1)
            )
            key = Op.Transpose(
                Op.Reshape(key, (1, sequence, heads, head_dim)), (0, 2, 3, 1)
            )
            query, key = Op.Custom(
                (query, key, rope_sin, rope_cos),
                (
                    f"qwen3vl_vision_l{layer:02d}_q_rope",
                    f"qwen3vl_vision_l{layer:02d}_k_rope",
                ),
                "ApplyRope",
                json.dumps(
                    {
                        "useFp16": True,
                        "inputLayout": "BHDN",
                        "qOutputLayout": "BHND",
                        "kOutputLayout": "BHDN",
                    },
                    separators=(",", ":"),
                ),
            )
            q3 = Op.Reshape(query, (heads, sequence, head_dim))
            k3 = Op.Reshape(key, (heads, head_dim, sequence))
            v3 = Op.Reshape(
                Op.Transpose(
                    Op.Reshape(value, (1, sequence, heads, head_dim)),
                    (0, 2, 1, 3),
                ),
                (heads, sequence, head_dim),
            )
            scores = Op.MatMul(q3, k3, transA=False, transB=False)
            scores = Op.Mul(
                scores, context.GetParamSymbol(prefix + ".attention.scale")
            )
            if softmax_fp32:
                scores = Op.Cast(scores, TFDataType.TFDL_FLOAT)
            probability = Op.Softmax(scores, axis=2)
            if softmax_fp32:
                probability = Op.Cast(probability, TFDataType.TFDL_FLOAT16)
            attention = Op.MatMul(probability, v3, transA=False, transB=False)
            attention = Op.Reshape(attention, (1, heads, sequence, head_dim))
            attention = Op.Reshape(
                Op.Transpose(attention, (0, 2, 1, 3)), (1, sequence, hidden_size)
            )
            hidden = Op.Add(residual, linear(attention, prefix + ".proj"))

            residual = hidden
            norm2 = Op.LayerNorm2(
                hidden,
                context.GetParamSymbol(prefix + ".norm2.weight"),
                context.GetParamSymbol(prefix + ".norm2.bias"),
                axis=2,
                eps=1e-6,
            )
            mlp = linear(norm2, prefix + ".fc1")
            mlp = Op.GeLU(mlp)
            hidden = Op.Add(residual, linear(mlp, prefix + ".fc2"))
            if layer in contract.deepstack_visual_indexes:
                merger_index = contract.deepstack_visual_indexes.index(layer)
                deep_outputs.append(
                    merger(hidden, f"deepstack_{merger_index}", True)
                )
        main_output = merger(hidden, "main", False)

    inputs = [str(pixels)]
    outputs = [str(main_output), *(str(value) for value in deep_outputs)]
    context.SetOutputs(outputs)
    audit = _audit_bsc_matmuls(context, symbols, contract, source_shapes)
    if not audit["ok"]:
        raise RuntimeError(
            "BSC MatMul topology audit failed: "
            + json.dumps(audit["invalid_projections"][:4])
        )
    context.vision_symbols = symbols
    context.vision_topology_audit = audit
    return context, inputs, outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--grid-h", type=int, default=32)
    parser.add_argument("--grid-w", type=int, default=18)
    parser.add_argument(
        "--softmax-fp32",
        action="store_true",
        help="cast scaled Q@K scores to FP32 for Softmax, then restore FP16 for A@V",
    )
    parser.add_argument(
        "--addon-path",
        default=str(Path(__file__).resolve().parents[3] / "AddonOps/build/libTFDLAddOn.so"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from TFDL2.utils import LoadCustomOp

    started = time.perf_counter()
    model_root = Path(args.model_path).resolve()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    contract = ModelContract.from_model(model_root)
    validate_grid(args.grid_h, args.grid_w, contract.spatial_merge_size)
    LoadCustomOp(str(Path(args.addon_path)))
    weights = _load_all_weights(
        SafeTensorIndex(model_root), contract, args.grid_h, args.grid_w
    )
    context, inputs, outputs = build_bsc_matmul_graph(
        contract,
        args.grid_h,
        args.grid_w,
        weights,
        softmax_fp32=args.softmax_fp32,
    )
    profile = f"g{args.grid_h}x{args.grid_w}-fp16-bsc-matmul-splitqkv"
    if args.softmax_fp32:
        profile += "-softmaxfp32"
    base = output / f"vision_{profile}"
    context.Dump(str(base))

    symbols_file = output / "symbols.single.json"
    audit_file = output / "topology.audit.json"
    symbols_file.write_text(json.dumps(context.vision_symbols, indent=2))
    audit_file.write_text(json.dumps(context.vision_topology_audit, indent=2))
    files = sorted(
        path
        for path in output.glob(base.name + "*")
        if path.is_file()
        and (
            path.name.endswith(".fb")
            or path.name.endswith(".param.fb")
            or path.name.endswith(".weights.fb64")
        )
    )
    if not files:
        raise FileNotFoundError(f"TFContext.Dump produced no FB for {base}")
    manifest = {
        "format": "qwen3-vl-vision-single-v1",
        "profile": profile,
        "layout": "single-logical-graph-external-parameters",
        "model_path": str(model_root),
        "grid_h": args.grid_h,
        "grid_w": args.grid_w,
        "sequence": args.grid_h * args.grid_w,
        "merged_sequence": args.grid_h * args.grid_w // contract.spatial_merge_size**2,
        "precision": "fp16",
        "quantization": None,
        "graph_construction": "direct-TFContext-Vit.py-style",
        "main_trunk_layout": "BSC",
        "projection_input_layout": "A[B,S,K]",
        "parameterized_linear_lowering": "MatMul",
        "projection_layout": "A[B,S,K] @ W[K,N]",
        "projection_trans_a": False,
        "projection_trans_b": False,
        "weight_layout": "pre-transposed W[K,N]",
        "qkv_projection_layout": "split-q-k-v",
        "runtime_transpose_policy": (
            "attention layout boundaries only; no projection input/output transpose"
        ),
        "attention_core": (
            "QK-scale-CastFp32-Softmax-CastFp16-AV"
            if args.softmax_fp32
            else "QK-scale-Softmax-AV"
        ),
        "numerics": {
            "activation_dtype": "FP16",
            "weight_dtype": "FP16",
            "qk_softmax_av_dtype": "FP16",
            "softmax_fp32": args.softmax_fp32,
        },
        "topology_audit": {
            key: value
            for key, value in context.vision_topology_audit.items()
            if key != "projections"
        },
        "diagnostics": {
            "symbols": symbols_file.name,
            "topology_audit": audit_file.name,
        },
        "inputs": inputs,
        "outputs": {
            "main": outputs[0],
            **{f"deepstack_{index}": name for index, name in enumerate(outputs[1:])},
        },
        "files": [
            {"name": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in files
        ],
        "total_bytes": sum(path.stat().st_size for path in files),
        "build_seconds": time.perf_counter() - started,
    }
    (output / "manifest.single.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
