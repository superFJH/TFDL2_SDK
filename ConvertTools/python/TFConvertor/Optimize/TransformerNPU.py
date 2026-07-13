from ..TFConvertor import TFConvertor
import math
import numpy as np
from typing import Optional, Tuple


def MergeAdjacentConcatParamsOnnx(Model: TFConvertor) -> int:
    """Fold adjacent constant Concat inputs into one parameter tensor.

    Distilled DeiT prepends both cls_token and dist_token as separate constant
    inputs. The SDK quantizer can align one constant prefix with the activation
    input, but a second constant input remains float and makes quantized Concat
    construction fail. Combining adjacent constants preserves the float graph
    exactly and presents the same two-input pattern as ordinary ViT.
    """

    merged = 0
    replaced_params = set()
    for node in Model.nodes.values():
        if node.op != "Concat" or len(node.inputs) < 2:
            continue
        inputs = list(node.inputs)
        axis = int(node.attr.get("axis", 0))
        rewritten = []
        index = 0
        group_id = 0
        while index < len(inputs):
            if inputs[index] not in Model.params:
                rewritten.append(inputs[index])
                index += 1
                continue
            end = index + 1
            while end < len(inputs) and inputs[end] in Model.params:
                end += 1
            names = inputs[index:end]
            arrays = [Model.params[name] for name in names]
            if len(arrays) < 2 or not all(isinstance(array, np.ndarray) for array in arrays):
                rewritten.extend(names)
                index = end
                continue
            rank = arrays[0].ndim
            normalized_axis = axis % rank
            compatible = all(
                array.ndim == rank
                and array.dtype == arrays[0].dtype
                and all(
                    left == right
                    for dim, (left, right) in enumerate(zip(arrays[0].shape, array.shape))
                    if dim != normalized_axis
                )
                for array in arrays[1:]
            )
            if not compatible:
                rewritten.extend(names)
                index = end
                continue
            base = node.outputs[0] if node.outputs else node.name
            merged_name = f"{base}__merged_concat_param_{group_id}"
            while merged_name in Model.params:
                group_id += 1
                merged_name = f"{base}__merged_concat_param_{group_id}"
            Model.params[merged_name] = np.ascontiguousarray(np.concatenate(arrays, axis=normalized_axis))
            rewritten.append(merged_name)
            replaced_params.update(names)
            merged += len(names) - 1
            group_id += 1
            index = end
        node.inputs = rewritten

    used = {name for node in Model.nodes.values() for name in node.inputs}
    for name in replaced_params:
        if name not in used:
            Model.params.pop(name, None)
    if merged:
        print(f"[TransformerNPUOptimize] merge {merged} adjacent Concat constant inputs")
    return merged


def _tensor_shapes(Model: TFConvertor) -> dict:
    shapes = {}
    try:
        import onnx

        inferred = onnx.shape_inference.infer_shapes(Model.onnxmodel)
        values = []
        values.extend(inferred.graph.input)
        values.extend(inferred.graph.value_info)
        values.extend(inferred.graph.output)
        for value in values:
            tensor_type = value.type.tensor_type
            if not tensor_type.HasField("shape"):
                continue
            dims = []
            for dim in tensor_type.shape.dim:
                if dim.dim_value > 0:
                    dims.append(int(dim.dim_value))
                else:
                    dims.append(-1)
            if dims:
                shapes[value.name] = dims
    except Exception:
        pass

    def known_size(shape):
        if shape is None or any(dim <= 0 for dim in shape):
            return None
        return int(np.prod(shape))

    def resolve_reshape(target, input_shape):
        if target is None:
            return None
        result = [int(dim) for dim in target]
        if input_shape is not None:
            for index, dim in enumerate(result):
                if dim == 0 and index < len(input_shape) and input_shape[index] > 0:
                    result[index] = input_shape[index]
        unknown = [index for index, dim in enumerate(result) if dim < 0]
        input_size = known_size(input_shape)
        if len(unknown) == 1 and input_size is not None:
            known = int(np.prod([dim for dim in result if dim > 0]))
            if known > 0 and input_size % known == 0:
                result[unknown[0]] = input_size // known
        return result

    def better(name, candidate):
        if not candidate:
            return False
        current = shapes.get(name)
        if current is None:
            shapes[name] = candidate
            return True
        merged = []
        changed = False
        if len(current) != len(candidate):
            return False
        for old, new in zip(current, candidate):
            if old <= 0 and new > 0:
                merged.append(int(new))
                changed = True
            else:
                merged.append(int(old))
        if changed:
            shapes[name] = merged
        return changed

    # Keep shapes recovered by an earlier optimizer/refresh iteration.  ONNX
    # inference starts from the original symbolic graph each time and would
    # otherwise discard the concrete window batch/sequence geometry already
    # attached to fused attention projections.
    for name, candidate in getattr(Model, "tensor_shapes", {}).items():
        better(name, list(candidate))

    passthrough_ops = {
        "Add", "Sub", "Mul", "Div", "Mod", "GeLU", "Swish", "Relu", "ReLU",
        "Softmax", "LayerNormalization", "LayerNorm", "Identity", "Cast", "Scale",
    }
    nodes = list(Model.nodes.values())
    # onnxslim leaves some statically resolvable Swin window dimensions as
    # symbolic. Propagate the fixed Reshape/Transpose/MatMul topology so the
    # NPU rewrite never substitutes a semantically incorrect dimension of 1.
    for _ in range(8):
        changed = False
        for node in nodes:
            input_shapes = [
                shapes.get(name, list(Model.params[name].shape) if name in Model.params else None)
                for name in node.inputs
            ]
            candidates = []
            if node.op == "Reshape" and input_shapes:
                target = None
                if len(node.inputs) > 1 and node.inputs[1] in Model.params:
                    target = Model.params[node.inputs[1]].reshape(-1).tolist()
                elif node.outputs:
                    target = shapes.get(node.outputs[0])
                candidates = [resolve_reshape(target, input_shapes[0])]
            elif node.op == "Transpose" and input_shapes and input_shapes[0] is not None:
                perm = node.attr.get("perm", list(reversed(range(len(input_shapes[0])))))
                candidates = [[input_shapes[0][index] for index in perm]]
            elif node.op in ("ReduceMean", "ReduceSum", "ReduceMax", "ReduceMin") and input_shapes and input_shapes[0] is not None:
                axes = node.attr.get("axes")
                if axes is None and len(node.inputs) > 1 and node.inputs[1] in Model.params:
                    axes = Model.params[node.inputs[1]].reshape(-1).tolist()
                if axes is not None:
                    input_rank = len(input_shapes[0])
                    normalized = {int(axis) % input_rank for axis in axes}
                    keepdims = bool(node.attr.get("keepdims", 1))
                    if keepdims:
                        candidates = [[
                            1 if index in normalized else dim
                            for index, dim in enumerate(input_shapes[0])
                        ]]
                    else:
                        candidates = [[
                            dim for index, dim in enumerate(input_shapes[0]) if index not in normalized
                        ]]
            elif node.op in passthrough_ops and input_shapes:
                activation_shapes = [shape for name, shape in zip(node.inputs, input_shapes) if name not in Model.params and shape]
                candidates = [max(activation_shapes, key=len) if activation_shapes else input_shapes[0]]
            elif node.op in ("MatMul", "Gemm") and input_shapes:
                left = input_shapes[0]
                existing = shapes.get(node.outputs[0]) if node.outputs else None
                if left is not None and existing is not None and len(existing) == len(left):
                    candidate = list(left[:-1]) + [existing[-1]]
                    candidates = [candidate]
                elif left is not None and len(input_shapes) > 1 and input_shapes[1] is not None:
                    right = input_shapes[1]
                    if len(left) >= 2 and len(right) >= 2:
                        candidates = [list(left[:-1]) + [right[-1]]]
            elif node.op == "Squeeze" and input_shapes and input_shapes[0] is not None:
                axes = node.attr.get("axes")
                if axes is None and len(node.inputs) > 1 and node.inputs[1] in Model.params:
                    axes = Model.params[node.inputs[1]].reshape(-1).tolist()
                if axes is not None:
                    normalized = {int(axis) % len(input_shapes[0]) for axis in axes}
                    candidates = [[dim for index, dim in enumerate(input_shapes[0]) if index not in normalized]]
            elif node.op == "Unsqueeze" and input_shapes and input_shapes[0] is not None:
                axes = node.attr.get("axes")
                if axes is None and len(node.inputs) > 1 and node.inputs[1] in Model.params:
                    axes = Model.params[node.inputs[1]].reshape(-1).tolist()
                if axes is not None:
                    result = list(input_shapes[0])
                    for axis in sorted(int(value) for value in axes):
                        result.insert(axis, 1)
                    candidates = [result]
            elif node.op == "Split" and input_shapes and input_shapes[0] is not None:
                candidates = [list(input_shapes[0]) for _ in node.outputs]
            elif node.op == "Concat" and input_shapes and all(shape is not None for shape in input_shapes):
                rank = len(input_shapes[0])
                axis = int(node.attr.get("axis", 0)) % rank
                if all(len(shape) == rank for shape in input_shapes):
                    result = list(input_shapes[0])
                    if all(shape[axis] > 0 for shape in input_shapes):
                        result[axis] = sum(shape[axis] for shape in input_shapes)
                    candidates = [result]
            elif node.op == "Slice" and input_shapes and input_shapes[0] is not None and len(node.inputs) >= 3:
                if all(name in Model.params for name in node.inputs[1:] if name):
                    starts = Model.params[node.inputs[1]].reshape(-1).tolist()
                    ends = Model.params[node.inputs[2]].reshape(-1).tolist()
                    axes = (
                        Model.params[node.inputs[3]].reshape(-1).tolist()
                        if len(node.inputs) > 3 and node.inputs[3] else list(range(len(starts)))
                    )
                    steps = (
                        Model.params[node.inputs[4]].reshape(-1).tolist()
                        if len(node.inputs) > 4 and node.inputs[4] else [1] * len(starts)
                    )
                    result = list(input_shapes[0])
                    valid = True
                    for axis, start, end, step in zip(axes, starts, ends, steps):
                        axis = int(axis) % len(result)
                        if result[axis] <= 0:
                            valid = False
                            break
                        start, stop, step = slice(int(start), int(end), int(step)).indices(result[axis])
                        result[axis] = len(range(start, stop, step))
                    if valid:
                        candidates = [result]
            elif node.op == "Pad" and input_shapes and input_shapes[0] is not None and len(node.inputs) > 1:
                if node.inputs[1] in Model.params:
                    pads = Model.params[node.inputs[1]].reshape(-1).tolist()
                    rank = len(input_shapes[0])
                    if len(pads) == 2 * rank:
                        candidates = [[
                            dim + int(pads[index]) + int(pads[index + rank]) if dim > 0 else dim
                            for index, dim in enumerate(input_shapes[0])
                        ]]

            for output, candidate in zip(node.outputs, candidates):
                changed = better(output, candidate) or changed
        if not changed:
            break

    input_batches = [attr["shape"][0] for attr in Model.inputs.values() if attr.get("shape")]
    default_batch = int(input_batches[0]) if input_batches else 1
    linear_groups = {}
    linear_entries = []
    for node in nodes:
        if not _is_param_linear(Model, node):
            continue
        shape = shapes.get(node.inputs[0])
        output_shape = shapes.get(node.outputs[0]) if node.outputs else None
        # Transformer projections operate on rank-3 token tensors. Rank-4
        # Swin MLP inputs are already handled by regular inferred shapes.
        rank = len(shape) if shape is not None else (len(output_shape) if output_shape is not None else 0)
        if rank != 3:
            continue
        weight = Model.params[node.inputs[1]]
        known_channel = shape[-1] if shape is not None and shape[-1] > 0 else int(min(weight.shape))
        role = "attention" if "/attn/" in node.name else ("mlp" if "/mlp/" in node.name else "other")
        # Do not mix equal channel widths from different hierarchical stages
        # (for example Swin stage-0 fc2 has C=384 while stage-2 fc1 also has
        # C=384, but their token counts are 3136 and 196 respectively).
        scope = node.name.split("/blocks/")[0]
        group = linear_groups.setdefault((scope, known_channel, role), {"batches": [], "seqs": []})
        if shape is not None and shape[0] > 0:
            group["batches"].append(int(shape[0]))
        if shape is not None and shape[1] > 0:
            group["seqs"].append(int(shape[1]))
        linear_entries.append((node, scope, known_channel, role))

    for node, scope, channels, role in linear_entries:
        group = linear_groups[(scope, channels, role)]
        if not group["seqs"]:
            continue
        # Blocks in the same Swin stage share C and window size. A concrete
        # first block therefore resolves later shifted blocks whose shape-only
        # roll/pad chain stayed symbolic after onnxslim.
        batch = (max(group["batches"]) if role == "attention" else min(group["batches"])) if group["batches"] else default_batch
        # Window attention uses the smallest repeated token group for a given
        # channel width; larger rank-3 entries at the same C are flattened
        # spatial MLP/downsample projections and already have concrete shapes.
        seq_len = min(group["seqs"]) if role == "attention" else max(group["seqs"])
        current = shapes.get(node.inputs[0])
        if current is not None and current[0] > 0 and current[1] > 0:
            # Preserve fully known non-window projections.
            continue
        shapes[node.inputs[0]] = [batch, seq_len, channels]
        if node.outputs:
            old_output = shapes.get(node.outputs[0])
            out_channels = old_output[-1] if old_output is not None and old_output[-1] > 0 else int(max(Model.params[node.inputs[1]].shape))
            shapes[node.outputs[0]] = [batch, seq_len, out_channels]
    if hasattr(Model, "tensor_shapes"):
        Model.tensor_shapes.update({name: list(shape) for name, shape in shapes.items() if shape})
    return shapes


def RefreshTransformerShapesOnnx(Model: TFConvertor) -> dict:
    """Re-run fixed-shape propagation after shape constants are folded."""
    shapes = _tensor_shapes(Model)
    for node in Model.nodes.values():
        if node.op not in ("LinearConv", "MLPLinearConv") or not node.inputs or not node.outputs:
            continue
        input_shape = shapes.get(node.inputs[0])
        if input_shape is None or len(input_shape) < 2 or any(dim <= 0 for dim in input_shape):
            continue
        conv_input_shape = [1] + list(input_shape) if len(input_shape) == 2 else list(input_shape)
        out_channels = int(node.attr["out_channels"])
        output_shape = list(input_shape[:-1]) + [out_channels]
        _set_linear_conv_attrs(
            node,
            conv_input_shape,
            output_shape,
            int(conv_input_shape[-1]),
            out_channels,
        )
        shapes[node.outputs[0]] = output_shape
        Model.tensor_shapes[node.outputs[0]] = output_shape
    return shapes


def _consumers(nodes):
    users = {}
    for node in nodes:
        for iname in node.inputs:
            users.setdefault(iname, []).append(node)
    return users


def _producer(nodes):
    result = {}
    for node in nodes:
        for oname in node.outputs:
            result[oname] = node
    return result


def _is_param_linear(Model: TFConvertor, node) -> bool:
    if node.op not in ("MatMul", "Gemm"):
        return False
    if len(node.inputs) < 2 or node.inputs[1] not in Model.params:
        return False
    weight = Model.params[node.inputs[1]]
    return isinstance(weight, np.ndarray) and weight.ndim == 2 and node.inputs[0] not in Model.params


def _is_transformer_like(Model: TFConvertor) -> bool:
    nodes = list(Model.nodes.values())
    users = _consumers(nodes)
    has_attention_softmax = False
    for node in nodes:
        if node.op not in ("Softmax", "MaskSoftmax"):
            continue
        for out in node.outputs:
            if any(user.op == "MatMul" for user in users.get(out, [])):
                has_attention_softmax = True
                break
        if has_attention_softmax:
            break
    linear_count = sum(1 for node in nodes if _is_param_linear(Model, node))
    already_rewritten = any(
        node.op in ("LinearConv", "QKVLinearConv", "AttentionOutLinearConv", "MLPLinearConv")
        for node in nodes
    )
    return has_attention_softmax and (linear_count >= 4 or already_rewritten)


def _factor_token_grid(seq_len: int, prefer_h: Optional[int] = None, prefer_w: Optional[int] = None) -> Tuple[int, int]:
    if seq_len <= 0:
        return 1, 1
    if prefer_h is None or prefer_h <= 0:
        prefer_h = int(math.sqrt(seq_len))
    if prefer_w is None or prefer_w <= 0:
        prefer_w = max(seq_len // max(prefer_h, 1), 1)
    target_ratio = float(prefer_h) / float(max(prefer_w, 1))
    best = (1, seq_len)
    best_score = float("inf")
    for h in range(1, int(math.sqrt(seq_len)) + 1):
        if seq_len % h != 0:
            continue
        for cand_h, cand_w in ((h, seq_len // h), (seq_len // h, h)):
            ratio = float(cand_h) / float(cand_w)
            aspect_score = abs(math.log(ratio / target_ratio))
            skinny_penalty = float(max(cand_h, cand_w)) / float(min(cand_h, cand_w))
            score = aspect_score * 4.0 + skinny_penalty * 0.1
            if score < best_score:
                best = (cand_h, cand_w)
                best_score = score
    return best


def _linear_conv_weight(Model: TFConvertor, node, input_shape):
    weight_name = node.inputs[1]
    weight = Model.params[weight_name]
    in_channels = input_shape[-1] if input_shape else None

    if node.op == "Gemm":
        conv_weight = weight
    elif in_channels is not None and weight.shape[0] == in_channels:
        conv_weight = np.transpose(weight, (1, 0))
    elif in_channels is not None and weight.shape[1] == in_channels:
        conv_weight = weight
    else:
        conv_weight = np.transpose(weight, (1, 0))

    if conv_weight.ndim != 2:
        return None
    Model.params[weight_name] = np.ascontiguousarray(conv_weight[:, :, np.newaxis, np.newaxis])
    return conv_weight.shape[0], conv_weight.shape[1]


def _normalize_bias(Model: TFConvertor, node, out_channels: int) -> None:
    if len(node.inputs) < 3:
        return
    bias_name = node.inputs[2]
    if bias_name not in Model.params:
        return
    bias = Model.params[bias_name]
    if not isinstance(bias, np.ndarray):
        return
    Model.params[bias_name] = np.ascontiguousarray(bias.reshape(-1).astype(bias.dtype))
    if Model.params[bias_name].shape[0] != out_channels:
        raise ValueError(f"{node.name} bias shape {bias.shape} does not match out_channels={out_channels}")


def _set_linear_conv_attrs(node, input_shape, output_shape, in_channels: int, out_channels: int) -> None:
    token_dims = input_shape[1:-1]
    seq_len = int(np.prod(token_dims))
    prefer_h = token_dims[0] if len(token_dims) >= 2 else None
    prefer_w = token_dims[1] if len(token_dims) >= 2 else None
    seq_h, seq_w = _factor_token_grid(seq_len, prefer_h=prefer_h, prefer_w=prefer_w)
    node.attr["batch"] = int(input_shape[0])
    node.attr["input_shape"] = [int(v) for v in input_shape]
    node.attr["output_shape"] = [int(v) for v in output_shape]
    node.attr["seq_len"] = int(seq_len)
    node.attr["seq_h"] = int(seq_h)
    node.attr["seq_w"] = int(seq_w)
    node.attr["in_channels"] = int(in_channels)
    node.attr["out_channels"] = int(out_channels)


def _is_supported_mlp_activation(node) -> bool:
    return node.op in ("GeLU", "Swish", "Relu", "Relu6", "ReLU")


def _trace_head_projection(head_tensor: str, prod: dict):
    path = []
    current = head_tensor
    for _ in range(3):
        node = prod.get(current)
        if node is None:
            return None
        if _is_supported_layout_node(node):
            path.append(node)
            current = node.inputs[0]
            continue
        if _is_raw_param_linear(node):
            return node, path
        return None
    return None


def _is_supported_layout_node(node) -> bool:
    return node.op in ("Reshape", "Transpose")


def _is_raw_param_linear(node) -> bool:
    return node is not None and node.op in ("MatMul", "Gemm")


def _fuse_packed_qkv(Model: TFConvertor, shapes: dict) -> int:
    """Split timm's packed QKV Gemm into three 1x1 Conv projections.

    onnxslim flattens CaiT's [B, S, C] projection to a rank-2 Gemm and then
    emits Reshape -> 5-D Transpose -> three scalar Gathers.  The native Gather
    path does not preserve that 5-D layout reliably.  Producing the three head
    tensors directly is both simpler and equivalent.
    """

    nodes = list(Model.nodes.values())
    users = _consumers(nodes)
    prod = _producer(nodes)
    remove = []
    fused = 0
    for node in nodes:
        if not _is_param_linear(Model, node) or not node.outputs:
            continue
        pre = prod.get(node.inputs[0])
        if pre is None or pre.op != "Reshape" or not pre.inputs:
            continue
        post_users = users.get(node.outputs[0], [])
        if len(post_users) != 1 or post_users[0].op != "Reshape":
            continue
        post = post_users[0]
        transpose_users = users.get(post.outputs[0], [])
        if len(transpose_users) != 1 or transpose_users[0].op != "Transpose":
            continue
        transpose = transpose_users[0]
        gathers = users.get(transpose.outputs[0], [])
        if len(gathers) != 3 or any(item.op != "Gather" or item.attr.get("axis", 0) != 0 for item in gathers):
            continue

        indexed = []
        for gather in gathers:
            if len(gather.inputs) < 2 or gather.inputs[1] not in Model.params:
                break
            index = np.asarray(Model.params[gather.inputs[1]]).reshape(-1)
            if index.size != 1:
                break
            indexed.append((int(index[0]), gather))
        if len(indexed) != 3 or sorted(index for index, _ in indexed) != [0, 1, 2]:
            continue
        indexed.sort(key=lambda item: item[0])

        input_shape = shapes.get(pre.inputs[0])
        head_shapes = [shapes.get(gather.outputs[0]) for _, gather in indexed]
        weight = Model.params[node.inputs[1]]
        if input_shape is None or len(input_shape) < 3 or any(shape is None or len(shape) != 4 for shape in head_shapes):
            continue
        out_channels = int(head_shapes[0][1] * head_shapes[0][3])
        if weight.ndim != 2 or weight.shape[0] != 3 * out_channels:
            continue

        bias_parts = None
        bias_names = ["ISNULL", "ISNULL", "ISNULL"]
        if len(node.inputs) > 2 and node.inputs[2] in Model.params:
            bias = np.asarray(Model.params[node.inputs[2]]).reshape(-1)
            if bias.size != 3 * out_channels:
                continue
            bias_parts = np.split(bias, 3)

        # Always create private parameters: onnxslim intentionally ties equal
        # zero biases across blocks, so shrinking the original initializer for
        # the first Q branch would corrupt every following packed projection.
        weight_names = [f"{node.name}:{prefix}Weight" for prefix in ("Q", "K", "V")]
        for weight_name, part in zip(weight_names, np.split(weight, 3, axis=0)):
            Model.params[weight_name] = np.ascontiguousarray(part[:, :, np.newaxis, np.newaxis])
        if bias_parts is not None:
            bias_names = [f"{node.name}:{prefix}Bias" for prefix in ("Q", "K", "V")]
            for bias_name, part in zip(bias_names, bias_parts):
                Model.params[bias_name] = np.ascontiguousarray(part)

        node.op = "QKVLinearConv"
        node.inputs = [
            pre.inputs[0], weight_names[0], bias_names[0], weight_names[1],
            bias_names[1], weight_names[2], bias_names[2],
        ]
        node.outputs = [gather.outputs[0] for _, gather in indexed]
        _set_linear_conv_attrs(node, input_shape, input_shape[:-1] + [out_channels], input_shape[-1], out_channels)
        for prefix, shape in zip(("q", "k", "v"), head_shapes):
            _output_head_attrs(prefix, node, shape)
        remove.extend([pre.name, post.name, transpose.name])
        remove.extend(gather.name for _, gather in indexed)
        fused += 1

    for name in remove:
        Model.nodes.pop(name, None)
    return fused


def _find_score_matmul(softmax_node, prod: dict):
    source = prod.get(softmax_node.inputs[0]) if softmax_node.inputs else None
    if source is None:
        return None
    if source.op == "MatMul":
        return source
    if source.op in ("Mul", "Add", "Sub", "Div"):
        for iname in source.inputs:
            candidate = prod.get(iname)
            if candidate is not None and candidate.op == "MatMul":
                return candidate
    return None


def _as_bias_input(node) -> str:
    return node.inputs[2] if len(node.inputs) > 2 else "ISNULL"


def _output_head_attrs(prefix: str, node, shape: list) -> None:
    node.attr[f"{prefix}_shape"] = [int(v) for v in shape]
    if len(shape) == 4:
        node.attr[f"{prefix}_layout"] = "BHSD" if shape[2] >= shape[3] else "BHDS"
    else:
        node.attr[f"{prefix}_layout"] = "UNKNOWN"


def _fuse_attention_qkv(Model: TFConvertor, shapes: dict) -> int:
    nodes = list(Model.nodes.values())
    users = _consumers(nodes)
    prod = _producer(nodes)
    remove = []
    fused = 0

    for softmax in nodes:
        if softmax.op not in ("Softmax", "MaskSoftmax"):
            continue
        av_users = [user for out in softmax.outputs for user in users.get(out, []) if user.op == "MatMul"]
        if len(av_users) != 1:
            continue
        av_matmul = av_users[0]
        score_matmul = _find_score_matmul(softmax, prod)
        if score_matmul is None or len(score_matmul.inputs) < 2 or len(av_matmul.inputs) < 2:
            continue

        traced = [
            _trace_head_projection(score_matmul.inputs[0], prod),
            _trace_head_projection(score_matmul.inputs[1], prod),
            _trace_head_projection(av_matmul.inputs[1], prod),
        ]
        if any(item is None for item in traced):
            continue
        q_node, q_path = traced[0]
        k_node, k_path = traced[1]
        v_node, v_path = traced[2]
        if len({q_node.name, k_node.name, v_node.name}) != 3:
            continue
        if not all(_is_param_linear(Model, node) for node in (q_node, k_node, v_node)):
            continue
        if not (q_node.inputs[0] == k_node.inputs[0] == v_node.inputs[0]):
            continue

        input_shape = shapes.get(q_node.inputs[0])
        q_shape = shapes.get(score_matmul.inputs[0])
        k_shape = shapes.get(score_matmul.inputs[1])
        v_shape = shapes.get(av_matmul.inputs[1])
        if input_shape is None or q_shape is None or k_shape is None or v_shape is None or len(input_shape) < 3:
            continue

        q_info = _linear_conv_weight(Model, q_node, input_shape)
        k_info = _linear_conv_weight(Model, k_node, input_shape)
        v_info = _linear_conv_weight(Model, v_node, input_shape)
        if q_info is None or k_info is None or v_info is None:
            continue
        q_out, in_channels = q_info
        k_out, _ = k_info
        v_out, _ = v_info
        if not (q_out == k_out == v_out):
            continue
        _normalize_bias(Model, q_node, q_out)
        _normalize_bias(Model, k_node, k_out)
        _normalize_bias(Model, v_node, v_out)

        q_node.op = "QKVLinearConv"
        q_node.inputs = [
            q_node.inputs[0],
            q_node.inputs[1],
            _as_bias_input(q_node),
            k_node.inputs[1],
            _as_bias_input(k_node),
            v_node.inputs[1],
            _as_bias_input(v_node),
        ]
        q_node.outputs = [score_matmul.inputs[0], score_matmul.inputs[1], av_matmul.inputs[1]]
        _set_linear_conv_attrs(q_node, input_shape, input_shape[:-1] + [q_out], in_channels, q_out)
        _output_head_attrs("q", q_node, q_shape)
        _output_head_attrs("k", q_node, k_shape)
        _output_head_attrs("v", q_node, v_shape)

        remove.extend([k_node.name, v_node.name])
        remove.extend(node.name for node in q_path + k_path + v_path)
        fused += 1

    for name in remove:
        Model.nodes.pop(name, None)
    return fused


def _trace_attention_out_projection(av_output: str, users: dict):
    path = []
    current = av_output
    for _ in range(3):
        next_users = users.get(current, [])
        if len(next_users) != 1:
            return None
        node = next_users[0]
        if _is_supported_layout_node(node):
            path.append(node)
            current = node.outputs[0]
            continue
        if _is_raw_param_linear(node):
            return node, path
        return None
    return None


def _known_upstream_batch(tensor_name: str, prod: dict, shapes: dict, params: dict, depth: int = 16):
    if depth <= 0:
        return None
    shape = shapes.get(tensor_name)
    if shape is not None and len(shape) >= 3 and shape[0] > 0:
        return int(shape[0])
    node = prod.get(tensor_name)
    if node is None:
        return None
    for input_name in node.inputs:
        if input_name in params:
            continue
        batch = _known_upstream_batch(input_name, prod, shapes, params, depth - 1)
        if batch is not None:
            return batch
    return None


def _annotate_window_reverse_shapes(Model, projection_output, users, shapes, batch, seq_len, channels):
    def only_user_with_op(tensor_name, op):
        matches = [node for node in users.get(tensor_name, []) if node.op == op]
        return matches[0] if len(matches) == 1 else None

    reshape_window = only_user_with_op(projection_output, "Reshape")
    if reshape_window is None:
        return
    reshape_grid = only_user_with_op(reshape_window.outputs[0], "Reshape")
    if reshape_grid is None:
        return
    transpose = only_user_with_op(reshape_grid.outputs[0], "Transpose")
    if transpose is None:
        return
    reshape_spatial = only_user_with_op(transpose.outputs[0], "Reshape")
    if reshape_spatial is None:
        return

    spatial_shape = None
    frontier = list(reshape_spatial.outputs)
    visited = set(frontier)
    for _ in range(12):
        next_frontier = []
        for tensor_name in frontier:
            for user in users.get(tensor_name, []):
                if user.op == "Add":
                    for input_name in user.inputs:
                        if input_name == tensor_name:
                            continue
                        candidate = shapes.get(input_name)
                        if candidate is not None and len(candidate) == 4 and all(dim > 0 for dim in candidate):
                            spatial_shape = candidate
                            break
                if spatial_shape is not None:
                    break
                if user.op in ("Slice", "Concat", "Identity"):
                    for output in user.outputs:
                        if output not in visited:
                            visited.add(output)
                            next_frontier.append(output)
            if spatial_shape is not None:
                break
        if spatial_shape is not None:
            break
        frontier = next_frontier
    if spatial_shape is None:
        return

    spatial_batch, height, width, _ = spatial_shape
    window_h, window_w = _factor_token_grid(seq_len)
    grid_h = int(math.ceil(height / window_h))
    grid_w = int(math.ceil(width / window_w))
    annotations = {
        reshape_window.outputs[0]: [batch, window_h, window_w, channels],
        reshape_grid.outputs[0]: [spatial_batch, grid_h, grid_w, window_h, window_w, channels],
        transpose.outputs[0]: [spatial_batch, grid_h, window_h, grid_w, window_w, channels],
        reshape_spatial.outputs[0]: [spatial_batch, grid_h * window_h, grid_w * window_w, channels],
    }
    shapes.update(annotations)
    if hasattr(Model, "tensor_shapes"):
        Model.tensor_shapes.update(annotations)


def _fuse_attention_output(Model: TFConvertor, shapes: dict) -> int:
    nodes = list(Model.nodes.values())
    users = _consumers(nodes)
    prod = _producer(nodes)
    remove = []
    fused = 0
    known_geometry = {}

    for softmax in nodes:
        if softmax.op not in ("Softmax", "MaskSoftmax"):
            continue
        av_users = [user for out in softmax.outputs for user in users.get(out, []) if user.op == "MatMul"]
        if len(av_users) != 1:
            continue
        av_matmul = av_users[0]
        if not av_matmul.outputs:
            continue
        traced = _trace_attention_out_projection(av_matmul.outputs[0], users)
        if traced is None:
            continue
        proj_node, path = traced
        if not _is_param_linear(Model, proj_node):
            continue

        attn_shape = shapes.get(av_matmul.outputs[0])
        input_shape = shapes.get(proj_node.inputs[0])
        output_shape = shapes.get(proj_node.outputs[0])
        if attn_shape is None or input_shape is None or output_shape is None or len(input_shape) < 3:
            continue
        weight_info = _linear_conv_weight(Model, proj_node, input_shape)
        if weight_info is None:
            continue
        out_channels, in_channels = weight_info
        _normalize_bias(Model, proj_node, out_channels)

        # onnxslim can leave the window batch/head dimension symbolic in Swin
        # even though the export input is static. Recover it from the upstream
        # packed-QKV projection and the known attention geometry.
        batch = input_shape[0] if input_shape[0] > 0 else _known_upstream_batch(av_matmul.outputs[0], prod, shapes, Model.params)
        seq_len = input_shape[1] if len(input_shape) == 3 and input_shape[1] > 0 else None
        num_heads = attn_shape[1] if len(attn_shape) == 4 and attn_shape[1] > 0 else known_geometry.get(in_channels)
        if num_heads is not None:
            known_geometry[in_channels] = int(num_heads)
        if batch is not None and num_heads is not None and seq_len is not None and in_channels % num_heads == 0:
            input_shape = [batch, seq_len, in_channels]
            output_shape = [batch, seq_len, out_channels]
            attn_shape = [batch, num_heads, seq_len, in_channels // num_heads]
            if hasattr(Model, "tensor_shapes"):
                Model.tensor_shapes[av_matmul.outputs[0]] = list(attn_shape)
                Model.tensor_shapes[proj_node.inputs[0]] = list(input_shape)
                Model.tensor_shapes[proj_node.outputs[0]] = list(output_shape)
            _annotate_window_reverse_shapes(
                Model,
                proj_node.outputs[0],
                users,
                shapes,
                batch,
                seq_len,
                out_channels,
            )

        proj_node.op = "AttentionOutLinearConv"
        proj_node.inputs = [av_matmul.outputs[0], proj_node.inputs[1], _as_bias_input(proj_node)]
        _set_linear_conv_attrs(proj_node, input_shape, output_shape, in_channels, out_channels)
        proj_node.attr["attn_shape"] = [int(v) for v in attn_shape]
        proj_node.attr["attn_layout"] = "BHSD" if len(attn_shape) == 4 and attn_shape[2] >= attn_shape[3] else "BHDS"
        remove.extend(node.name for node in path)
        fused += 1

    for name in remove:
        Model.nodes.pop(name, None)
    return fused


def _fuse_mlp_linear_chain(Model: TFConvertor, shapes: dict) -> int:
    nodes = list(Model.nodes.values())
    users = _consumers(nodes)
    remove = []
    fused = 0

    for first in nodes:
        if first.name in remove or not _is_param_linear(Model, first):
            continue
        first_users = users.get(first.outputs[0], []) if first.outputs else []
        if len(first_users) != 1 or not _is_supported_mlp_activation(first_users[0]):
            continue
        act = first_users[0]
        act_users = users.get(act.outputs[0], []) if act.outputs else []
        if len(act_users) != 1 or not _is_param_linear(Model, act_users[0]):
            continue
        second = act_users[0]

        input_shape = shapes.get(first.inputs[0])
        output_shape = shapes.get(second.outputs[0])
        if input_shape is None or output_shape is None or len(input_shape) < 2:
            continue
        if len(input_shape) == 2:
            input_shape = [1] + list(input_shape)

        first_weight_info = _linear_conv_weight(Model, first, input_shape)
        if first_weight_info is None:
            continue
        second_input_shape = shapes.get(second.inputs[0]) or (input_shape[:-1] + [first_weight_info[0]])
        second_weight_info = _linear_conv_weight(Model, second, second_input_shape)
        if second_weight_info is None:
            continue

        mid_channels, in_channels = first_weight_info
        out_channels, _ = second_weight_info
        _normalize_bias(Model, first, mid_channels)
        _normalize_bias(Model, second, out_channels)

        first.op = "MLPLinearConv"
        first.outputs = second.outputs
        first.inputs = [
            first.inputs[0],
            first.inputs[1],
            first.inputs[2] if len(first.inputs) > 2 else "ISNULL",
            second.inputs[1],
            second.inputs[2] if len(second.inputs) > 2 else "ISNULL",
        ]
        _set_linear_conv_attrs(first, input_shape, output_shape, in_channels, out_channels)
        first.attr["mid_channels"] = int(mid_channels)
        first.attr["activation"] = "ReLU" if act.op == "Relu" else act.op
        remove.extend([act.name, second.name])
        fused += 1

    for name in remove:
        Model.nodes.pop(name, None)
    return fused


def RemoveSoftmaxMatMulPassThroughOnnx(Model: TFConvertor) -> None:
    nodes = list(Model.nodes.values())
    users = _consumers(nodes)
    prod = _producer(nodes)
    remove = []
    for node in nodes:
        if node.op not in ("Dropout", "Identity", "Cast"):
            continue
        if len(node.inputs) < 1 or len(node.outputs) == 0:
            continue
        source = prod.get(node.inputs[0])
        if source is None or source.op not in ("Softmax", "MaskSoftmax"):
            continue
        consumers = users.get(node.outputs[0], [])
        if not consumers or any(user.op != "MatMul" for user in consumers):
            continue
        for user in consumers:
            user.inputs = [node.inputs[0] if iname == node.outputs[0] else iname for iname in user.inputs]
        remove.append(node.name)

    for name in remove:
        Model.nodes.pop(name, None)


def TransformerNPUOptimizeOnnx(Model: TFConvertor) -> int:
    if not _is_transformer_like(Model):
        return 0

    shapes = _tensor_shapes(Model)
    fused_packed_qkv_count = _fuse_packed_qkv(Model, shapes)
    fused_qkv_count = _fuse_attention_qkv(Model, shapes)
    fused_attn_out_count = _fuse_attention_output(Model, shapes)
    fused_mlp_count = _fuse_mlp_linear_chain(Model, shapes)
    converted = 0
    for node in list(Model.nodes.values()):
        if not _is_param_linear(Model, node):
            continue
        input_shape = shapes.get(node.inputs[0])
        output_shape = shapes.get(node.outputs[0])
        if input_shape is None or len(input_shape) < 2:
            continue
        if len(input_shape) == 2:
            input_shape = [1] + list(input_shape)
        if int(np.prod(input_shape[1:-1])) <= 0:
            continue

        weight_info = _linear_conv_weight(Model, node, input_shape)
        if weight_info is None:
            continue
        out_channels, in_channels = weight_info
        _normalize_bias(Model, node, out_channels)

        node.op = "LinearConv"
        _set_linear_conv_attrs(node, input_shape, output_shape or (input_shape[:-1] + [out_channels]), in_channels, out_channels)
        converted += 1

    if fused_packed_qkv_count or fused_qkv_count or fused_attn_out_count or fused_mlp_count or converted:
        print(
            f"[TransformerNPUOptimize] fuse {fused_packed_qkv_count} packed QKV and {fused_qkv_count} QKV groups, "
            f"{fused_attn_out_count} attention output projections, "
            f"{fused_mlp_count} MLP chains, "
            f"convert {converted} standalone MatMul/Gemm linear ops to 1x1 Conv"
        )
    return fused_packed_qkv_count * 3 + fused_qkv_count * 3 + fused_attn_out_count + fused_mlp_count * 2 + converted
