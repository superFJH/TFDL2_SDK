#!/usr/bin/env python3
"""Exact Qwen3-VL visual ordering, MRoPE and DeepStack contracts."""

from __future__ import annotations

from typing import Iterable

import numpy as np

from checkpoint import ModelContract


def validate_grid(grid_h: int, grid_w: int, merge_size: int = 2) -> None:
    if grid_h <= 0 or grid_w <= 0:
        raise ValueError("grid dimensions must be positive")
    if grid_h % merge_size or grid_w % merge_size:
        raise ValueError("grid dimensions must be divisible by spatial_merge_size")


def block_order_spatial_positions(
    grid_h: int, grid_w: int, merge_size: int = 2
) -> np.ndarray:
    """Return the official Qwen3-VL patch order as [S, row/column]."""
    validate_grid(grid_h, grid_w, merge_size)
    positions = []
    for block_h in range(grid_h // merge_size):
        for block_w in range(grid_w // merge_size):
            for inner_h in range(merge_size):
                for inner_w in range(merge_size):
                    positions.append(
                        (block_h * merge_size + inner_h, block_w * merge_size + inner_w)
                    )
    return np.asarray(positions, dtype=np.int64)


def vision_rope(
    grid_h: int,
    grid_w: int,
    head_dim: int,
    merge_size: int = 2,
    theta: float = 10_000.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Match Qwen3VLVisionModel.rot_pos_emb and apply_rotary_pos_emb_vision."""
    if head_dim % 4:
        raise ValueError("vision head_dim must be divisible by four")
    positions = block_order_spatial_positions(grid_h, grid_w, merge_size)
    rotary_dim = head_dim // 2
    inv_freq = 1.0 / (
        theta ** (np.arange(0, rotary_dim, 2, dtype=np.float32) / rotary_dim)
    )
    row = positions[:, 0:1].astype(np.float32) * inv_freq[None]
    col = positions[:, 1:2].astype(np.float32) * inv_freq[None]
    frequencies = np.concatenate((row, col), axis=-1)
    angles = np.concatenate((frequencies, frequencies), axis=-1)
    sin = np.sin(angles).reshape(1, 1, grid_h * grid_w, head_dim)
    cos = np.cos(angles).reshape(1, 1, grid_h * grid_w, head_dim)
    return np.ascontiguousarray(sin, dtype=np.float32), np.ascontiguousarray(
        cos, dtype=np.float32
    )


def interpolated_vision_position_embedding(
    source: np.ndarray,
    grid_h: int,
    grid_w: int,
    merge_size: int = 2,
) -> np.ndarray:
    """Match Qwen3VLVisionModel.fast_pos_embed_interpolate for one grid."""
    validate_grid(grid_h, grid_w, merge_size)
    table = np.asarray(source, dtype=np.float32)
    if table.ndim != 2:
        raise ValueError(f"position table must be [N,D], got {table.shape}")
    side = int(round(table.shape[0] ** 0.5))
    if side * side != table.shape[0]:
        raise ValueError("vision position table must be square")
    h = np.linspace(0, side - 1, grid_h, dtype=np.float32)
    w = np.linspace(0, side - 1, grid_w, dtype=np.float32)
    h0 = h.astype(np.int64)
    w0 = w.astype(np.int64)
    h1 = np.minimum(h0 + 1, side - 1)
    w1 = np.minimum(w0 + 1, side - 1)
    dh = h - h0
    dw = w - w0
    output = np.empty((grid_h, grid_w, table.shape[1]), dtype=np.float32)
    for row in range(grid_h):
        for col in range(grid_w):
            output[row, col] = (
                table[h0[row] * side + w0[col]] * (1 - dh[row]) * (1 - dw[col])
                + table[h0[row] * side + w1[col]] * (1 - dh[row]) * dw[col]
                + table[h1[row] * side + w0[col]] * dh[row] * (1 - dw[col])
                + table[h1[row] * side + w1[col]] * dh[row] * dw[col]
            )
    m = merge_size
    output = (
        output.reshape(grid_h // m, m, grid_w // m, m, table.shape[1])
        .transpose(0, 2, 1, 3, 4)
        .reshape(grid_h * grid_w, table.shape[1])
    )
    return np.ascontiguousarray(output[None], dtype=np.float32)


def text_mrope(
    position_ids: np.ndarray,
    head_dim: int,
    theta: float,
    mrope_section: Iterable[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Build Qwen3VLTextRotaryEmbedding's interleaved 3D sin/cos tables."""
    positions = np.asarray(position_ids, dtype=np.float32)
    if positions.ndim == 2:
        positions = np.broadcast_to(positions[None], (3,) + positions.shape)
    if positions.ndim != 3 or positions.shape[0] != 3:
        raise ValueError(f"position_ids must be [3,B,S] or [B,S], got {positions.shape}")
    section = tuple(int(v) for v in mrope_section)
    if len(section) != 3 or sum(section) != head_dim // 2:
        raise ValueError("invalid mrope_section")
    inv_freq = 1.0 / (
        theta ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim)
    )
    frequencies = positions[..., None] * inv_freq[None, None, None, :]
    mixed = frequencies[0].copy()
    for dimension, offset in ((1, 1), (2, 2)):
        stop = section[dimension] * 3
        mixed[..., offset:stop:3] = frequencies[dimension, ..., offset:stop:3]
    angles = np.concatenate((mixed, mixed), axis=-1)
    return np.ascontiguousarray(np.sin(angles), dtype=np.float32)[:, None], np.ascontiguousarray(
        np.cos(angles), dtype=np.float32
    )[:, None]


def _expanded_video_grids(video_grid_thw: np.ndarray | None) -> np.ndarray | None:
    if video_grid_thw is None:
        return None
    grid = np.asarray(video_grid_thw, dtype=np.int64).reshape(-1, 3)
    expanded = np.repeat(grid, grid[:, 0], axis=0)
    expanded[:, 0] = 1
    return expanded


def multimodal_position_ids(
    input_ids: np.ndarray,
    attention_mask: np.ndarray,
    contract: ModelContract,
    *,
    image_grid_thw: np.ndarray | None = None,
    video_grid_thw: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """NumPy port of Qwen3VLModel.get_rope_index for batch size one."""
    ids = np.asarray(input_ids, dtype=np.int64)
    mask = np.asarray(attention_mask, dtype=np.int64)
    if ids.ndim != 2 or ids.shape[0] != 1 or ids.shape != mask.shape:
        raise ValueError("input_ids and attention_mask must have shape [1,S]")
    images = None if image_grid_thw is None else np.asarray(image_grid_thw, dtype=np.int64).reshape(-1, 3)
    videos = _expanded_video_grids(video_grid_thw)
    if images is None and videos is None:
        one_d = np.cumsum(mask, axis=-1) - 1
        one_d[mask == 0] = 1
        maximum = one_d.max(axis=-1, keepdims=True)
        return np.broadcast_to(one_d[None], (3,) + one_d.shape).copy(), maximum + 1 - int(mask.sum())

    valid = ids[0, mask[0] == 1].tolist()
    starts = [index for index, token in enumerate(valid) if token == contract.vision_start_token_id]
    image_count = sum(
        index + 1 < len(valid) and valid[index + 1] == contract.image_token_id
        for index in starts
    )
    video_count = sum(
        index + 1 < len(valid) and valid[index + 1] == contract.video_token_id
        for index in starts
    )
    if image_count != (0 if images is None else images.shape[0]):
        raise ValueError(f"prompt/image grid count mismatch: {image_count} vs {0 if images is None else images.shape[0]}")
    if video_count != (0 if videos is None else videos.shape[0]):
        raise ValueError(f"prompt/video grid count mismatch: {video_count} vs {0 if videos is None else videos.shape[0]}")

    spans: list[np.ndarray] = []
    cursor = 0
    image_index = 0
    video_index = 0
    remaining_images = image_count
    remaining_videos = video_count
    for _ in range(image_count + video_count):
        try:
            image_at = valid.index(contract.image_token_id, cursor) if remaining_images else len(valid) + 1
        except ValueError:
            image_at = len(valid) + 1
        try:
            video_at = valid.index(contract.video_token_id, cursor) if remaining_videos else len(valid) + 1
        except ValueError:
            video_at = len(valid) + 1
        if image_at < video_at:
            assert images is not None
            grid_t, grid_h, grid_w = images[image_index]
            image_index += 1
            remaining_images -= 1
            visual_at = image_at
        else:
            assert videos is not None
            grid_t, grid_h, grid_w = videos[video_index]
            video_index += 1
            remaining_videos -= 1
            visual_at = video_at
        llm_t = int(grid_t)
        llm_h = int(grid_h) // contract.spatial_merge_size
        llm_w = int(grid_w) // contract.spatial_merge_size
        text_length = visual_at - cursor
        start = int(max((item.max() for item in spans), default=-1)) + 1
        if text_length:
            spans.append(np.broadcast_to(np.arange(text_length)[None], (3, text_length)) + start)
        temporal = np.broadcast_to(
            np.arange(llm_t)[:, None], (llm_t, llm_h * llm_w)
        ).reshape(-1)
        height = np.broadcast_to(
            np.arange(llm_h)[None, :, None], (llm_t, llm_h, llm_w)
        ).reshape(-1)
        width = np.broadcast_to(
            np.arange(llm_w)[None, None, :], (llm_t, llm_h, llm_w)
        ).reshape(-1)
        spans.append(np.stack((temporal, height, width)) + text_length + start)
        cursor = visual_at + llm_t * llm_h * llm_w
    if cursor < len(valid):
        start = int(max((item.max() for item in spans), default=-1)) + 1
        text_length = len(valid) - cursor
        spans.append(np.broadcast_to(np.arange(text_length)[None], (3, text_length)) + start)
    positions_valid = np.concatenate(spans, axis=1).astype(np.int64)
    if positions_valid.shape[1] != len(valid):
        raise ValueError(
            f"position assembly produced {positions_valid.shape[1]} values for {len(valid)} tokens"
        )
    positions = np.ones((3, 1, ids.shape[1]), dtype=np.int64)
    positions[:, 0, mask[0] == 1] = positions_valid
    delta = np.asarray([[int(positions_valid.max()) + 1 - len(valid)]], dtype=np.int64)
    return positions, delta


def inject_deepstack(
    hidden: np.ndarray, visual_mask: np.ndarray, features: np.ndarray
) -> np.ndarray:
    value = np.asarray(hidden)
    mask = np.asarray(visual_mask, dtype=bool)
    feature = np.asarray(features, dtype=value.dtype)
    if value.ndim != 3 or mask.shape != value.shape[:2]:
        raise ValueError("hidden must be [B,S,D] and visual_mask [B,S]")
    if feature.shape != (int(mask.sum()), value.shape[-1]):
        raise ValueError(
            f"DeepStack feature shape {feature.shape}, expected {(int(mask.sum()), value.shape[-1])}"
        )
    result = value.copy()
    result[mask] += feature
    return result

