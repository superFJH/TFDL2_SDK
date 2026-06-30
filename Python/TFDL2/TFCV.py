# -*- encoding: utf-8 -*-
"""
TFDL2.TFCV — 流式推理 (video / image stream) 的 Python 入口.

内部封装了 _tfcv pybind 模块 (C++ StreamInfer: source + executor + worker 线程绑定).

加载: 完全自包含 —— 运行时库 (libTFCV/libtfdec/libtfenc/libTFDL2_LITE_C_API/libNPUxx)
由安装时的 rpath ($ORIGIN/lib) 解析, 无需 LD_LIBRARY_PATH. 详见 _ensure_loaded().
"""
from __future__ import annotations

import os
import json
from enum import IntEnum
from typing import Optional, TypedDict, Union

import numpy as np

_LOADED = False


def _ensure_loaded():
    """导入 _tfcv 前的加载处理.

    背景: libTFCV.so 内部有一处未被实际调用的废弃 ffmpeg 符号 (av_bitstream_filter_filter),
    其捆绑的 ffmpeg 已移除该符号. C++ 可执行文件默认懒绑定故无碍; 但 CPython import 扩展用
    RTLD_NOW 会立即解析它而失败.

    解法: 用 RTLD_LAZY 导入 _tfcv, 让该符号不被立即解析. 整条库链
    (libTFCV -> libtfdec/libtfenc; TFDL::tlog 来自 libTFDL2_LITE_C_API) 全部由 _tfcv 扩展
    自带的 rpath ($ORIGIN/lib) 解析 —— 因此无需 LD_LIBRARY_PATH, 安装包完全自包含.
    """
    global _LOADED
    if _LOADED:
        return
    import sys
    old = sys.getdlopenflags()
    LAZY = getattr(os, "RTLD_LAZY", 1)
    try:
        sys.setdlopenflags(LAZY)
        # 触发 _tfcv (及其依赖 libTFCV 等) 的加载; LAZY 下废弃符号不解析, rpath 解析整条链
        from . import _tfcv  # noqa: F401
    finally:
        sys.setdlopenflags(old)
    _LOADED = True


_ensure_loaded()

from . import _tfcv  # noqa: E402


class DECODER_FLAGS(IntEnum):
    """解码输出通道顺序 (与 TFCV::DECODER_FLAGS 一致, 直接当 int 用)."""
    BGR = 0
    RGB = 1
    Gray = 2


# 视频流编码类型字符串 (VideoCapture.media_type / StreamInfer.media_type 取值)
MEDIA_H264 = "h264"
MEDIA_HEVC = "hevc"
MEDIA_H265 = "h265"
MEDIA_PNG = "png"
MEDIA_BMP = "bmp"


class StreamResult(TypedDict, total=False):
    """StreamInfer.poll() 的返回元素 (超时返回 None)."""
    status: int          # 1=正常, 0=流结束(EOS), -1=错误
    frame_index: int     # 该帧序号 (从 0 起)
    frame: np.ndarray    # 原始分辨率帧 HxWxC uint8 (keep_frame=False 时缺失)
    outputs: list        # list[np.ndarray] 模型各输出张量 (按其 dtype)
    quant: list          # list[(scale_list, zeropoint_list)] 各输出量化信息 (uint8 输出可据此反量化)
    error: str           # status==-1 时的错误描述
    crop: list           # preprocess="centercrop" 时: [left,top,cw,ch] 源帧中心裁剪区域
    letterbox: list      # preprocess="letterbox" 时: [ratio, pad_left, pad_top]


class StreamInfer:
    """流式推理: C++ 内部绑定 {解码源 + Executor + worker 线程}.

    worker 线程在 C++ 里循环 解码 -> 填输入张量 -> forward, 把 (输出张量 + 原始帧) 入队;
    Python 侧 `poll()` 取结果, 在 Python 里做后处理与事件判断. 多路 = 多个 StreamInfer.
    """

    def __init__(
        self,
        model: str,
        config: Union[str, dict],
        source_kind: str,
        source: Union[str, list],
        media_type: str = MEDIA_H264,
        decoder_path: str = "",
        fps: int = 0,
        loop: bool = True,
        flags: int = DECODER_FLAGS.RGB,
        scale: Optional[list] = None,
        means: Optional[list] = None,
        input_name: str = "",
        queue_size: int = 8,
        keep_frame: bool = True,
        preprocess: str = "stretch",
        pad_value: int = 114,
    ) -> None:
        """
        Args:
            model:         .fb 模型路径.
            config:        executor 编译配置, str(JSON) 或 dict. dict 会被 json.dumps.
                           如 {"UseHardware": True, "FrugalMode": True, "Core": [-1]}.
            source_kind:   "video" 或 "image".
            source:        video -> RTSP/文件 url(str); image -> 图片路径 list[str].
            media_type:    视频编码 "h264"/"hevc"/"h265" (image 忽略).
            decoder_path:  硬件解码器 "/dev/mv500" 等; "" = 软件解码.
            fps:           目标帧率 (>0 生效). 多路安防可设低帧率(内部只解 I 帧).
            loop:          流结束是否自动重连/重放.
            flags:         解码输出通道顺序 DECODER_FLAGS.BGR/RGB/Gray.
            scale:         executor 内归一化的 scale (如 [1/255]*3); 与 means 同时为空则不设.
            means:         executor 内归一化的 mean (如 [0,0,0]).
            input_name:    输入张量名; "" 自动取模型第一个输入名.
            queue_size:    结果队列容量 (背压: 满 worker 等待).
            keep_frame:    是否在结果里携带原始分辨率帧 (画框/显示用).
            preprocess:    空间预处理模式:
                             "stretch"    — 拉伸缩放到 tensor 的 HxW (默认, 不保比例).
                             "centercrop" — ImageNet 风格: 源帧中心取宽高比=tensor 的最大区域再 resize.
                                            poll 结果带 crop=[left,top,cw,ch] 供坐标反映射.
                             "letterbox"  — YOLO 风格: 等比缩放 fit 进 tensor, 居中 pad.
                                            poll 结果带 letterbox=[ratio,pad_left,pad_top] 供坐标反映射.
            pad_value:     letterbox 的填充值 (默认 114, YOLO 惯例).
        """
        cfg = json.dumps(config) if isinstance(config, dict) else config
        self._inner = _tfcv.StreamInfer(
            model, cfg, source_kind, source,
            media_type, decoder_path, fps, loop, int(flags),
            list(scale) if scale else [], list(means) if means else [],
            input_name, queue_size, keep_frame,
            preprocess, int(pad_value),
        )

    def start(self) -> None:
        """启动 worker 线程, 开始解码+推理."""
        self._inner.start()

    def stop(self) -> None:
        """停止 worker 线程并关闭解码源."""
        self._inner.stop()

    def poll(self, timeout_ms: int = 1000) -> Optional[StreamResult]:
        """取一个结果 (阻塞最多 timeout_ms 毫秒). 队列空且超时返回 None; 否则返回 StreamResult."""
        return self._inner.poll(timeout_ms)

    def is_running(self) -> bool:
        """worker 线程是否仍在运行."""
        return self._inner.is_running()

    def stats(self) -> dict:
        """返回 {produced, consumed, queue_len, queue_cap, running}."""
        return self._inner.stats()


class ImgReader:
    """图片解码器 (JPEG/PNG/BMP). open(path) 后即完成解码, frame() dump 成 numpy.

    用于在 Python 里自定义预处理: frame() 拿到 HxWxC numpy -> resize/归一化/量化 ->
    executor.GetInputs()[0].fromNumpy(...) -> exe(). 多张图需对每张重新 open().
    """

    def __init__(self) -> None:
        self._inner = _tfcv.ImgReader()

    def open(self, path: str, flags: int = DECODER_FLAGS.BGR) -> None:
        """打开并解码图片. flags: 通道顺序 DECODER_FLAGS.BGR/RGB/Gray."""
        self._inner.open(path, int(flags))

    def frame(
        self,
        scale: float = 1.0,
        crop: Optional[tuple] = None,
    ) -> np.ndarray:
        """dump 当前图为 HxWxC uint8 numpy (解码即 scale+crop, 硬解路径近乎零成本).

        Args:
            scale: 等比缩放系数 (1.0=原尺寸). 输出高/宽 = 原始 × scale.
            crop:  (left, top, width, height) 在 scale 之后的帧上裁剪; None=不裁.
        """
        return self._inner.frame(scale, crop if crop is not None else (-1, -1, -1, -1))

    def into_tensor(
        self,
        tensor,
        preprocess: str = "stretch",
        pad_value: int = 114,
    ) -> dict:
        """解码当前图 + 空间预处理 + 写进 tensor (写的是原始像素 NCHW uint8).

        Args:
            tensor:     executor 的输入张量 (executor.GetInputs()[0]). 用其 shape 定目标 HxW,
                        用其 fromNumpy 写入. 归一化由 executor.SetPreprocess 做.
            preprocess: "stretch" / "centercrop" / "letterbox" (与 StreamInfer 同义).
            pad_value:  letterbox 填充值 (默认 114).

        Returns:
            meta dict: centercrop -> {"crop":[left,top,cw,ch]};
                       letterbox  -> {"letterbox":[ratio,pad_left,pad_top]}; stretch -> {}.
        """
        return self._inner.into_tensor(tensor, preprocess, int(pad_value))

    def width(self) -> int:
        """源帧原始宽 (不受 scale/crop 影响)."""
        return self._inner.width()

    def height(self) -> int:
        """源帧原始高 (不受 scale/crop 影响)."""
        return self._inner.height()


class VideoCapture:
    """视频流解码器 (RTSP/文件, 可硬件解码). read()/frame() 取帧, 预处理交给 Python."""

    def __init__(self, media_type: str = MEDIA_H264, decoder_path: str = "") -> None:
        """
        Args:
            media_type:   "h264"/"hevc"/"h265".
            decoder_path: 硬件解码器 "/dev/mv500" 等; "" = 软件解码.
        """
        self._inner = _tfcv.VideoCapture(media_type, decoder_path)

    def open(self, url: str, flags: int = DECODER_FLAGS.BGR) -> None:
        """打开视频源 (RTSP/文件). flags: 通道顺序."""
        self._inner.open(url, int(flags))

    def set_fps(self, fps: int) -> None:
        """设置目标帧率 (>0 生效). 多路安防设低帧率可只解 I 帧, 降低解码开销."""
        self._inner.set_fps(fps)

    def read(self) -> int:
        """读取下一帧. 返回 1=成功 0=流结束 -1=错误(可跳过/重连)."""
        return self._inner.read()

    def frame(
        self,
        scale: float = 1.0,
        crop: Optional[tuple] = None,
    ) -> np.ndarray:
        """dump 当前帧为 HxWxC uint8 numpy (read() 成功后调用, 解码即 scale+crop).

        Args:
            scale: 等比缩放系数 (1.0=原尺寸).
            crop:  (left, top, width, height) 在 scale 之后的帧上裁剪; None=不裁.
        """
        return self._inner.frame(scale, crop if crop is not None else (-1, -1, -1, -1))

    def into_tensor(
        self,
        tensor,
        preprocess: str = "stretch",
        pad_value: int = 114,
    ) -> dict:
        """解码当前帧 + 空间预处理 + 写进 tensor (read() 成功后调用).

        Args:
            tensor:     executor 的输入张量 (executor.GetInputs()[0]).
            preprocess: "stretch" / "centercrop" / "letterbox".
            pad_value:  letterbox 填充值 (默认 114).

        Returns:
            meta dict: centercrop -> {"crop":[left,top,cw,ch]};
                       letterbox  -> {"letterbox":[ratio,pad_left,pad_top]}; stretch -> {}.
        """
        return self._inner.into_tensor(tensor, preprocess, int(pad_value))

    def width(self) -> int:
        return self._inner.width()

    def height(self) -> int:
        return self._inner.height()

    def fps(self) -> int:
        return self._inner.fps()

    def close(self) -> None:
        """关闭视频源."""
        self._inner.close()


__all__ = [
    "StreamInfer", "ImgReader", "VideoCapture",
    "DECODER_FLAGS", "StreamResult",
    "MEDIA_H264", "MEDIA_HEVC", "MEDIA_H265", "MEDIA_PNG", "MEDIA_BMP",
]
