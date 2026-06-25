# -*- encoding: utf-8 -*-
"""
PP_OCRv6 端到端 OCR 引擎 (TFDL2 推理)

封装文本检测 (det) 与文本识别 (rec) 两个量化模型, 提供:
    engine = PP OCREngine()
    result = engine.ocr(bgr_image)

返回: 每一行的 [{ "bbox": [x1,y1,x2,y2], "polygon": [[x,y]x4], "text": "...", "score": float }]
       bbox / polygon 均已归一化到 [0,1] (相对输入图像宽高).

关键约定 (通过 model_dump 实测得到):
  - det 输入 [1,3,H,W] uint8, qscale≈0.0187, qzp=114; W,H 为动态, 运行时按图像宽高比
        从 4 个预设画布 (16:9/1:1/2:1/1:2) 中挑最接近的一个, letterbox 进入.
        输入归一化为 ImageNet mean/std 的 float, 再用 (f/qscale+qzp) 量化回 uint8.
  - det 输出 Sigmoid [1,1,H,W] uint8, qscale≈1/255, qzp=0 => 直接 u*qscale 即概率图.
  - rec 输入 [1,3,48,W] uint8, qscale≈1/127.5, qzp=127; 宽度动态, 从预设
        [80,160,320,640,1024] 中按文本条自然宽度挑能容纳的最小一个. 输入归一化到 [-1,1] 再量化.
  - rec 输出 Softmax [1,T,18710] uint8, qscale≈1/255, qzp=0 => u*qscale 即概率 (T 随宽度变化).
  - rec 字符表: index 0 = CTC blank; index 1..18708 = embedding.yaml 中 character_dict;
        index 18709 = 空格 (PaddleOCR CTCLabelDecode 约定: ["blank"] + dict + [" "]).

并发识别: det 完后若有多行, 用 n_rec_instances 个独立 rec 实例 + 线程池并发识别.
  前提是 TFDL2 的 Forward 已释放 GIL (见 Python/TFDL2/TFDL2_PythonWrap.cpp 的 Forward,
  在 ForwardExecutor/ForwardExecutorAlone 外包了 py::gil_scoped_release). 否则线程会被 GIL
  串行化, 拿不到加速. 实测 8 线程跑 8 行约 7.9x 加速.
"""
import os
import queue
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import cv2
import yaml

from TFDL2 import TFContext, TFExecutor
from TFDL2.Common import TFDataType


# ---------------------------------------------------------------------------
# 路径与常量
# ---------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
DET_FB = os.path.join(HERE, "PP_OCRv6_det_medium.quant.fb")
REC_FB = os.path.join(HERE, "PP_OCRv6_rec_medium.quant.fb")
EMBED_YAML = os.path.join(HERE, "embedding.yaml")

# ImageNet 归一化 (det 输入)
DET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
DET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# 检测 / 识别超参 (PP-OCR DB + CTC 默认值)
DET_THRESH = 0.3        # 概率图二值化阈值
DET_BOX_THRESH = 0.5    # 框内平均分阈值 (过滤低质量框)
DET_UNCLIP_RATIO = 1.5  # DB 框扩张比例
DET_MIN_SIZE = 3        # 过滤过小的框 (原始图坐标, 像素)

# 检测预设画布 (W, H), 对应 16:9 / 1:1 / 2:1 / 1:2;
# 运行时按图像宽高比 (ow/oh) 挑比例最接近的一个, 然后 letterbox 进入该画布.
DET_PRESETS = [(1024, 576), (640, 640), (1024, 512), (512, 1024)]

REC_IMG_H = 48          # rec 输入固定高度
# 识别预设宽度 (高度固定 48); 运行时按文本条自然宽度 tgt_w 挑能容纳它的最小一个
# (>= tgt_w, 超过最大宽度则取最大并等比降采样, 保证不截断).
REC_PRESETS_W = [80, 160, 320, 640, 1024]


# ---------------------------------------------------------------------------
# 工具: uint8<->float 量化互转 (按张量自带 qscale/qzp)
# ---------------------------------------------------------------------------
def _quantize_to_uint8(float_arr, qscale, qzp):
    """float -> uint8, 复刻 SDK 反量化公式 float=(u-qzp)*qscale 的逆运算."""
    u = np.round(float_arr / qscale + qzp)
    u = np.clip(u, 0, 255).astype(np.uint8)
    return u


def _dequant(tensor):
    """TFDL2 输出张量 -> float32 numpy.

    uint8 量化输出用 (u-qzp)*qscale 还原; float 输出直接取值.
    """
    arr = tensor.toNumpy()
    if tensor.dtype == TFDataType.TFDL_UINT8:
        qscale = np.array(tensor.qscale, dtype=np.float32)
        qzp = np.array(tensor.qzeropoint, dtype=np.float32)
        # qscale/qzp 可能是逐通道 list, 广播到 arr 形状.
        return (arr.astype(np.float32) - qzp) * qscale
    return arr.astype(np.float32)


# ---------------------------------------------------------------------------
# DB 检测后处理
# ---------------------------------------------------------------------------
def _get_mini_boxes(contour):
    """返回最小外接矩形的 4 个角点 (顺序: 左下,左上,右上,右下近似) 及短边."""
    rect = cv2.minAreaRect(contour)
    points = cv2.boxPoints(rect)
    points = points[np.argsort(points[:, 0]), :]  # 按 x 排序
    index_1, index_4 = 0, 3
    if points[1][1] > points[0][1]:
        index_1, index_4 = 0, 3
    else:
        index_1, index_4 = 1, 2
    box_1 = points[index_1]
    box_4 = points[index_4]
    # 按 y 排序剩下的两个
    rest = np.delete(points, [index_1, index_4], axis=0)
    rest = rest[np.argsort(rest[:, 1]), :]
    box_2, box_3 = rest[0], rest[1]
    out = np.stack([box_1, box_2, box_3, box_4])
    w = float(np.linalg.norm(box_2 - box_1))
    h = float(np.linalg.norm(box_4 - box_1))
    return out, min(w, h)


def _unclip(box, ratio=1.5):
    """L2 距离扩张: 沿最小外接矩形的两条轴各向外扩张 dist = area*ratio/perimeter."""
    box = np.asarray(box, dtype=np.float32).reshape(-1, 2)
    rect = cv2.minAreaRect(box)
    (cx, cy), (w, h), ang = rect
    area = w * h
    peri = 2 * (w + h)
    dist = area * ratio / peri if peri > 0 else 0.0
    new_rect = ((cx, cy), (w + 2 * dist, h + 2 * dist), ang)
    return cv2.boxPoints(new_rect)


def _boxes_from_bitmap(pred, bitmap, dest_w, dest_h,
                       thresh=DET_BOX_THRESH, min_size=DET_MIN_SIZE):
    """从概率图 pred 中取出文本框 (返回原始图坐标的 4 点多边形)."""
    height, width = pred.shape
    mask = (bitmap > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    rx = dest_w / float(width)
    ry = dest_h / float(height)

    boxes = []
    for contour in contours:
        pts, sside = _get_mini_boxes(contour)
        if sside < min_size:
            continue
        # 框内概率均值 (按整张图的整数坐标取)
        x0 = int(np.clip(np.floor(pts[:, 0].min()), 0, width - 1))
        x1 = int(np.clip(np.ceil(pts[:, 0].max()), 0, width))
        y0 = int(np.clip(np.floor(pts[:, 1].min()), 0, height - 1))
        y1 = int(np.clip(np.ceil(pts[:, 1].max()), 0, height))
        score = float(pred[y0:y1, x0:x1].mean()) if x1 > x0 and y1 > y0 else 0.0
        if score < thresh:
            continue
        box = _unclip(pts, DET_UNCLIP_RATIO)
        box, sside = _get_mini_boxes(box)
        if sside < min_size + 2:
            continue
        box = np.array(box, dtype=np.float32)
        box[:, 0] = np.clip(box[:, 0], 0, width)
        box[:, 1] = np.clip(box[:, 1], 0, height)
        box[:, 0] *= rx
        box[:, 1] *= ry
        boxes.append(box.tolist())
    return boxes


# ---------------------------------------------------------------------------
# 识别预处理: 透视校正 + 高度归一
# ---------------------------------------------------------------------------
def _order_points_clockwise(pts):
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)  # y - x
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def _get_rotate_crop_image(img, points):
    """从 4 点多边形抠出水平文本条 (透视变换校正)."""
    pts = np.array(points, dtype=np.float32).reshape(4, 2)
    left = int(np.floor(pts[:, 0].min()))
    top = int(np.floor(pts[:, 1].min()))
    right = int(np.ceil(pts[:, 0].max()))
    bottom = int(np.ceil(pts[:, 1].max()))
    left, top = max(left, 0), max(top, 0)
    right = min(right, img.shape[1])
    bottom = min(bottom, img.shape[0])
    img_crop = img[top:bottom, left:right, :].copy()
    pts = pts - np.array([[left, top]], dtype=np.float32)

    rect = _order_points_clockwise(pts)
    w1 = np.linalg.norm(rect[0] - rect[1])
    w2 = np.linalg.norm(rect[2] - rect[3])
    h1 = np.linalg.norm(rect[0] - rect[3])
    h2 = np.linalg.norm(rect[1] - rect[2])
    w = int(max(w1, w2))
    h = int(max(h1, h2))
    w, h = max(w, 1), max(h, 1)
    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(img_crop, M, (w, h), borderValue=(0, 0, 0))


def _resize_rec_to_height(crop):
    """把文本条等比 resize 到高 48, 返回 (tgt_w, resized[48, tgt_w, 3])."""
    h, w = crop.shape[:2]
    ratio = float(w) / float(h) if h > 0 else 0.0
    tgt_w = max(1, int(round(REC_IMG_H * ratio)))
    resized = cv2.resize(crop, (tgt_w, REC_IMG_H))
    return tgt_w, resized


# ---------------------------------------------------------------------------
# OCR 引擎
# ---------------------------------------------------------------------------
class PP_OCREngine(object):
    def __init__(self,
                 det_fb=DET_FB, rec_fb=REC_FB, embed_yaml=EMBED_YAML,
                 use_hardware=False, prewarm=True,
                 n_det_instances=2, n_rec_instances=8):
        # ---- 字符表: blank(0) + dict(1..N) + space(末位) ----
        with open(embed_yaml, "r", encoding="utf-8") as f:
            emb = yaml.safe_load(f)
        char_dict = emb["character_dict"]
        self.chars = ["blank"] + list(char_dict) + [" "]
        self.blank_idx = 0

        self.det_fb = det_fb
        self.rec_fb = rec_fb
        self.use_hardware = use_hardware

        # det 多实例池: 每个实例内部按 (W,H) 编译自己的 executor.
        # det forward 不可重入 (共享输入张量/内部状态), 故并发 ocr() 必须用不同实例.
        self.n_det_instances = max(1, int(n_det_instances))
        self._det_caches = [{} for _ in range(self.n_det_instances)]  # list[ dict{(W,H): exe} ]
        self._det_pool = queue.Queue()
        for i in range(self.n_det_instances):
            self._det_pool.put(i)

        # rec 多实例池: N 个独立实例, 每个实例内部按宽度懒编译自己的 executor.
        # GIL 已在 TFDL2_PythonWrap.cpp 的 Forward 中释放, 不同实例可在不同线程并发.
        self.n_rec_instances = max(1, int(n_rec_instances))
        self._rec_caches = [{} for _ in range(self.n_rec_instances)]  # list[ dict{W: exe} ]
        self._rec_pool = queue.Queue()
        for i in range(self.n_rec_instances):
            self._rec_pool.put(i)
        self._rec_thread_pool = ThreadPoolExecutor(max_workers=self.n_rec_instances)

        # 预热: det 每个实例预热所有预设画布; rec 每个实例预热全部宽度
        # (任意实例都可能被任意线程取走处理任意尺寸/宽度的输入, 故每个实例都要备齐所有档位).
        if prewarm:
            for i in range(self.n_det_instances):
                for (w, h) in DET_PRESETS:
                    self._det_executor(i, w, h)
            for i in range(self.n_rec_instances):
                for w in REC_PRESETS_W:
                    self._rec_executor(i, w)

    # ---------------- 执行体缓存 ----------------
    def _det_executor(self, inst_idx, W, H):
        """取第 inst_idx 个 det 实例里 (W,H) 的 executor (按需编译)."""
        key = (W, H)
        cache = self._det_caches[inst_idx]
        exe = cache.get(key)
        if exe is None:
            ctx = TFContext(path=self.det_fb)
            opt = {
                "UseHardware": self.use_hardware,
                "FrugalMode": True,
                "Core": [-1],
                "cpuLimit" : 16,
                "compileMod": 0,
                "useCache": True,
                "ignoreDepthwise": True,
                "optimize": {"MakeAlign": True, "AttnSoftmaxImpl": True},
                "InputShape": [{"NodeName": "TFDL_Placeholder_0",
                                "Shape": [1, 3, H, W]}],
            }
            exe = TFExecutor(context=ctx, config=opt)
            cache[key] = exe
        return exe

    def _rec_executor(self, inst_idx, W):
        """取第 inst_idx 个实例里宽度为 W 的 executor (按需编译)."""
        cache = self._rec_caches[inst_idx]
        exe = cache.get(W)
        if exe is None:
            ctx = TFContext(path=self.rec_fb)
            opt = {
                "UseHardware": self.use_hardware,
                "FrugalMode": True,
                "Core": [-1 if inst_idx % 2 ==0 else -2],
                "cpuLimit" : 16,
                "compileMod": 0,
                "useCache": True,
                "ignoreDepthwise": True,
                "optimize": {"MakeAlign": True, "AttnSoftmaxImpl": True},
                "InputShape": [{"NodeName": "TFDL_Placeholder_0",
                                "Shape": [1, 3, REC_IMG_H, W]}],
            }
            exe = TFExecutor(context=ctx, config=opt)
            cache[W] = exe
        return exe

    # ---------------- 检测 ----------------
    def _detect(self, img, timing=None):
        """返回原图坐标下的 4 点多边形列表. timing: 可选 dict, 记录各阶段耗时."""
        oh, ow = img.shape[:2]
        # 按图像宽高比挑最接近的预设画布 (W, H)
        img_aspect = ow / float(oh)
        PW, PH = min(DET_PRESETS, key=lambda wh: abs(wh[0] / float(wh[1]) - img_aspect))

        # 从 det 实例池取一个实例 (并发 ocr() 不会撞同一个 executor)
        inst_idx = self._det_pool.get()
        try:
            exe = self._det_executor(inst_idx, PW, PH)
            din = exe.GetInputs()[0]
            qscale = float(np.array(din.qscale[0], dtype=np.float32))
            qzp = int(din.qzeropoint[0])

            t0 = time.perf_counter()
            # letterbox: 保持长宽比缩放, 居中 pad 到预设画布
            scale = min(PH / float(oh), PW / float(ow))
            nh, nw = int(round(oh * scale)), int(round(ow * scale))
            resized = cv2.resize(img, (nw, nh))
            canvas = np.full((PH, PW, 3), 0, dtype=np.uint8)
            top = (PH - nh) // 2
            left = (PW - nw) // 2
            canvas[top:top + nh, left:left + nw, :] = resized
            pad_top, pad_left = top, left
            
            # BGR -> RGB, 归一化, 量化 因为模型已经量化过所以这一步不需要了
            #rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
            #norm = (rgb.astype(np.float32) / 255.0 - DET_MEAN) / DET_STD
            bgr = canvas
            norm = bgr
            chw = norm.transpose(2, 0, 1)[None]  # 1,3,H,W
            #u = _quantize_to_uint8(chw, qscale, qzp)
            din.fromNumpy(chw)
            t_prep = time.perf_counter() - t0

            t0 = time.perf_counter()
            outs = exe()
            t_fwd = time.perf_counter() - t0

            t0 = time.perf_counter()
            prob = _dequant(outs[0])[0, 0]  # H,W 概率图
            bitmap = prob > DET_THRESH
            ph, pw = prob.shape
            boxes = _boxes_from_bitmap(prob, bitmap, pw, ph,
                                       thresh=DET_BOX_THRESH,
                                       min_size=DET_MIN_SIZE)
            # 逆 letterbox: 原图坐标 = (画布坐标 - padding) / 缩放比
            poly_orig = []
            for box in boxes:
                box = np.array(box, dtype=np.float32)
                box[:, 0] = (box[:, 0] - pad_left) / scale
                box[:, 1] = (box[:, 1] - pad_top) / scale
                box[:, 0] = np.clip(box[:, 0], 0, ow)
                box[:, 1] = np.clip(box[:, 1], 0, oh)
                poly_orig.append(box.tolist())
            t_post = time.perf_counter() - t0
        finally:
            self._det_pool.put(inst_idx)   # 归还 det 实例

        if timing is not None:
            timing["det_prep"] = t_prep
            timing["det_fwd"] = t_fwd
            timing["det_post"] = t_post
        return poly_orig

    # ---------------- 识别 ----------------
    def _recognize(self, img, box, inst_idx, timing=None):
        """对单个文本框做识别, 返回 (text, score). 使用第 inst_idx 个 rec 实例."""
        t_all = time.perf_counter()
        try:
            crop = _get_rotate_crop_image(img, box)
        except Exception:
            if timing is not None:
                timing.update(prep=0.0, fwd=0.0, decode=0.0, total=0.0)
            return "", 0.0
        if crop is None or crop.size == 0:
            if timing is not None:
                timing.update(prep=0.0, fwd=0.0, decode=0.0, total=0.0)
            return "", 0.0

        t0 = time.perf_counter()
        tgt_w, resized = _resize_rec_to_height(crop)  # resized: (48, tgt_w, 3)
        # 选最接近且能容纳的预设宽度 (>= tgt_w); 超过最大宽度则取最大并等比降采样
        pw = next((p for p in REC_PRESETS_W if p >= tgt_w), REC_PRESETS_W[-1])
        if tgt_w >= pw:
            rec_in_img = cv2.resize(crop, (pw, REC_IMG_H))  # 等比降采样到 pw
        else:
            pad = np.zeros((REC_IMG_H, pw - tgt_w, 3), dtype=resized.dtype)
            rec_in_img = np.concatenate([resized, pad], axis=1)
        # RGB, 归一化到 [-1,1], 量化回 uint8 提供的模型已经量化过了不需要外部量化
        #rgb = cv2.cvtColor(rec_in_img, cv2.COLOR_BGR2RGB)
        #f = (rgb.astype(np.float32) / 127.5 - 1.0)  # [-1,1]
        bgr = rec_in_img
        f = bgr
        chw = f.transpose(2, 0, 1)[None]
        exe = self._rec_executor(inst_idx, pw)
        rin = exe.GetInputs()[0]
        #qscale = float(np.array(rin.qscale[0], dtype=np.float32))
        #qzp = int(rin.qzeropoint[0])
        #u = _quantize_to_uint8(chw, qscale, qzp)
        rin.fromNumpy(chw)
        t_prep = time.perf_counter() - t0

        t0 = time.perf_counter()
        outs = exe()
        pred = _dequant(outs[0])[0]  # (T, num_classes) 概率
        t_fwd = time.perf_counter() - t0

        t0 = time.perf_counter()
        idx = pred.argmax(axis=1)
        prob = pred.max(axis=1)
        # CTC 解码: 去重相邻相同, 去除 blank
        chars, confs = [], []
        last = -1
        for i, c in enumerate(idx):
            if c == self.blank_idx:
                last = c
                continue
            if c == last:
                continue
            if 0 <= c < len(self.chars):
                chars.append(self.chars[c])
                confs.append(float(prob[i]))
            last = c
        text = "".join(chars)
        score = float(np.mean(confs)) if confs else 0.0
        t_decode = time.perf_counter() - t0

        if timing is not None:
            timing["prep"] = t_prep
            timing["fwd"] = t_fwd
            timing["decode"] = t_decode
            timing["total"] = time.perf_counter() - t_all
        return text, score

    # ---------------- 端到端 ----------------
    def ocr(self, img, drop_empty=True, return_timing=False):
        """img: BGR uint8.

        返回 lines 列表 (默认); return_timing=True 时返回 (lines, timing).

        每行: {"bbox":[x1,y1,x2,y2], "polygon":[[x,y]x4], "text":str, "score":float}
              bbox / polygon 均归一化到 [0,1].

        det 完后若有多行, 用 n_rec_instances 个线程并发识别, 每个线程独占一个 rec 实例.
        det 本身用 n_det_instances 个实例池, 支持多个 ocr() 并发调用.
        """
        wall0 = time.perf_counter()
        timing = {}
        if img is None or img.size == 0:
            return ([], timing) if return_timing else []

        oh, ow = img.shape[:2]

        det_t = {}
        t_det0 = time.perf_counter()
        polys = self._detect(img, timing=det_t)
        timing["det_total"] = time.perf_counter() - t_det0
        timing["det_prep"] = det_t.get("det_prep", 0.0)
        timing["det_fwd"] = det_t.get("det_fwd", 0.0)
        timing["det_post"] = det_t.get("det_post", 0.0)

        # 按从上到下排序 (取中心 y)
        polys = sorted(polys, key=lambda b: float(np.mean([p[1] for p in b])))

        # 并发识别: 每行一个任务, 每个任务从实例池取一个实例, 跑完归还
        texts = [None] * len(polys)
        scores = [None] * len(polys)
        line_timings = [None] * len(polys)

        def work(i, box):
            inst_idx = self._rec_pool.get()       # 阻塞直到有空闲实例
            lt = {}
            try:
                t, s = self._recognize(img, box, inst_idx, timing=lt)
            except Exception:
                t, s = "", 0.0
                lt = {"prep": 0.0, "fwd": 0.0, "decode": 0.0, "total": 0.0}
            finally:
                texts[i], scores[i] = t, s
                line_timings[i] = lt
                self._rec_pool.put(inst_idx)      # 归还

        t_rec0 = time.perf_counter()
        futs = [self._rec_thread_pool.submit(work, i, p) for i, p in enumerate(polys)]
        for f in futs:
            f.result()
        timing["rec_total"] = time.perf_counter() - t_rec0

        # 各阶段汇总: rec 并行, 这里把所有行的时间相加 = rec 实际占用的计算量
        valid = [lt for lt in line_timings if lt]
        timing["rec_prep_sum"] = sum(lt["prep"] for lt in valid)
        timing["rec_fwd_sum"] = sum(lt["fwd"] for lt in valid)
        timing["rec_decode_sum"] = sum(lt["decode"] for lt in valid)
        timing["rec_busy_sum"] = sum(lt["total"] for lt in valid)
        timing["rec_per_line_max"] = max((lt["total"] for lt in valid), default=0.0)
        timing["n_lines"] = len(polys)

        lines = []
        for poly, text, score in zip(polys, texts, scores):
            if drop_empty and not text:
                continue
            arr = np.array(poly, dtype=np.float32)
            x1 = float(arr[:, 0].min())
            y1 = float(arr[:, 1].min())
            x2 = float(arr[:, 0].max())
            y2 = float(arr[:, 1].max())
            lines.append({
                "bbox": [round(x1 / ow, 6), round(y1 / oh, 6),
                         round(x2 / ow, 6), round(y2 / oh, 6)],
                "polygon": [[round(float(x) / ow, 6), round(float(y) / oh, 6)]
                            for x, y in poly],
                "text": text,
                "score": round(float(score), 4),
            })
        timing["wall"] = time.perf_counter() - wall0
        timing["n_lines_out"] = len(lines)
        return (lines, timing) if return_timing else lines
