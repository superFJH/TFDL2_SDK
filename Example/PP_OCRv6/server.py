# -*- encoding: utf-8 -*-
"""
PP_OCRv6 OCR HTTP 服务 (FastAPI)

用法:
    # 激活已安装 TFDL2 的虚拟环境
    source /root/thinkforce/bin/activate
    python server.py --host 0.0.0.0 --port 8800

接口:
    POST /ocr
        - multipart/form-data 字段 file: 上传 pdf / png / jpg / jpeg
        - 返回 JSON: 每页每行文字的归一化 bbox + 文字内容

    GET  /        健康检查 / 简易说明
    GET  /docs    FastAPI 自动生成的交互文档

返回 JSON 结构:
    {
      "num_pages": 1,
      "pages": [
        {
          "page": 1,
          "width": 1240,          // 该页图像像素宽 (便于把归一化 bbox 还原成像素)
          "height": 1754,
          "lines": [
            {
              "bbox":    [x1, y1, x2, y2],   // 归一化 [0,1], 轴对齐外接框
              "polygon": [[x,y], ...4],      // 归一化 [0,1], 文本行四角点
              "text":    "识别出的文字",
              "score":   0.99
            }
          ]
        }
      ]
    }
"""
import argparse
import base64
import io
import os

import numpy as np
import cv2
import fitz  # PyMuPDF
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse, HTMLResponse
from PIL import Image, ImageDraw, ImageFont

from ocr_engine import PP_OCREngine


# PDF 渲染分辨率 (DPI) 与最大边长限制 (避免超大图导致推理过慢)
PDF_DPI = 200
PDF_MAX_SIDE = 2000

# 全局 OCR 引擎 (启动时编译一次, 复用)
ENGINE: PP_OCREngine = None

app = FastAPI(title="PP_OCRv6 OCR Service", version="1.0")


# ---------------------------------------------------------------------------
# 文件解码: pdf / png / jpg -> list[(page_idx, BGR uint8 图像)]
# ---------------------------------------------------------------------------
def _is_pdf(data: bytes) -> bool:
    return data[:5] == b"%PDF-"


def _render_pdf(data: bytes):
    pages = []
    doc = fitz.open(stream=data, filetype="pdf")
    zoom = PDF_DPI / 72.0
    matrix = fitz.Matrix(zoom, zoom)
    for i, page in enumerate(doc):
        try:
            pix = page.get_pixmap(matrix=matrix, alpha=False)
        except Exception:
            continue
        img = np.frombuffer(pix.samples, dtype=np.uint8)
        img = img.reshape(pix.height, pix.width, pix.n)
        # PyMuPDF 默认 RGB -> 转 BGR
        if pix.n >= 3:
            img = img[:, :, :3][:, :, ::-1]
        img = np.ascontiguousarray(img)
        # 限制最大边长, 等比缩小
        img = _limit_max_side(img, PDF_MAX_SIDE)
        pages.append((i, img))
    doc.close()
    return pages


def _limit_max_side(img, max_side):
    h, w = img.shape[:2]
    if max(h, w) <= max_side:
        return img
    scale = max_side / float(max(h, w))
    return cv2.resize(img, (int(round(w * scale)), int(round(h * scale))))


def _decode_image(data: bytes):
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("无法解码图像, 请上传 png/jpg/jpeg 文件")
    return [(0, img)]


# ---------------------------------------------------------------------------
# 可视化 (vis=true): 在与原图同尺寸的空白画布上画 bbox + 识别文字
# ---------------------------------------------------------------------------
# 相邻行 (按阅读顺序) 取不同颜色 —— 调色板循环即可保证相邻不同色.
VIS_PALETTE = [
    (220, 50, 50),    # red
    (40, 110, 220),   # blue
    (30, 165, 70),    # green
    (240, 145, 15),   # orange
    (150, 50, 190),   # purple
    (0, 160, 170),    # teal
    (215, 70, 155),   # magenta
    (130, 90, 35),    # brown
    (70, 70, 220),    # indigo
]

_CJK_FONT_CANDIDATES = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
]
_CJK_FONT_CACHE = {}


def _load_cjk_font(size: int):
    """加载一个支持中文的 TrueType 字体 (按尺寸缓存); 找不到则回退默认字体."""
    size = max(10, int(size))
    if size in _CJK_FONT_CACHE:
        return _CJK_FONT_CACHE[size]
    for p in _CJK_FONT_CANDIDATES:
        if os.path.exists(p):
            font = ImageFont.truetype(p, size)
            break
    else:
        font = ImageFont.load_default()
    _CJK_FONT_CACHE[size] = font
    return font


def _render_vis(img, lines):
    """在空白 (白底) 画布上绘制每行 bbox 矩形 + 框内识别文字 (文字大小填充框高).

    - bbox 画成**轴对齐矩形** (取归一化 bbox [x1,y1,x2,y2]), 半透明填充 + 实心粗描边, 不会交叉.
    - 文字画在框内, 字号按框高自适应, 垂直方向基本填满 bbox.
    - 相邻行 (按阅读顺序) 用不同颜色.

    lines 为 engine.ocr 返回的结构 (bbox 已归一化).
    """
    h, w = img.shape[:2]
    canvas = Image.new("RGB", (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas, "RGBA")

    for i, ln in enumerate(lines):
        col = VIS_PALETTE[i % len(VIS_PALETTE)]
        # 归一化 bbox -> 像素矩形
        x1, y1 = ln["bbox"][0] * w, ln["bbox"][1] * h
        x2, y2 = ln["bbox"][2] * w, ln["bbox"][3] * h
        bh = y2 - y1

        # 字号按框高自适应 (留少量上下边距), 文字高度基本填充 bbox
        fs = max(10, int(bh * 0.85))
        fs = (fs // 2) * 2  # 量化到偶数, 命中字体缓存
        font = _load_cjk_font(fs)
        border = max(2, int(bh * 0.06))

        # 半透明填充 + 实心粗描边 (轴对齐矩形)
        draw.rectangle([x1, y1, x2, y2], fill=col + (40,), outline=col, width=border)

        # 文字: 框内, 左对齐留小边距, 垂直居中
        text = ln.get("text", "") or ""
        if text:
            try:
                tx0, ty0, _, ty1 = draw.textbbox((0, 0), text, font=font)
                th = ty1 - ty0
            except Exception:
                tx0 = ty0 = 0
                th = fs
            tx = x1 + max(2, int(bh * 0.05)) - tx0
            ty = y1 + (bh - th) / 2.0 - ty0
            draw.text((tx, ty), text, fill=col, font=font)

    buf = io.BytesIO()
    canvas.save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# 路由
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse(
        "<h3>PP_OCRv6 OCR Service</h3>"
        "<p>POST <code>/ocr</code> 上传 pdf/png/jpg, 返回每行文字的归一化 bbox 与文字.</p>"
        "<p>交互文档: <a href='/docs'>/docs</a></p>"
    )


@app.post("/ocr")
def ocr(file: UploadFile = File(...),
        vis: bool = Query(False, description="为 true 时, 每页额外返回渲染图 (vis_image, base64 data URL): "
                                             "在原图同尺寸空白画布上画 bbox + 识别文字, 相邻行不同色, 便于肉眼 debug"),
        timing: bool = Query(False, description="为 true 时, 每页额外返回 timing: 各阶段耗时 (秒), 便于优化定位")):
    # 注意: 用同步 def (而非 async def), FastAPI 会把整个函数放到线程池里跑,
    # 多个请求可并发处理 (引擎 det/rec 均用实例池, 线程安全).
    data = file.file.read()
    if not data:
        raise HTTPException(status_code=400, detail="空文件")

    try:
        pages = _render_pdf(data) if _is_pdf(data) else _decode_image(data)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"文件解析失败: {e}")

    if not pages:
        raise HTTPException(status_code=400, detail="未解析出任何页面")

    result_pages = []
    for page_idx, img in pages:
        h, w = img.shape[:2]
        if timing:
            lines, t = ENGINE.ocr(img, return_timing=True)
        else:
            lines, t = ENGINE.ocr(img), None
        page = {
            "page": page_idx + 1,
            "width": int(w),
            "height": int(h),
            "lines": lines,
        }
        if timing:
            page["timing"] = {k: (round(v, 4) if isinstance(v, float) else v)
                              for k, v in t.items()}
        if vis:
            png = _render_vis(img, lines)
            page["vis_image"] = "data:image/png;base64," + base64.b64encode(png).decode("ascii")
        result_pages.append(page)

    return JSONResponse({"num_pages": len(result_pages), "pages": result_pages})


# ---------------------------------------------------------------------------
# 启动
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="PP_OCRv6 OCR FastAPI 服务")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8800)
    parser.add_argument("--hardware", action="store_true",
                        help="使用 NPU 硬件推理 (默认 CPU 仿真)")
    parser.add_argument("--n-det", type=int, default=2,
                        help="det 实例池大小 (支持并发请求数上限, 默认 2)")
    parser.add_argument("--n-rec", type=int, default=8,
                        help="rec 实例池大小 (单请求内行级并发, 默认 8)")
    args = parser.parse_args()

    global ENGINE
    print("[server] 加载 OCR 引擎 (编译模型, 约数秒) ...")
    ENGINE = PP_OCREngine(use_hardware=args.hardware,
                          n_det_instances=args.n_det, n_rec_instances=args.n_rec)
    print(f"[server] 引擎就绪 (det 池={args.n_det}, rec 池={args.n_rec}), "
          f"监听 http://{args.host}:{args.port}  (docs: /docs)")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
