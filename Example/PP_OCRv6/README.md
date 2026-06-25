# PP_OCRv6 OCR 服务 (TFDL2)

基于 TFDL2 SDK 量化模型 (`PP_OCRv6_det_medium.quant.fb` 检测 + `PP_OCRv6_rec_medium.quant.fb` 识别)
搭建的 HTTP OCR 服务。用户上传 **PDF / PNG / JPG**, 服务返回每行文字的**归一化 bbox** 与**文字内容**。

## 文件说明

| 文件 | 作用 |
|------|------|
| `ocr_engine.py` | OCR 引擎: det (DB 后处理) + rec (CTC 解码), 封装 TFDL2 推理 |
| `server.py` | FastAPI 服务, 提供 `POST /ocr` 接口 |
| `embedding.yaml` | rec 字符字典 (`character_dict`), 用于把 rec 输出映射回字符 |
| `PP_OCRv6_det_medium.quant.fb` | 文本检测量化模型 |
| `PP_OCRv6_rec_medium.quant.fb` | 文本识别量化模型 |

## 环境准备

使用已安装 TFDL2 的虚拟环境, 并补装服务依赖:

```bash
source /root/thinkforce/bin/activate
pip install fastapi uvicorn pymupdf   # opencv/numpy/pyyaml 已随 TFDL2 安装
```

## 启动服务

```bash
cd Example2/PP_OCRv6
python server.py --host 0.0.0.0 --port 8800
# NPU 硬件加速 (若有芯片):  python server.py --hardware
```

服务启动后访问 `http://<host>:8800/docs` 可看交互文档。

## 调用示例

```bash
# 图片
curl -s -F "file=@demo.png" http://localhost:8800/ocr | python -m json.tool

# PDF
curl -s -F "file=@demo.pdf" http://localhost:8800/ocr | python -m json.tool

# 打开可视化 debug: 在 ?vis=true 时, 每页额外返回 vis_image (base64 data URL),
# 为与原图同尺寸的空白画布上画的 bbox + 识别文字, 相邻行不同色。
curl -s -F "file=@demo.png" "http://localhost:8800/ocr?vis=true" \
  | python -c "import json,sys,base64; d=json.load(sys.stdin); b=d['pages'][0]['vis_image']; open('vis.png','wb').write(base64.b64decode(b.split(',')[1]))"
# 然后直接打开 vis.png 查看
```

Python 调用:

```python
import requests
r = requests.post("http://localhost:8800/ocr",
                  files={"file": open("demo.pdf", "rb")})
data = r.json()
for page in data["pages"]:
    for line in page["lines"]:
        print(line["bbox"], line["text"])   # bbox 已归一化到 [0,1]
```

## 返回 JSON 结构

```jsonc
{
  "num_pages": 1,
  "pages": [
    {
      "page": 1,
      "width": 1240,                 // 页面像素宽 (可把归一化 bbox 还原成像素)
      "height": 1754,
      "lines": [
        {
          "bbox":    [0.05, 0.10, 0.40, 0.15],   // 归一化轴对齐外接框 [x1,y1,x2,y2]
          "polygon": [[0.05,0.10], [0.40,0.10], [0.40,0.15], [0.05,0.15]], // 四角点
          "text":    "识别出的文字",
          "score":   0.99
        }
      ]
    }
  ]
}
```

### 可视化 debug (`?vis=true`)

请求 `POST /ocr?vis=true` 时, 每页会额外多一个 `vis_image` 字段 (base64 data URL):
在与原图**同尺寸的空白白底画布**上, 按阅读顺序把每行的 bbox 多边形 + 识别文字画出来,
**相邻行使用不同颜色** (调色板循环), 并带同色文字背景条, 便于肉眼核对检测框与识别结果。

```jsonc
{ "pages": [ { "page":1, "width":800, "height":600, "lines":[...],
              "vis_image": "data:image/png;base64,iVBORw0KGgo..." } ] }
```

## 关键实现约定 (已通过模型实测校准)

- **det 输入** `[1,3,H,W] uint8`, `qscale≈0.0187, qzp=114`：输入做 ImageNet mean/std 归一化后,
  用 `(f/qscale+qzp)` 量化回 uint8 再喂入。**画布按图像宽高比从 4 个预设里挑最接近的一个**,
  再 letterbox 居中放入 (避免固定方形画布浪费分辨率):
  `(1024,576)`16:9 / `(960,960)`1:1 / `(1024,512)`2:1 / `(512,1024)`1:2。
- **det 输出** `Sigmoid [1,1,H,W] uint8, qscale≈1/255`：`u*qscale` 即文本概率图, 二值阈值 0.3,
  DB 取框 (box 阈值 0.5, unclip 1.5)。
- **rec 输入** `[1,3,48,W] uint8, qscale≈1/127.5, qzp=127`：**宽度按文本条自然宽度从预设
  `[80,160,320,640,1024]` 里挑能容纳的最小一个** (短文本用小宽度, 长文本用大宽度, 超过最大值则等比降采样), 文本条归一化到 `[-1,1]` 再量化。
- **rec 输出** `Softmax [1,T,18710] uint8` (时间步 T 随宽度变化, W/8)：CTC 解码。字符表 = `["blank"] + character_dict + [" "]`,
  即 index 0 为 CTC blank, `1..18708` 对应字典字符, `18709` 为空格 (PaddleOCR `CTCLabelDecode` 约定)。

> 当前 `--hardware` 未启用时为 CPU 仿真, 单页多行耗时约数秒~十几秒; 接入 NPU 硬件 (`--hardware`) 可大幅加速。

### 并发识别 (8 实例 + 线程池)

- det 完成后若有多行, OCR 引擎会用 8 个独立 rec 实例 + 线程池**并发识别**, 每个线程独占一个实例。
- **前提**: 已修改 `Python/TFDL2/TFDL2_PythonWrap.cpp` 的 `Forward`, 在 `ForwardExecutor/ForwardExecutorAlone` 外包了
  `py::gil_scoped_release` 释放 GIL (pybind11 默认全程持有 GIL, 否则多线程会被串行化)。改完需 `cd Python && pip install .` 重新编译。
- 实测: 8 行 rec 由串行 5.8s → 并发 0.73s (**~7.9x 加速**), 且每个 (实例, 输入) 结果确定、无数据竞争。
- 注: CPU 仿真下 det 在大画布 (960×960) 上本身就慢 (~11s, 随面积线性增长), 是当前单页耗时主要来源; rec 并发主要压缩了识别那一段。接入 NPU 后 det/rec 都会快很多。
