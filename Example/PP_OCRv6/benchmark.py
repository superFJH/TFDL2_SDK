#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PP_OCRv6 OCR 服务性能压测脚本

对运行中的 OCR 服务 (server.py) 发压, 测量各并发档位下的:
  - QPS (吞吐)
  - 延迟分布 (mean / p50 / p90 / p95 / p99 / min / max)
  - 成功率
  - 实际识别出的文本行总数 (工作量)

负载模型: closed model —— 维持固定 C 个并发工作线程, 每个线程不停发请求.
两种跑法:
  --requests N   每个并发档位共发 N 个请求 (默认)
  --duration S   每个并发档位持续 S 秒

用法:
  # 1. 先起服务 (另一个终端):
  #    source /root/thinkforce/bin/activate
  #    cd Example2/PP_OCRv6 && python server.py --port 8800
  #
  # 2. 压测 (默认自生成 3 张样图, 并发 1/2/4/8):
  python benchmark.py --url http://127.0.0.1:8800/ocr
  # 指定测试文件 + 自定义并发与请求数:
  python benchmark.py --files a.png b.pdf --concurrency 1,4,8 --requests 16
  # 按时长跑:
  python benchmark.py --duration 60 --concurrency 1,8
  # 结果存盘:
  python benchmark.py --out result.json
"""
import argparse
import json
import os
import statistics
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests


# ---------------------------------------------------------------------------
# 样图自生成 (未提供 --files 时用), 覆盖不同 det 预设比例
# ---------------------------------------------------------------------------
def _gen_samples(out_dir):
    from PIL import Image, ImageDraw, ImageFont
    import numpy as np
    import cv2
    font_paths = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    ]
    fp = next((p for p in font_paths if os.path.exists(p)), None)
    paths = []

    def make(path, W, H, lines, fsize):
        img = Image.new("RGB", (W, H), (255, 255, 255))
        d = ImageDraw.Draw(img)
        font = ImageFont.truetype(fp, fsize) if fp else ImageFont.load_default()
        gap = H // (len(lines) + 1)
        for i, t in enumerate(lines):
            d.text((40, gap * (i + 1) - fsize // 2), t, fill=(0, 0, 0), font=font)
        cv2.imwrite(path, np.array(img)[:, :, ::-1])
        paths.append(path)

    # 接近 1:1 (800x600), 16:9, 2:1, 1:2 各一张, 混合中英文
    make(os.path.join(out_dir, "s_square.png"), 800, 600,
         ["Hello OCR Test 123", "人工智能视觉芯片", "ThinkForce NPU 2026",
          "深度学习推理加速器", "PP-OCRv6 Detection"], 34)
    make(os.path.join(out_dir, "s_portrait.png"), 660, 1100,
         ["量化模型INT8推理", "Mountain View CA", "ResNet Transformer LLM",
          "第一行中文识别测试", "第二行 English Line"], 32)
    make(os.path.join(out_dir, "s_wide.png"), 1600, 360,
         ["Wide Banner Text Line One", "Second Banner Line Here"], 40)
    return paths


def _load_samples(files, gen_dir):
    """返回 list[(filename, bytes)]"""
    if files:
        samples = []
        for f in files:
            with open(f, "rb") as fh:
                samples.append((os.path.basename(f), fh.read()))
        return samples
    gen = _gen_samples(gen_dir)
    samples = []
    for f in gen:
        with open(f, "rb") as fh:
            samples.append((os.path.basename(f), fh.read()))
    return samples


# ---------------------------------------------------------------------------
# 单次请求
# ---------------------------------------------------------------------------
def _do_request(session, url, samples, idx, timeout):
    fname, content = samples[idx % len(samples)]
    t0 = time.perf_counter()
    ok = False
    status = -1
    lines = 0
    err = None
    try:
        r = session.post(url, files={"file": (fname, content)}, timeout=timeout)
        status = r.status_code
        if status == 200:
            try:
                d = r.json()
                ok = True
                lines = sum(len(p.get("lines", [])) for p in d.get("pages", []))
            except Exception as e:
                err = "bad-json: %s" % e
        else:
            err = "http-%d" % status
    except Exception as e:
        err = "%s: %s" % (type(e).__name__, e)
    dt = time.perf_counter() - t0
    return {"ok": ok, "status": status, "lat": dt, "lines": lines, "err": err}


# ---------------------------------------------------------------------------
# 百分位
# ---------------------------------------------------------------------------
def _pct(sorted_lats, p):
    if not sorted_lats:
        return float("nan")
    k = (len(sorted_lats) - 1) * p / 100.0
    f = int(k)
    c = min(f + 1, len(sorted_lats) - 1)
    return sorted_lats[f] + (sorted_lats[c] - sorted_lats[f]) * (k - f)


def _summary(records, wall):
    lats = sorted(r["lat"] for r in records)
    oks = [r for r in records if r["ok"]]
    ok_lats = sorted(r["lat"] for r in oks)
    n = len(records)
    n_ok = len(oks)
    qps = n_ok / wall if wall > 0 else 0.0
    total_lines = sum(r["lines"] for r in oks)
    return {
        "requests": n,
        "ok": n_ok,
        "fail": n - n_ok,
        "succ_pct": 100.0 * n_ok / n if n else 0.0,
        "wall_s": wall,
        "qps": qps,
        "lat_min": lats[0] if lats else float("nan"),
        "lat_mean": statistics.mean(lats) if lats else float("nan"),
        "lat_p50": _pct(ok_lats, 50) if ok_lats else float("nan"),
        "lat_p90": _pct(ok_lats, 90) if ok_lats else float("nan"),
        "lat_p95": _pct(ok_lats, 95) if ok_lats else float("nan"),
        "lat_p99": _pct(ok_lats, 99) if ok_lats else float("nan"),
        "lat_max": lats[-1] if lats else float("nan"),
        "total_lines": total_lines,
    }


# ---------------------------------------------------------------------------
# 单个并发档位
# ---------------------------------------------------------------------------
def _run_level(session, url, samples, concurrency, n_requests, duration, timeout,
               warmup):
    # 预热 (不计入统计)
    for i in range(min(warmup, max(1, len(samples)))):
        try:
            _do_request(session, url, samples, i, timeout)
        except Exception:
            pass

    records = []
    t_start = time.perf_counter()

    if duration:
        stop = threading.Event()

        def loop(base):
            i = base
            while not stop.is_set():
                records.append(_do_request(session, url, samples, i, timeout))
                i += concurrency

        threads = [threading.Thread(target=loop, args=(k,)) for k in range(concurrency)]
        for t in threads:
            t.start()
        time.sleep(duration)
        stop.set()
        for t in threads:
            t.join()
    else:
        with ThreadPoolExecutor(max_workers=concurrency) as ex:
            futs = [ex.submit(_do_request, session, url, samples, i, timeout)
                    for i in range(n_requests)]
            for fut in as_completed(futs):
                records.append(fut.result())

    wall = time.perf_counter() - t_start
    return _summary(records, wall), records


# ---------------------------------------------------------------------------
# 打印
# ---------------------------------------------------------------------------
def _fmt(x, unit="s", w=7, prec=3):
    if x != x:  # nan
        return "n/a".rjust(w)
    return ("%." + str(prec) + "f%s") % (x, unit) if False else \
        ("%.*f" % (prec, x)).rjust(w)


def _print_table(rows):
    hdr = ["C", "reqs", "ok", "succ%", "QPS", "lat_mean",
           "p50", "p90", "p95", "p99", "max", "lines"]
    print("  " + " | ".join(h.rjust(9 if i >= 4 else 6) for i, h in enumerate(hdr)))
    print("  " + "-+-".join("-" * (9 if i >= 4 else 6) for i in range(len(hdr))))
    for r in rows:
        cells = [
            str(r["concurrency"]).rjust(6),
            str(r["requests"]).rjust(6),
            str(r["ok"]).rjust(6),
            ("%5.1f" % r["succ_pct"]).rjust(6),
            ("%7.3f" % r["qps"]).rjust(9),
            ("%7.2f" % r["lat_mean"]).rjust(9),
            ("%7.2f" % r["lat_p50"]).rjust(9),
            ("%7.2f" % r["lat_p90"]).rjust(9),
            ("%7.2f" % r["lat_p95"]).rjust(9),
            ("%7.2f" % r["lat_p99"]).rjust(9),
            ("%7.2f" % r["lat_max"]).rjust(9),
            str(r["total_lines"]).rjust(9),
        ]
        print("  " + " | ".join(cells))


def main():
    ap = argparse.ArgumentParser(description="PP_OCRv6 OCR 服务压测")
    ap.add_argument("--url", default="http://127.0.0.1:8800/ocr", help="服务 /ocr 地址")
    ap.add_argument("--files", nargs="*", default=None,
                    help="测试文件 (png/jpg/pdf); 不给则自动生成 3 张样图")
    ap.add_argument("--concurrency", default="1,2,4,8",
                    help="并发档位, 逗号分隔, 如 1,4,8")
    ap.add_argument("--requests", type=int, default=8,
                    help="每档位总请求数 (count 模式, 默认)")
    ap.add_argument("--duration", type=float, default=0,
                    help="每档位持续秒数 (>0 时启用 duration 模式, 覆盖 --requests)")
    ap.add_argument("--warmup", type=int, default=2, help="每档位预热请求数 (不计入)")
    ap.add_argument("--timeout", type=float, default=300, help="单请求超时秒")
    ap.add_argument("--out", default=None, help="结果存为 JSON 文件")
    args = ap.parse_args()

    levels = [int(x) for x in args.concurrency.split(",") if x.strip()]
    max_c = max(levels)

    # 共享 session + 大连接池 (线程安全)
    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(
        pool_connections=max_c + 4, pool_maxsize=max_c + 4, max_retries=0)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    # 连通性 + 样图
    gen_dir = tempfile.mkdtemp(prefix="ocr_bench_")
    samples = _load_samples(args.files, gen_dir)
    print("样图: %s" % ", ".join(name for name, _ in samples))
    try:
        probe = session.get(args.url.rsplit("/", 1)[0] + "/", timeout=args.timeout)
        print("服务连通: HTTP %d" % probe.status_code)
    except Exception as e:
        print("!! 无法连接服务 %s: %s — 请先启动 server.py" % (args.url, e))
        sys.exit(1)

    mode = ("duration=%gs" % args.duration) if args.duration > 0 else ("requests=%d" % args.requests)
    print("\n压测目标: %s\n并发档位: %s\n模式: %s (warmup=%d)\n" %
          (args.url, levels, mode, args.warmup))

    all_rows = []
    all_detail = []
    for c in levels:
        n = args.requests
        d = args.duration
        print(">>> 并发 %d ..." % c)
        summ, _records = _run_level(session, args.url, samples, c, n, d,
                                    args.timeout, args.warmup)
        summ["concurrency"] = c
        all_rows.append(summ)
        print("    完成: %d reqs, wall=%.1fs, QPS=%.3f, mean=%.2fs, p95=%.2fs, lines=%d\n"
              % (summ["requests"], summ["wall_s"], summ["qps"],
                 summ["lat_mean"], summ["lat_p95"], summ["total_lines"]))

    print("=" * 110)
    print("汇总 (延迟单位: 秒)")
    print("=" * 110)
    _print_table(all_rows)

    # 简单结论
    print("\n解读:")
    base = all_rows[0]["qps"] if all_rows else 0
    peak = max((r["qps"] for r in all_rows), default=0)
    peak_c = next((r["concurrency"] for r in all_rows if r["qps"] == peak), "?")
    print("  - 基线并发 %s QPS=%.3f; 峰值并发 %s QPS=%.3f"
          % (all_rows[0]["concurrency"] if all_rows else "?", base, peak_c, peak))
    if base > 0 and peak <= base * 1.15:
        print("  - QPS 基本不随并发增长 => 服务侧串行处理 (瓶颈在单请求耗时, 非客户端并发).")
    else:
        print("  - QPS 随并发上升 => 服务侧可并发处理, 并发从 %s 提升到 %s, QPS x%.2f."
              % (all_rows[0]["concurrency"] if all_rows else "?", peak_c,
                 peak / base if base else 0))
    # 尾延迟随并发的变化
    p95_low = all_rows[0]["lat_p95"] if all_rows else float("nan")
    p95_high = all_rows[-1]["lat_p95"] if all_rows else float("nan")
    if p95_high > p95_low * 1.5:
        print("  - 并发升高 p95 从 %.2fs -> %.2fs, 排队明显 (请求在服务侧堆积)." % (p95_low, p95_high))

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"url": args.url, "mode": mode, "levels": all_rows}, f,
                      indent=2, ensure_ascii=False)
        print("\n结果已保存: %s" % args.out)


if __name__ == "__main__":
    main()
