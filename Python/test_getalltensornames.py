# -*- coding: utf-8 -*-
"""
测试 TFContext.GetAllTensorNames 新增 API。

API 说明 (include/TFDL2_C_API.h):
    std::set<std::string> GetAllTensorNames(TFContext);
    输出 Context 内全部 Node 的输出 tensor name。

Python 侧 (TFDL2/TFContext.py):
    def GetAllTensorNames(self) -> list:
        return super(TFContext, self)._GetAllTensorNames()

本测试包含两部分:
  1. 合成小图: 构造已知结构的图 (Placeholder -> Add/Mul/Relu), 校验返回的
     tensor name 集合与预期完全一致。
  2. 真实模型: 加载 .fb, 校验返回类型为 list[str] 且非空, 并与
     GetInputSymbols / GetOutSymbols 的结果在集合层面吻合。
"""

import os
import sys
from TFDL2 import TFContext, Op
from TFDL2.Common import TFDataType

GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"

_passed = 0
_failed = 0


def check(cond, msg):
    global _passed, _failed
    if cond:
        _passed += 1
        print(f"  {GREEN}[PASS]{RESET} {msg}")
    else:
        _failed += 1
        print(f"  {RED}[FAIL]{RESET} {msg}")


def test_synthetic_graph():
    print("\n=== 测试 1: 合成小图 GetAllTensorNames ===")
    ctx = TFContext("test_ctx")
    with ctx:
        a = Op.Placeholder2(ctx, shape=(1, 4), outDatatype=TFDataType.TFDL_FLOAT)
        b = Op.Placeholder2(ctx, shape=(1, 4), outDatatype=TFDataType.TFDL_FLOAT)
        s = Op.Add(a, b)          # a + b
        m = Op.Mul(s, a)          # s * a
        r = Op.ReLU(m)            # relu(m)
        # 记录每个节点输出 tensor 的名字 (TFSymbol -> str 即 tensor name)
        node_out_names = [str(x) for x in (s, m, r)]
        ctx.SetOutputs([str(r)])

    names = ctx.GetAllTensorNames()

    check(isinstance(names, list), f"返回类型为 list (实际: {type(names).__name__})")
    check(all(isinstance(n, str) for n in names),
          f"所有元素均为 str (共 {len(names)} 个)")

    # 每个节点的输出 tensor name 都应出现在返回集合里
    missing = [n for n in node_out_names if n not in set(names)]
    check(not missing,
          f"包含图里所有节点输出 tensor name {node_out_names} (缺失: {missing})")

    # 输入 placeholder 的名字也应在集合内
    in_syms = [str(x) for x in ctx.GetInputSymbols()]
    check(set(in_syms).issubset(set(names)),
          f"GetInputSymbols 结果 {in_syms} 是 GetAllTensorNames 的子集")

    # 输出 symbol 也应在集合内
    out_syms = [str(x) for x in ctx.GetOutSymbols()]
    check(set(out_syms).issubset(set(names)),
          f"GetOutSymbols 结果 {out_syms} 是 GetAllTensorNames 的子集")

    # 结果应无重复 (C++ 侧是 std::set)
    check(len(names) == len(set(names)), "返回结果无重复 (符合 std::set 语义)")

    print(f"\n  共返回 {len(names)} 个 tensor name, 全部示例:")
    for n in names:
        print(f"    - {n}")
    return ctx


def test_real_model(fb_path):
    print(f"\n=== 测试 2: 真实模型 {os.path.basename(fb_path)} ===")
    if not os.path.exists(fb_path):
        print(f"  {RED}[SKIP]{RESET} 模型文件不存在: {fb_path}")
        return

    ctx = TFContext(path=fb_path)
    names = ctx.GetAllTensorNames()

    check(isinstance(names, list), f"返回类型为 list (实际: {type(names).__name__})")
    check(len(names) > 0, f"返回非空 (共 {len(names)} 个 tensor name)")
    check(all(isinstance(n, str) for n in names), "所有元素均为 str")
    check(len(names) == len(set(names)), "返回结果无重复")

    in_syms = [str(x) for x in ctx.GetInputSymbols()]
    out_syms = [str(x) for x in ctx.GetOutSymbols()]
    check(set(in_syms).issubset(set(names)),
          f"输入 {in_syms} 是返回集合的子集")
    check(set(out_syms).issubset(set(names)),
          f"输出 {out_syms} 是返回集合的子集")

    print(f"\n  输入 tensor: {in_syms}")
    print(f"  输出 tensor: {out_syms}")
    print(f"  全部 tensor 数量: {len(names)}")


if __name__ == "__main__":
    # 测试 1: 合成图
    test_synthetic_graph()

    # 测试 2: 真实 .fb 模型 (取一个较小的量化模型)
    # 优先用 SDK 仓库内的绝对路径, 兼容脚本被拷贝到其它目录运行的情况
    repo_convert = "/root/NPU-SDK/TFDL2_SDK/ConvertTools/python"
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(repo_convert, "arc_recognition2.quant.fb"),
        os.path.join(repo_convert, "arc_recognition.quant.fb"),
        os.path.join(here, "..", "ConvertTools", "python", "arc_recognition2.quant.fb"),
        os.path.join(here, "..", "ConvertTools", "python", "arc_recognition.quant.fb"),
    ]
    for c in candidates:
        if os.path.exists(c):
            test_real_model(c)
            break

    print(f"\n{'=' * 50}")
    print(f"{GREEN}通过: {_passed}{RESET}  {RED}失败: {_failed}{RESET}")
    sys.exit(1 if _failed else 0)
