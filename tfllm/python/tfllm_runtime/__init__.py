"""TFLLM 引导包: 定位随包安装的 SDK 树, 把控制台命令转发给 sdk/bin 下的原生产物.

安装布局 (pip install .):
    site-packages/tfllm_runtime/
        __init__.py
        sdk/                     # 完整 TFLLM 树 (bin/ lib/ share/ ...) + libNPU40T.so
源码/editable 布局回退到本包上两级 (即 tfllm/ 目录本身).

所有控制台脚本 (tfllm, tfllm-run, tfllm_profile, ...) 共用 launch(): 按 argv[0]
的 basename 找到 sdk/bin 下同名可执行文件并 execve 替换进程. 原生二进制的
RUNPATH ($ORIGIN/../lib) 与 bin/tfllm 启动器的相对布局均不需修改.
"""
import os
import sys
from pathlib import Path


def sdk_root():
    """返回 TFLLM SDK 根目录; 找不到时抛 RuntimeError."""
    here = Path(__file__).resolve().parent
    for candidate in (here / "sdk", here.parents[2]):
        if (candidate / "bin").is_dir() and (candidate / "lib").is_dir():
            return candidate
    raise RuntimeError("TFLLM SDK 树缺失; 请重新 pip install 本包")


def commands():
    """列出 sdk/bin 下可作为控制台命令的可执行文件名."""
    return sorted(p.name for p in (sdk_root() / "bin").iterdir() if p.is_file())


def launch():
    name = os.path.basename(sys.argv[0])
    binary = sdk_root() / "bin" / name
    if not binary.is_file():
        raise SystemExit(f"tfllm_runtime: 未知命令 {name}; 可用: {' '.join(commands())}")
    os.execve(binary, [str(binary), *sys.argv[1:]], env=os.environ)
