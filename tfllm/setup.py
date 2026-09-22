# -*- coding: utf-8 -*-
"""
TFLLM pip 安装脚本.

TFLLM 本体是 C/C++ 产物 (bin/ lib/ share/ ...), 由父项目 CMake `make install`
铺设成相对布局; 本脚本把铺设好的树连同 SDK 根的 libNPU40T.so 一起装进
Python 环境, 并为 bin/ 下每个可执行文件生成同名控制台命令:

    cd tfllm && pip install .
    tfllm chat /models/Qwen3-8B

布局 (site-packages):
    tfllm_runtime/
        __init__.py        # 引导: 按命令名 exec sdk/bin/<name>
        sdk/               # 完整 TFLLM 树 + libNPU40T.so (test/ 除外)
    <venv>/bin/tfllm, tfllm-run, tfllm_profile, ...

原生二进制 RUNPATH 为 $ORIGIN/../lib:$ORIGIN/../../lib, libtfllm-chat.so 为
$ORIGIN:$ORIGIN/../../lib, 因此把 libNPU40T.so 复制进 sdk/lib 后无需改 rpath.
"""
import os
import shutil

from setuptools import setup
from setuptools.command.build_py import build_py as _build_py

HERE = os.path.dirname(os.path.abspath(__file__))   # .../TFDL2_SDK/tfllm
SDK = os.path.dirname(HERE)                          # SDK 根 (含 lib/libNPU40T.so)

# 装进包里的 SDK 子目录; test/ 是板端对照/benchmark 程序, pip 运行时不需要.
SDK_SUBDIRS = ("bin", "lib", "include", "schema", "llama", "share")
SDK_FILES = ("README.md",)
# sdk 树里的驱动库; 与 libtfllm-chat.so 同目录, 由其 $ORIGIN RUNPATH 解析.
DRIVERS = ("libNPU40T.so",)
EXCLUDE_NAMES = {"__pycache__"}


def copy_sdk_tree(destination):
    """把 TFLLM 铺设树复制到 build_lib/tfllm_runtime/sdk 下."""
    for name in SDK_SUBDIRS:
        src = os.path.join(HERE, name)
        if not os.path.isdir(src):
            raise SystemExit(f"[setup] 缺少 {src}; 请先在父项目执行 make install (WITH_TFLLM=ON)")
        shutil.copytree(src, os.path.join(destination, name),
                        ignore=shutil.ignore_patterns(*EXCLUDE_NAMES),
                        dirs_exist_ok=True)
    for name in SDK_FILES:
        shutil.copy2(os.path.join(HERE, name), os.path.join(destination, name))
    for name in DRIVERS:
        driver = os.path.join(SDK, "lib", name)
        if os.path.isfile(driver):
            shutil.copy2(driver, os.path.join(destination, "lib", name))
        else:
            print(f"[setup] 警告: 未找到 {driver}, NPU 后端将不可用")


class build_py(_build_py):
    def run(self):
        super().run()
        for package in self.packages:
            if package == "tfllm_runtime":
                copy_sdk_tree(os.path.join(self.build_lib, package, "sdk"))


def console_scripts():
    # 所有命令共用 launch(), 按 venv/bin 里的脚本名分发到 sdk/bin 同名文件.
    bin_dir = os.path.join(HERE, "bin")
    if not os.path.isdir(bin_dir):
        raise SystemExit(f"[setup] 缺少 {bin_dir}; 请先在父项目执行 make install (WITH_TFLLM=ON)")
    names = sorted(n for n in os.listdir(bin_dir)
                   if os.path.isfile(os.path.join(bin_dir, n)) and not n.startswith("."))
    print(f"[setup] 控制台命令: {' '.join(names)}")
    return [f"{n} = tfllm_runtime:launch" for n in names]


setup(
    name="tfllm",
    version=os.environ.get("TFLLM_VERSION", "0.1.0"),
    description="ThinkForce LLM runtime (NPU prefill + llama.cpp decode) with a Python launcher",
    long_description="TFLLM pip 包: 内嵌 CMake 安装出的 TFLLM 树与 NPU 驱动库, 提供 tfllm chat/serve 等命令.",
    long_description_content_type="text/markdown",
    package_dir={"": "python"},
    packages=["tfllm_runtime"],
    cmdclass={"build_py": build_py},
    entry_points={"console_scripts": console_scripts()},
    # 仅 serve/chat 必需; 下载 provider 与 HF 转换依赖按需另装 (见 share/tfllm/python/requirements.txt).
    install_requires=["jinja2>=3.1,<4"],
    python_requires=">=3.9",
    platforms=["linux"],
)
