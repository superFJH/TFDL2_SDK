# -*- coding: utf-8 -*-
"""
TFDL2 Python 包构建脚本.

- 头文件/库一律取自 SDK 根的 include/ 与 lib/ (不再用 Python/TFDL2 下重复的副本).
- 自动按 CPU 核心类型判定芯片: Cortex-A77(0xd0d)->NPU40T, Cortex-A53(0xd03)->NPU10T.
  可用环境变量 CHIP=NPU40T/NPU10T 覆盖.
- 安装时把所需运行时 .so 从 SDK 复制到 安装目录/TFDL2/lib, 扩展模块 rpath=$ORIGIN/lib,
  使安装后的包自包含.

构建:  cd Python && pip install .
"""
import os
import glob
import shutil

from setuptools import setup, find_packages, Extension
from setuptools.command.install import install as _install
import pybind11

HERE = os.path.dirname(os.path.abspath(__file__))   # .../Python
SDK = os.path.dirname(HERE)                          # SDK 根
INCLUDE = os.path.join(SDK, "include")               # 头文件 (TFCV/, TFDL2_C_API.h, ...)
LIB = os.path.join(SDK, "lib")                       # libTFDL2_LITE_C_API.so, libNPU{CHIP}.so
CV_NPU = lambda chip: os.path.join(LIB, "CV_" + chip)  # lib/CV_NPU40T 等

# ---- 芯片自动识别 (CPU part) ----
# Cortex-A53=0xd03 A57=0xd07 A72=0xd08 A73=0xd09  -> NPU10T (TF16110)
# Cortex-A76=0xd0b A77=0xd0d A78=0xd41 Neoverse-N1=0xd0c -> NPU40T (TF7000)
_CHIP_PARTS = {
    "0xd03": "NPU10T", "0xd07": "NPU10T", "0xd08": "NPU10T", "0xd09": "NPU10T",
    "0xd0d": "NPU40T", "0xd0b": "NPU40T", "0xd41": "NPU40T", "0xd0c": "NPU40T",
}


def detect_chip():
    env = os.environ.get("CHIP") or os.environ.get("TFCV_CHIP")
    if env:
        return env
    try:
        with open("/proc/cpuinfo", "r") as f:
            for line in f:
                if "CPU part" in line:
                    part = line.split(":", 1)[-1].strip().lower()
                    return _CHIP_PARTS.get(part, "NPU40T")
    except Exception:
        pass
    return "NPU40T"  # 探测失败默认 40T


CHIP = detect_chip()
CV = CV_NPU(CHIP)
print("[setup] 芯片 = %s  (CPU 探测; 可用 CHIP= 环境变量覆盖)" % CHIP)
print("[setup] include = %s" % INCLUDE)
print("[setup] lib     = %s" % LIB)
print("[setup] cv      = %s" % CV)

# ---- 扩展模块 ----
core = Extension(
    name="TFDL2.TFDL2",
    sources=["TFDL2/TFDL2_PythonWrap.cpp"],
    include_dirs=[INCLUDE, pybind11.get_include()],
    libraries=["TFDL2_LITE_C_API"],
    library_dirs=[LIB],
    extra_compile_args=["-O3", "-std=c++14"],
    extra_link_args=["-Wl,--disable-new-dtags", "-Wl,-rpath,$ORIGIN/lib"],
    language="c++",
)

ext_modules = [core]
if os.path.isfile(os.path.join(CV, "libTFCV.so")):
    ext_modules.append(Extension(
        name="TFDL2._tfcv",
        sources=["TFDL2/TFCV_PythonWrap.cpp"],
        include_dirs=[INCLUDE, pybind11.get_include()],
        # _tfcv 只直接引用 TFCV:: 和 TFDL2_C_API 函数; CV 子库 (tfdec/tfenc/tfg/tfgs/mk_api,
        # 不同芯片集合不同) 是 libTFCV 的传递依赖, 由 rpath 在运行时解析, 无需在此 -l.
        libraries=["TFCV", "TFDL2_LITE_C_API"],
        library_dirs=[CV, LIB],
        extra_compile_args=["-O2", "-std=c++14"],
        # 运行时库都复制到 $ORIGIN/lib, 故只指该目录
        extra_link_args=["-Wl,--disable-new-dtags", "-Wl,-rpath,$ORIGIN/lib"],
        language="c++",
    ))
    print("[setup] 将构建 _tfcv (TFCV 流式推理)")
else:
    print("[setup] 跳过 _tfcv (未找到 %s/libTFCV.so)" % CV)

# ---- 安装时从 SDK 复制运行时库到 安装目录/TFDL2/lib (rpath=$ORIGIN/lib) ----
RUNTIME_LIBS = ["libTFDL2_LITE_C_API.so", "lib%s.so" % CHIP]   # 核心 + 芯片 NPU 驱动


class CustomInstall(_install):
    def run(self):
        super().run()
        dest = os.path.join(self.install_lib, "TFDL2", "lib")
        os.makedirs(dest, exist_ok=True)
        # 清掉上次安装残留的 .so (如旧的另一芯片 NPU 驱动 / 旧的 CV 子库), 保证本次只装所需集合
        for stale in glob.glob(os.path.join(dest, "*.so")):
            os.remove(stale)
        copied = []
        for name in RUNTIME_LIBS:
            src = os.path.join(LIB, name)
            if os.path.isfile(src):
                shutil.copy2(src, dest)
                copied.append(name)
        # CV 子库集合随芯片不同 (40T: tfdec/tfenc/tfgs/mk_api; 10T: tfdec/tfg/tfgs/mk_api),
        # 故直接复制 CV 目录下全部 .so, 不写死列表.
        if os.path.isdir(CV):
            for src in glob.glob(os.path.join(CV, "*.so")):
                shutil.copy2(src, dest)
                copied.append("CV/" + os.path.basename(src))
        print("[install] 复制运行时库到 %s: %s" % (dest, copied))


setup(
    name="TFDL2",
    version="2.0.0",
    description="熠知 TFDL2 Python package",
    author="ThinkForce.Inc",
    author_email="feng.jianhao@think-force.com",
    ext_modules=ext_modules,
    setup_requires=["pybind11"],
    install_requires=["numpy"],
    packages=find_packages(),
    package_dir={"": "."},
    package_data={"TFDL2": ["**/*.py"]},   # 运行时 .so 由 CustomInstall 从 SDK 复制
    include_package_data=False,
    cmdclass={"install": CustomInstall},
)
