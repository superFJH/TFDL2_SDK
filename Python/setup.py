import os
import sys
from setuptools import setup,find_packages
from setuptools.command.install import install

# 获取当前 Python 版本
python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
'''
# 支持的 Python 版本和对应的 .so 文件
SUPPORTED_VERSIONS = {
    "3.6": "TFDL2_3_6.so",
    "3.8": "TFDL2_3_8.so",
    "3.10": "TFDL2_3_10.so",
}

# 检查当前 Python 版本是否支持
if python_version not in SUPPORTED_VERSIONS:
    raise RuntimeError(f"Unsupported Python version: {python_version}. Supported versions are: {list(SUPPORTED_VERSIONS.keys())}")

# 获取对应的 .so 文件
so_file = SUPPORTED_VERSIONS[python_version]

# 自定义安装类
class CustomInstall(install):
    def run(self):
        # 调用父类的安装方法
        super().run()

        # 创建软链接
        target_so = os.path.join(self.install_lib, "TFDL2", so_file)
        link_so = os.path.join(self.install_lib, "TFDL2", "TFDL2.so")

        # 删除已存在的软链接
        if os.path.exists(link_so):
            os.remove(link_so)

        # 创建软链接
        os.symlink(target_so, link_so)
        print(f"Created symlink: {link_so} -> {target_so}")

# 打包并安装
setup(
    name="TFDL2",
    version="1.0.0",
    description="熠知 TFDL2 Python package with version-specific .so files",
    author="ThinkForce.Inc",
    author_email="feng.jianhao@think-force.com",
    packages=find_packages(),  # 自动查找所有包（包括子目录）
    package_dir={"": "."},  # 包的根目录
    package_data={
        "TFDL2": [so_file, "**/*.py", "**/*.so"],  # 包含所有 .so 文件和子目录中的 .py 文件
    },
    include_package_data=True,
    cmdclass={"install": CustomInstall},  # 自定义安装类
)
'''
from setuptools import setup, Extension
import pybind11
import glob

# 获取 lib/ 目录下所有库文件（.so 和 .a）
lib_dir = os.path.join(os.path.dirname(__file__), "TFDL2/lib")
libs = [os.path.basename(f).replace(".so", "").replace(".a", "").replace("lib", "") 
        for f in glob.glob(os.path.join(lib_dir, "lib*"))]
# 获取头文件路径（假设 TFDL2/include 在项目根目录下）
include_dir = os.path.abspath("TFDL2/include")
# 定义扩展模块
module = Extension(
    name="TFDL2.TFDL2",       # 导入路径：my_module.core
    sources=["TFDL2/TFDL2_PythonWrap.cpp"],
    include_dirs=[include_dir,pybind11.get_include()],  # pybind11 头文件路径
    libraries=libs,             # 需要链接的库名（去掉前缀 `lib` 和后缀 `.so/.a`）
    library_dirs=[lib_dir],      # 库文件搜索路径
    extra_compile_args=['-O3','-std=c++14'],  # 可选：优化选项
    extra_link_args=["-Wl,--disable-new-dtags","-Wl,-rpath=$ORIGIN/lib"],  # 相对路径
    language="c++",              # 如果是 C++ 代码
)
# 打包
setup(
    name="TFDL2",
    version="1.2.0",
    description="熠知 TFDL2 Python package with version-specific .so files",
    author="ThinkForce.Inc",
    author_email="feng.jianhao@think-force.com",
    ext_modules=[module],
    setup_requires=['pybind11'],  # 自动安装 pybind11
    install_requires=["numpy"],  # 仍声明运行时依赖
    packages=find_packages(),  # 自动查找所有包（包括子目录）
    package_dir={"": "."},  # 包的根目录
    package_data={
        "TFDL2": ["**/*.py", "**/*.so"],  # 包含所有 .so 文件和子目录中的 .py 文件
    },
    include_package_data=True,
)

