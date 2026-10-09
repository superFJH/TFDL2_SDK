# 安装预编译 TFLLM SDK 到 Python 环境

此目录的 `setup.py` 打包 **CMake 已经安装的 SDK**，不编译源码。
源码模板位于 `TFLLM/packaging`，每次 `make install`/`cmake --install` 自动输出到 SDK 的 `tfllm/`。

```sh
# 构建配置中开启 TFLLM_WITH_TORCH=ON，才能包含 tfllm.torch 和原生 Torch 库。
# CLI chat/serve 另需 TFLLM_WITH_LLAMA=ON。完成构建、安装后：
cd /path/to/SDK/tfllm
python -m pip install .
# 如需同时安装 PyTorch/safetensors 依赖：
python -m pip install '.[torch]'
```

`.[torch]` 只选择 Python 依赖，不能给缺少 Torch 产物的 SDK 补编译。
已有板端专用 PyTorch 时可沿用现有安装；离线部署先准备好 setuptools>=77、
jinja2 及所需运行依赖，再使用 `python -m pip install . --no-build-isolation --no-deps`。
Transformers/Diffusers、视觉/视频及模型转换依赖仍按对应功能文档另装，不强制所有 CLI 用户安装 Torch。

安装后的布局：

```text
site-packages/
  tfllm/torch/                 # 构建开启 Torch 时直接可导入的 Python 包
  tfllm_runtime/
    __init__.py               # 轻量命令启动器，不导入 Torch
    sdk/                      # 原有 bin/lib/include/schema/llama/share 相对布局
      lib/libtfllm-torch.so
      lib/libNPU40T.so         # 从 SDK 根 lib/ 复制（若存在）
```

SDK 的 `test/` 和 Python bytecode 不打入 wheel。普通 Python 不再需要设置 `PYTHONPATH`
或 `TFLLM_TORCH_LIBRARY`，后者仍可显式覆盖默认库位置：

```python
import torch
from tfllm.torch import NpuRuntime, optimize, optimization_report

with NpuRuntime() as runtime:
    optimize(model.cpu().half().eval(), runtime=runtime)
    with torch.inference_mode():
        output = model(**inputs)
    print(optimization_report(model))
```

`bin/` 中的可执行文件生成同名 console_scripts，包括存在时的 `tfllm`、`tfllm_profile`、
`tfllm-run`。Python launcher 使用当前 Python 环境的解释器启动，不依赖 shell 激活 venv；
原生程序通过 exec 启动，保持参数、退出码和信号语义。NPU 库沿用 SDK 的相对 RUNPATH；
系统 ABI、驱动设备节点及其他平台依赖仍需与构建环境匹配。

分发时使用 wheel：

```sh
python -m pip wheel . --no-deps -w /path/to/wheels
```

这是包含原生二进制的 `py3-none-<platform>` wheel，不是 `py3-none-any`。
默认平台来自打包机器，通常应在目标板同平台环境打包；跨平台打包者必须显式设置
`TFLLM_WHEEL_PLATFORM=linux_aarch64` 等真实目标标签，并自行验证二进制和系统 ABI。
设置标签不会重编译二进制，也不代表符合 manylinux。规范见
[Python wheel 格式](https://packaging.python.org/en/latest/specifications/binary-distribution-format/)。
该预编译安装树不支持 sdist 或 editable 安装。

更新 SDK 后，已经通过 pip 安装的 `tfllm` 不会自动更新：wheel 在
`tfllm_runtime/sdk` 内保存了独立副本。先完成新版 SDK 的 `cmake --install`，
再在原 Python 环境重新安装，即使包版本仍为 `0.1.0`：

```sh
python3 -m pip install --force-reinstall --no-deps /path/to/SDK/tfllm
hash -r
```

离线环境另加 `--no-build-isolation`，沿用前文的打包依赖。也可直接运行
`/path/to/SDK/tfllm/bin/tfllm` 使用安装树。`type -a tfllm` 能查看当前 shell
选中的命令；仅更新 SDK 中的库或 Python 文件不会更新旧 wheel。

打包源码/原生 Torch 库必须配套；缺少任意一侧会直接报错。
更新 Conv 对齐切分/stalling 诊断功能时，需要同步重建 `libNPU40T`（新增只读诊断接口）和 TFLLM，
重新 `make install` 后再从安装树安装 Python 包；仅更新 `.py` 文件不生效。
运行时可用 `TFLLM_NPU_DUMP_DIR=/path/to/debug` 指定故障文件目录，详见 [Torch 诊断说明](TORCH.md#npu-stalling--超时诊断文件)。
从启用 Torch 切换到关闭 Torch 的构建时，CMake 不会删除旧安装产物，请安装到新的空 prefix，
否则旧文件仍会被当作 SDK 的一部分。重复 wheel 构建会清理本包的生成目录，避免 build/ 残留进入新 wheel。
