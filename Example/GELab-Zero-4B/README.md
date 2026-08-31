# GELab-Zero-4B 固化部署

本目录只维护当前已验证的部署组合；`deploy/model` 是唯一模型契约：

GELab 的 Python 实现、prefill 图构建、部署 wheel 与运行时配置均在本目录内；
运行和打包不会查找其他 `Example/` 项目。

| 阶段 | 当前产物 | 执行设备 |
| --- | --- | --- |
| vision | `vision/single-g32x18-fp16-bsc-matmul-runtime` | NPU FP16 |
| prefill | `prefill/s256-fp16-masksoftmax-grouped-gqa` | NPU FP16 |
| decode | `decode/llmdecode-q4_k_m` | ARM CPU llama.cpp + KleidiAI |
| tokenizer / final head | `runtime` | CPU；BF16 final RMSNorm + tied embedding |

没有 ONNX decoder、旧 Conv vision、量化 vision 或历史模型 bucket 的运行入口。

## 启动 API

```bash
source .venv-tfdl-linux/bin/activate

PYTHON=.venv-tfdl-linux/bin/python bash deploy/install_llmdecode.sh
bash deploy/run.sh --host 0.0.0.0 --port 8000
```

服务启动时构建一个 vision executor、36 个 prefill executor，并加载一个常驻
Q4_K_M GGUF decode worker。无需原始 checkpoint 或 `--model-path`。

KV 的有效长度逐请求计算：`S_valid = attention_mask.sum()`。NPU prefill 的
物理 bucket 虽为 256，传给 llama.cpp 前只会导入每层
`[1, Hkv, :S_valid, D]` 的紧凑 FP16 前缀；右侧 pad 永不进入 KV cache。

## 校验与打包

```bash
python deploy/verify_assets.py --allow-missing-model
python deploy/package_project.py --output /tmp/gelab-package
```

默认包只复制 `deployment.json` 选中的模型目录、runtime assets、TFDL addon
和目标 ARM wheel；不复制 safetensors checkpoint。目标机安装 wheel 后使用上面
的 `deploy/run.sh` 启动即可。

## 重新导出当前固定结构

仅在更新模型权重时执行 vision、prefill 与 runtime 导出；decode GGUF 是已验证
的固定产物，不由本项目重新导出：

```bash
python deploy/export_models.py --model-path MODEL_DIR --stage vision
python deploy/export_models.py --model-path MODEL_DIR --stage prefill
python deploy/export_models.py --model-path MODEL_DIR --stage runtime
```

`deployment.json` 是 vision / prefill `TFExecutor` 完整配置与 decode 配置的
唯一来源；修改模型路径或执行策略后应重新运行 `verify_assets.py`。

`deploy/wheels/` 必须包含一个与目标 ARM/Python ABI 匹配的
`tfdl_llmdecode-*.whl`；它随 GELab 项目打包，不从其他 Example 查找。
