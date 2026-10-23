# MiniCPM-V 4.7 35B-A3B

TFLLM 原生支持 `MiniCPMV4_7ForConditionalGeneration` 的 **Qwen3.5 MoE 语言模型 + SigLIP 视觉编码器**，提供文本、单图、多图及图片切片输入。NPU 执行视觉和 prefill 中的矩阵乘法/Attention；CPU 保留 FP32 路由、归一化和 DeltaNet 状态计算，并通过 llama.cpp 接力 decode。

官方模型为 40 层、hidden=2048、256 个专家，每 token 选择 8 个专家，另有共享专家；30 层采用线性注意力、10 层采用全注意力。支持部分旋转的 interleaved 三轴 MRoPE，以及缩略图、切片标记和换行的 canvas 坐标。视觉在第 6 层后插入 2x2 窗口 Attention/MLP，再通过最终 2x2 MLP 压缩；同时支持省略中间窗口模块的 4x 模式。

参考：[官方项目](https://github.com/OpenBMB/MiniCPM-V)、[官方 35B-A3B 配置](https://huggingface.co/openbmb/MiniCPM-V-4.7-35B-A3B/blob/main/config.json)、[Transformers 原生实现](https://github.com/huggingface/transformers/tree/main/src/transformers/models/minicpmv4_7)。

## 权重和内存

导入统计全部专家及视觉参数，`qwen35moe` 上限为 40B；其他结构仍保持 30B 上限。A3B 是每 token 的激活规模，不能据此推算完整权重只有 3B。

原生包使用 schema v4：路由器及共享专家 gate 保留 FP32，普通投影沿用 UINT8 行量化，路由专家按每 32 个值一个 FP16 scale 的 Q4 block 保存。GGUF Q4_0 专家可直接保存原始 block；浮点专家逐行转换，逐个专家写入包，避免一次展开整层专家。`--f32` 可生成用于精度对照的浮点包。普通 v1/v2/v3 包继续可读。

执行按所选专家聚合 token；一个专家执行完即释放其 NPU packing，避免累积 256 个专家的展开副本。当前没有实现官方 LES 专家预测/预取，也没有验证其移动设备内存、速度指标。专家权重仍需存在于本地 GGUF/包文件中；文件映射不等于全部权重都无需物理内存。

## 使用

重新构建已有 TFLLM 构建目录并安装，要求 `TFLLM_WITH_LLAMA=ON`；板端启用 `TFLLM_WITH_NPU40T=ON`。固定 llama.cpp revision 不变，转换扩展由 `apply_bridge.py` 安装。

准备好本地官方 HF 目录后，可以使用现有入口，自动生成语言 GGUF、视觉 mmproj 和原生包：

```sh
tfllm chat --model /path/to/MiniCPM-V-4.7-35B-A3B \
  --backend npu --mmproj auto --image /path/to/image.jpg

tfllm serve --model /path/to/MiniCPM-V-4.7-35B-A3B \
  --backend npu --mmproj auto --context 4096 --max-num-seqs 1
```

HF 导入自动排除 MTP；未启用投机解码。也可使用本地 `qwen35moe` GGUF 配套视觉 GGUF：

```sh
tfllm chat --model /path/to/model-Q4_0.gguf \
  --backend npu --mmproj /path/to/mmproj.gguf --image /path/to/image.jpg
```

转换需要额外输出空间；应在有足够空间的设备上完成。模型 ID 下载入口不在本次验证中使用。当前专家 GGUF 要求分别保存 gate/up/down，未支持 fused gate-up 专家张量。

## 验证范围

`tfllm-minicpm47-hf` 创建官方 Transformers 小配置随机模型，经实际 HF→GGUF→原生包转换，对照 4x/16x 非方形视觉特征、2x2 切片 canvas、相邻多图、图文 prefill 和共享状态 decode。tokenizer 使用本地生成的测试词表，不下载模型或词表。端到端对照使用路由间隔明确的小配置权重：top-k 在分数接近时会随 FP16/量化误差改变选择，不能把该测试误差当作完整 checkpoint 的精度保证。

`tfllm-qwen35moe-handoff` 覆盖不同 prefill 长度、CPU/NPU 后端接力、前缀重用、失败回滚、重放、独立会话与 Q4 专家包；`tfllm-moe` 覆盖 FP32 top-k/shared gate、未选专家不执行、失败释放和 Q4 行尾/padding；`tfllm-moe-npu`/`_profile` 用注入 SDK 执行生产 NPU 路径，检查结果和重复执行时权重 packing 不增长。

这些验证不包含完整 35B checkpoint 或真实 NPU 板端性能。当前原生 MiniCPM-V 4.7 视频、MTP 和 LES 预取尚未接入；视频请求明确拒绝，图片多切片可用。
