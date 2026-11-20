"""Breeze HF -> GGUF, with direct BF16/F32 quantization and bounded tensor memory.

The upstream mapping is pinned beside this file. Two passes write tensor metadata
then payloads; no whole-model F16 intermediate or duplicate weight spool is used.
"""
import argparse
from collections import Counter
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import tempfile


def upstream():
    root = Path(__file__).resolve().parent
    scripts = root / "upstream" / "scripts" if (root / "upstream").is_dir() else root
    sys.path.insert(0, str(scripts))
    # Use the same gguf implementation as TFLLM's pinned llama checkout.
    base = root.parents[2] if len(root.parents) > 2 else root
    for package in (base / "share/tfllm/llama/gguf-py",
                    base / "Third_Party/llama.cpp/gguf-py",
                    base / "../Third_Party/llama.cpp/gguf-py",
                    root.parent / "llama/gguf-py"):
        if package.is_dir():
            sys.path.insert(0, str(package.resolve()))
            break
    spec = importlib.util.spec_from_file_location("tfllm_breeze_conversion", scripts / "convert_hf_to_gguf.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_source(directory, module):
    config = json.loads((directory / "config.json").read_text())
    if "BreezeForConditionalGeneration" not in config.get("architectures", []):
        raise ValueError("expected BreezeForConditionalGeneration HF checkpoint")
    if config.get("backbone_config", {}).get("model_type") != "qwen3":
        raise ValueError("Breeze requires the nested Qwen3 backbone_config")
    bb = config["backbone_config"]
    if bb.get("rope_scaling") or bb.get("use_sliding_window") or bb.get("attention_bias") or bb.get("mlp_bias"):
        raise ValueError("unsupported Breeze backbone attention/RoPE configuration")
    if config.get("num_codebooks") != 16:
        raise ValueError("Breeze currently requires 16 codebooks")
    audio = directory / "audio_tokenizer"
    if not (audio / "config.json").is_file():
        raise ValueError("Breeze checkpoint must include audio_tokenizer/config.json and its safetensors")
    # Check offsets, dimensions, shard identity and dtype before exporting.
    total = 0
    for folder in (directory, audio):
        reader = module.ShardReader(folder)
        for name, filename in reader.weight_map.items():
            file = (folder / filename).resolve()
            if Path(filename).is_absolute() or ".." in Path(filename).parts:
                raise ValueError("safetensors shard escapes its checkpoint directory")
            st = reader.handle(filename)
            info = st.header[name]
            shape = info["shape"]
            if not shape or any(type(x) is not int or x < 1 for x in shape):
                raise ValueError("invalid tensor dimensions: " + name)
            size = {"F32": 4, "F16": 2, "BF16": 2}.get(info["dtype"])
            start, end = info["data_offsets"]
            count = math.prod(shape)
            if not size or start < 0 or end-start != size*count or end > file.stat().st_size-st.data_start:
                raise ValueError("invalid/truncated safetensors payload: " + name)
            total += count
            if total > 30_000_000_000:
                raise ValueError("Breeze checkpoint exceeds 30B total parameters")
    return config


def convert(directory, output, dtype="q8_0", depth_dtype="preserve", backbone_dtype="preserve"):
    precisions = {"q4_0", "q8_0", "f16", "f32"}
    if dtype not in precisions or any(p not in precisions | {"preserve"} for p in (depth_dtype, backbone_dtype)):
        raise ValueError("invalid Breeze export precision")
    directory, output = Path(directory).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError("Breeze output already exists: " + str(output))
    module = upstream()
    np, gguf = module.np, module.gguf
    config = validate_source(directory, module)
    import convert_codec
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".breeze-", suffix=".gguf", dir=output.parent)
    os.close(fd)
    writer = gguf.GGUFWriter(temporary, module.ARCH)
    counts = Counter()
    planning = True
    order = []
    index = 0

    def add(_writer, name, array, _dtype):
        nonlocal index
        array = np.ascontiguousarray(array, dtype=np.float32)
        if name.startswith("te.") and name.endswith("_norm.weight"):
            array = array + np.float32(1)
        if not np.isfinite(array).all():
            raise ValueError("nonfinite Breeze tensor: " + name)
        precision = depth_dtype if name.startswith("dd.") and depth_dtype != "preserve" else dtype
        if name.startswith("bb.") and backbone_dtype != "preserve":
            precision = backbone_dtype
        sensitive = (array.ndim <= 1 or name.endswith("embed") or "codebook" in name or
                     ".conv." in name or name.endswith(".dw.weight") or name.endswith("sample.weight"))
        quantized = precision in {"q4_0", "q8_0"} and not sensitive and array.ndim == 2 and array.shape[-1] % 32 == 0
        target = gguf.GGMLQuantizationType[precision.upper()] if quantized else (
            gguf.GGMLQuantizationType.F32 if sensitive or precision == "f32" else gguf.GGMLQuantizationType.F16)
        if planning:
            block, type_size = gguf.GGML_QUANT_SIZES[target]
            size = array.size // block * type_size
            writer.add_tensor_info(name, array.shape, np.dtype(np.float32), size, raw_dtype=target)
            order.append((name, tuple(array.shape), target))
            counts[target.name] += 1
        else:
            if order[index] != (name, tuple(array.shape), target):
                raise ValueError("checkpoint tensor layout changed during conversion")
            index += 1
            encoded = gguf.quants.quantize(array, target) if quantized else array.astype(
                np.float32 if target == gguf.GGMLQuantizationType.F32 else np.float16)
            if not quantized and not np.isfinite(encoded).all():
                raise ValueError("Breeze tensor overflows export precision: " + name)
            writer.write_tensor_data(encoded)

    def tensors():
        reader = module.ShardReader(directory)
        module.write_text_encoder(writer, reader, config, dtype)
        module.write_backbone(writer, reader, config, dtype)
        module.write_depth(writer, reader, config, dtype)
        audio_reader = module.ShardReader(directory / "audio_tokenizer")

        class AudioReader:
            def names(self):
                return audio_reader.keys()

            def get(self, name):
                return audio_reader.get(name)

        codec_config = json.loads((directory / "audio_tokenizer/config.json").read_text())
        convert_codec.write_codec(lambda name, data: add(writer, name, data, dtype), AudioReader(), codec_config)
        return codec_config

    module.add = add
    try:
        module.write_metadata(writer, config)
        module.write_tokenizer(writer, directory)
        codec_config = tensors()
        convert_codec.codec_metadata(writer, codec_config)
        writer.add_string("tfllm.breeze.source_architecture", "BreezeForConditionalGeneration")
        writer.add_string("tfllm.breeze.quantization", dtype)
        writer.add_string("tfllm.breeze.depth_quantization", depth_dtype)
        writer.add_string("tfllm.breeze.backbone_quantization", backbone_dtype)
        writer.add_string("tfllm.breeze.runtime_revision", "155e60818433ea6322367aff8f8436ba790f6fbe")
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_ti_data_to_file()
        planning = False
        tensors()
        writer.close()
        # Publish only a complete file, and never overwrite a racing exporter.
        os.link(temporary, output)
        return {"gguf": str(output), "quantization": dtype, "depth_quantization": depth_dtype, "backbone_quantization": backbone_dtype,
                "tensors": dict(counts), "size_bytes": output.stat().st_size}
    finally:
        writer.close()
        Path(temporary).unlink(missing_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("--output", "-o", required=True, type=Path)
    parser.add_argument("--dtype", choices=["q4_0", "q8_0", "f16", "f32"], default="q8_0")
    parser.add_argument("--depth-dtype", choices=["preserve", "q4_0", "q8_0", "f16", "f32"], default="preserve")
    parser.add_argument("--backbone-dtype", choices=["preserve", "q4_0", "q8_0", "f16", "f32"], default="preserve")
    args = parser.parse_args()
    print(json.dumps(convert(args.model_dir, args.output, args.dtype, args.depth_dtype, args.backbone_dtype)))
