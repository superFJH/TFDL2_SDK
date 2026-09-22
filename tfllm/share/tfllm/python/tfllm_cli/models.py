"""Resolve immutable model assets; conversions never modify source snapshots."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import struct
import sys
import tempfile

CACHE_VERSION = 3
VISION_ARCHITECTURES = {"Qwen3VLForConditionalGeneration", "Qwen2VLForConditionalGeneration",
                        "Qwen2_5_VLForConditionalGeneration", "LlavaForConditionalGeneration",
                        "LlavaNextForConditionalGeneration", "Mistral3ForConditionalGeneration",
                        "InternVLChatModel", "InternVLForConditionalGeneration",
                        "Gemma3ForConditionalGeneration", "LocateAnythingForConditionalGeneration"}


def check_safetensors(files):
    parameters, names = 0, set()
    for file in files:
        with file.open("rb") as f:
            length = f.read(8)
            if len(length)!=8:
                raise ValueError("Truncated safetensors: " + str(file))
            size = struct.unpack("<Q", length)[0]
            if size>64*1024*1024 or size>file.stat().st_size-8:
                raise ValueError("Invalid safetensors header: " + str(file))
            header = json.loads(f.read(size))
        for name, tensor in header.items():
            if name == "__metadata__":
                continue
            if name in names:
                raise ValueError("Duplicate tensor: " + name)
            names.add(name)
            item_size = {"F16": 2, "BF16": 2, "F32": 4}.get(tensor.get("dtype"))
            if not item_size:
                raise ValueError("Only floating safetensors are currently accepted: " + name)
            count = 1
            for dim in tensor["shape"]:
                if type(dim) is not int or dim<1:
                    raise ValueError("Invalid tensor shape: " + name)
                count *= dim
            begin, end = tensor["data_offsets"]
            if begin<0 or end-begin!=count*item_size or end>file.stat().st_size-size-8:
                raise ValueError("Truncated/invalid tensor payload: " + name)
            parameters += count
            if parameters>30_000_000_000:
                raise ValueError("Model exceeds 30B total parameters")
    return parameters


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_manifest(directory):
    value = read_json(directory / "manifest.json")
    if value.get("tfllm_cache_version") != CACHE_VERSION:
        raise ValueError("Unsupported TFLLM cache version; re-import the original model directory/GGUF with this TFLLM version")
    if value.get("decode_quantization") not in {"source", "q4_0"}:
        raise ValueError("Missing/invalid decode policy in TFLLM cache")
    for field in ("package", "gguf") + (("mmproj",) if value.get("mmproj") else ()):
        value[field] = str((directory / value[field]).resolve())
    if value.get("source_gguf"):
        value["source_gguf"] = str((directory / value["source_gguf"]).resolve())
    if not all(Path(value["package"] + ext).is_file() for ext in (".tfllm", ".weights", ".commands")) or not all(p.is_file() for p in gguf_shards(Path(value["gguf"]))):
        raise ValueError("Incomplete TFLLM cache entry; keep its package and all decode GGUF shards")
    if value.get("mmproj") and not Path(value["mmproj"]).is_file():
        raise ValueError("Missing mmproj asset in TFLLM cache")
    return value


def download(model, source, revision=None, gguf_file=None, mmproj="auto"):
    # A normal Transformers snapshot does not need every GGUF quantization
    # variant that a repository might also publish.
    patterns = ["*.json", "*.jinja", "*.model", "*.txt", "*.tiktoken"]
    patterns += [gguf_file] if gguf_file else ["*.safetensors"]
    if gguf_file and mmproj != "none":
        patterns += ["mmproj*.gguf"] if mmproj == "auto" else [Path(mmproj).name]
    kwargs = dict(allow_patterns=patterns)
    if revision:
        kwargs["revision"] = revision
    try:
        if source == "modelscope":
            from modelscope import snapshot_download
            return Path(snapshot_download(model, **kwargs))
        from huggingface_hub import snapshot_download
        return Path(snapshot_download(repo_id=model, **kwargs))
    except ImportError as e:
        raise RuntimeError(f"Install download support: {sys.executable} -m pip install "
                           + ("modelscope" if source == "modelscope" else "huggingface_hub")) from e


def gguf_shards(path):
    match = re.fullmatch(r"(.+)-(\d{5})-of-(\d{5})\.gguf", path.name)
    if not match:
        return [path]
    if int(match[2]) != 1:
        raise ValueError("Provide the first GGUF shard (-00001-of-...).")
    count = int(match[3])
    if count < 1 or count > 10000:
        raise ValueError("Invalid GGUF shard count")
    return [path.with_name(f"{match[1]}-{i:05d}-of-{count:05d}.gguf") for i in range(1, count+1)]


def architectures(config):
    # Recent Transformers saves Qwen2/2.5-VL's wrapper name in text_config.
    return config.get("architectures") or config.get("text_config", {}).get("architectures") or []


def source_files(path, gguf_file=None):
    if path.is_file():
        if path.suffix.lower() != ".gguf":
            raise ValueError("Model file must be GGUF; safetensors input requires its model directory")
        files = gguf_shards(path)
        if not all(p.is_file() for p in files):
            raise ValueError("Missing GGUF shard")
        return path, files, False
    if gguf_file:
        candidates = sorted(path.glob(gguf_file))
    elif (path / "config.json").is_file() and list(path.glob("*.safetensors")):
        candidates = []
    else:
        candidates = sorted(p for p in path.glob("*.gguf") if not p.name.lower().startswith("mmproj"))
    if candidates:
        candidates = [p for p in candidates if not re.search(r"-\d{5}-of-\d{5}\.gguf$", p.name)
                      or "-00001-of-" in p.name]
        if len(candidates) != 1:
            raise ValueError("Multiple GGUF models: select a file path or --gguf-file PATTERN (include all shards)")
        return source_files(candidates[0])
    if not (path / "config.json").is_file():
        raise ValueError("Expected config.json plus safetensors, or a GGUF model")
    config = read_json(path / "config.json")
    # Do not silently produce a llama fallback or load arbitrary Python model code.
    supported = {"LlamaForCausalLM", "Qwen2ForCausalLM", "Qwen3ForCausalLM", "Qwen3VLForConditionalGeneration",
                 "GemmaForCausalLM", "Gemma2ForCausalLM", "Gemma3ForCausalLM"} | VISION_ARCHITECTURES
    if not set(architectures(config)) & supported:
        raise ValueError("No native TFLLM importer for architectures=" + str(config.get("architectures")))
    for cfg in (config, config.get("text_config", {}), config.get("llm_config", {}), config.get("vision_config", {})):
        if isinstance(cfg,dict) and any(cfg.get(k,0) for k in ("num_experts", "num_local_experts", "n_routed_experts", "num_experts_per_tok")):
            raise ValueError("TFLLM supports dense models only; MoE is unsupported")
    text_config = config.get("text_config", {})
    if not isinstance(text_config, dict):
        raise ValueError("Invalid text_config: expected an object")
    if config.get("quantization_config") or text_config.get("quantization_config"):
        raise ValueError("Quantized safetensors import is not validated; use floating safetensors or a supported quantized GGUF")
    if "LocateAnythingForConditionalGeneration" in architectures(config):
        vision = config.get("vision_config", {})
        if (text_config.get("model_type") != "qwen2" or vision.get("model_type") != "moonvit"
                or vision.get("merge_kernel_size") != [2, 2]
                or config.get("use_backbone_lora") or config.get("use_llm_lora")):
            raise ValueError("LocateAnything requires dense Qwen2 + 2D MoonViT with 2x2 merge and merged LoRA weights")
    if "Qwen3VLForConditionalGeneration" in architectures(config):
        # The pinned converter extracts the language decoder and keeps its
        # qwen3vl GGUF identity; the vision weights are converted separately.
        rope = text_config.get("rope_parameters") or text_config.get("rope_scaling") or {}
        if rope.get("mrope_interleaved") is False:
            raise ValueError("Qwen3-VL native text import requires interleaved MRoPE")
    weights = sorted(path.glob("*.safetensors"))
    if not weights:
        raise ValueError("Missing safetensors weights")
    check_safetensors(weights)
    index = path / "model.safetensors.index.json"
    if index.is_file():
        for name in set(read_json(index)["weight_map"].values()):
            file = (path / name).resolve()
            if not file.is_relative_to(path.resolve()) or not file.is_file():
                raise ValueError("Missing/invalid weight shard: " + name)
    metadata = sorted(p for p in path.rglob("*") if p.is_file() and
                      p.suffix in {".json", ".jinja", ".model", ".txt", ".tiktoken"} and ".cache" not in p.relative_to(path).parts)
    return path, weights + metadata, True


def fingerprint(files):
    # Content is checked by the GGUF/package binding. This startup cache key
    # detects source replacement without rereading tens of GB every launch.
    return [[str(p.resolve()), p.stat().st_size, p.stat().st_mtime_ns,
             p.stat().st_ctime_ns, p.stat().st_ino] for p in files]


def run(command):
    print("TFLLM import: " + " ".join(map(str, command)), file=sys.stderr, flush=True)
    subprocess.run(list(map(str, command)), check=True, stdout=sys.stderr)


def prepare(args, bin_dir, converter_dir):
    selection = getattr(args, "mmproj", "auto")
    original = Path(args.model).expanduser()
    if not original.exists():
        if original.is_absolute() or args.model.startswith("."):
            raise ValueError("Model path does not exist: " + args.model)
        original = download(args.model, args.model_source or "huggingface", args.revision, args.gguf_file, selection)
    original = original.resolve()
    if original.is_dir() and (original / "manifest.json").is_file():
        manifest = load_manifest(original)
        if selection == "none":
            manifest.pop("mmproj", None)
        elif selection != "auto":
            file = Path(selection).expanduser().resolve()
            if not file.is_file():
                raise ValueError("Missing --mmproj file")
            manifest["mmproj"] = str(file)
        return manifest
    source, files, hf = source_files(original, args.gguf_file)
    mmproj = None
    export_vision = False
    directory = source if hf else source.parent
    if selection == "auto":
        if hf:
            config = read_json(source / "config.json")
            export_vision = bool(set(architectures(config)) & VISION_ARCHITECTURES) and (
                (source / "preprocessor_config.json").is_file() or (source / "processor_config.json").is_file()
                or "InternVLChatModel" in architectures(config))
        else:
            candidates = sorted(directory.glob("mmproj*.gguf"))
            if len(candidates)>1:
                raise ValueError("Multiple mmproj files: select --mmproj FILE or --mmproj none")
            mmproj = candidates[0] if candidates else None
    elif selection != "none":
        mmproj = Path(selection).expanduser()
        if not mmproj.is_file():
            candidates = sorted(directory.glob(selection)) if not mmproj.is_absolute() else []
            if len(candidates)!=1:
                raise ValueError("--mmproj must select one existing GGUF file")
            mmproj = candidates[0]
        mmproj = mmproj.resolve()
    if mmproj:
        files = files + [mmproj]
    converter = bin_dir / "tfllm-convert"
    if not converter.is_file():
        raise RuntimeError("Missing tfllm-convert; build/install with TFLLM_WITH_LLAMA=ON")
    revision_file = converter_dir / "REVISION"
    version = revision_file.read_text().strip() if revision_file.is_file() else "checkout"
    key_data = [CACHE_VERSION, version, fingerprint(files), fingerprint([converter]), "u8-row-symmetric", "vision-dense-v2", export_vision, str(mmproj), "bf16-decode-q4_0-v1"]
    # Converter changes must invalidate already compiled HF packages too.
    if hf:
        key_data.append(fingerprint(sorted((converter_dir / "conversion").glob("*.py"))))
    key = hashlib.sha256(json.dumps(key_data).encode()).hexdigest()
    cache = Path(args.cache_dir).expanduser().resolve()
    cache.mkdir(parents=True, exist_ok=True)
    destination = cache / key
    with (cache / (key + ".lock")).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (destination / "manifest.json").is_file():
            try:
                manifest = load_manifest(destination)
                print("TFLLM import cache hit: " + str(destination), file=sys.stderr)
                return manifest
            except (ValueError, KeyError, OSError):
                pass
        temporary = Path(tempfile.mkdtemp(prefix=key + ".tmp-", dir=cache))
        try:
            gguf = source
            if hf:
                script = converter_dir / "convert_hf_to_gguf.py"
                if not script.is_file():
                    raise RuntimeError("Missing installed HF converter")
                gguf = temporary / "source.gguf"
                try:
                    run([sys.executable, script, source, "--outtype", "auto", "--outfile", gguf])
                except subprocess.CalledProcessError as e:
                    raise RuntimeError("HF conversion failed (see converter output). Install its pinned dependencies with: "
                        f"{sys.executable} -m pip install -r {converter_dir / 'requirements/requirements-convert_hf_to_gguf.txt'}") from e
            if export_vision:
                mmproj = temporary / "mmproj.gguf"
                run([sys.executable, script, source, "--mmproj", "--outtype", "auto", "--outfile", mmproj])
            decode = temporary / "decode-q4_0.gguf"
            run([converter, gguf, temporary / "model", "--require-native", "--decode-q4-if-bf16", decode])
            # If input changed while converting, do not publish this snapshot.
            if fingerprint(files) != key_data[2]:
                raise RuntimeError("Source model changed during conversion; retry")
            generation = read_json(source / "generation_config.json") if hf and (source / "generation_config.json").is_file() else {}
            template = None
            if hf:
                if (source / "chat_template.jinja").is_file():
                    template = (source / "chat_template.jinja").read_text()
                elif (source / "chat_template.json").is_file():
                    template = read_json(source / "chat_template.json").get("chat_template")
                elif (source / "tokenizer_config.json").is_file():
                    template = read_json(source / "tokenizer_config.json").get("chat_template")
            manifest = dict(tfllm_cache_version=CACHE_VERSION, package="model",
                            gguf=decode.name if decode.is_file() else "source.gguf" if hf else str(gguf),
                            source_gguf="source.gguf" if hf else str(gguf),
                            decode_quantization="q4_0" if decode.is_file() else "source",
                            source=str(source), source_fingerprint=key_data[2], generation_config=generation,
                            chat_template=template, mmproj="mmproj.gguf" if export_vision else str(mmproj) if mmproj else None)
            (temporary / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
            if destination.exists():
                shutil.rmtree(destination)  # An incomplete entry under this key's lock.
            os.replace(temporary, destination)
            print("TFLLM import ready: " + str(destination), file=sys.stderr)
            return load_manifest(destination)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
