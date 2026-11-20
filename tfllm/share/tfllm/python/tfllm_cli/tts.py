"""Breeze synthesis frontend. Runtime inference does not import Torch/NumPy."""
import array
import ctypes
import fcntl
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import threading
import wave

CACHE_VERSION = 1


def gguf_architecture(path):
    """Read only GGUF metadata, with bounded reads even for malformed files."""
    with Path(path).open("rb") as stream:
        def read(n):
            if n < 0 or stream.tell()+n > min(Path(path).stat().st_size, 64*1024*1024):
                raise ValueError("invalid/oversized GGUF metadata")
            data = stream.read(n)
            if len(data) != n:
                raise ValueError("truncated GGUF metadata")
            return data

        def integer(fmt):
            return struct.unpack("<"+fmt, read(struct.calcsize(fmt)))[0]

        def string():
            return read(integer("Q")).decode("utf-8")

        sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}

        def skip(kind):
            if kind == 8:
                read(integer("Q"))
            elif kind == 9:
                item, count = integer("I"), integer("Q")
                if count > 10_000_000 or item == 9:
                    raise ValueError("invalid GGUF array")
                if item in sizes:
                    read(count*sizes[item])
                else:
                    for _ in range(count):
                        skip(item)
            elif kind in sizes:
                read(sizes[kind])
            else:
                raise ValueError("unknown GGUF metadata type")

        if read(4) != b"GGUF" or integer("I") not in (2, 3):
            raise ValueError("expected GGUF v2/v3")
        integer("Q")
        count = integer("Q")
        if count > 100000:
            raise ValueError("oversized GGUF metadata")
        for _ in range(count):
            key, kind = string(), integer("I")
            if key == "general.architecture":
                if kind != 8:
                    raise ValueError("GGUF architecture must be a string")
                return string()
            skip(kind)
    raise ValueError("GGUF architecture is missing")


def is_local_breeze(model):
    path = Path(model).expanduser()
    if path.is_file() and path.suffix.lower() == ".gguf":
        return gguf_architecture(path) == "breeze-tts-2"
    if path.is_dir():
        if (path / "breeze.json").is_file():
            return True
        if (path / "config.json").is_file():
            return "BreezeForConditionalGeneration" in json.loads((path / "config.json").read_text()).get("architectures", [])
    return False


def library_path(bin_dir, profile=False):
    suffix = "_profile" if profile else ""
    for folder in (bin_dir.parent / "lib", bin_dir):
        for ext in ("so", "dylib"):
            file = folder / f"libtfllm-breeze{suffix}.{ext}"
            if file.is_file():
                return file
    raise RuntimeError("Missing Breeze runtime; build/install TFLLM_WITH_BREEZE=ON")


class Engine:
    @staticmethod
    def check(library, backend):
        lib = ctypes.CDLL(str(library))
        if lib.tfllm_breeze_abi_version() != 1:
            raise RuntimeError("Unsupported Breeze runtime ABI")
        if backend == "npu" and not lib.tfllm_breeze_has_npu():
            raise RuntimeError("Breeze NPU support is not compiled in; rebuild with TFLLM_WITH_NPU40T=ON or use --backend cpu for host testing")
        return lib

    def __init__(self, library, model, options):
        self.lib = self.check(library, options["backend"])
        self.lib.tfllm_breeze_last_error.restype = ctypes.c_char_p
        self.lib.tfllm_breeze_create.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
        self.lib.tfllm_breeze_create.restype = ctypes.c_void_p
        self.lib.tfllm_breeze_info.argtypes = [ctypes.c_void_p]
        self.lib.tfllm_breeze_info.restype = ctypes.c_char_p
        self.lib.tfllm_breeze_destroy.argtypes = [ctypes.c_void_p]
        self.lib.tfllm_breeze_validate_request.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        self.callback = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.POINTER(ctypes.c_float), ctypes.c_size_t, ctypes.c_void_p)
        self.lib.tfllm_breeze_generate.argtypes = [ctypes.c_void_p, ctypes.c_char_p, self.callback, ctypes.c_void_p]
        self.lib.tfllm_breeze_generate.restype = ctypes.c_char_p
        self.pointer = self.lib.tfllm_breeze_create(os.fsencode(model), json.dumps(options, allow_nan=False).encode())
        if not self.pointer:
            raise RuntimeError(self.lib.tfllm_breeze_last_error().decode())
        self.lock = threading.Lock()
        self.info = json.loads(self.lib.tfllm_breeze_info(self.pointer))

    def validate(self, request):
        with self.lock:
            if not self.pointer:
                raise RuntimeError("Breeze engine is closed")
            if self.lib.tfllm_breeze_validate_request(self.pointer, json.dumps(request, allow_nan=False).encode()):
                raise ValueError(self.lib.tfllm_breeze_last_error().decode())

    def generate(self, request, on_pcm):
        errors = []

        def emit(data, count, _user):
            try:
                values = array.array("f")
                values.frombytes(ctypes.string_at(data, count*4))
                samples = array.array("h", (max(-32768, min(32767, int(x*32767))) for x in values))
                if sys.byteorder != "little":
                    samples.byteswap()
                return int(on_pcm(samples.tobytes()) is False)
            except BaseException as exc:
                errors.append(exc)
                return 1

        with self.lock:
            if not self.pointer:
                raise RuntimeError("Breeze engine is closed")
            value = self.lib.tfllm_breeze_generate(self.pointer, json.dumps(request, allow_nan=False).encode(), self.callback(emit), None)
            if errors:
                raise errors[0]
            if not value:
                raise RuntimeError(self.lib.tfllm_breeze_last_error().decode())
            return json.loads(value)

    def close(self):
        with self.lock:
            if self.pointer:
                self.lib.tfllm_breeze_destroy(self.pointer)
                self.pointer = None


def converter_path(bin_dir):
    installed = bin_dir.parent / "share/tfllm/breeze/convert.py"
    source = Path(__file__).resolve().parents[2] / "adapters/breeze/convert.py"
    for file in (installed, source):
        if file.is_file():
            return file
    raise RuntimeError("Missing Breeze HF converter; reinstall TFLLM_WITH_BREEZE=ON")


def prepare(args, bin_dir):
    from .models import download, fingerprint
    path = Path(args.model).expanduser()
    if not path.exists():
        if path.is_absolute() or args.model.startswith("."):
            raise ValueError("Breeze model path does not exist")
        path = download(args.model, args.model_source or "huggingface", args.revision, args.gguf_file, "none")
    path = path.resolve()
    if path.is_dir() and (path / "snapshots").is_dir():
        ref = path / "refs/main"
        choices = [path / "snapshots" / ref.read_text().strip()] if ref.is_file() else list((path / "snapshots").iterdir())
        choices = [p for p in choices if (p / "config.json").is_file() or list(p.glob("*.gguf"))]
        if len(choices) != 1:
            raise ValueError("Select one Breeze HF snapshot explicitly")
        path = choices[0].resolve()
    if path.is_dir() and (path / "breeze.json").is_file():
        manifest = json.loads((path / "breeze.json").read_text())
        if manifest.get("version") != CACHE_VERSION:
            raise ValueError("Unsupported Breeze cache version; re-import the original model")
        path = path / manifest["gguf"]
    if path.is_dir() and (args.gguf_file or not (path / "config.json").is_file()):
        files = list(path.glob(args.gguf_file or "*.gguf"))
        if len(files) != 1:
            raise ValueError("Select one Breeze GGUF with --gguf-file or an explicit file path")
        path = files[0]
    if path.is_file():
        if gguf_architecture(path) != "breeze-tts-2":
            raise ValueError("expected a breeze-tts-2 GGUF")
        return path
    if not is_local_breeze(path) or not (path / "audio_tokenizer/config.json").is_file():
        raise ValueError("Breeze HF import requires config.json, tokenizer.json, safetensors and audio_tokenizer/")
    files = sorted(p for p in path.rglob("*") if p.is_file() and p.suffix in {".json", ".safetensors"} and ".cache" not in p.relative_to(path).parts)
    converter = converter_path(bin_dir)
    scripts = converter.parent / "upstream/scripts" if (converter.parent / "upstream").is_dir() else converter.parent
    implementation = [converter, scripts / "convert_hf_to_gguf.py", scripts / "convert_codec.py"]
    backbone_quantization = getattr(args, "backbone_quantization", "preserve")
    key = hashlib.sha256(json.dumps([CACHE_VERSION, fingerprint(files),
        [hashlib.sha256(p.read_bytes()).hexdigest() for p in implementation],
        args.tts_quantization, args.depth_quantization, backbone_quantization]).encode()).hexdigest()
    root = Path(args.cache_dir).expanduser().resolve() / "breeze"
    root.mkdir(parents=True, exist_ok=True)
    target = root / key
    with (root / (key+".lock")).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not (target / "breeze.json").is_file():
            with tempfile.TemporaryDirectory(prefix=".import-", dir=root) as temp:
                temp = Path(temp)
                command = [sys.executable, str(converter), str(path), "--output", str(temp / "model.gguf"),
                           "--dtype", args.tts_quantization, "--depth-dtype", args.depth_quantization,
                           "--backbone-dtype", backbone_quantization]
                print("TFLLM Breeze import: " + " ".join(command), file=sys.stderr, flush=True)
                subprocess.run(command, check=True, stdout=sys.stderr)
                (temp / "breeze.json").write_text(json.dumps({"version": CACHE_VERSION, "gguf": "model.gguf",
                    "source": str(path), "quantization": args.tts_quantization, "depth_quantization": args.depth_quantization,
                    "backbone_quantization": backbone_quantization}))
                temp.rename(target)
    model = target / "model.gguf"
    if not model.is_file() or gguf_architecture(model) != "breeze-tts-2":
        raise ValueError("Incomplete Breeze cache; re-import after removing its incomplete directory")
    return model


def run(args, profile_build=False):
    if args.command not in {"speak", "serve"}:
        raise ValueError("Breeze supports speak and serve --task tts")
    if args.command == "speak" and (not args.text or not args.output):
        raise ValueError("speak requires --text and --output")
    if bool(args.ref_audio) != bool(args.ref_text):
        raise ValueError("Breeze voice cloning requires both --ref-audio and --ref-text")
    if args.command == "serve" and (args.ref_audio or args.ref_text or args.text or args.output):
        raise ValueError("serve accepts synthesis text per request")
    if args.image or args.video or args.audio or args.language or args.mmproj != "auto":
        raise ValueError("Breeze synthesis uses --ref-audio/--ref-text for reference voices")
    if not 1 <= args.threads <= 256 or not 0 <= args.port <= 65535 or not 1 <= args.max_pending_requests <= 1024:
        raise ValueError("Invalid Breeze threads, port or queue limit")
    stage_threads = {}
    for stage in ("vocoder", "text", "backbone", "depth"):
        count = getattr(args, f"tts_{stage}_threads", None)
        if count is None:
            count = args.threads
        if not 1 <= count <= 256:
            raise ValueError(f"Breeze --tts-{stage}-threads must be in 1..256")
        stage_threads[f"{stage}_threads"] = count
    bin_dir = Path(sys.argv[0]).resolve().parent
    library = library_path(bin_dir, profile_build)
    Engine.check(library, args.backend)
    model = prepare(args, bin_dir)
    options = dict(backend=args.backend, threads=args.threads, npu_stages=args.npu_stages,
                   chip=args.chip, gpu_memory_utilization=args.gpu_memory_utilization,
                   **stage_threads, fuse_projections=getattr(args,"tts_projection_fusion","on")=="on")
    options["npu_decode"]=getattr(args,"tts_backbone_decode","npu")=="npu"
    options.update(lookahead_segments=getattr(args,"tts_lookahead",0),
                   lookahead_split_chars=getattr(args,"tts_segment_chars",120),
                   lookahead_first_chars=getattr(args,"tts_first_segment_chars",60))
    options["startup_buffer_ms"]=getattr(args,"tts_startup_buffer_ms",0)
    options["cpu_repack_backbone"]=getattr(args,"tts_backbone_repack","off")=="on"
    options["vocoder_fused_snake"]=getattr(args,"tts_vocoder_fusion","off")=="on"
    options["dialogue_pipeline"]=getattr(args,"tts_dialogue",False)
    options["pipeline_frames"]=getattr(args,"tts_pipeline_frames",2)
    repack=getattr(args,"tts_cpu_repack","auto")
    if repack!="auto":options["cpu_repack"]=repack=="on"
    if getattr(args, "profile", None):
        if not 1 <= args.profile_max_events <= 10000000:
            raise ValueError("--profile-max-events must be in 1..10000000")
        args.profile.mkdir(parents=True, exist_ok=True)
        options.update(profile_dir=tempfile.mkdtemp(prefix="breeze-", dir=args.profile.resolve()),
                       profile_max_events=args.profile_max_events)
    engine = Engine(library, model, options)
    name = args.served_model_name or "breeze-tts-2"
    defaults = {key: getattr(args, key) for key in ("temperature", "top_k", "top_p", "max_tokens", "seed") if getattr(args, key) is not None}
    defaults.update(cfg_scale=args.cfg_scale, instruction=args.instruction or "Speak clearly and naturally.",
                    chunk_first=args.chunk_first, chunk_max=args.chunk_max)
    try:
        print("TFLLM_TTS_READY " + json.dumps(dict(model=name, **engine.info)), file=sys.stderr, flush=True)
        if args.command == "serve":
            from .tts_server import make_server
            server = make_server(engine, args.host, args.port, name, args.api_key, args.max_pending_requests, defaults)
            try:
                print(f"TFLLM_TTS_HTTP http://{args.host}:{server.server_port}", file=sys.stderr, flush=True)
                server.serve_forever()
            finally:
                server.server_close()
            return 0
        request = dict(defaults, text=args.text)
        if args.ref_audio:
            request.update(ref_audio=str(args.ref_audio.expanduser().resolve()), ref_text=args.ref_text or "")
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        fd, temp = tempfile.mkstemp(prefix=".speech-", suffix=".wav", dir=output.parent)
        os.close(fd)
        try:
            with wave.open(temp, "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(engine.info["sample_rate"])
                stats = engine.generate(request, wav.writeframesraw)
            if not stats["samples"] or stats["cancelled"]:
                raise RuntimeError("Breeze generation was cancelled or returned no audio")
            os.replace(temp, output)
        finally:
            Path(temp).unlink(missing_ok=True)
        print("TFLLM_TTS_REQUEST " + json.dumps(dict(output=str(output), **stats)), file=sys.stderr, flush=True)
        return 0
    finally:
        engine.close()
