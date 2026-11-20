import argparse
import hashlib
import json
import math
import os
import signal
from pathlib import Path
import sys
import tempfile

from .chat import Chat
from .models import prepare
from .runtime import Engine, library_path
from .server import make_server


def npu_jit_cache_dir(cache_dir, manifest):
    cache = Path(cache_dir).expanduser().resolve()
    package = Path(manifest["package"]).resolve()
    # Converted models already have a content/version-keyed cache directory.
    if package.parent.parent == cache:
        return str(package.parent / "npu-jit")
    # A directly opened package may be read-only. Keep its recipes under the
    # selected cache root too; recipes contain geometry, not model weights.
    identity = json.dumps([str(package), manifest.get("mmproj") or ""])
    key = hashlib.sha256(identity.encode()).hexdigest()
    return str(cache / "npu-jit" / key)


def main(profile_build=False):
    parser = argparse.ArgumentParser(prog="tfllm_profile" if profile_build else "tfllm", description="Native TFLLM NPU prefill with llama.cpp CPU decode")
    parser.add_argument("command", choices=["serve", "chat", "transcribe", "speak"])
    parser.add_argument("--task", choices=["auto", "tts"], default="auto", help="Select Breeze synthesis for serve; speak selects it automatically")
    parser.add_argument("model", help="Transformers directory, GGUF file/directory, prepared cache directory, or repository ID")
    parser.add_argument("--model-source", choices=["huggingface", "modelscope"])
    parser.add_argument("--revision")
    parser.add_argument("--gguf-file", help="Select a GGUF download/model pattern, including all shards")
    parser.add_argument("--cache-dir", default=os.environ.get("TFLLM_CACHE_DIR", "~/.cache/tfllm"))
    parser.add_argument("--backend", choices=["npu", "cpu"], default="npu", help="CPU is explicit host testing, never an automatic fallback")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--mmproj", default="auto", help="Vision/audio GGUF; auto discovers/converts it")
    parser.add_argument("--audio", type=Path, help="Local audio file for transcribe (WAV/FLAC/MP3/OGG)")
    parser.add_argument("--text", help="Text for speak")
    parser.add_argument("--instruction", help="Breeze voice/delivery description")
    parser.add_argument("--output", type=Path, help="Output WAV for speak")
    parser.add_argument("--ref-audio", type=Path, help="Reference WAV for Breeze voice cloning")
    parser.add_argument("--ref-text", help="Exact reference-audio transcript")
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--chunk-first", type=int, default=1)
    parser.add_argument("--chunk-max", type=int, default=25)
    parser.add_argument("--npu-stages", default="te,bb", help="Breeze accelerated Linear stages: te,bb,dd,codec; default te,bb")
    parser.add_argument("--tts-quantization", choices=["q4_0", "q8_0", "f16", "f32"], default="q8_0", help="Breeze HF export precision; existing GGUF is preserved")
    parser.add_argument("--tts-vocoder-threads", type=int, help="Breeze vocoder CPU threads; default follows --threads")
    parser.add_argument("--tts-text-threads", type=int, help="Breeze text encoder CPU threads; default follows --threads")
    parser.add_argument("--tts-backbone-threads", type=int, help="Breeze backbone CPU threads; default follows --threads")
    parser.add_argument("--tts-depth-threads", type=int, help="Breeze depth decoder CPU threads; default follows --threads")
    parser.add_argument("--tts-lookahead", type=int, choices=[0,1,2,4], default=0, help="Breeze sentence batch size; 0 keeps the original sequential generation")
    parser.add_argument("--tts-segment-chars", type=int, default=120, help="Breeze lookahead text budget; CJK characters count as 3")
    parser.add_argument("--tts-first-segment-chars", type=int, default=60, help="Breeze first sentence budget for the shared voice reference")
    parser.add_argument("--tts-startup-buffer-ms", type=int, default=0, help="Breeze initial audio buffer in milliseconds (0..10000); generation continues while it fills")
    parser.add_argument("--tts-backbone-repack", choices=["on","off"], default="off", help="Pack CPU-only Breeze backbone weights for ARM matrix kernels")
    parser.add_argument("--tts-vocoder-fusion", choices=["on","off"], default="off", help="Fuse CPU Breeze SnakeBeta activations")
    parser.add_argument("--tts-dialogue", action="store_true", help="Overlap CPU speech-code generation and vocoder; requires zero startup buffer and lookahead=0")
    parser.add_argument("--tts-pipeline-frames", type=int, choices=[1,2,4], default=2, help="Dialogue packet size in 80 ms frames")
    parser.add_argument("--tts-cpu-repack", choices=["auto", "on", "off"], default="auto", help="Breeze DD Q4/Q8 CPU layout; auto enables below 16 threads on supported ARM CPUs")
    parser.add_argument("--tts-projection-fusion", choices=["on", "off"], default="on", help="Combine Breeze QKV and gate/up NPU projections")
    parser.add_argument("--tts-backbone-decode", choices=["npu", "cpu"], default="npu", help="Breeze backbone decode route; CPU retains GGUF weights while NPU handles prefill")
    parser.add_argument("--depth-quantization", choices=["preserve", "q4_0", "q8_0", "f16", "f32"], default="preserve", help="Breeze HF depth precision override; sensitive F32 tensors remain F32")
    parser.add_argument("--backbone-quantization", choices=["preserve", "q4_0", "q8_0", "f16", "f32"], default="preserve", help="Breeze HF backbone precision override; sensitive F32 tensors remain F32")
    parser.add_argument("--language", help="Transcription language (e.g. zh, en, Chinese); omitted means auto detection")
    parser.add_argument("--response-format", choices=["text", "json", "verbose_json"], default="text")
    parser.add_argument("--max-audio-seconds", type=int, default=120, help="Maximum audio duration per request (1..120)")
    parser.add_argument("--max-image-tokens", type=int, default=1024, help="Maximum merged visual tokens per image (1..4096)")
    parser.add_argument("--image", type=Path, action="append", default=[], help="Attach a local image to the first chat turn (repeatable)")
    parser.add_argument("--max-video-tokens", type=int, default=2048, help="Total merged visual tokens across videos per request (1..4096)")
    parser.add_argument("--video-max-frames", type=int, default=32, help="Maximum sampled frames per video (2..64)")
    parser.add_argument("--video-fps", type=float, default=2, help="Target sampling rate; frames cover the whole clip when capped (0..30]")
    parser.add_argument("--video", type=Path, action="append", default=[], help="Attach a local video to the first chat turn (repeatable, Qwen3-VL)")
    parser.add_argument("--chip", type=int, default=0)
    parser.add_argument("--npu-jit-shapes", type=Path, help="Precompile additional GEMM/attention geometries from npu_jit_shapes.py before READY")
    parser.add_argument("--context", type=int, default=4096)
    parser.add_argument("--threads", type=int, default=4, help="total CPU decode threads shared by all sequences")
    parser.add_argument("--kv-cache-type", choices=["auto", "q8_0", "f16"], default="auto",
                        help="persistent K/V storage; auto uses Q8_0 for head dimensions divisible by 32, otherwise F16")
    parser.add_argument("--max-num-seqs", type=int, default=4)
    parser.add_argument("--max-pending-requests", type=int, default=64)
    parser.add_argument("--chat-template", type=Path)
    parser.add_argument("--chat-template-kwargs", default="{}", help='JSON, e.g. {"enable_thinking":false}')
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--system", default="")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--served-model-name")
    parser.add_argument("--api-key", default=os.environ.get("TFLLM_API_KEY"))
    if profile_build:
        parser.add_argument("--profile", type=Path, help="Capture native and llama cb_eval operator spans; save CSV and JSONL per request in a run subdirectory")
        parser.add_argument("--profile-max-events", type=int, default=100000, help="Bound each request's trace (default: 100000 events)")
    elif any(a.split("=", 1)[0] in ("--profile", "--profile-max-events") for a in sys.argv[1:]):
        parser.error("profiling is not compiled in; use tfllm_profile with --profile")
    args = parser.parse_args()
    def interrupt(_signum, _frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupt)
    try:
        from . import tts
        if args.command == "speak" or args.task == "tts" or tts.is_local_breeze(args.model):
            return tts.run(args, profile_build)
        if args.text or args.instruction or args.output or args.ref_audio or args.ref_text:
            raise ValueError("--text/--instruction/--output/--ref-audio/--ref-text require Breeze speak or serve --task tts")
        if not 1 <= args.max_audio_seconds <= 120:
            raise ValueError("--max-audio-seconds must be in 1..120")
        if args.command == "transcribe":
            if not args.audio or args.image or args.video:
                raise ValueError("transcribe requires --audio FILE")
            from .asr import audio_dependencies, language_name, local_audio
            language_name(args.language)
            audio_dependencies()
            audio_data = local_audio(args.audio)
        elif args.audio or args.language:
            raise ValueError("--audio and --language are for transcribe; serve accepts uploaded audio")
        if not 0<args.gpu_memory_utilization<=1 or not 1<=args.max_num_seqs<=64 or args.max_pending_requests<args.max_num_seqs:
            raise ValueError("Invalid memory utilization or concurrency limits")
        if args.context<64 or args.context%64 or not 1<=args.threads<=256 or not 0<=args.chip<=9 or not 0<=args.port<=65535:
            raise ValueError("Invalid context, threads, chip or port")
        if not 1<=args.max_image_tokens<=4096 or (args.command == "serve" and args.image):
            raise ValueError("Invalid image token limit; --image is for chat, serve accepts image_url content")
        if not 1<=args.max_video_tokens<=4096 or not 2<=args.video_max_frames<=64 or not math.isfinite(args.video_fps) or not 0<args.video_fps<=30:
            raise ValueError("Invalid video token/frame/fps limits")
        if len(args.video)>4 or (args.command=="serve" and args.video):
            raise ValueError("--video is for chat (at most 4); serve accepts video_url content")
        kwargs = json.loads(args.chat_template_kwargs)
        if not isinstance(kwargs, dict) or set(kwargs) & {"messages", "bos_token", "eos_token", "add_generation_prompt"}:
            raise ValueError("Invalid --chat-template-kwargs")
        # Fail missing dependencies/backend before downloading or converting GBs.
        try:
            import jinja2  # noqa: F401
        except ImportError as e:
            raise RuntimeError(f"Install chat dependencies: {sys.executable} -m pip install 'jinja2>=3.1,<4'") from e
        bin_dir = Path(sys.argv[0]).resolve().parent
        library = library_path(bin_dir, profile_build)
        Engine.check(library, args.backend, profile_build)
        profile_options = {}
        if profile_build:
            if not 1 <= args.profile_max_events <= 10000000:
                raise ValueError("--profile-max-events must be in 1..10000000")
            if args.profile:
                args.profile.mkdir(parents=True, exist_ok=True)
                directory = tempfile.mkdtemp(prefix="run-", dir=args.profile.resolve())
                profile_options = dict(profile_dir=directory, profile_max_events=args.profile_max_events)
                print(f"TFLLM_PROFILE directory={directory}", file=sys.stderr, flush=True)
        installed = bin_dir.parent / "share" / "tfllm" / "llama"
        converter_dir = installed if installed.is_dir() else Path(os.environ["TFLLM_LLAMA_CONVERTER"])
        manifest = prepare(args, bin_dir, converter_dir)
        if manifest.get("modality") == "audio":
            from .asr import audio_dependencies
            audio_dependencies()
        print(f"TFLLM decode: weights={manifest['decode_quantization']} threads={args.threads} gguf={manifest['gguf']}",
              file=sys.stderr, flush=True)
        engine_options = dict(package=manifest["package"], gguf=manifest["gguf"], backend=args.backend,
                                     npu_jit_warmup=args.command == "serve",
                                     context=args.context, threads=args.threads, max_sequences=args.max_num_seqs,
                                     kv_cache_type=args.kv_cache_type,
                                     mmproj=manifest.get("mmproj") or "", max_image_tokens=args.max_image_tokens,
                                     max_video_tokens=args.max_video_tokens, video_max_frames=args.video_max_frames, video_fps=args.video_fps,
                                     max_audio_seconds=args.max_audio_seconds,
                                     gpu_memory_utilization=args.gpu_memory_utilization, chip=args.chip, **profile_options)
        if args.backend == "npu" and args.command == "serve":
            engine_options["npu_jit_cache_dir"] = npu_jit_cache_dir(args.cache_dir, manifest)
        if args.npu_jit_shapes:
            if args.backend != "npu":
                raise ValueError("--npu-jit-shapes requires --backend npu")
            shape_file = json.loads(args.npu_jit_shapes.read_text())
            if not isinstance(shape_file, dict) or shape_file.get("version") != 1:
                raise ValueError("NPU JIT shape file requires version=1")
            shapes = shape_file.get("shapes")
            if not isinstance(shapes, list) or len(shapes) > 4096:
                raise ValueError("NPU JIT shape file requires at most 4096 shapes")
            for s in shapes:
                if not isinstance(s, dict) or set(s) != {"m", "k", "n", "cores"} or any(type(v) is not int for v in s.values()):
                    raise ValueError("Each NPU JIT shape requires integer m, k, n, cores")
                if not (16 <= s["m"] <= 1024 and s["m"] % 16 == 0 and
                        all(64 <= s[d] <= 16384 and s[d] % 64 == 0 for d in ("k", "n")) and s["cores"] in (2, 8)):
                    raise ValueError("NPU JIT shape exceeds supported dimensions/alignment")
            engine_options["npu_jit_shapes"] = shapes
        template = args.chat_template.read_text() if args.chat_template else None
        defaults = {k: getattr(args, k) for k in ("max_tokens", "temperature", "top_p", "top_k", "seed")}
        if args.command == "transcribe":
            from .asr import Transcriber, native_stdout_to_stderr, response_body
            with native_stdout_to_stderr():
                engine = Engine(library, engine_options)
                try:
                    chat = Chat(engine.info, manifest, template, defaults, kwargs)
                    result = Transcriber(engine, chat).transcribe(audio_data, args.language, args.system)
                    if "error" in result:
                        raise RuntimeError(result["error"])
                    if result.get("finish_reason") == "cancelled":
                        raise RuntimeError("Transcription cancelled")
                    print("TFLLM_REQUEST " + json.dumps({k: v for k, v in result.items() if k != "text"}), file=sys.stderr, flush=True)
                finally:
                    engine.close()
            output = response_body(result, args.response_format)
            print(output if isinstance(output, str) else json.dumps(output, ensure_ascii=False, allow_nan=False))
            return 0
        engine = Engine(library, engine_options)
        try:
            # Direct ASR GGUF imports may not carry the HF modality manifest.
            if "audio" in engine.info.get("modalities", []):
                from .asr import audio_dependencies
                audio_dependencies()
            chat = Chat(engine.info, manifest, template, defaults, kwargs)
            model = args.served_model_name or args.model.rstrip("/").split("/")[-1]
            print(f"TFLLM_READY model={model} architecture={engine.info['architecture']} modalities={','.join(engine.info['modalities'])} backend={engine.info['backend']} context={args.context} max_sequences={args.max_num_seqs} decode_scheduler={engine.info.get('decode_scheduler', 'unknown')} decode_threads={args.threads} profile_compiled={int(profile_build)} profiling={int(bool(profile_options))}", file=sys.stderr, flush=True)
            if args.command == "serve":
                server = make_server(engine, chat, args.host, args.port, model, args.max_pending_requests, args.api_key,
                                     audio_prompt=args.system)
                print(f"TFLLM listening http://{args.host}:{server.server_port}/v1", file=sys.stderr, flush=True)
                try:
                    server.serve_forever(poll_interval=0.2)
                except KeyboardInterrupt:
                    pass
                finally:
                    server.server_close()
            else:
                if "audio" in engine.info.get("modalities", []):
                    raise ValueError("Use tfllm transcribe MODEL --audio FILE or serve /v1/audio/transcriptions for ASR")
                messages = [dict(role="system", content=args.system)] if args.system else []
                print("TFLLM chat: /clear resets history, /exit quits.", file=sys.stderr)
                attachments = list(args.image)
                videos = list(args.video)
                while True:
                    try:
                        text = input("You> ")
                    except (EOFError, KeyboardInterrupt):
                        break
                    if text == "/exit":
                        break
                    if text == "/clear":
                        messages = [dict(role="system", content=args.system)] if args.system else []
                        continue
                    if not text.strip():
                        continue
                    content = text
                    if attachments or videos:
                        from .images import local_image
                        from .videos import local_video
                        content = [dict(type="image_url", image_url=dict(url=local_image(p))) for p in attachments]
                        content.extend(dict(type="video_url",video_url=dict(url=local_video(p))) for p in videos)
                        content.append(dict(type="text", text=text))
                        attachments = []; videos = []
                    messages.append(dict(role="user", content=content))
                    try:
                        print("Assistant> ", end="", flush=True)
                        result = engine.generate(chat.request(dict(messages=messages)), lambda t: print(t, end="", flush=True))
                        print()
                        if "error" in result:
                            raise RuntimeError(result["error"])
                        messages.append(dict(role="assistant", content=result["text"]))
                    except (Exception, KeyboardInterrupt) as e:
                        messages.pop()
                        print("\nTFLLM: " + str(e), file=sys.stderr)
        finally:
            engine.close()
        return 0
    except Exception as e:
        print("TFLLM error: " + str(e), file=sys.stderr)
        return 1
