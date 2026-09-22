import argparse
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


def main(profile_build=False):
    parser = argparse.ArgumentParser(prog="tfllm_profile" if profile_build else "tfllm", description="Native TFLLM NPU prefill with llama.cpp CPU decode")
    parser.add_argument("command", choices=["serve", "chat"])
    parser.add_argument("model", help="Transformers directory, GGUF file/directory, prepared cache directory, or repository ID")
    parser.add_argument("--model-source", choices=["huggingface", "modelscope"])
    parser.add_argument("--revision")
    parser.add_argument("--gguf-file", help="Select a GGUF download/model pattern, including all shards")
    parser.add_argument("--cache-dir", default=os.environ.get("TFLLM_CACHE_DIR", "~/.cache/tfllm"))
    parser.add_argument("--backend", choices=["npu", "cpu"], default="npu", help="CPU is explicit host testing, never an automatic fallback")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--mmproj", default="auto", help="Vision GGUF; auto discovers/converts it, none disables images/video")
    parser.add_argument("--max-image-tokens", type=int, default=1024, help="Maximum merged visual tokens per image (1..4096)")
    parser.add_argument("--image", type=Path, action="append", default=[], help="Attach a local image to the first chat turn (repeatable)")
    parser.add_argument("--max-video-tokens", type=int, default=2048, help="Total merged visual tokens across videos per request (1..4096)")
    parser.add_argument("--video-max-frames", type=int, default=32, help="Maximum sampled frames per video (2..64)")
    parser.add_argument("--video-fps", type=float, default=2, help="Target sampling rate; frames cover the whole clip when capped (0..30]")
    parser.add_argument("--video", type=Path, action="append", default=[], help="Attach a local video to the first chat turn (repeatable, Qwen3-VL)")
    parser.add_argument("--chip", type=int, default=0)
    parser.add_argument("--context", type=int, default=4096)
    parser.add_argument("--threads", type=int, default=4, help="total CPU decode threads shared by all sequences")
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
        parser.add_argument("--profile", type=Path, help="Capture operator spans; create a run subdirectory here with one CSV per request")
        parser.add_argument("--profile-max-events", type=int, default=100000, help="Bound each request's trace (default: 100000 events)")
    elif any(a.split("=", 1)[0] in ("--profile", "--profile-max-events") for a in sys.argv[1:]):
        parser.error("profiling is not compiled in; use tfllm_profile with --profile")
    args = parser.parse_args()
    def interrupt(_signum, _frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupt)
    try:
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
        print(f"TFLLM decode: weights={manifest['decode_quantization']} threads={args.threads} gguf={manifest['gguf']}",
              file=sys.stderr, flush=True)
        engine = Engine(library, dict(package=manifest["package"], gguf=manifest["gguf"], backend=args.backend,
                                     context=args.context, threads=args.threads, max_sequences=args.max_num_seqs,
                                     mmproj=manifest.get("mmproj") or "", max_image_tokens=args.max_image_tokens,
                                     max_video_tokens=args.max_video_tokens, video_max_frames=args.video_max_frames, video_fps=args.video_fps,
                                     gpu_memory_utilization=args.gpu_memory_utilization, chip=args.chip, **profile_options))
        try:
            template = args.chat_template.read_text() if args.chat_template else None
            defaults = {k: getattr(args, k) for k in ("max_tokens", "temperature", "top_p", "top_k", "seed")}
            chat = Chat(engine.info, manifest, template, defaults, kwargs)
            model = args.served_model_name or args.model.rstrip("/").split("/")[-1]
            print(f"TFLLM_READY model={model} architecture={engine.info['architecture']} modalities={','.join(engine.info['modalities'])} backend={engine.info['backend']} context={args.context} max_sequences={args.max_num_seqs} decode_scheduler={engine.info.get('decode_scheduler', 'unknown')} decode_threads={args.threads} profile_compiled={int(profile_build)} profiling={int(bool(profile_options))}", file=sys.stderr, flush=True)
            if args.command == "serve":
                server = make_server(engine, chat, args.host, args.port, model, args.max_pending_requests, args.api_key)
                print(f"TFLLM listening http://{args.host}:{server.server_port}/v1", file=sys.stderr, flush=True)
                try:
                    server.serve_forever(poll_interval=0.2)
                except KeyboardInterrupt:
                    pass
                finally:
                    server.server_close()
            else:
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
