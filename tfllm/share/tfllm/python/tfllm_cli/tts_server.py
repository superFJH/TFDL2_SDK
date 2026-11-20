"""Single-engine Breeze WAV / chunked PCM HTTP service."""
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import tempfile
import threading
import wave


def make_server(engine, host, port, model, api_key=None, max_pending=64, defaults=None):
    slots = threading.BoundedSemaphore(max_pending)
    defaults = defaults or {}

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def authorized(self):
            return not api_key or hmac.compare_digest(self.headers.get("Authorization", ""), "Bearer "+api_key)

        def reply(self, status, payload):
            body = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if not self.authorized():
                self.reply(401, {"error": "invalid API key"})
            elif self.path == "/health":
                self.reply(200, {"status": "ok", "model": model, **engine.info})
            elif self.path == "/v1/models":
                self.reply(200, {"object": "list", "data": [{"id": model, "object": "model", "owned_by": "tfllm"}]})
            else:
                self.reply(404, {"error": "unknown route"})

        def do_POST(self):
            started = False
            acquired = False
            try:
                if not self.authorized():
                    self.close_connection = True
                    return self.reply(401, {"error": "invalid API key"})
                if self.path != "/v1/audio/speech":
                    self.close_connection = True
                    return self.reply(404, {"error": "unknown route"})
                size = int(self.headers.get("Content-Length", "0"))
                if not 1 <= size <= 1024*1024 or self.headers.get("Transfer-Encoding"):
                    self.close_connection = True
                    return self.reply(413, {"error": "JSON body must have Content-Length in 1..1048576"})
                data = json.loads(self.rfile.read(size))
                if not isinstance(data, dict):
                    raise ValueError("expected a JSON object")
                if data.get("model", model) != model:
                    raise ValueError("unknown model")
                if any(key in data for key in ("ref_audio", "ref_text")):
                    raise ValueError("reference WAV paths are available only through the local speak command")
                if data.get("voice", "breeze") not in ("breeze", "default"):
                    raise ValueError("voice must be breeze/default; use instructions for voice design")
                text = data.get("input", data.get("text"))
                if not isinstance(text, str) or not text or len(text.encode()) > 65536:
                    raise ValueError("input must be nonempty text up to 65536 UTF-8 bytes")
                instruction = data.get("instructions", data.get("instruction", defaults.get("instruction", "Speak clearly and naturally.")))
                if not isinstance(instruction, str) or len(instruction.encode()) > 8192:
                    raise ValueError("instructions must be text up to 8192 UTF-8 bytes")
                request = dict(defaults, text=text, instruction=instruction)
                for key in ("cfg_scale", "temperature", "top_k", "top_p", "seed", "max_tokens", "chunk_first", "chunk_max"):
                    if key in data:
                        value = data[key]
                        if type(value) not in (int, float) or not math.isfinite(value):
                            raise ValueError("invalid numeric sampling option: "+key)
                        if key in {"seed", "top_k", "max_tokens", "chunk_first", "chunk_max"} and type(value) is not int:
                            raise ValueError(key+" must be an integer")
                        request[key] = value
                fmt = data.get("response_format", "wav")
                if fmt not in ("wav", "pcm"):
                    raise ValueError("response_format must be wav or pcm (mono 24 kHz PCM16LE)")
                if not slots.acquire(blocking=False):
                    return self.reply(429, {"error": "synthesis queue is full"})
                acquired = True
                engine.validate(request)
                if fmt == "pcm":
                    def emit(chunk):
                        nonlocal started
                        if not started:
                            self.send_response(200)
                            self.send_header("Content-Type", "application/octet-stream")
                            self.send_header("Transfer-Encoding", "chunked")
                            self.send_header("X-Audio-Sample-Rate", str(engine.info["sample_rate"]))
                            self.end_headers()
                            started = True
                        self.wfile.write(f"{len(chunk):x}\r\n".encode()+chunk+b"\r\n")
                        self.wfile.flush()
                    stats = engine.generate(request, emit)
                    if not started:
                        raise RuntimeError("generation returned no audio")
                    self.wfile.write(b"0\r\n\r\n")
                else:
                    with tempfile.SpooledTemporaryFile(max_size=8*1024*1024) as output:
                        with wave.open(output, "wb") as wav:
                            wav.setnchannels(1)
                            wav.setsampwidth(2)
                            wav.setframerate(engine.info["sample_rate"])
                            stats = engine.generate(request, wav.writeframesraw)
                        if not stats["samples"] or stats["cancelled"]:
                            raise RuntimeError("generation returned no audio")
                        size = output.tell()
                        output.seek(0)
                        self.send_response(200)
                        self.send_header("Content-Type", "audio/wav")
                        self.send_header("Content-Length", str(size))
                        self.end_headers()
                        started = True
                        while chunk := output.read(65536):
                            self.wfile.write(chunk)
            except (BrokenPipeError, ConnectionResetError):
                self.close_connection = True
            except Exception as exc:
                if started:
                    self.close_connection = True
                else:
                    self.reply(400 if isinstance(exc, (ValueError, KeyError, TypeError)) else 500, {"error": str(exc)})
            finally:
                if acquired:
                    slots.release()

    return ThreadingHTTPServer((host, port), Handler)
