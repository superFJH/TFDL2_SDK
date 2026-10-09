"""Text Chat Completions transport. NPU execution stays in the shared C++ engine."""
import hmac
from email import policy
from email.parser import BytesParser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import select
import socket
import socketserver
import sys
import threading
import time
import uuid


def usage(result):
    p, c = result.get("prompt_tokens", 0), result.get("completion_tokens", 0)
    return dict(prompt_tokens=p, completion_tokens=c, total_tokens=p+c,
                prompt_tokens_details=dict(cached_tokens=result.get("cached_tokens", 0)))


def audio_form(content_type, raw):
    if not isinstance(content_type, str) or not content_type.lower().startswith("multipart/form-data;"):
        raise ValueError("Audio transcription requires multipart/form-data with a file field")
    message = BytesParser(policy=policy.default).parsebytes(
        b"Content-Type: " + content_type.encode("ascii") + b"\r\nMIME-Version: 1.0\r\n\r\n" + raw)
    if not message.is_multipart() or message.defects:
        raise ValueError("Invalid multipart audio body/boundary")
    fields, data = {}, None
    for i, part in enumerate(message.iter_parts()):
        if i >= 16 or part.is_multipart() or part.defects or part.get_content_disposition() != "form-data":
            raise ValueError("Invalid multipart audio field")
        name = part.get_param("name", header="content-disposition")
        if name in fields or (name == "file" and data is not None):
            raise ValueError("Duplicate audio field: " + str(name))
        payload = part.get_payload(decode=True)
        if name == "file":
            if not part.get_filename() or not payload:
                raise ValueError("file must contain an audio upload")
            data = payload
        elif name in {"model", "language", "prompt", "response_format", "temperature"}:
            if len(payload) > 16384:
                raise ValueError("Audio field exceeds 16384 bytes")
            fields[name] = payload.decode("utf-8")
        else:
            raise ValueError("Unsupported audio field: " + str(name))
    if data is None:
        raise ValueError("Missing audio file")
    if "temperature" in fields:
        fields["temperature"] = float(fields["temperature"])
    return fields, data


def make_server(engine, chat, host, port, model, max_pending=64, api_key=None, audio_prompt=""):
    admission = threading.BoundedSemaphore(max_pending)

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        def setup(self):
            super().setup()
            self.connection.settimeout(30)

        def json(self, status, body):
            data = json.dumps(body, ensure_ascii=False, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def error(self, status, message):
            self.json(status, {"error": {"message": message, "type": "invalid_request_error" if status<500 else "server_error"}})

        def authorized(self):
            if api_key and not hmac.compare_digest(self.headers.get("Authorization", ""), "Bearer " + api_key):
                self.close_connection = True
                self.error(401, "Invalid API key")
                return False
            return True

        def disconnected(self):
            readable, _, _ = select.select([self.connection], [], [], 0)
            return bool(readable) and not self.connection.recv(1, socket.MSG_PEEK)

        def transcribe(self, raw):
            from .asr import Transcriber, response_body
            fields, data = audio_form(self.headers.get("Content-Type"), raw)
            if fields.get("model", model) != model:
                self.error(404, "Unknown model")
                return
            format = fields.get("response_format", "json")
            if format not in {"json", "text", "verbose_json"}:
                raise ValueError("response_format must be json, text or verbose_json; timestamps are not supported")
            result = Transcriber(engine, chat).transcribe(data, fields.get("language"),
                        fields.get("prompt", audio_prompt), fields.get("temperature"), self.disconnected)
            if result.get("finish_reason") == "cancelled":
                self.close_connection = True
                return
            if "error" in result:
                self.error(result.get("status", 500), result["error"])
                return
            print("TFLLM_REQUEST " + json.dumps({k: v for k, v in result.items() if k != "text"}), file=sys.stderr, flush=True)
            output = response_body(result, format)
            if format != "text":
                self.json(200, output)
                return
            encoded = output.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def do_GET(self):
            if not self.authorized():
                return
            if self.path == "/health":
                self.json(200, dict(status="ready", backend=engine.info["backend"]))
            elif self.path == "/v1/models":
                self.json(200, dict(object="list", data=[dict(id=model, object="model", created=0, owned_by="tfllm")]))
            else:
                self.error(404, "Unknown endpoint")

        def do_POST(self):
            if not self.authorized():
                return
            started = False
            if self.path not in {"/v1/chat/completions", "/v1/audio/transcriptions"}:
                self.close_connection = True
                self.error(404, "Unknown endpoint")
                return
            if not admission.acquire(blocking=False):
                self.close_connection = True
                self.error(429, "Request queue is full")
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length<=0 or length>32*1024*1024 or self.headers.get("Transfer-Encoding"):
                    self.close_connection = True
                    self.error(413, "Provide Content-Length with a body of at most 32 MiB")
                    return
                raw = self.rfile.read(length)
                if len(raw)!=length:
                    raise ValueError("Incomplete request body")
                if self.path == "/v1/audio/transcriptions":
                    self.transcribe(raw)
                    return
                body = json.loads(raw)
                if not isinstance(body, dict):
                    raise ValueError("Request must be a JSON object")
                if body.get("model", model)!=model:
                    self.error(404, "Unknown model")
                    return
                request = chat.request(body)
                stream = body.get("stream", False)
                identifier = "chatcmpl-" + uuid.uuid4().hex
                created = int(time.time())
                def chunk(delta, finish=None):
                    return dict(id=identifier, object="chat.completion.chunk", created=created, model=model,
                                choices=[dict(index=0, delta=delta, finish_reason=finish)])
                def sse(value):
                    data = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, allow_nan=False)
                    self.wfile.write(("data: " + data + "\n\n").encode())
                    self.wfile.flush()
                def begin():
                    nonlocal started
                    if started:
                        return
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                    self.send_header("Cache-Control", "no-cache")
                    self.send_header("Connection", "close")
                    self.end_headers()
                    self.close_connection = True
                    started = True
                    sse(chunk(dict(role="assistant", content="")))
                def output(text):
                    if stream:
                        begin()
                        sse(chunk(dict(content=text)))
                result = engine.generate(request, output, self.disconnected)
                if result.get("finish_reason") == "cancelled":
                    self.close_connection = True
                    return
                if "error" in result:
                    if started:
                        sse({"error": {"message": result["error"]}})
                        sse("[DONE]")
                    else:
                        self.error(result.get("status", 500), result["error"])
                    return
                print("TFLLM_REQUEST " + json.dumps({k: v for k, v in result.items() if k!="text"}), file=sys.stderr, flush=True)
                if stream:
                    begin()
                    sse(chunk({}, result["finish_reason"]))
                    if body.get("stream_options", {}).get("include_usage"):
                        final = chunk({})
                        final.update(choices=[], usage=usage(result))
                        sse(final)
                    sse("[DONE]")
                else:
                    self.json(200, dict(id=identifier, object="chat.completion", created=created, model=model,
                                      choices=[dict(index=0, message=dict(role="assistant", content=result["text"]),
                                                    finish_reason=result["finish_reason"])], usage=usage(result)))
            except (BrokenPipeError, ConnectionResetError, TimeoutError):
                self.close_connection = True  # callback exception stops generation
            except (ValueError, TypeError, KeyError) as e:
                if not started:
                    self.error(400, str(e))
            except Exception as e:
                if not started:
                    self.error(500, str(e))
            finally:
                admission.release()

    # Non-daemon request threads: server_close waits before the engine is freed.
    class Server(ThreadingHTTPServer):
        def server_bind(self):
            # HTTPServer normally performs a reverse DNS lookup here. It is
            # unnecessary for serving and can stall offline NPU boards.
            socketserver.TCPServer.server_bind(self)
            self.server_name, self.server_port = self.server_address[:2]
    server = Server((host, port), Handler)
    server.daemon_threads = False
    return server
