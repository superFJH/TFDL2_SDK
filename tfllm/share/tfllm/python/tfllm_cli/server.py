"""Text Chat Completions transport. NPU execution stays in the shared C++ engine."""
import hmac
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


def make_server(engine, chat, host, port, model, max_pending=64, api_key=None):
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
            if self.path != "/v1/chat/completions":
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
                    self.error(413, "Provide Content-Length with a JSON body of at most 32 MiB")
                    return
                raw = self.rfile.read(length)
                if len(raw)!=length:
                    raise ValueError("Incomplete request body")
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
                def disconnected():
                    readable, _, _ = select.select([self.connection], [], [], 0)
                    return bool(readable) and not self.connection.recv(1, socket.MSG_PEEK)
                result = engine.generate(request, output, disconnected)
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
