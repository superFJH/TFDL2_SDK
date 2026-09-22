import datetime
import json
import math


class Chat:
    def __init__(self, info, manifest, template=None, defaults=None, template_kwargs=None):
        try:
            from jinja2 import nodes
            from jinja2.ext import Extension
            from jinja2.sandbox import ImmutableSandboxedEnvironment
        except ImportError as e:
            raise RuntimeError("Install chat dependencies: python3 -m pip install 'jinja2>=3.1,<4'") from e

        class Generation(Extension):
            tags = {"generation"}
            def parse(self, parser):
                token = next(parser.stream)
                body = parser.parse_statements(["name:endgeneration"], drop_needle=True)
                return nodes.CallBlock(self.call_method("render"), [], [], body).set_lineno(token.lineno)
            def render(self, caller):
                return caller()

        source = template or manifest.get("chat_template") or info.get("chat_template")
        if isinstance(source, list):
            source = {x["name"]: x["template"] for x in source}
        if isinstance(source, dict):
            source = source.get("default")
        if not isinstance(source, str) or not source:
            raise ValueError("Model has no default chat template; supply --chat-template FILE")
        env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=[Generation, "jinja2.ext.loopcontrols"])
        def fail(message):
            raise ValueError(message)
        env.globals.update(raise_exception=fail, strftime_now=lambda fmt: datetime.datetime.now(datetime.timezone.utc).strftime(fmt))
        env.filters["tojson"] = lambda value, **kwargs: json.dumps(value, ensure_ascii=kwargs.pop("ensure_ascii", False), **kwargs)
        self.template = env.from_string(source)
        self.info = info
        from .video_adapters import VideoPreprocessor
        self.videos = VideoPreprocessor()
        generation = manifest.get("generation_config") or {}
        default_temperature = 0 if info.get("generation_mode") == "slow" else 0.8
        self.defaults = dict(max_tokens=generation.get("max_new_tokens", 256),
                             temperature=generation.get("temperature", default_temperature) if generation.get("do_sample", True) else 0,
                             top_p=generation.get("top_p", 0.95), top_k=generation.get("top_k", 40),
                             repetition_penalty=generation.get("repetition_penalty", 1.0))
        self.defaults.update({k: v for k, v in (defaults or {}).items() if v is not None})
        self.kwargs = template_kwargs or {}
        self.eos = generation.get("eos_token_id", [])
        if isinstance(self.eos, int):
            self.eos = [self.eos]
        if self.eos is None:
            self.eos = []

    def request(self, body):
        supported = {"model", "messages", "stream", "stream_options", "max_tokens", "max_completion_tokens",
                     "temperature", "top_p", "top_k", "repetition_penalty", "frequency_penalty", "presence_penalty",
                     "seed", "stop", "n", "chat_template_kwargs", "user", "tools", "tool_choice", "parallel_tool_calls"}
        unknown = set(body) - supported
        if unknown:
            raise ValueError("Unsupported request fields: " + ", ".join(sorted(unknown)))
        # OpenAI-compatible clients can send these fields even for ordinary
        # chat. Accept inactive tool settings, but never pretend that an active
        # tool request has been fulfilled by an unstructured text reply.
        tools = body.get("tools")
        if tools is not None and (not isinstance(tools, list) or any(not isinstance(t, dict) for t in tools)):
            raise ValueError("tools must be a list of tool definitions or null")
        choice = body.get("tool_choice")
        if choice not in (None, "none", "auto") or (tools and choice != "none"):
            raise ValueError("Tool calling is not implemented; disable tools/MCP/web search in the client, or set tool_choice='none'")
        parallel = body.get("parallel_tool_calls")
        if parallel is not None and type(parallel) is not bool:
            raise ValueError("parallel_tool_calls must be boolean or null")
        if body.get("n", 1) != 1:
            raise ValueError("Only n=1 is supported")
        if not isinstance(body.get("stream", False), bool):
            raise ValueError("stream must be boolean")
        options = body.get("stream_options", {})
        if not isinstance(options, dict) or set(options) - {"include_usage"} or type(options.get("include_usage", False)) is not bool:
            raise ValueError("Unsupported stream_options")
        messages = body.get("messages")
        if not isinstance(messages, list) or not messages or len(messages)>4096:
            raise ValueError("messages must be a nonempty list")
        video_count = sum(p.get("type")=="video_url" for m in messages if isinstance(m,dict)
                          for p in (m.get("content") if isinstance(m.get("content"),list) else []) if isinstance(p,dict))
        if video_count and ("video" not in self.info.get("modalities",[]) or "video" not in self.info):
            raise ValueError("This model has no native video adapter; currently Qwen3-VL with matching mmproj is supported")
        if video_count>4:
            raise ValueError("Videos are limited to 4 user content parts per request")
        normalized = []
        images = []
        media = []
        for m in messages:
            if not isinstance(m, dict) or m.get("role") not in {"system", "user", "assistant"} or set(m)-{"role", "content"}:
                raise ValueError("Only system/user/assistant text messages are supported")
            content = m.get("content")
            if isinstance(content, list):
                parts = []
                for part in content:
                    if not isinstance(part,dict):
                        raise ValueError("Invalid message content part")
                    if part.get("type")=="text" and isinstance(part.get("text"),str) and not set(part)-{"type","text"}:
                        parts.append(part)
                    elif part.get("type")=="image_url":
                        if "image" not in self.info.get("modalities",[]):
                            raise ValueError("Only text input is enabled; load a matching mmproj")
                        if m["role"]!="user" or len(images)>=16:
                            raise ValueError("Images are limited to 16 user content parts")
                        from .images import prepare_image
                        config = dict(self.info["vision"])
                        if video_count:
                            config["input_encoding"]="fp16_chw"
                        images.append(prepare_image(part,config))
                        if video_count:
                            media.append(dict(images[-1],kind="image",frames=1))
                            parts.append(dict(type="text",text="<|vision_start|><|image_pad|><|vision_end|>"))
                        elif self.info["vision"].get("projector")=="locateanything":
                            # The official processor numbers images globally,
                            # then inserts one span per image (no MRoPE).
                            parts.append(dict(type="text",text=f"<image {len(images)}><img><IMG_CONTEXT></img>"))
                        elif self.info["vision"].get("input_encoding")=="rgb8":
                            parts.append(dict(type="text",text="<__media__>"))
                        else:
                            parts.append(dict(type="image"))
                    elif part.get("type")=="video_url":
                        if m["role"]!="user":
                            raise ValueError("Videos must be user content parts")
                        from .video_adapters import qwen_video_text
                        config = dict(self.info["video"])
                        # Share the request budget equally across video parts,
                        # including history; each adapter may use fewer tokens.
                        config["max_video_tokens"] //= video_count
                        spans, _ = self.videos.prepare(part,config)
                        media.extend(dict(s) for s in spans)
                        parts.append(dict(type="text",text=qwen_video_text(spans)))
                    else:
                        raise ValueError("Only text, image_url and video_url content parts are supported")
                content = parts if any(p["type"]=="image" for p in parts) else "".join(p["text"] for p in parts)
            if not isinstance(content,(str,list)):
                raise ValueError("Message content must be text or content parts")
            normalized.append(dict(role=m["role"], content=content))
        kwargs = body.get("chat_template_kwargs", {})
        if not isinstance(kwargs, dict) or set(kwargs) & {"messages", "bos_token", "eos_token", "add_generation_prompt", "tools", "tool_choice", "parallel_tool_calls"}:
            raise ValueError("Invalid chat_template_kwargs")
        values = dict(self.kwargs, **kwargs)
        values.update(messages=normalized, bos_token=self.info["bos_token"], eos_token=self.info["eos_token"], add_generation_prompt=True,
                      tools=None, tool_choice="none", parallel_tool_calls=False)
        result = dict(self.defaults)
        for key in ("max_tokens", "temperature", "top_p", "top_k", "repetition_penalty", "frequency_penalty", "presence_penalty", "seed"):
            if key in body:
                result[key] = body[key]
        if "max_completion_tokens" in body:
            if "max_tokens" in body:
                raise ValueError("Specify only one of max_tokens and max_completion_tokens")
            result["max_tokens"] = body["max_completion_tokens"]
        for key in ("max_tokens", "top_k", "seed"):
            if key in result and (type(result[key]) is not int):
                raise ValueError(key + " must be an integer")
        for key in ("temperature", "top_p", "repetition_penalty", "frequency_penalty", "presence_penalty"):
            if key in result and (type(result[key]) not in (int, float) or not math.isfinite(result[key])):
                raise ValueError(key + " must be a number")
        stop = body.get("stop")
        if stop is None:
            stop = []
        if isinstance(stop, str):
            stop = [stop]
        if not isinstance(stop, list) or len(stop)>16 or any(not isinstance(s, str) or not s or len(s.encode())>4096 for s in stop):
            raise ValueError("stop must contain at most 16 nonempty strings")
        result.update(prompt=self.template.render(**values), add_special=False, stop=stop, eos_token_ids=self.eos)
        if media:
            result["media"] = media
        elif images:
            result["images"] = images
        return result
