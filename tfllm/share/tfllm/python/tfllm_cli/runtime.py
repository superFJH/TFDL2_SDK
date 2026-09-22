import codecs
import ctypes
import json
from pathlib import Path


class Engine:
    def __init__(self, library, options):
        self.lib = ctypes.CDLL(str(library))
        for name in ("capabilities", "error"):
            getattr(self.lib, "tfllm_chat_" + name).restype = ctypes.c_char_p
        self.lib.tfllm_chat_create.argtypes = [ctypes.c_char_p]
        self.lib.tfllm_chat_create.restype = ctypes.c_void_p
        self.lib.tfllm_chat_info.argtypes = [ctypes.c_void_p]
        self.lib.tfllm_chat_info.restype = ctypes.c_char_p
        self.lib.tfllm_chat_destroy.argtypes = [ctypes.c_void_p]
        self.callback_type = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t)
        self.lib.tfllm_chat_generate.argtypes = [ctypes.c_void_p, ctypes.c_char_p, self.callback_type, ctypes.c_void_p]
        self.lib.tfllm_chat_generate.restype = ctypes.c_char_p
        self.pointer = self.lib.tfllm_chat_create(json.dumps(options).encode())
        if not self.pointer:
            raise RuntimeError(json.loads(self.lib.tfllm_chat_error())["error"])
        self.info = json.loads(self.lib.tfllm_chat_info(self.pointer))

    @staticmethod
    def check(library, backend, profile_build=None):
        lib = ctypes.CDLL(str(library))
        lib.tfllm_chat_capabilities.restype = ctypes.c_char_p
        caps = json.loads(lib.tfllm_chat_capabilities())
        if caps["abi"] != 1:
            raise RuntimeError("Unsupported TFLLM chat ABI")
        if profile_build is not None and caps.get("profile", False) != profile_build:
            raise RuntimeError("TFLLM launcher/library profile variant mismatch; reinstall the matching SDK")
        if backend == "npu" and not caps["npu"]:
            raise RuntimeError("This SDK has no NPU chat backend. Build TFLLM_WITH_NPU40T=ON; CPU testing requires explicit --backend cpu")

    def generate(self, request, on_text=lambda text: None, cancelled=lambda: False):
        decoder = codecs.getincrementaldecoder("utf-8")("replace")
        callback_errors = []

        def emit(_user, data, size):
            try:
                if cancelled():
                    return 0
                if size:
                    text = decoder.decode(ctypes.string_at(data, size))
                    if text:
                        on_text(text)
                return 1
            except BaseException as e:
                callback_errors.append(e)
                return 0
        result = json.loads(self.lib.tfllm_chat_generate(self.pointer, json.dumps(request).encode(), self.callback_type(emit), None))
        if callback_errors:
            raise callback_errors[0]
        tail = decoder.decode(b"", final=True)
        if tail and not cancelled():
            on_text(tail)
        return result

    def close(self):
        if self.pointer:
            self.lib.tfllm_chat_destroy(self.pointer)
            self.pointer = None


def library_path(bin_dir, profile_build=False):
    suffix = "_profile" if profile_build else ""
    for directory in (bin_dir.parent / "lib", bin_dir):
        for name in (f"libtfllm-chat{suffix}.so", f"libtfllm-chat{suffix}.dylib"):
            if (directory / name).is_file():
                return directory / name
    raise RuntimeError(f"Missing libtfllm-chat{suffix}; build/install TFLLM_WITH_LLAMA=ON")
