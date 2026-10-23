"""MiniCPM-V 4.7 export against TFLLM's pinned llama.cpp converter.

The 4.7 vision weights retain the 4.6 GGUF tensor names/preprocessor. TFLLM
executes those weights itself and constructs the new canvas MRoPE positions.
No remote modeling code or checkpoint downloads are involved.
"""
from .base import ModelBase, gguf
from .qwen import Qwen3_5MoeTextModel
from .minicpm import MiniCPMV4_6VisionModel


@ModelBase.register("MiniCPMV4_7ForConditionalGeneration")
class MiniCpm47Text(Qwen3_5MoeTextModel):
    model_arch = gguf.MODEL_ARCH.QWEN35MOE

    @classmethod
    def filter_tensors(cls, item):
        name, gen = item
        if name.startswith(("model.merger.", "model.vision_tower.", "mtp.", "model.mtp.")):
            return None
        return super().filter_tensors((name, gen))

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        self.gguf_writer.add_string("tfllm.source_architecture", "minicpmv4_7")
        self.gguf_writer.add_string("tfllm.mrope_mode", "canvas")


@ModelBase.register("MiniCPMV4_7ForConditionalGeneration")
class MiniCpm47Vision(MiniCPMV4_6VisionModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.global_config.get("merger_times", 1) != 1 or self.global_config.get("merge_kernel_size", [2, 2]) != [2, 2]:
            raise ValueError("MiniCPM-V 4.7 requires one final 2x2 merge")
        if self.global_config.get("mrope_mode", "canvas") != "canvas":
            raise ValueError("MiniCPM-V 4.7 requires canvas MRoPE")

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        self.gguf_writer.add_string("tfllm.source_architecture", "minicpmv4_7")
