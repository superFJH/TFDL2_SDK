"""PaddleOCR's original checkpoints and Transformers' native wrapper.

Only tensor names/configuration are adapted; checkpoint Python is never loaded.
"""
from .base import ModelBase
from .ernie import PaddleOCRModel, PaddleOCRVisionModel


@ModelBase.register("TFLLMPaddleText")
class TFLLMPaddleText(PaddleOCRModel):
    model_arch = PaddleOCRModel.model_arch

    @classmethod
    def filter_tensors(cls, item):
        name, tensor = item
        if name.startswith(("model.visual.", "model.projector.", "vision_model.", "mlp_AR.")):
            return None
        name = name.replace("model.language_model.", "model.", 1)
        return super().filter_tensors((name, tensor))


@ModelBase.register("TFLLMPaddleVision")
class TFLLMPaddleVision(PaddleOCRVisionModel):
    model_arch = PaddleOCRVisionModel.model_arch

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hparams_vision["rms_norm_eps"] = self.hparams_vision.get("layer_norm_eps", 1e-6)

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        # The native Transformers wrapper changed the interpolation convention;
        # the official 1.6 checkpoint still uses the original tensor layout.
        native = self.global_config.get("text_config", {}).get("model_type") == "paddleocr_vl_text"
        self.gguf_writer.add_bool("clip.vision.position_align_corners", native)
        self.gguf_writer.add_uint32("clip.vision.spatial_merge_size", 2)

    @classmethod
    def filter_tensors(cls, item):
        name, tensor = item
        name = name.replace("model.visual.vision_model.", "model.vision_model.", 1)
        name = name.replace("vision_model.vision_model.", "model.vision_model.", 1)
        name = name.replace("model.projector.", "mlp_AR.", 1)
        return super().filter_tensors((name, tensor))
