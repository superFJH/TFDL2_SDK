"""Lazy, offline-first official model factories."""
import importlib
from pathlib import Path

import torch

FAMILIES = ("sam", "sam2", "sam3", "efficienttam", "depth-anything", "depth-anything-v2", "depth-anything-v3")
ALIASES = {"sam1": "sam", "sam2.1": "sam2", "efficient-tam": "efficienttam",
           "depthanything": "depth-anything", "depthanything-v1": "depth-anything",
           "depth-anything-v1": "depth-anything", "depthanything-v2": "depth-anything-v2",
           "depthanything-v3": "depth-anything-v3", "da1": "depth-anything",
           "da2": "depth-anything-v2", "da3": "depth-anything-v3"}


def family_name(value):
    name = str(value).lower().replace("_", "-")
    name = ALIASES.get(name, name)
    if name not in FAMILIES:
        raise ValueError(f"unknown vision family {value!r}; choose {', '.join(FAMILIES)}")
    return name


def dependency(name):
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as error:
        raise ImportError(f"Install the official model's dependencies before loading {name}: missing {error.name}") from error


def build(family, checkpoint, *, provider="auto", task="image", config=None, encoder=None,
          local_files_only=True, model_options=None, processor_options=None):
    family = family_name(family)
    if task not in {"image", "video", "depth"}:
        raise ValueError("task must be image, video or depth")
    if family.startswith("depth-") and task == "video":
        raise ValueError("use Depth Anything V3 inference(images) for multi-view inputs; this is not a video tracker")
    if not family.startswith("depth-") and task == "depth":
        raise ValueError("SAM families do not implement depth estimation")
    checkpoint = str(checkpoint)
    path = Path(checkpoint)
    if local_files_only and not path.exists():
        raise FileNotFoundError(f"local checkpoint does not exist: {checkpoint}")
    if provider == "auto":
        provider = "official" if family in {"efficienttam", "depth-anything-v3"} or path.is_file() else "transformers"
    if provider not in {"official", "transformers"}:
        raise ValueError("provider must be auto, official or transformers")
    options, processor_opts = dict(model_options or {}), dict(processor_options or {})
    if any(k in options for k in ("device", "device_map", "compile", "vos_optimized", "torch_dtype", "dtype", "load_from_HF")):
        raise ValueError("vision loaders select CPU/eval, disable compilation and preserve source precision")
    if provider == "transformers":
        if task == "video" or family in {"efficienttam", "depth-anything-v3"}:
            raise ValueError(f"{family}/{task} requires the official provider")
        hf = dependency("transformers")
        cls = {"sam": "SamModel", "sam2": "Sam2Model", "sam3": "Sam3Model"}.get(family)
        if cls is None:
            # HF uses depth_anything for both V1 and V2; checkpoint config selects the backbone.
            model_cls, processor_cls = hf.AutoModelForDepthEstimation, hf.AutoImageProcessor
        else:
            model_cls, processor_cls = getattr(hf, cls), hf.AutoProcessor
        model = model_cls.from_pretrained(checkpoint, local_files_only=local_files_only, **options).cpu().eval()
        expected = "depth_anything" if family.startswith("depth-") else family
        if model.config.model_type != expected:
            raise ValueError(f"{family} requires {expected} architecture, found {model.config.model_type}")
        processor = processor_cls.from_pretrained(checkpoint, local_files_only=local_files_only, **processor_opts)
        return model, processor, "transformers"
    if family == "sam":
        if task == "video":
            raise ValueError("SAM1 has no video memory; select SAM2 or EfficientTAM")
        sam = dependency("segment_anything")
        model = sam.sam_model_registry[encoder or "vit_b"](checkpoint=checkpoint, **options).cpu().eval()
        return model, sam.SamPredictor(model, **processor_opts), "official"
    if family in {"sam2", "efficienttam"}:
        if not config:
            raise ValueError(f"{family} requires its official model config and checkpoint")
        prefix = "sam2" if family == "sam2" else "efficient_track_anything"
        kind = "sam2" if family == "sam2" else "efficienttam"
        builders = dependency(f"{prefix}.build_{kind}")
        builder = getattr(builders, f"build_{kind}" + ("_video_predictor" if task == "video" else ""))
        overrides = list(options.pop("hydra_overrides_extra", []))
        overrides.append("++model.compile_image_encoder=False")
        model = builder(config, checkpoint, device="cpu", hydra_overrides_extra=overrides, **options).cpu().eval()
        if task == "video":
            return model, model, "official"
        predictors = dependency(f"{prefix}.{kind}_image_predictor")
        cls = getattr(predictors, "SAM2ImagePredictor" if family == "sam2" else "EfficientTAMImagePredictor")
        return model, cls(model, **processor_opts), "official"
    if family == "sam3":
        if task == "video":
            raise ValueError("official SAM3 multi-GPU video predictor is CUDA-specific; this CPU/NPU adapter supports image/text/box prompts")
        builders = dependency("sam3.model_builder")
        model = builders.build_sam3_image_model(checkpoint_path=checkpoint, device="cpu", load_from_HF=False,
                                               compile=False, **options).cpu().eval()
        cls = dependency("sam3.model.sam3_image_processor").Sam3Processor
        return model, cls(model, device="cpu", **processor_opts), "official"
    if family == "depth-anything-v3":
        cls = dependency("depth_anything_3.api").DepthAnything3
        model = cls.from_pretrained(checkpoint, local_files_only=local_files_only, **options).cpu().eval()
        return model, model, "official"
    if family == "depth-anything":
        raise ValueError("load V1 from a local Transformers directory or attach an already loaded official DepthAnything model")
    cls = dependency("depth_anything_v2.dpt").DepthAnythingV2
    shapes = {"vits": (64, [48, 96, 192, 384]), "vitb": (128, [96, 192, 384, 768]),
              "vitl": (256, [256, 512, 1024, 1024]), "vitg": (384, [1536, 1536, 1536, 1536])}
    encoder = encoder or "vits"
    if encoder not in shapes:
        raise ValueError("Depth Anything V2 encoder must be vits, vitb, vitl or vitg")
    features, channels = shapes[encoder]
    model = cls(encoder=encoder, features=features, out_channels=channels, **options)
    weights = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if "state_dict" in weights:
        weights = weights["state_dict"]
    model.load_state_dict(weights, strict=True)
    return model.cpu().eval(), model, "official"
