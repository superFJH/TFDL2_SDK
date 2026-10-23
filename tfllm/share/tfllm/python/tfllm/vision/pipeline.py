"""Stateful segmentation and depth inference with observable TFLLM routing."""
import contextlib
import inspect
import threading

import torch
from torch import nn
from torch.nn import functional as F

from ..torch import NpuRuntime, optimize, restore, optimization_report
from ..torch.integrations import _SDPAMode
from ..torch.routing import AttentionRoute
from . import attention
from .loaders import FAMILIES, family_name, build, dependency


def supported_models():
    return FAMILIES


class VisionSession:
    """Calls and generator steps are scoped to one model and one runtime.

    Use the session's methods instead of calling the raw processor/model if
    functional SDPA should be routed. Original result/state objects are retained.
    """
    def __init__(self, model, family, runtime, processor, provider, task, owns_runtime=False):
        self.model, self.family, self.runtime = model, family, runtime
        self.processor = processor if processor is not None else model
        self.provider, self.task, self._owns_runtime = provider, task, owns_runtime
        self._lock, self._closed = threading.RLock(), False
        self._image, self._image_inputs, self._image_embeddings = None, None, None

    @property
    def report(self):
        return optimization_report(self.model)

    def _invoke(self, function, *args, **kwargs):
        with self._lock:
            if self._closed:
                raise RuntimeError("vision session is closed")
            state = self.model._tfllm_optimization
            scope=contextlib.nullcontext() if getattr(state,'vision_module_attention',False) and not getattr(self.runtime,'attention_fast',False) else _SDPAMode(AttentionRoute(state,self.family+'.sdpa'))
            active=getattr(state,'vision_attention_context',None)
            token=active.set(True) if active is not None else None
            try:
                with torch.inference_mode(),scope:
                    result = function(*args, **kwargs)
            finally:
                if active is not None:active.reset(token)
        if inspect.isgenerator(result):
            # Do not keep inference_mode/TorchFunctionMode active across yield:
            # unrelated caller operations must never inherit our NPU routing.
            def steps():
                try:
                    while True:
                        try:
                            value = self._invoke(next, result)
                        except StopIteration:
                            return
                        yield value
                finally:
                    result.close()
            return steps()
        return result

    def __call__(self, *args, **kwargs):
        return self._invoke(self.model, *args, **kwargs)

    def invoke(self, method, *args, **kwargs):
        if method.startswith("_"):
            raise ValueError("invoke expects a public predictor method")
        return self._invoke(getattr(self.processor, method), *args, **kwargs)

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        target = getattr(self.processor, name)
        if callable(target):
            return lambda *args, **kwargs: self._invoke(target, *args, **kwargs)
        return target

    def set_image(self, image, **kwargs):
        if self.provider != "transformers":
            if self.family == "sam3":
                import numpy as np
                from PIL import Image
                # The official SAM3 processor treats ndarray's last axes as
                # H/W. RGB HWC arrays need PIL to preserve original geometry.
                if isinstance(image, np.ndarray) and image.ndim == 3 and image.shape[-1] == 3:
                    if image.dtype != np.uint8:
                        raise ValueError("SAM3 HWC arrays must be RGB uint8; use PIL or an official CHW tensor otherwise")
                    image = Image.fromarray(image)
            return self.invoke("set_image", image, **kwargs)
        if self.family == "sam3":
            raise ValueError("HF SAM3 uses segment(image, text_prompt=...); use the official provider for stateful processor methods")
        with self._lock:
            inputs = self.processor(images=image, return_tensors="pt", **kwargs)
            embeddings = self._invoke(self.model.get_image_embeddings, inputs["pixel_values"].cpu())
            self._image, self._image_inputs, self._image_embeddings = image, inputs, embeddings

    def predict(self, **kwargs):
        if self.provider != "transformers":
            return self.invoke("predict", **kwargs)
        with self._lock:
            if self._image_embeddings is None:
                raise RuntimeError("call set_image before predict")
            multimask = kwargs.pop("multimask_output", True)
            inputs = self.processor(images=self._image, return_tensors="pt", **kwargs)
            model_inputs = {k: v.cpu() for k, v in inputs.items()
                            if k not in {"pixel_values", "original_sizes", "reshaped_input_sizes"}}
            output = self._invoke(self.model, image_embeddings=self._image_embeddings,
                                  multimask_output=multimask, **model_inputs)
            sizes = {"original_sizes": inputs["original_sizes"]}
            if self.family == "sam":
                sizes["reshaped_input_sizes"] = inputs["reshaped_input_sizes"]
            masks = self.processor.post_process_masks(output.pred_masks, **sizes)
            return masks, output.iou_scores, output.pred_masks

    def segment(self, image, *, point_coords=None, point_labels=None, box=None, text_prompt=None,
                multimask_output=True):
        with self._lock:
            return self._segment(image, point_coords=point_coords, point_labels=point_labels,
                                 box=box, text_prompt=text_prompt, multimask_output=multimask_output)

    def _segment(self, image, *, point_coords, point_labels, box, text_prompt, multimask_output):
        if self.family.startswith("depth-") or self.task == "video":
            raise ValueError("segment is an image SAM entry; video sessions expose their official state methods")
        if self.family == "sam3":
            if text_prompt is None:
                raise ValueError("segment for SAM3 requires text_prompt; geometry prompts use official processor methods")
            if any(x is not None for x in (point_coords, point_labels, box)):
                raise ValueError("use SAM3 processor's geometric prompt API explicitly")
            if self.provider == "transformers":
                inputs = self.processor(images=image, text=text_prompt, return_tensors="pt")
                output = self._invoke(self.model, **{k: v.cpu() for k, v in inputs.items() if isinstance(v, torch.Tensor)})
                return self.processor.post_process_instance_segmentation(output, target_sizes=inputs["original_sizes"])[0]
            state = self.set_image(image)
            return self.invoke("set_text_prompt", state=state, prompt=text_prompt)
        if point_coords is None and box is None:
            raise ValueError("provide point_coords or box")
        if point_coords is not None and point_labels is None:
            raise ValueError("point_coords require point_labels")
        self.set_image(image)
        if self.provider == "transformers":
            import numpy as np
            prompts = {}
            if point_coords is not None:
                prompts["input_points"] = [[np.asarray(point_coords).tolist()]]
                prompts["input_labels"] = [[np.asarray(point_labels).tolist()]]
            if box is not None:
                prompts["input_boxes"] = [[np.asarray(box).tolist()]]
            # multimask_output is a model argument, not image preprocessing.
            masks, scores, low = self.predict(multimask_output=multimask_output, **prompts)
        else:
            masks, scores, low = self.predict(point_coords=point_coords, point_labels=point_labels,
                                             box=box, multimask_output=multimask_output)
        return {"masks": masks, "scores": scores, "low_res_masks": low}

    def infer_image(self, raw_image, input_size=518):
        if self.family not in {"depth-anything", "depth-anything-v2"}:
            raise ValueError("infer_image accepts BGR images for V1/V2; V3 uses inference(images)")
        if input_size <= 0:
            raise ValueError("input_size must be positive")
        if self.provider == "transformers":
            inputs = self.processor(images=raw_image[..., ::-1].copy(), return_tensors="pt")
            output = self._invoke(self.model, **{k: v.cpu() for k, v in inputs.items()})
            depth = self.processor.post_process_depth_estimation(output, target_sizes=[raw_image.shape[:2]])[0]["predicted_depth"]
            return depth.float().cpu().numpy()
        # Preserve the official OpenCV resize/normalization/align_corners path
        # without image2tensor's implicit CUDA/MPS device selection.
        cv2 = dependency("cv2")
        prefix = "depth_anything" if self.family == "depth-anything" else "depth_anything_v2"
        transforms = dependency(prefix + ".util.transform")
        transform = [transforms.Resize(width=input_size, height=input_size, resize_target=False,
                                       keep_aspect_ratio=True, ensure_multiple_of=14, resize_method="lower_bound",
                                       image_interpolation_method=cv2.INTER_CUBIC),
                     transforms.NormalizeImage(mean=[.485, .456, .406], std=[.229, .224, .225]),
                     transforms.PrepareForNet()]
        sample = {"image": cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB) / 255.0}
        for operation in transform:
            sample = operation(sample)
        pixels = torch.from_numpy(sample["image"]).unsqueeze(0).cpu()
        depth = self._invoke(self.model, pixels)
        return F.interpolate(depth[:, None].float(), raw_image.shape[:2], mode="bilinear",
                             align_corners=True)[0, 0].cpu().numpy()

    def close(self):
        with self._lock:
            if self._closed:
                return
            restore(self.model)
            if self._owns_runtime:
                self.runtime.close()
            self._image = self._image_inputs = self._image_embeddings = None
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def attach(model, *, family, runtime, processor=None, provider="official", task="image", strict=False,
           activation_policy="auto", predicate=None):
    """Attach an existing CPU/eval model; keep FP32 weights and official state.

    strict applies to recognized projection/attention routes. Other model
    operations remain in Torch and are not certified by strict mode.
    """
    family = family_name(family)
    if not isinstance(model, nn.Module) or model.training:
        raise ValueError("attach requires an eval torch.nn.Module")
    if any(t.device.type != "cpu" for t in list(model.parameters()) + list(model.buffers())):
        raise ValueError("move the model to CPU before attaching TFLLM")
    if any(hasattr(m, "_orig_mod") or hasattr(m.forward, "_torchdynamo_orig_callable") for m in model.modules()):
        raise ValueError("disable torch.compile before attach; vision loaders disable compile_image_encoder automatically")
    if provider not in {"official", "transformers"} or task not in {"image", "video", "depth"}:
        raise ValueError("invalid provider or task")
    def projection_filter(path, module):
        if predicate is not None and not predicate(path,module):return False
        # DA3 explicitly computes camera geometry in FP32. Its small pose/FOV
        # heads must not be quantized along with the large visual backbone.
        return family != "depth-anything-v3" or not {"cam_dec", "cam_enc", "gs_head", "gs_adapter"}.intersection(path.split("."))
    optimize(model, runtime=runtime, strict=strict,
             operators=("linear", "conv2d", "conv_transpose2d", "gelu", "silu", "sigmoid", "tanh",
                        "mish", "softplus", "elu", "selu"),
             precision="fp16", predicate=projection_filter, activation_policy=activation_policy)
    try:
        state = model._tfllm_optimization
        attention_ids = attention.install(model, state)
        state.report.scope = "SAM/Depth Anything CPU model; external FP16/FP32 projections/activations and scoped SDPA; functional/custom code is not fully audited"
        state.report.add(family + ".sdpa", "attention", "routed", "scoped FAST SDPA with mask/bias" if getattr(runtime,'attention_fast',False) else "scoped SDPA; finite positional bias remains original")
        for path, module in model.named_modules():
            if isinstance(module, (nn.LayerNorm, nn.Embedding, nn.BatchNorm2d)):
                state.report.add(path, type(module).__name__, "retained", "official Torch operation")
            if "attention" in type(module).__name__.lower() and id(module) not in attention_ids:
                state.report.add(path, "attention", "retained", "original attention module; its SDPA calls may route at runtime")
        return VisionSession(model, family, runtime, processor, provider, task)
    except Exception:
        restore(model)
        raise


def load(family, checkpoint, *, runtime=None, backend="npu", chip=0, strict=False, activation_policy="auto", predicate=None, **options):
    """Load official/HF local weights, then attach TFLLM. Downloads are opt-in."""
    model, processor, provider = build(family, checkpoint, **options)
    owns_runtime = runtime is None
    if owns_runtime:
        runtime = NpuRuntime(backend=backend, chip=chip)
    try:
        session = attach(model, family=family, runtime=runtime, processor=processor, provider=provider,
                         task=options.get("task", "image"), strict=strict, activation_policy=activation_policy, predicate=predicate)
        session._owns_runtime = owns_runtime
        return session
    except Exception:
        if owns_runtime:
            runtime.close()
        raise
