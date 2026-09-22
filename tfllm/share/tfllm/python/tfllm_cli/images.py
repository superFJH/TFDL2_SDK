"""Qwen3-VL still-image preprocessing. C++ owns all vision/decoder execution."""
import base64
import io
import math
from pathlib import Path


def local_image(path):
    data = Path(path).expanduser().read_bytes()
    if len(data)>12*1024*1024:
        raise ValueError("Image file exceeds 12 MiB")
    return "data:image/png;base64," + base64.b64encode(data).decode("ascii")


def prepare_image(part, config):
    try:
        import numpy as np
        from PIL import Image, ImageOps
    except ImportError as e:
        raise RuntimeError("Image input requires Pillow and numpy; install TFLLM's Python requirements") from e
    if set(part)-{"type", "image_url"} or not isinstance(part.get("image_url"), dict) or set(part["image_url"])-{"url", "detail"}:
        raise ValueError("Invalid image_url content part")
    url = part["image_url"].get("url", "")
    if not isinstance(url,str) or not url.startswith("data:image/") or ";base64," not in url:
        raise ValueError("image_url currently requires a base64 data:image URL; use --image PATH for local chat")
    if part["image_url"].get("detail", "auto") != "auto":
        raise ValueError("Only image detail=auto is supported; configure --max-image-tokens")
    head, encoded = url.split(",",1)
    if len(encoded)>16*1024*1024 or not head.endswith(";base64"):
        raise ValueError("Image payload exceeds 12 MiB or has invalid encoding")
    try:
        data = base64.b64decode(encoded, validate=True)
        image = Image.open(io.BytesIO(data))
        if image.width*image.height>32*1024*1024 or image.width<1 or image.height<1 or getattr(image,"n_frames",1)!=1:
            raise ValueError("Image must be a single frame of at most 32 megapixels")
        image = ImageOps.exif_transpose(image).convert("RGB")
    except (OSError, Image.DecompressionBombError) as e:
        raise ValueError("Invalid or oversized image") from e
    w,h = image.size
    if max(w,h)/min(w,h)>200:
        raise ValueError("Image aspect ratio exceeds 200")
    if config.get("input_encoding")=="rgb8":
        return dict(height=h,width=w,encoding="rgb8",pixels=base64.b64encode(image.tobytes()).decode("ascii"))
    factor=config["patch"]*config["merge"]
    budget=config["max_image_tokens"]
    locate = config.get("projector") == "locateanything"
    if locate:
        # MoonViT-SO: first bound unmerged patches, then resize UP to full
        # merge cells. Preserve the official two bicubic resizes; the CLI's
        # merged-token cap additionally bounds ceil-rounding overshoot below.
        patches=(w//config["patch"])*(h//config["patch"])
        if patches>budget*config["merge"]**2:
            scale=math.sqrt(budget*config["merge"]**2/patches)
            w,h=max(1,int(w*scale)),max(1,int(h*scale))
            image=image.resize((w,h),Image.Resampling.BICUBIC)
    # Qwen smart-resize: round both axes to complete merge cells, then bound
    # total pixels while keeping aspect ratio. No letterboxing or patch loss.
    rounding=math.ceil if locate else round
    rw=max(factor,rounding(w/factor)*factor)
    rh=max(factor,rounding(h/factor)*factor)
    if (rw//factor)*(rh//factor)>budget:
        scale=math.sqrt(w*h/(budget*factor*factor))
        rw=max(factor,math.floor(w/scale/factor)*factor)
        rh=max(factor,math.floor(h/scale/factor)*factor)
        # Extreme aspect ratios plus a one-cell minimum can exceed the budget.
        if (rw//factor)*(rh//factor)>budget:
            if rw>rh:
                rw=factor*max(1,budget//(rh//factor))
            else:
                rh=factor*max(1,budget//(rw//factor))
    if locate and max(rw,rh)//config["patch"]>=512:
        raise ValueError("LocateAnything image exceeds its 512-patch position grid")
    image=image.resize((rw,rh), Image.Resampling.BICUBIC)
    pixels=np.asarray(image,dtype=np.float32)/255.0
    pixels=(pixels-np.asarray(config["mean"],dtype=np.float32))/np.asarray(config["std"],dtype=np.float32)
    payload=np.ascontiguousarray(pixels.transpose(2,0,1),dtype="<f2").tobytes()
    return dict(height=rh,width=rw,pixels=base64.b64encode(payload).decode("ascii"))
