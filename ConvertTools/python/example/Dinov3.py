"""DINOv3 TFDL2 converter entrypoint.

This is the Op-builder/int8-range flow for DINOv3 ViT models. It uses the
generic ViT builder plus DINOv3 defaults: 2D RoPE and no absolute position
embedding. ``ApplyRope`` must be available when dumping/running the TFDL graph.
"""

from Vit import main


if __name__ == "__main__":
    main(default_arch="dinov3")
