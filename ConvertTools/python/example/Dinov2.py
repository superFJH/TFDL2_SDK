"""DINOv2 TFDL2 converter entrypoint.

This wraps Vit.py with ``--arch dinov2``. The shared implementation converts all
parameterized Transformer projections to 1x1 Conv, collects GPU min/max ranges
with an equivalent PyTorch graph, and maps those ranges onto the Op-built graph.
"""

from Vit import main


if __name__ == "__main__":
    main(default_arch="dinov2")
