"""Local backend: runs the vendored inference module from the authors' demo Space on a
CUDA GPU. Requires `pip install -e '.[local]'` and roughly 17 GB of VRAM at 1024^2.

The first run downloads Qwen-Image-Edit-2509 (VAE + 20B transformer, ~40 GB) and the
four Marigold V2 LoRA checkpoints (~1.8 GB each), then quantizes the transformer to
4-bit, which takes a few minutes. bitsandbytes is CUDA-only, so this does not run on
Apple Silicon; use the remote backend there.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image


class LocalMarigoldV2:
    def __init__(self, device: str | None = None):
        import torch

        from . import marigoldv2_inference as m

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cpu":
            raise RuntimeError(
                "local backend needs a CUDA GPU (4-bit bitsandbytes); use --backend remote"
            )
        self.model = m.MarigoldV2(m.BASE_MODEL_URI, m.MODEL_URI, torch.device(device))

    def __call__(self, image_path: Path) -> dict[str, Image.Image]:
        image = Image.open(image_path).convert("RGB")
        return self.model(image)
