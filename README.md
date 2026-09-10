# Marigold V2 test pipeline

Drop images in, get Marigold V2 outputs out: depth, see-through depth, surface
normals, and albedo. Meant for eyeballing the model, not for production.

Model: [huawei-bayerlab/marigold-v2-0](https://huggingface.co/huawei-bayerlab/marigold-v2-0)
(LoRA adapters on Qwen-Image-Edit-2509). Paper site:
[huawei-bayerlab/marigold-v2-web](https://huggingface.co/spaces/huawei-bayerlab/marigold-v2-web).
Live demo whose API the remote backend calls: [toshas/Marigold-V2](https://huggingface.co/spaces/toshas/Marigold-V2).

## Setup

```bash
uv venv -p 3.12 && source .venv/bin/activate
uv pip install -e .
```

## Run

```bash
marigold-test --input inputs --output outputs/run1          # folder
marigold-test --input some/photo.jpg --output outputs/one   # single image
marigold-test --input inputs --modality Depth --limit 2     # subset
open outputs/run1/index.html                                # side-by-side sheet
```

Each image gets its own folder under the output dir with `input.*`, `depth.png`,
`see_through_depth.png`, `normals.png`, `albedo.png`. `results.json` records what
succeeded and how long each call took. `index.html` shows everything in a grid.

## Backends

| Backend | Where it runs | Needs | Speed |
|---|---|---|---|
| `remote` (default) | Authors' demo Space on HF ZeroGPU | internet | ~20-30 s per image, all 4 outputs |
| `local` | Your CUDA GPU | Linux, ~17 GB VRAM, `uv pip install -e '.[local]'` | first load downloads ~45 GB and quantizes for a few minutes, then a few seconds per image |

The remote backend is the one that works on a Mac. Apple Silicon cannot run the
local backend: the 20B transformer is loaded 4-bit through bitsandbytes, which is
CUDA-only, and the unquantized bf16 weights alone are ~40 GB.

ZeroGPU quota is the real limit of the remote backend. Every call reserves up to
180 s of GPU time, and an anonymous IP gets only a few minutes per day, so expect
roughly 3 images before a quota error, then a 24 h wait. A free Hugging Face
account raises the daily quota, and PRO gives 40 min/day:

```bash
export HF_TOKEN=hf_...   # from https://huggingface.co/settings/tokens
```

Quota errors are reported immediately and the batch moves on to the next image;
rerun the same command later and only the failed images need redoing (delete the
finished folders or point `--output` at a new dir).

## Notes on the outputs

- Depth is affine-invariant (unknown scale and shift). The PNGs are the demo's
  Spectral colormap; "Depth" and "See-through Depth" share one color scale.
- Normals are camera-space unit vectors mapped to RGB. Albedo is linear RGB shown as sRGB.
- The remote backend only returns rendered PNGs. For raw float arrays, run the
  local backend or the authors' repo (`scripts/infer.py` writes `.npy`).
- `marigold_test/marigoldv2_inference.py` is vendored unchanged from the demo Space
  (Apache 2.0) so the local backend needs none of the training code.
