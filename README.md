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
| `mps` | Apple Silicon GPU | macOS, `uv pip install -e '.[mps]'`, ~48 GB disk for weights | ~2 min per image for all 4 outputs at 1536 px on an M3 Max (36 GB); see below |

The `local` backend cannot run on Apple Silicon: it loads the 20B transformer
4-bit through bitsandbytes, which is CUDA-only. The `mps` backend exists for that
case.

### The `mps` backend (disk streaming)

Marigold V2 is a single-step model: one transformer pass per modality. So instead
of holding the ~40 GB bf16 transformer in memory, `marigold_test/mps.py` builds it
with every block weight as an empty placeholder and installs forward hooks that
load each of the 60 blocks right before it runs and free it right after. A
background thread reads block i+1 into a pinned staging buffer while the GPU
computes block i. Reads bypass the page cache (a 40 GB scan cannot fit in it and
would only evict everything else), and the LoRA checkpoint for each modality is
read from disk when needed rather than held in RAM. Peak process memory is under
10 GB, and swap stays flat.

```bash
uv pip install -e '.[mps]'
hf download Qwen/Qwen-Image-Edit-2509 --include "transformer/*" --include "vae/*" --include "model_index.json"
hf download huawei-bayerlab/marigold-v2-0 --include "depth/Log-stage2/*" --include "depth/Log-layered/*" \
    --include "normals/*" --include "albedo/*" --include "qwen_text_embeddings/*512_*"
marigold-test --input inputs --output outputs/mps --backend mps            # native res, long side capped at 1536
marigold-test --input inputs --output outputs/mps768 --backend mps --max-side 768   # faster smoke test
```

Measured on an M3 Max with 36 GB at 1536x1024: about 28 s per modality, of which
roughly 16 s is GPU compute and the rest is the 40 GB read (the SSD does it in
~9 s, overlapped with compute). At `--max-side 768` a modality takes about 9 s.
Numerics are the bf16 base weights rather than the demo's 4-bit base, so outputs
are close to the demo but not identical.

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
