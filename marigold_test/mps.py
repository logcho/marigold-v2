"""Apple Silicon backend: streams the 20B Qwen-Image-Edit transformer from disk.

The transformer is built with every block weight as a zero-size placeholder on the
GPU. Forward hooks on each of the 60 transformer blocks copy that block's weights
in right before it runs and drop them right after. Disk reads go through two
preallocated pinned staging buffers: a background thread reads block i+1 into one
buffer while the GPU computes block i from the other. Reads bypass the page cache
(F_NOCACHE) because a 40 GB sequential scan cannot fit in it anyway and would only
evict everything else. About 1.3 GB of staging plus ~0.7 GB of live block weights
are resident; the LoRA adapter, VAE, and the small in/out projections stay on the
GPU. Marigold V2 is single-step, so each modality costs exactly one 40 GB read.

Preprocessing, LoRA loading, decoding, and visualization reuse the authors'
inference module unchanged. Numerics are the bf16 base weights rather than the
4-bit bitsandbytes base the demo runs on.
"""

from __future__ import annotations

import os

# Lift the Metal working-set cap (default ~75% of unified memory). Must precede torch import.
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

import fcntl  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from concurrent.futures import Future, ThreadPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import torch  # noqa: E402
from accelerate import init_empty_weights  # noqa: E402
from diffusers import AutoencoderKLQwenImage, QwenImageTransformer2DModel  # noqa: E402
from huggingface_hub import snapshot_download  # noqa: E402
from peft import LoraConfig  # noqa: E402
from PIL import Image  # noqa: E402
from safetensors.torch import load_file  # noqa: E402

from . import marigoldv2_inference as m  # noqa: E402

BLOCK_PREFIX = "transformer_blocks."
DTYPES = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}


class ShardReader:
    """Large sequential, page-cache-bypassing reads of tensors from the safetensors shards."""

    def __init__(self, transformer_dir: Path):
        index = json.loads((transformer_dir / "diffusion_pytorch_model.safetensors.index.json").read_text())
        self.files = {}
        self.meta: dict[str, tuple[str, torch.dtype, tuple[int, ...], int, int]] = {}
        for name in sorted(set(index["weight_map"].values())):
            f = open(transformer_dir / name, "rb", buffering=0)
            if hasattr(fcntl, "F_NOCACHE"):
                fcntl.fcntl(f.fileno(), fcntl.F_NOCACHE, 1)
            self.files[name] = f
            header_len = int.from_bytes(f.read(8), "little")
            header = json.loads(f.read(header_len))
            base = 8 + header_len
            for key, info in header.items():
                if key != "__metadata__":
                    start, end = info["data_offsets"]
                    self.meta[key] = (name, DTYPES[info["dtype"]], tuple(info["shape"]), base + start, base + end)

    def __contains__(self, key: str) -> bool:
        return key in self.meta

    def ranges(self, keys: list[str]) -> list[tuple[str, int, int, list[str]]]:
        """Group keys by shard into (shard, start, end, keys) byte ranges."""
        by_shard: dict[str, list[str]] = {}
        for key in keys:
            by_shard.setdefault(self.meta[key][0], []).append(key)
        return [
            (name, min(self.meta[k][3] for k in ks), max(self.meta[k][4] for k in ks), ks)
            for name, ks in by_shard.items()
        ]

    def span(self, keys: list[str]) -> int:
        return sum(end - start for _, start, end, _ in self.ranges(keys))

    def read_into(self, keys: list[str], staging: torch.Tensor) -> dict[str, torch.Tensor]:
        """Read all of `keys` into the uint8 `staging` tensor; return views into it."""
        out: dict[str, torch.Tensor] = {}
        offset = 0
        for name, start, end, ks in self.ranges(keys):
            view = memoryview(staging.numpy())[offset : offset + (end - start)]
            f = self.files[name]
            f.seek(start)
            got = 0
            while got < len(view):
                n = f.readinto(view[got:])
                if not n:
                    raise EOFError(f"{name}: short read at {start + got}")
                got += n
            for k in ks:
                _, dtype, shape, s, e = self.meta[k]
                out[k] = staging[offset + s - start : offset + e - start].view(dtype).view(shape)
            offset += end - start
        return out

    def read(self, key: str) -> torch.Tensor:
        staging = torch.empty(self.span([key]), dtype=torch.uint8)
        return self.read_into([key], staging)[key].clone()


def _replace_param(model: torch.nn.Module, name: str, tensor: torch.Tensor) -> None:
    # `.data` cannot be reassigned across tensor types (meta -> mps), so swap the Parameter object.
    parent_name, _, attr = name.rpartition(".")
    parent = model.get_submodule(parent_name) if parent_name else model
    setattr(parent, attr, torch.nn.Parameter(tensor, requires_grad=False))


class BlockStreamer:
    """Loads one transformer block's base weights per forward call, prefetching the next."""

    def __init__(self, model: QwenImageTransformer2DModel, reader: ShardReader, device: torch.device):
        self.device = device
        self.reader = reader
        self.blocks = list(model.transformer_blocks)
        self.block_params: list[list[tuple[torch.nn.Parameter, str]]] = []
        for i, block in enumerate(self.blocks):
            entries = []
            for name, param in block.named_parameters():
                if "lora_" in name:
                    continue
                key = f"{BLOCK_PREFIX}{i}." + name.replace(".base_layer", "")
                if key not in reader:
                    raise KeyError(f"no checkpoint weight for {key}")
                entries.append((param, key))
            self.block_params.append(entries)
        max_bytes = max(reader.span([k for _, k in entries]) for entries in self.block_params)
        self.staging = [torch.empty(max_bytes, dtype=torch.uint8, pin_memory=True) for _ in range(2)]
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.pending: dict[int, Future] = {}
        self.disk_wait = 0.0  # time the GPU sat idle waiting for the disk
        self.gpu_wait = 0.0  # time the disk sat idle waiting for the GPU
        for i, block in enumerate(self.blocks):
            block.register_forward_pre_hook(self._make_pre_hook(i))
            block.register_forward_hook(self._make_post_hook(i))

    def _read_block(self, i: int) -> dict[str, torch.Tensor]:
        return self.reader.read_into([key for _, key in self.block_params[i]], self.staging[i % 2])

    def _prefetch(self, i: int) -> None:
        if 0 <= i < len(self.blocks) and i not in self.pending:
            self.pending[i] = self.pool.submit(self._read_block, i)

    def _make_pre_hook(self, i: int):
        def hook(module, args):
            t0 = time.perf_counter()
            future = self.pending.pop(i, None)
            tensors = future.result() if future is not None else self._read_block(i)
            t1 = time.perf_counter()
            # Block i-1 must be done with its staging buffer before the prefetch below reuses it.
            torch.mps.synchronize()
            t2 = time.perf_counter()
            for param, key in self.block_params[i]:
                param.data = tensors[key].to(self.device, non_blocking=True)
            self._prefetch(i + 1)
            self.disk_wait += t1 - t0
            self.gpu_wait += t2 - t1

        return hook

    def _make_post_hook(self, i: int):
        def hook(module, args, output):
            for param, _ in self.block_params[i]:
                param.data = torch.empty(0, device=self.device, dtype=param.dtype)
            if i == len(self.blocks) - 1:
                self._prefetch(0)  # next pass starts at block 0

        return hook

    def reset_stats(self) -> None:
        self.disk_wait = self.gpu_wait = 0.0


def build_streaming_transformer(base_dir: Path, device: torch.device) -> tuple[QwenImageTransformer2DModel, BlockStreamer]:
    transformer_dir = base_dir / "transformer"
    config = json.loads((transformer_dir / "config.json").read_text())
    config = {k: v for k, v in config.items() if not k.startswith("_")}
    with init_empty_weights():
        model = QwenImageTransformer2DModel.from_config(config)
    model.add_adapter(
        LoraConfig(r=128, lora_alpha=128, target_modules=m.LORA_TARGET_MODULES, init_lora_weights=False)
    )
    reader = ShardReader(transformer_dir)

    # Materialize every parameter on the device once. LoRA params are placeholders filled by
    # load_checkpoint; resident modules (img_in, txt_in, time embedding, norm_out, proj_out) get
    # their real weights; block base weights become zero-size tensors the streamer swaps in and out.
    for name, param in list(model.named_parameters()):
        if "lora_" in name:
            tensor = torch.empty(param.shape, device=device, dtype=torch.bfloat16)
        elif name.startswith(BLOCK_PREFIX):
            tensor = torch.empty(0, device=device, dtype=torch.bfloat16)
        else:
            tensor = reader.read(name.replace(".base_layer", "")).to(device, torch.bfloat16)
        _replace_param(model, name, tensor)
    for name, buf in list(model.named_buffers()):
        parent_name, _, attr = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        parent._buffers[attr] = buf.to(device)
    model.requires_grad_(False)
    model.eval()
    return model, BlockStreamer(model, reader, device)


def load_prompt_embeds(uri_model: str, prefix: str, device: torch.device):
    # Same as the authors' loader but with map_location so CUDA-saved tensors load on a Mac.
    embeds = torch.load(m.resolve_file(uri_model, f"qwen_text_embeddings/{prefix}_prompt_embeds.pt"), map_location="cpu")
    mask = torch.load(m.resolve_file(uri_model, f"qwen_text_embeddings/{prefix}_prompt_mask.pt"), map_location="cpu")
    return embeds[:1].to(device, torch.bfloat16), mask[:1].to(device) > 0


class LazyCheckpoints(dict):
    """Reads a LoRA checkpoint (1.8 GB) from disk on each access instead of holding all four in RAM."""

    def __init__(self, uri_model: str):
        super().__init__()
        self.uri_model = uri_model

    def __getitem__(self, name: str) -> dict[str, torch.Tensor]:
        subfolder, _ = m.CHECKPOINTS[name]
        return load_file(m.resolve_file(self.uri_model, f"{subfolder}/trainables.safetensors"))


class StreamingMarigoldV2(m.MarigoldV2):
    """Authors' MarigoldV2 with the transformer replaced by the disk-streaming build."""

    def __init__(self, uri_base: str = m.BASE_MODEL_URI, uri_model: str = m.MODEL_URI, device: str = "mps"):
        self.device = torch.device(device)
        base_dir = Path(uri_base) if os.path.isdir(uri_base) else Path(
            snapshot_download(uri_base, allow_patterns=["transformer/*", "vae/*", "model_index.json"])
        )
        t0 = time.perf_counter()
        self.vae = AutoencoderKLQwenImage.from_pretrained(
            str(base_dir), subfolder="vae", torch_dtype=torch.bfloat16, use_safetensors=True
        ).to(self.device)
        self.transformer, self.streamer = build_streaming_transformer(base_dir, self.device)
        self.prompt_embeds = {
            prefix: load_prompt_embeds(uri_model, prefix, self.device) for prefix in set(m.PROMPT_EMBEDS.values())
        }
        self.checkpoints = LazyCheckpoints(uri_model)
        self.vae_decoder_state = {
            k: v.cpu() for k, v in self.vae.state_dict().items() if k.startswith(("decoder.", "post_quant_conv."))
        }
        print(f"streaming model ready in {time.perf_counter() - t0:.1f}s on {self.device}", flush=True)

    def load_checkpoint(self, name: str):
        torch.mps.empty_cache()  # release cached activation buffers from the previous pass
        super().load_checkpoint(name)


class MpsMarigoldV2:
    def __init__(self, max_side: int | None = None):
        if not torch.backends.mps.is_available():
            raise RuntimeError("Apple GPU (MPS) backend not available in this torch build")
        if max_side:
            m.PROCESSING_MAX_LONG_SIDE = max_side
        self.model = StreamingMarigoldV2()

    def __call__(self, image_path: Path) -> dict[str, Image.Image]:
        image = Image.open(image_path).convert("RGB")
        streamer = self.model.streamer
        streamer.reset_stats()
        out = self.model(image)
        print(f"  GPU idle waiting for disk: {streamer.disk_wait:.1f}s; disk idle waiting for GPU: {streamer.gpu_wait:.1f}s", flush=True)
        torch.mps.empty_cache()
        return out
