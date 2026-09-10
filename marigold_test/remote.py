"""Remote backend: sends images to the authors' public demo (toshas/Marigold-V2 on
Hugging Face ZeroGPU) and downloads the four rendered outputs.

Works from any machine, no GPU needed. One call runs all four models on the image
(~20-30 s). ZeroGPU has a per-user quota; set HF_TOKEN to raise it.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

from PIL import Image

SPACE = "toshas/Marigold-V2"
ENDPOINT = "/on_process_first"
CALL_TIMEOUT_S = 180
POLL_S = 3


class RemoteMarigoldV2:
    def __init__(self, space: str = SPACE, hf_token: str | None = None):
        from gradio_client import Client

        token = hf_token or os.environ.get("HF_TOKEN")
        self.client = Client(space, token=token, verbose=False)

    def _call_once(self, image_path: Path) -> dict[str, Image.Image]:
        from gradio_client import handle_file

        job = self.client.submit(
            image_slider=[handle_file(str(image_path)), None],
            modality_selector_left=None,
            modality_selector_right=None,
            api_name=ENDPOINT,
        )
        start = time.time()
        last_code = None
        while not job.done():
            if time.time() - start > CALL_TIMEOUT_S:
                job.cancel()
                raise TimeoutError(f"no result after {CALL_TIMEOUT_S}s (last status: {last_code})")
            status = job.status()
            if status.code != last_code:
                extra = f" (queue rank {status.rank}/{status.queue_size})" if status.rank is not None else ""
                print(f"  {status.code.name.lower()}{extra}", flush=True)
                last_code = status.code
            time.sleep(POLL_S)
        gallery = job.result()[0]
        return {
            item["caption"]: Image.open(item["image"]).convert("RGB")
            for item in gallery
            if item["caption"] != "Input"
        }

    def __call__(self, image_path: Path, retries: int = 2) -> dict[str, Image.Image]:
        last_error: Exception | None = None
        for attempt in range(retries + 1):
            try:
                return self._call_once(image_path)
            except Exception as e:  # noqa: BLE001 - surface any Space/quota error after retries
                last_error = e
                if "quota" in str(e).lower():
                    raise RuntimeError(
                        f"ZeroGPU quota exhausted: {e}\n"
                        "  Log in with a Hugging Face token (export HF_TOKEN=hf_...) for a larger daily quota, "
                        "or use --backend local on a CUDA machine."
                    ) from e
                if attempt < retries:
                    wait = 10 * (attempt + 1)
                    print(f"  remote call failed ({type(e).__name__}: {e}); retrying in {wait}s", flush=True)
                    time.sleep(wait)
        raise RuntimeError(f"remote inference failed for {image_path.name}: {last_error}")
