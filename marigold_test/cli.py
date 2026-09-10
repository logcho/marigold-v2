"""Run images through Marigold V2 and save every output plus an HTML contact sheet.

Examples:
    marigold-test --input inputs --output outputs/run1
    marigold-test --input photo.jpg --backend local
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from PIL import Image

from . import MODALITIES
from .report import write_report

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def _collect_images(inp: Path) -> list[Path]:
    if inp.is_file():
        return [inp]
    return sorted(p for p in inp.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def _slug(name: str) -> str:
    return name.lower().replace(" ", "_").replace("-", "_")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", "-i", required=True, help="Image file or folder of images")
    parser.add_argument("--output", "-o", default="outputs/latest", help="Output folder")
    parser.add_argument("--backend", choices=["remote", "local"], default="remote")
    parser.add_argument(
        "--modality",
        choices=["all", *MODALITIES],
        default="all",
        help="Which outputs to keep (the model always computes all four)",
    )
    parser.add_argument("--limit", type=int, help="Only process the first N images")
    args = parser.parse_args()

    images = _collect_images(Path(args.input))
    if args.limit:
        images = images[: args.limit]
    if not images:
        raise SystemExit(f"no images found under {args.input}")

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    keep = MODALITIES if args.modality == "all" else [args.modality]

    if args.backend == "remote":
        from .remote import RemoteMarigoldV2

        model = RemoteMarigoldV2()
    else:
        from .local import LocalMarigoldV2

        model = LocalMarigoldV2()

    print(f"{len(images)} image(s) -> {out_dir} via {args.backend} backend")
    rows = []
    for idx, path in enumerate(images, 1):
        stem = path.stem
        img_dir = out_dir / stem
        img_dir.mkdir(exist_ok=True)
        files = {}
        row = {"name": path.name, "files": files}
        t0 = time.perf_counter()
        try:
            input_copy = img_dir / f"input{path.suffix.lower()}"
            Image.open(path).convert("RGB").save(input_copy)
            files["Input"] = f"{stem}/{input_copy.name}"
            results = model(path)
            for name in keep:
                if name in results:
                    fn = f"{_slug(name)}.png"
                    results[name].save(img_dir / fn)
                    files[name] = f"{stem}/{fn}"
            row["seconds"] = round(time.perf_counter() - t0, 1)
            print(f"[{idx}/{len(images)}] {path.name}: {', '.join(k for k in files if k != 'Input')} ({row['seconds']}s)")
        except Exception as e:  # noqa: BLE001 - keep the batch going, record the failure
            row["error"] = f"{type(e).__name__}: {e}"
            print(f"[{idx}/{len(images)}] {path.name}: FAILED {row['error']}")
        rows.append(row)

    (out_dir / "results.json").write_text(json.dumps(rows, indent=2))
    report = write_report(out_dir, rows, title=f"Marigold V2 ({args.backend}) - {args.input}")
    ok = sum("error" not in r for r in rows)
    print(f"done: {ok}/{len(rows)} succeeded. Report: {report.resolve()}")


if __name__ == "__main__":
    main()
