"""Write a self-contained HTML contact sheet: one row per input, one column per output."""

from __future__ import annotations

import html
from pathlib import Path

from . import MODALITIES


def write_report(output_dir: Path, rows: list[dict], title: str = "Marigold V2 test run") -> Path:
    cols = ["Input", *MODALITIES]
    parts = [
        "<!doctype html><meta charset='utf-8'>",
        f"<title>{html.escape(title)}</title>",
        "<style>body{font-family:system-ui,sans-serif;background:#111;color:#eee;margin:16px}"
        "table{border-collapse:collapse}th{text-align:left;padding:6px 8px;font-weight:600;color:#bbb}"
        "td{padding:6px 8px;vertical-align:top}img{max-width:320px;height:auto;display:block;background:#222}"
        ".name{font-size:12px;color:#999;margin-top:4px;max-width:320px;word-break:break-all}"
        ".err{color:#f66;font-size:13px}</style>",
        f"<h2>{html.escape(title)}</h2>",
        "<table><tr>" + "".join(f"<th>{html.escape(c)}</th>" for c in cols) + "</tr>",
    ]
    for row in rows:
        parts.append("<tr>")
        for col in cols:
            rel = row["files"].get(col)
            if rel:
                parts.append(
                    f"<td><a href='{rel}' target='_blank'><img src='{rel}' loading='lazy'></a>"
                    + (f"<div class='name'>{html.escape(row['name'])}</div>" if col == "Input" else "")
                    + "</td>"
                )
            else:
                parts.append(f"<td class='err'>{html.escape(row.get('error', 'missing'))}</td>")
        parts.append("</tr>")
    parts.append("</table>")
    path = output_dir / "index.html"
    path.write_text("\n".join(parts))
    return path
