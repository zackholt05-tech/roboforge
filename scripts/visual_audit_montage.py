"""Before/after montages of the Session 20 Observatory visual audit (DECISIONS D-136, D-137).

Reads the viewport tiles that ``make web-audit`` wrote under
``logs/session20/visual_audit/{before,after}/`` and, for each section and each viewer capture,
writes one JPEG with the two states side by side. It only stitches, scales and labels existing
screenshots; nothing is rendered or measured here.

Usage: ``python scripts/visual_audit_montage.py [--root logs/session20/visual_audit]``
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw

TOPBAR_PX = 56  # the sticky navigation bar, repeated at the top of every tile
OVERLAP_PX = 24  # scroll overlap between two tiles (website/e2e/audit-s20.spec.ts)
COLUMN_PX = 720
GAP_PX = 16
LABEL_PX = 28
BACKGROUND = (30, 33, 37)
INK = (238, 231, 215)


def stitch(tiles: list[Path], height_px: int | None) -> Image.Image:
    """Stack the tiles of one section into one tall image, dropping the repeated top bar."""
    images = [Image.open(t).convert("RGB") for t in tiles]
    parts = [images[0]] + [
        im.crop((0, TOPBAR_PX + OVERLAP_PX, im.width, im.height)) for im in images[1:]
    ]
    out = Image.new("RGB", (parts[0].width, sum(p.height for p in parts)), BACKGROUND)
    y = 0
    for p in parts:
        out.paste(p, (0, y))
        y += p.height
    if height_px is not None:
        out = out.crop((0, 0, out.width, min(out.height, height_px + TOPBAR_PX)))
    return out


def side_by_side(before: Image.Image, after: Image.Image) -> Image.Image:
    """Scale both states to one column width and place them next to each other with labels."""
    cols = []
    for im in (before, after):
        scale = COLUMN_PX / im.width
        cols.append(im.resize((COLUMN_PX, max(1, round(im.height * scale))), Image.LANCZOS))
    height = max(c.height for c in cols) + LABEL_PX
    out = Image.new("RGB", (2 * COLUMN_PX + GAP_PX, height), BACKGROUND)
    draw = ImageDraw.Draw(out)
    for i, (name, col) in enumerate(zip(("before", "after"), cols, strict=True)):
        x = i * (COLUMN_PX + GAP_PX)
        draw.text((x + 6, 8), name, fill=INK)
        out.paste(col, (x, LABEL_PX))
    return out


def main() -> None:
    """Write ``montage/<section>.jpg`` and ``montage/viewers_<name>.jpg`` under the audit root."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="logs/session20/visual_audit")
    ap.add_argument("--size", default="1440x900")
    args = ap.parse_args()
    root = Path(args.root)
    tags = ("before", "after")
    index = {t: json.loads((root / t / "index.json").read_text()) for t in tags}
    words = {t: json.loads((root / t / "word_counts.json").read_text()) for t in tags}
    out_dir = root / "montage"
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for key in index["before"]["tiles"]:
        size, section = key.split("/")
        if size != args.size or key not in index["after"]["tiles"]:
            continue
        sides = []
        for tag in ("before", "after"):
            tiles = [root / tag / f for f in index[tag]["tiles"][key]]
            sides.append(stitch(tiles, words[tag][size]["sections"][section]["height_px"]))
        path = out_dir / f"{section}.jpg"
        side_by_side(*sides).save(path, quality=62, optimize=True)
        written.append(path.name)
    for shot in sorted((root / "before" / "viewers").glob("*.jpg")):
        other = root / "after" / "viewers" / shot.name
        if not other.exists():
            continue
        path = out_dir / f"viewers_{shot.name}"
        side_by_side(Image.open(shot).convert("RGB"), Image.open(other).convert("RGB")).save(
            path, quality=62, optimize=True
        )
        written.append(path.name)
    print(f"{len(written)} montages -> {out_dir}")


if __name__ == "__main__":
    main()
