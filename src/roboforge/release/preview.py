"""README preview: a side-by-side GIF of one recorded head-to-head pair (DECISIONS D-103).

Source: the committed Session 10 MuJoCo renders of recorded held-out episodes
(`reports/session10/observatory/media/*.mp4`). The pair is the D-089 rule 3(a) case on Can:
seed 0 diffusion U-Net and seed 0 BC-MLP on the same held-out start, the first start (sorted
state ID) on which both succeed, labelled "representative" by the selection. The state ID is
read from `selection.json`, and each clip's outcome, checkpoint and evaluation run from the
export's `index.json`; nothing about the episodes is typed here.

Transformations, all recorded in the sidecar: decode the two MP4s, keep every second frame
(20 fps to 10 fps), place them side by side on a dark band, draw text labels, and hold the
shorter clip's last frame until the longer one ends. Nothing is re-simulated or re-rendered.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import imageio.v3 as iio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from roboforge.paths import REPO_ROOT
from roboforge.provenance import sha256_file

OBS = REPO_ROOT / "reports" / "session10" / "observatory"
OUT_GIF = REPO_ROOT / "docs" / "media" / "readme_preview.gif"
OUT_JSON = REPO_ROOT / "docs" / "media" / "readme_preview.json"
TASK = "can"
CASE = "a_both_succeed"
PALETTE = 96  # colours of the single shared palette
FRAME_STEP = 2  # keep every second frame: the renders are 20 fps, the GIF is 10 fps
LABEL_H = 58
GAP = 6
BG = (24, 25, 27)
FG = (238, 233, 222)
DIM = (160, 156, 148)
OK = (122, 201, 143)
NAMES = {"diffusion_unet": "Diffusion U-Net", "bc_mlp": "BC-MLP"}


def select_pair(obs: Path = OBS) -> list[dict[str, Any]]:
    """The two clip records of the rule-3(a) Can pair, diffusion first, from the export."""
    sel = json.loads((obs / "selection.json").read_text())
    h2h = sel["head_to_head"][TASK]
    case = h2h[CASE]
    if case["kind"] != "representative":
        raise ValueError(f"{TASK}/{CASE} is {case['kind']!r}, expected 'representative'")
    index = json.loads((obs / "index.json").read_text())
    by_id = {c["id"]: c for c in index["clips"]}
    out = []
    for fam in h2h["pair"]:
        cid = f"{TASK}-{fam}-s{h2h['seed']}-{case['state_id']}"
        clip = by_id[cid]
        if clip["eval_run_id"] != h2h["eval_run_ids"][fam]:
            raise ValueError(f"{cid}: evaluation run differs from the selection")
        if case["categories"][fam] != "success" or not clip["outcome"]["success"]:
            raise ValueError(f"{cid}: the selection says both succeed")
        out.append(clip)
    return out


def _font(size: int) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    return ImageFont.load_default(size=size)


def _label(clip: dict[str, Any], width: int) -> Image.Image:
    img = Image.new("RGB", (width, LABEL_H), BG)
    d = ImageDraw.Draw(img)
    name = NAMES.get(clip["family"], clip["family"])
    d.text((8, 5), f"{name}  (seed {clip['seed']})", font=_font(15), fill=FG)
    o = clip["outcome"]
    verdict = f"success at step {o['first_success_step']}" if o["success"] else "failure"
    d.text((8, 25), verdict, font=_font(13), fill=OK if o["success"] else (230, 110, 98))
    d.text((8, 41), f"ckpt {clip['checkpoint']['sha256'][:8]}", font=_font(11), fill=DIM)
    return img


def _frames(path: Path) -> list[np.ndarray]:
    return [f for i, f in enumerate(iio.imiter(path)) if i % FRAME_STEP == 0]


def build(obs: Path = OBS, out_gif: Path = OUT_GIF, out_json: Path = OUT_JSON) -> dict[str, Any]:
    """Write the GIF and its provenance sidecar; return the sidecar."""
    clips = select_pair(obs)
    media = [obs / "media" / f"{c['id']}_frontview.mp4" for c in clips]
    seqs = [_frames(p) for p in media]
    h, w = seqs[0][0].shape[:2]
    n = max(len(s) for s in seqs)
    labels = [_label(c, w) for c in clips]
    footer = Image.new("RGB", (2 * w + GAP, 22), BG)
    fd = ImageDraw.Draw(footer)
    fd.text(
        (8, 4),
        f"MuJoCo render of recorded held-out episodes · {TASK.capitalize()} · same start "
        f"({clips[0]['state_id']})",
        font=_font(11),
        fill=DIM,
    )
    rgb = []
    for i in range(n):
        canvas = Image.new("RGB", (2 * w + GAP, LABEL_H + h + 22), BG)
        for k, seq in enumerate(seqs):
            x = k * (w + GAP)
            canvas.paste(labels[k], (x, 0))
            canvas.paste(Image.fromarray(seq[min(i, len(seq) - 1)]), (x, LABEL_H))
        canvas.paste(footer, (0, LABEL_H + h))
        rgb.append(canvas)
    # One palette for every frame (from a strip of sampled frames), so the encoder can store
    # only the changed region of each frame.
    sample = rgb[:: max(1, n // 8)]
    strip = Image.new("RGB", (rgb[0].width, rgb[0].height * len(sample)))
    for k, im in enumerate(sample):
        strip.paste(im, (0, k * rgb[0].height))
    pal = strip.quantize(colors=PALETTE, method=Image.Quantize.MEDIANCUT, dither=0)
    frames = [im.quantize(palette=pal, dither=0) for im in rgb]
    out_gif.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        out_gif,
        save_all=True,
        append_images=frames[1:],
        duration=1000 // (20 // FRAME_STEP),
        loop=0,
        optimize=False,
        disposal=1,
    )
    side = {
        "schema": "roboforge.readme_preview/1",
        "decision": "D-103",
        "rule": "D-089 rule 3(a): seed 0 diffusion_unet vs seed 0 bc_mlp on the same held-out "
        "start; the first state_id on which both succeed (selection kind 'representative')",
        "selection": {
            "path": "reports/session10/observatory/selection.json",
            "sha256": sha256_file(obs / "selection.json"),
        },
        "index": {
            "path": "reports/session10/observatory/index.json",
            "sha256": sha256_file(obs / "index.json"),
        },
        "sources": [
            {
                "path": p.relative_to(REPO_ROOT).as_posix(),
                "sha256": sha256_file(p),
                "clip_id": c["id"],
                "eval_run_id": c["eval_run_id"],
                "checkpoint_sha256": c["checkpoint"]["sha256"],
                "outcome": {k: c["outcome"][k] for k in ("success", "first_success_step")},
                "frames_used": len(s),
            }
            for p, c, s in zip(media, clips, seqs, strict=True)
        ],
        "transformation": f"decode; keep every {FRAME_STEP}nd frame (10 fps); side by side "
        "with a 6 px gap; text labels from index.json; the shorter clip holds its last frame; "
        f"one shared {PALETTE}-colour palette (median cut over sampled frames), no dithering",
        "output": {
            "path": out_gif.relative_to(REPO_ROOT).as_posix(),
            "sha256": sha256_file(out_gif),
            "frames": n,
            "size_px": [2 * w + GAP, LABEL_H + h + 22],
        },
    }
    out_json.write_text(json.dumps(side, indent=1, sort_keys=True) + "\n")
    return side


def check(obs: Path = OBS, out_gif: Path = OUT_GIF, out_json: Path = OUT_JSON) -> list[str]:
    """The committed GIF and sources still match the sidecar's hashes (no re-encode)."""
    if not out_json.exists() or not out_gif.exists():
        return ["README preview or its sidecar is missing (run make readme-preview)"]
    side = json.loads(out_json.read_text())
    problems = []
    for ref in [side["selection"], side["index"], *side["sources"], side["output"]]:
        p = REPO_ROOT / ref["path"]
        if not p.exists() or sha256_file(p) != ref["sha256"]:
            problems.append(f"README preview: {ref['path']} does not match its recorded SHA-256")
    want = [c["id"] for c in select_pair(obs)]
    if [s["clip_id"] for s in side["sources"]] != want:
        problems.append("README preview: the sidecar's clips are not the rule-3(a) pair")
    return problems
