"""Montages of recorded validation episodes, chosen by a fixed rule, for visual inspection.

    uv run --frozen python scripts/render_episodes.py \
        --summary reports/session6/suite/level1_core_lift.json

Selection rule (declared before rendering, never "the nicest clip"): for each family, among
the §7-selected checkpoints of the chosen configuration, ordered by training seed:

* **success**: the first successful episode, by sorted ``state_id``, of the lowest seed;
* **failure**: the first failed episode, by sorted ``state_id``, of the lowest seed that has
  one (none if every selected checkpoint succeeds on all 50 starts).

Each episode is drawn from its **recorded** physics states (``arrays.npz``), never
re-simulated: the validation snapshot is restored to compile its model, and
``ShadowRenderer`` draws each recorded state on scratch data, so nothing touches a live
rollout (KI-019). Tiles at 0, 25, 50, 75 and 100 % of the episode, ``agentview`` and
``frontview``. Frames are rendered on the platform running this script and are labelled
with it; they illustrate recorded states and are not evidence of success (the records are).
Writes ``<out-dir>/episodes_<family>.png`` and ``<out-dir>/episodes_index.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from roboforge.sim.backend import configure_mujoco_gl

configure_mujoco_gl()

import imageio.v2 as iio  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from roboforge.paths import REPO_ROOT, repo_relative  # noqa: E402
from roboforge.provenance import evaluation_platform, git_revision, sha256_file  # noqa: E402
from roboforge.sim.env_adapter import RoboForgeEnv  # noqa: E402
from roboforge.sim.eval_states import load_suite  # noqa: E402
from roboforge.sim.shadow_render import ShadowRenderer  # noqa: E402
from roboforge.sim.snapshot import restore  # noqa: E402

CAMERAS = ("agentview", "frontview")
FRACTIONS = (0.0, 0.25, 0.5, 0.75, 1.0)
SIZE = 192


def _label(img: np.ndarray, text: str) -> np.ndarray:
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    d.rectangle((0, 0, SIZE, 12), fill=(0, 0, 0))
    d.text((2, 0), text, fill=(255, 255, 255))
    return np.asarray(im)


def pick(fam: dict[str, Any], reports: Path) -> list[dict[str, Any]]:
    """Apply the selection rule to one family; return the chosen episodes."""
    runs = sorted([r for r in fam["runs"] if r["trial"] == fam["chosen_trial"]
                   and r["selection"]], key=lambda r: r["seed"])
    picks: list[dict[str, Any]] = []
    for want in ("success", "failure"):
        for r in runs:
            ev = r["selection"]["eval_run_id"]
            summ = json.loads((reports / "validation" / f"{ev}.json").read_text())
            eps = sorted(summ["episodes"], key=lambda e: e["state_id"])
            hit = next((e for e in eps if e["success"] == (want == "success")), None)
            if hit:
                picks.append({"kind": want, "family": fam["family"], "trial": r["trial"],
                              "seed": r["seed"],
                              "train_run_id": r["run_id"], "eval_run_id": ev,
                              "checkpoint_step": r["selection"]["step"],
                              "checkpoint_sha256": r["selection"]["sha256"],
                              "state_id": hit["state_id"], "steps": hit["steps"],
                              "stop_reason": hit["stop_reason"],
                              "physics_sha256": hit["physics_sha256"]})
                break
    return picks


def render(task: str, pick_: dict[str, Any], starts: dict[str, Any]) -> np.ndarray:
    """Rows = cameras, columns = episode fractions, from the recorded states."""
    rec_dir = REPO_ROOT / "runs" / pick_["eval_run_id"] / "episodes" / pick_["state_id"]
    record = json.loads((rec_dir / "record.json").read_text())
    if record["physics_sha256"] != pick_["physics_sha256"]:
        raise SystemExit(f"{rec_dir}: physics hash differs from the summary")
    states = np.load(rec_dir / "arrays.npz")["state"]
    env = RoboForgeEnv(task)
    try:
        restore(env.raw_env, starts[pick_["state_id"]])
        sh = ShadowRenderer(env.model)
        idx = [round(f * (len(states) - 1)) for f in FRACTIONS]
        rows = [np.concatenate([_label(sh.rgb(states[k], cam, SIZE, SIZE),
                                       f"{cam} k={k}") for k in idx], axis=1)
                for cam in CAMERAS]
        sh.close()
    finally:
        env.close()
    pick_["frames"] = idx
    return np.concatenate(rows, axis=0)


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--summary", type=Path, required=True)
    p.add_argument("--reports-dir", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, default=None)
    a = p.parse_args(argv)
    reports = a.reports_dir or a.summary.parent.parent
    out = a.out_dir or reports / "figures"
    out.mkdir(parents=True, exist_ok=True)
    summary = json.loads(a.summary.read_text())
    task = summary["task"]
    starts = {s.state_id: s for s in load_suite(task, "validation")}
    picks_all, outputs = [], {}
    for fam in summary["families"]:
        picks = pick(fam, reports)
        blocks = []
        for pk in picks:
            img = render(task, pk, starts)
            title = Image.new("RGB", (img.shape[1], 16), (20, 20, 20))
            ImageDraw.Draw(title).text(
                (4, 2), f"{pk['kind'].upper()} | {fam['family']} trial {pk['trial']} seed "
                f"{pk['seed']} step {pk['checkpoint_step']} | {pk['state_id']} | "
                f"{pk['steps']} steps, {pk['stop_reason']} | recorded states, VALIDATION ONLY",
                fill=(255, 255, 255))
            blocks += [np.asarray(title), img]
        if blocks:
            path = out / f"episodes_{fam['family']}.png"
            iio.imwrite(path, np.concatenate(blocks, axis=0))
            outputs[repo_relative(path)] = sha256_file(path)
        picks_all += picks
    index = {"schema": "roboforge.episode_montages/1",
             "rule": "per family, chosen configuration's selected checkpoints by seed: first "
                     "success (sorted state_id) of the lowest seed; first failure of the "
                     "lowest seed that has one",
             "label": "VALIDATION ONLY; tiles drawn from recorded states, not re-simulated",
             "render_platform": evaluation_platform(), "cameras": list(CAMERAS),
             "summary": {"path": repo_relative(a.summary), "sha256": sha256_file(a.summary)},
             "episodes": picks_all, "outputs": outputs, "git": git_revision(),
             "generated_utc": datetime.now(UTC).isoformat(timespec="seconds")}
    (out / "episodes_index.json").write_text(json.dumps(index, indent=1) + "\n")
    for k in outputs:
        print(k)
    return 0


if __name__ == "__main__":
    sys.exit(main())
