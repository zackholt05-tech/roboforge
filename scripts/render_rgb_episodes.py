"""RGB study: videos of recorded closed-loop episodes, chosen by a fixed rule (MASTER_SPEC S8).

    ROBOFORGE_EVIDENCE_SESSION=8 uv run --frozen python scripts/render_rgb_episodes.py

**Selection rule** (declared here before rendering, never "the nicest clip"):
* **held-out failure:** per family, the first failed held-out episode, in seed order then by
  sorted ``state_id``. Listed as "none" if the family has no failure.
* **held-out success:** per family, the first successful held-out episode of the lowest seed.
* **validation failure:** per family, the earliest checkpoint of the lowest seed with a failed
  validation episode, and its first failure by sorted ``state_id``. These are development
  checkpoints, shown because the held-out test has almost no failures.

**Rendering.** Each video is drawn from the episode's **recorded** physics states
(``arrays.npz``) by ``ShadowRenderer``, never re-simulated, the same path as
``render_episodes.py``. The panels are the policy's two cameras (``agentview``,
``robot0_eye_in_hand``) plus ``frontview``, at 224², 20 frames per second (the control rate).
Frames are drawn on the platform that runs this script; they illustrate the records, which
are the evidence.

**Held-out access.** Held-out starts load only through ``eval_states.load_suite``, after the
RGB gate (D-081) and only once all 6 held-out reports exist (post-test illustration).

**Writes** ``reports/session<N>/rgb_videos/*.mp4`` and ``index.json``.
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from typing import Any

from roboforge.sim.backend import configure_mujoco_gl

configure_mujoco_gl()

import imageio.v2 as iio  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from roboforge.eval import rgb_heldout as rh  # noqa: E402
from roboforge.paths import REPO_ROOT, evidence_reports_dir, repo_relative  # noqa: E402
from roboforge.provenance import evaluation_platform, git_revision, sha256_file  # noqa: E402
from roboforge.sim import eval_states as es  # noqa: E402
from roboforge.sim.env_adapter import RoboForgeEnv  # noqa: E402
from roboforge.sim.shadow_render import ShadowRenderer  # noqa: E402
from roboforge.sim.snapshot import restore  # noqa: E402

CAMERAS = ("agentview", "robot0_eye_in_hand", "frontview")
SIZE, FPS = 224, 20
RULE = ("held-out: per family, first failure (seed order, then sorted state_id) and first "
        "success of the lowest seed; validation: per family, the lowest seed's earliest "
        "checkpoint with a failure, first failure by sorted state_id")


def _runs(family: str) -> list[str]:
    return sorted((r for r in rh.SELECTIONS if rh.load_selection(r)["family"] == family),
                  key=lambda r: rh.load_selection(r)["seed"])


def _pick(kind: str, run_id: str, summ: dict[str, Any], want_success: bool,
          step: int) -> dict[str, Any] | None:
    eps = sorted(summ["episodes"], key=lambda e: e["state_id"])
    hit = next((e for e in eps if bool(e["success"]) == want_success), None)
    if hit is None:
        return None
    return {"kind": kind, "family": summ.get("family") or rh.load_selection(run_id)["family"],
            "seed": rh.load_selection(run_id)["seed"], "train_run_id": run_id,
            "checkpoint_step": step, "eval_run_id": summ["eval_run_id"],
            "records_dir": summ["records_dir"], "state_id": hit["state_id"],
            "success": bool(hit["success"]), "steps": hit["steps"],
            "stop_reason": hit["stop_reason"], "physics_sha256": hit["physics_sha256"]}


def picks() -> list[dict[str, Any]]:
    """Apply the selection rule."""
    out: list[dict[str, Any]] = []
    for fam in rh.FAMILY_ORDER:
        runs = _runs(fam)
        held = {r: json.loads((evidence_reports_dir("rgb_heldout") / f"{r}.json").read_text())
                for r in runs}
        fail = next((p for r in runs if (p := _pick("heldout_failure", r, held[r], False,
                                                    held[r]["checkpoint"]["step"]))), None)
        out.append(fail or {"kind": "heldout_failure", "family": fam, "state_id": None,
                            "note": "no failed held-out episode in this family"})
        r0 = runs[0]
        out.append(_pick("heldout_success", r0, held[r0], True, held[r0]["checkpoint"]["step"]))
        vfail = None
        for r in runs:
            vdir = REPO_ROOT / "reports" / "session8" / "rgb_validation" / r
            for p in sorted(vdir.glob("step_*.json")):
                s = json.loads(p.read_text())
                if s["successes"] < s["n_episodes"]:
                    full = json.loads((REPO_ROOT / s["records_dir"] / "manifest.json")
                                      .read_text())
                    vfail = _pick("validation_failure", r, full, False, s["checkpoint"]["step"])
                    break
            if vfail:
                break
        out.append(vfail or {"kind": "validation_failure", "family": fam, "state_id": None,
                             "note": "no failed validation episode in this family"})
    return out


def _label(img: np.ndarray, text: str) -> np.ndarray:
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    d.rectangle((0, 0, img.shape[1], 12), fill=(0, 0, 0))
    d.text((3, 0), text, fill=(255, 255, 255))
    return np.asarray(im)


def render(pk: dict[str, Any], starts: dict[str, Any]) -> str:
    """Write one MP4 of the recorded states; return its repo-relative path."""
    rec = REPO_ROOT / pk["records_dir"] / "episodes" / pk["state_id"]
    record = json.loads((rec / "record.json").read_text())
    if record["physics_sha256"] != pk["physics_sha256"]:
        raise SystemExit(f"{rec}: physics hash differs from the summary")
    states = np.load(rec / "arrays.npz")["state"]
    head = (f"{pk['kind'].upper()} | {pk['family']} s{pk['seed']} step {pk['checkpoint_step']}"
            f" | {pk['state_id']} | {pk['steps']} steps, {pk['stop_reason']} | recorded states")
    env = RoboForgeEnv(rh.TASK)
    frames = []
    try:
        restore(env.raw_env, starts[pk["state_id"]])
        sh = ShadowRenderer(env.model)
        for k, st in enumerate(states):
            row = np.concatenate([sh.rgb(st, c, SIZE, SIZE) for c in CAMERAS], axis=1)
            frames.append(_label(row, f"{head} | frame {k}"))
        sh.close()
    finally:
        env.close()
    out = evidence_reports_dir("rgb_videos")
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{pk['kind']}_{pk['family']}_s{pk['seed']}_{pk['state_id']}.mp4"
    iio.mimwrite(path, frames, fps=FPS, macro_block_size=8)
    return repo_relative(path)


def main() -> int:
    """CLI entrypoint (see the module docstring)."""
    if any(not (evidence_reports_dir("rgb_heldout") / f"{r}.json").exists()
           for r in rh.SELECTIONS):
        raise SystemExit("all 6 held-out reports must exist first (post-test illustration)")
    ok, problems = rh.gate_status()
    if not ok:
        raise SystemExit(f"RGB held-out gate shut: {problems}")
    chosen = picks()
    starts = {s.state_id: s for suite in ("validation", es.LOCKED_SUITE)
              for s in es.load_suite(rh.TASK, suite)
              if any(p and p.get("state_id") == s.state_id for p in chosen)}
    outputs = {}
    for pk in chosen:
        if pk and pk.get("state_id"):
            pk["video"] = render(pk, starts)
            outputs[pk["video"]] = sha256_file(REPO_ROOT / pk["video"])
            print(pk["video"], flush=True)
    index = {"schema": "roboforge.rgb_episode_videos/1", "rule": RULE,
             "label": "illustrations drawn from recorded states, not re-simulated; the "
                      "records and summaries are the evidence",
             "cameras": list(CAMERAS), "size": SIZE, "fps": FPS,
             "render_platform": evaluation_platform(), "episodes": chosen,
             "outputs": outputs, "git": git_revision(),
             "generated_utc": datetime.now(UTC).isoformat(timespec="seconds")}
    (evidence_reports_dir("rgb_videos") / "index.json").write_text(
        json.dumps(index, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
