"""Preregistered failure taxonomy (RESEARCH_PROTOCOL §14.5, DECISIONS D-065).

Labels every recorded episode from its arrays only (``roboforge.rollout/1`` records: body
poses, observations, executed actions, clip masks, outcome). Config:
``configs/eval/failure_taxonomy.yaml``. The rules are ordered, exhaustive and mutually
exclusive; the first one that matches labels the episode:

0. ``success``: the first-success rule fired.
1. ``engineering_error``: the episode stopped with an error (policy or simulator exception,
   invalid action). Engineering-incomplete, not task failure; still a failure in the
   primary metric.
2. ``gripper_never_closed``: no executed gripper command > 0.
3. ``no_grasp_drift`` / 4. ``no_grasp_near``: the task object was never *grasp-lifted*
   (rise >= 0.01 m above its rest height while the gripper is closed); the final
   end-effector-object distance is > 0.20 m (drift) or not (near).
5. ``timeout_holding``: grasp-lifted at the final frame (still carrying at the horizon).
6. ``in_goal_bin_not_success`` (Can): the can ends inside the goal bin's footprint, not
   held, and the success predicate is false (wrong quadrant or height).
7. ``dropped``: grasp-lifted at some frame but not at the end, and not covered above.

Definitions: rest height = minimum object z over frames 0..10; the gripper is *closed* at
frame k >= 1 when the executed gripper command of step k - 1 is > 0 (frame 0 is open).
"""

from __future__ import annotations

from functools import cache
from pathlib import Path
from typing import Any

import numpy as np

from roboforge.data.pipeline import load_specs
from roboforge.paths import configs_dir
from roboforge.provenance import sha256_json


@cache
def taxonomy_config() -> dict[str, Any]:
    """Load ``configs/eval/failure_taxonomy.yaml``."""
    import yaml

    return yaml.safe_load((configs_dir() / "eval" / "failure_taxonomy.yaml").read_text())


def taxonomy_sha256() -> str:
    """Content hash of the taxonomy config (recorded with every labelling)."""
    return sha256_json(taxonomy_config())


@cache
def _eef_slice(task: str) -> slice:
    lo, hi = load_specs(task)[0].slices()["robot0_eef_pos"]
    return slice(lo, hi)


def classify(meta: dict[str, Any], arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    """Category and attributes of one recorded episode."""
    cfg = taxonomy_config()
    task = meta["task"]
    out_c = meta["outcome"]
    names = meta["bodies"]["names"]
    obj = arrays["body_pos"][:, names.index(cfg["task_object_body"][task])]
    eef = arrays["obs"][:, _eef_slice(task)]
    ex = arrays["executed_action"]
    clip = arrays["clip_mask"]
    n_frames = len(obj)
    grip = ex[:, -1] > 0 if len(ex) else np.zeros(0, dtype=bool)
    closed = np.concatenate([[False], grip])[:n_frames]
    rest_z = float(obj[: int(cfg["rest_frames"]), 2].min())
    rise = obj[:, 2] - rest_z
    lifted = (rise >= float(cfg["rise_m"])) & closed
    dist_final = float(np.linalg.norm(eef[-1] - obj[-1]))
    motion_clip = clip[:, :-1].any(axis=1) if len(clip) else np.zeros(0, dtype=bool)
    attrs = {"steps": int(out_c["steps"]), "stop_reason": out_c["stop_reason"],
             "final_eef_object_dist_m": dist_final, "max_rise_m": float(rise.max()),
             "max_rise_while_closed_m": float(rise[closed].max()) if closed.any() else 0.0,
             "grasp_lifted_frames": int(lifted.sum()),
             "final_object_pos_m": obj[-1].tolist(), "eef_z_max_m": float(eef[:, 2].max()),
             "gripper_close_commands": int(grip.sum()),
             "motion_clip_fraction": float(motion_clip.mean()) if len(motion_clip) else 0.0,
             "first_motion_clip_step": (int(np.argmax(motion_clip)) if motion_clip.any()
                                        else None)}
    in_goal = None
    if task == "can":
        g = cfg["can_goal_bin"]
        bxy = arrays["body_pos"][-1, names.index(g["body"])][:2]
        hx, hy = (float(v) for v in g["half_size_xy_m"])
        in_goal = bool(abs(obj[-1, 0] - bxy[0]) < hx and abs(obj[-1, 1] - bxy[1]) < hy)
        attrs["final_object_in_goal_bin_footprint"] = in_goal
    if out_c["success"]:
        cat = "success"
    elif out_c["stop_reason"] == "error":
        cat = "engineering_error"
    elif not grip.any():
        cat = "gripper_never_closed"
    elif not lifted.any():
        cat = "no_grasp_drift" if dist_final > float(cfg["far_m"]) else "no_grasp_near"
    elif lifted[-1]:
        cat = "timeout_holding"
    elif in_goal:
        cat = "in_goal_bin_not_success"
    else:
        cat = "dropped"
    return {"category": cat, **attrs}


def classify_records(records_dir: Path) -> list[dict[str, Any]]:
    """Label every episode record under ``<records_dir>/episodes/`` (sorted by state ID)."""
    from roboforge.eval.rollout import RolloutRecord

    out = []
    for d in sorted((Path(records_dir) / "episodes").iterdir()):
        rec = RolloutRecord.load(d)
        out.append({"state_id": rec.meta["episode"]["initial_state"]["state_id"],
                    **classify(rec.meta, rec.arrays)})
    return out


def counts(labels: list[dict[str, Any]]) -> dict[str, int]:
    """Episodes per category, in taxonomy order (every category present, zeros included)."""
    cats = taxonomy_config()["categories"]
    unknown = {lab["category"] for lab in labels} - set(cats)
    if unknown:
        raise ValueError(f"labels outside the taxonomy: {unknown}")
    return {c: sum(lab["category"] == c for lab in labels) for c in cats}
