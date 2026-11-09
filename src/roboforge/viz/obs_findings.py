"""``findings.json`` for the Observatory: the frozen results, copied verbatim with citations.

The site never computes or types a statistic. This module copies the **generated** blocks of
the three frozen analyses (Level 1 ``reports/session7/results``, the RGB study
``reports/session8/rgb_results``, Level 2 ``reports/session9/level2_results``) into one file,
each block next to the path and SHA-256 of the file it came from. It adds only:

* **outcome strips**: per evaluated checkpoint, the category of each start, from the frozen
  ``taxonomy.json`` files (the per-episode labels the analyses counted);
* the **cube placement map** of the Level 2 sets: each start's initial cube position, read from
  frame 0 of its recorded episode, with the per-start success counts of both arms;
* **annotations** that the frozen record itself requires: the KI-049 erratum beside the Level 2
  training row it corrects, and the KI-034 ceiling note beside the Lift numbers.
"""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Any

import numpy as np
import yaml

from roboforge.paths import REPO_ROOT
from roboforge.viz.obs_select import L1_RESULTS, L1_TAXONOMY, L2_RESULTS, L2_TAXONOMY, source

FINDINGS_SCHEMA = "roboforge.observatory_findings/1"
RGB_RESULTS = "reports/session8/rgb_results/results.json"
RGB_TAXONOMY = "reports/session8/rgb_results/taxonomy.json"
L2_ERRATA = "reports/session9/level2_results/ERRATA.md"
NOTES = {
    "ki_034": "Lift is saturated: most learned families reach 100 % or close, so Lift cannot "
              "rank them on success (KNOWN_ISSUES KI-034). Can separates more.",
    "not_separated": "'not separated' means the 95 % interval of the paired difference "
                     "includes 0; it is not evidence of equivalence.",
    "privilege_level1": "Level 1 policies observe privileged simulator object state "
                        "(cube/can pose) and robot proprioception; they are not vision "
                        "policies.",
    "privilege_rgb": "RGB study: two 84² cameras + 9-D proprioception; no object state.",
    "privilege_level2": "Level 2: one 512-point world-frame cloud from depth + calibration of "
                        "two 84² cameras, + 9-D proprioception; sensor-like, no segmentation "
                        "or object pose (D-083). DP3-inspired, not a DP3 reproduction.",
    "ki_049": "Erratum (KNOWN_ISSUES KI-049, ERRATA.md): seed 0's generated training row is "
              "miscomputed (24.62 ms/step). Correct: 42.2 ms/step for the completed segment "
              "(35,000 steps in 1,477 s, 0.41 h); 0.70 h in total including the cut first "
              "segment. Billed amounts and every success, interval and separation are "
              "unaffected.",
}
KI049_CORRECTION = {"run_id": "s9_pc_diffusion_dp3_lift_seed0_8fd35613",
                    "generated_ms_per_step": 24.62, "corrected_ms_per_step": 42.2,
                    "completed_segment_wall_h": 0.41, "total_wall_h": 0.70,
                    "source": L2_ERRATA}


def _load(rel: str) -> Any:
    return json.loads((REPO_ROOT / rel).read_text())


def _strip(labels: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{"state_id": x["state_id"], "category": x["category"], "steps": x.get("steps")}
            for x in sorted(labels, key=lambda x: x["state_id"])]


def level1() -> dict[str, Any]:
    """Level 1 held-out blocks per task, verbatim, plus outcome strips."""
    r, t = _load(L1_RESULTS), _load(L1_TAXONOMY)
    out: dict[str, Any] = {"label": r["label"], "freeze": r["freeze"], "git": r["git"],
                           "generated_utc": r["generated_utc"], "tasks": {}}
    for task, block in r["tasks"].items():
        strips = [{"family": e["family"], "seed": e["seed"], "eval_run_id": e["eval_run_id"],
                   "outcomes": _strip(e["labels"])} for e in t["tasks"][task]]
        out["tasks"][task] = {k: block[k] for k in (
            "status", "n_states", "available_families", "success", "censored_episode_length",
            "per_checkpoint", "taxonomy_by_family", "cost_latency", "engineering")}
        out["tasks"][task]["strips"] = strips
    return out


def rgb() -> dict[str, Any]:
    """The RGB study's held-out blocks, verbatim, plus outcome strips."""
    r, t = _load(RGB_RESULTS), _load(RGB_TAXONOMY)
    by: dict[tuple[str, int], list] = defaultdict(list)
    for e in t["episodes"]:
        by[(e["family"], int(e["seed"]))].append(e)
    keep = ("label", "task", "n_states", "rgb_freeze", "decision", "git", "generated_utc",
            "success", "per_checkpoint", "taxonomy_by_family", "level1_context", "note",
            "validation_status", "engineering")
    return {**{k: r[k] for k in keep},
            "strips": [{"family": f, "seed": s, "outcomes": _strip(v)}
                       for (f, s), v in sorted(by.items())]}


def _cube_xy(records_dir: str, state_id: str) -> list[float]:
    meta = json.loads((REPO_ROOT / records_dir / "episodes" / state_id / "record.json")
                      .read_text())
    i = meta["bodies"]["names"].index("cube_main")
    with np.load(REPO_ROOT / records_dir / "episodes" / state_id / "arrays.npz") as z:
        p = z["body_pos"][0, i]
    return [round(float(p[0]), 5), round(float(p[1]), 5), round(float(p[2]), 5)]


def level2(selection: dict[str, Any]) -> dict[str, Any]:
    """Level 2 blocks verbatim, outcome strips, the cube placement map and the erratum."""
    r, t = _load(L2_RESULTS), _load(L2_TAXONOMY)
    by: dict[tuple[str, str, int], list] = defaultdict(list)
    for e in t["episodes"]:
        by[(e["arm"], e["suite"], int(e["seed"]))].append(e)
    strips = [{"arm": a, "suite": su, "seed": s, "outcomes": _strip(v)}
              for (a, su, s), v in sorted(by.items())]
    cfg = yaml.safe_load((REPO_ROOT / "configs/eval/robustness.yaml").read_text())
    shift = cfg["ood"]["ood_cube_shift"]
    placement_map = {"training_square_half_m": shift["training_half_range_m"],
                     "center_xy": shift["center_xy"], "ood_radius_m": shift["radius_m"],
                     "source": source("configs/eval/robustness.yaml"), "states": []}
    pc_dir = selection["pixels"]["arms"]["pc_diffusion"]
    for suite in ("robustness_base", "ood_cube_shift"):
        sids = sorted({e["state_id"] for e in t["episodes"] if e["suite"] == suite})
        for sid in sids:
            succ = {arm: sum(1 for s in range(3) for e in by[(arm, suite, s)]
                             if e["state_id"] == sid and e["category"] == "success")
                    for arm in ("pc_diffusion", "rgb_diffusion")}
            placement_map["states"].append({
                "suite": suite, "state_id": sid,
                "cube_initial_pos_m": _cube_xy(pc_dir[suite]["dir"], sid),
                "successes_of_3_seeds": succ})
    keep = ("label", "task", "level2_freeze", "decision", "approval", "git", "generated_utc",
            "placement", "heldout", "taxonomy_by_arm_and_set", "selections", "privilege",
            "power", "note")
    return {**{k: r[k] for k in keep}, "training_generated": r["training"],
            "training_erratum": KI049_CORRECTION, "strips": strips,
            "placement_map": placement_map}


def build_findings(selection: dict[str, Any]) -> dict[str, Any]:
    """The findings document (see the module docstring)."""
    return {"schema": FINDINGS_SCHEMA,
            "label": "frozen results, copied verbatim from the generated analyses (no value "
                     "on the site is typed by hand)",
            "sources": [source(p) for p in (L1_RESULTS, L1_TAXONOMY, RGB_RESULTS, RGB_TAXONOMY,
                                            L2_RESULTS, L2_TAXONOMY, L2_ERRATA)],
            "notes": NOTES, "level1": level1(), "rgb": rgb(), "level2": level2(selection)}
