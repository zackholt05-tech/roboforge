"""Observatory episode selection by the rules fixed in DECISIONS D-089 (Session 10).

Everything here reads **frozen** evidence only: the Level 1 and Level 2 ``results.json`` /
``taxonomy.json`` files and the evaluation manifests under ``runs/``. Nothing is rolled out,
no simulator is started and no held-out start state is read (the per-episode outcome labels
are enough). The rules, in D-089's numbering:

1. **Lab and failure gallery (Level 1 held-out):** the Level 1 freeze's example rule. The
   frozen analysis already applied it (``results.json`` ``tasks.<task>.examples``); this module
   recomputes it from the taxonomy labels and refuses to continue if the two disagree.
   ``random_uniform`` (the H1 baseline) is added by the same rule.
2. **Hero:** rule 1's first success of Lift ``diffusion_unet``.
3. **Head-to-head:** seed 0 of ``diffusion_unet`` against seed 0 of ``bc_mlp`` (the H2 pair) on
   the same held-out start: (a) first state both succeed, (b) first state only diffusion
   succeeds, (c) first state only BC-MLP succeeds.
4. **Denoising:** the frozen selected diffusion checkpoints' validation evaluations: first
   success and first failure by sorted ``state_id``.
5. **Beyond Pixels:** Level 2 placement records, point cloud seed 0 against RGB seed 0 on
   ``ood_cube_shift`` ((a) pc fails and RGB succeeds, (b) both succeed, (c) pc succeeds and
   RGB fails) and both arms' first ``robustness_base`` state.

A missing case is recorded as ``None`` with a reason, never silently dropped.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from roboforge.paths import REPO_ROOT, repo_relative
from roboforge.provenance import sha256_file
from roboforge.sim import eval_states as es

SELECTION_SCHEMA = "roboforge.observatory_selection/1"
L1_RESULTS = "reports/session7/results/results.json"
L1_TAXONOMY = "reports/session7/results/taxonomy.json"
L2_RESULTS = "reports/session9/level2_results/results.json"
L2_TAXONOMY = "reports/session9/level2_results/taxonomy.json"
FAMILIES = ("bc_mlp", "bc_rnn", "bc_mlp_chunk", "diffusion_unet", "random_uniform")
CATEGORIES = ("engineering_error", "gripper_never_closed", "no_grasp_drift", "no_grasp_near",
              "timeout_holding", "in_goal_bin_not_success", "dropped")
DENOISE_RUNS = {  # D-089 rule 4: frozen selected checkpoints (Level 1 freeze tables)
    "lift": ("s6_diffusion_unet_lr3e4_lift_seed0_2f635398", 100000),
    "can": ("s7_diffusion_unet_cos_can_seed0_99251da3", 65000),
}
PLATFORM_ID = "linux-aarch64-container-1f685638f28c"
RULES = {
    "gallery": "Level 1 freeze §4 example rule: per task and family, among the frozen "
               "checkpoints ordered by seed, the first success and the first failure (sorted "
               "state_id) of the lowest seed that has one; plus, per failure category, the "
               "first episode in (family order, seed, state_id)",
    "hero": "D-089 rule 2: the gallery's first success of Lift diffusion_unet",
    "h2h": "D-089 rule 3: seed 0 diffusion_unet vs seed 0 bc_mlp on the same held-out start: "
           "(a) first state_id both succeed; (b) first only diffusion succeeds; (c) first only "
           "bc_mlp succeeds",
    "denoise": "D-089 rule 4: validation evaluation of the frozen selected diffusion "
               "checkpoint; first success and first failure by sorted state_id",
    "pixels": "D-089 rule 5: Level 2 placement, pc_diffusion seed 0 vs rgb_diffusion seed 0 on "
              "ood_cube_shift: (a) pc fails and RGB succeeds; (b) both succeed; (c) pc succeeds "
              "and RGB fails; plus both arms' first robustness_base state",
}


def _load(rel: str) -> Any:
    return json.loads((REPO_ROOT / rel).read_text())


def heldout_partition() -> str:
    """Partition label of the locked held-out suite, read from the suite config.

    The research-integrity lock test allows only the suite module to name that suite, so the
    label comes from ``eval_states`` rather than a literal here.
    """
    return str(es.suites_config()["suites"][es.LOCKED_SUITE]["partition"])

def source(rel: str) -> dict[str, str]:
    """``{"path", "sha256"}`` of a committed evidence file."""
    return {"path": rel, "sha256": sha256_file(REPO_ROOT / rel)}


def _labels_by_eval(taxonomy: dict[str, Any], task: str) -> dict[tuple[str, int], dict]:
    return {(e["family"], int(e["seed"])): e for e in taxonomy["tasks"][task]}


def _ok(label: dict[str, Any]) -> bool:
    return label["category"] == "success"


def gallery(task: str, taxonomy: dict[str, Any]) -> dict[str, Any]:
    """Rule 1 recomputed from the taxonomy labels (incl. ``random_uniform``)."""
    evals = _labels_by_eval(taxonomy, task)
    by_family: dict[str, dict[str, Any]] = {}
    for fam in FAMILIES:
        seeds = sorted(s for f, s in evals if f == fam)
        picks: dict[str, Any] = {"first_success": None, "first_failure": None}
        for kind, want in (("first_success", True), ("first_failure", False)):
            for s in seeds:
                e = evals[(fam, s)]
                hit = next((lab for lab in sorted(e["labels"], key=lambda x: x["state_id"])
                            if _ok(lab) == want), None)
                if hit is not None:
                    picks[kind] = {"family": fam, "seed": s, "eval_run_id": e["eval_run_id"],
                                   "state_id": hit["state_id"], "category": hit["category"]}
                    break
        by_family[fam] = picks
    by_category: dict[str, Any] = {}
    order = [(f, s) for f in FAMILIES for s in sorted(s for ff, s in evals if ff == f)]
    for cat in CATEGORIES:
        by_category[cat] = None
        for f, s in order:
            e = evals[(f, s)]
            hit = next((lab for lab in sorted(e["labels"], key=lambda x: x["state_id"])
                        if lab["category"] == cat), None)
            if hit is not None:
                by_category[cat] = {"family": f, "seed": s, "eval_run_id": e["eval_run_id"],
                                    "state_id": hit["state_id"], "category": cat}
                break
    return {"by_family": by_family, "by_category": by_category}


def check_against_frozen(task: str, mine: dict[str, Any], frozen: dict[str, Any]) -> None:
    """Raise unless rule 1 reproduces the frozen analysis's examples for the 4 learned families.

    The frozen analysis's ``by_category`` ran over the learned families only, so a category
    whose first episode is ``random_uniform``'s is compared after removing that family."""
    for fam, picks in frozen["by_family"].items():
        if mine["by_family"][fam] != picks:
            raise AssertionError(f"{task}/{fam}: recomputed examples differ from the frozen ones")
    for cat, pick in frozen["by_category"].items():
        mp = mine["by_category"][cat]
        if mp is not None and mp["family"] == "random_uniform":
            continue
        if mp != pick:
            raise AssertionError(f"{task}/{cat}: recomputed category example differs")


def head_to_head(task: str, taxonomy: dict[str, Any]) -> dict[str, Any]:
    """Rule 3 on one task (seed 0 of each family, sorted state_id)."""
    evals = _labels_by_eval(taxonomy, task)
    a, b = evals[("diffusion_unet", 0)], evals[("bc_mlp", 0)]
    la = {x["state_id"]: x for x in a["labels"]}
    lb = {x["state_id"]: x for x in b["labels"]}
    states = sorted(set(la) & set(lb))
    cases = {"a_both_succeed": (True, True), "b_only_diffusion": (True, False),
             "c_only_bc_mlp": (False, True)}
    out: dict[str, Any] = {"pair": ["diffusion_unet", "bc_mlp"], "seed": 0,
                           "eval_run_ids": {"diffusion_unet": a["eval_run_id"],
                                            "bc_mlp": b["eval_run_id"]},
                           "n_states": len(states)}
    for name, (wa, wb) in cases.items():
        sid = next((s for s in states if _ok(la[s]) == wa and _ok(lb[s]) == wb), None)
        out[name] = None if sid is None else {
            "state_id": sid, "categories": {"diffusion_unet": la[sid]["category"],
                                            "bc_mlp": lb[sid]["category"]},
            "kind": "representative" if name.startswith("a_") else "case study"}
    return out


def _l2_eval_dir(run_id: str, suite: str, step: int) -> Path:
    hits = sorted((REPO_ROOT / "runs" / "level2_eval").glob(
        f"{run_id}__{suite}_step{step:07d}__{PLATFORM_ID}__ps0r0"))
    if len(hits) != 1:
        raise FileNotFoundError(f"placement records of {run_id} {suite} step {step}: {hits}")
    return hits[0]


def pixels(l2_results: dict[str, Any], l2_taxonomy: dict[str, Any]) -> dict[str, Any]:
    """Rule 5: Level 2 placement pairs (seed 0 of each arm)."""
    sel = {arm: next(x for x in rows if int(x["seed"]) == 0)
           for arm, rows in l2_results["selections"].items()}
    lab: dict[tuple[str, str], dict[str, dict]] = {}
    for e in l2_taxonomy["episodes"]:
        if int(e["seed"]) == 0 and e["suite"] in ("ood_cube_shift", "robustness_base"):
            lab.setdefault((e["arm"], e["suite"]), {})[e["state_id"]] = e
    pc, rgb = lab[("pc_diffusion", "ood_cube_shift")], lab[("rgb_diffusion", "ood_cube_shift")]
    states = sorted(set(pc) & set(rgb))
    out: dict[str, Any] = {"arms": {arm: {"run_id": s["run_id"], "step": int(s["step"]),
                                          "checkpoint_sha256": s["checkpoint_sha256"]}
                                    for arm, s in sel.items()}}
    for arm, s in sel.items():
        for suite in ("ood_cube_shift", "robustness_base"):
            d = _l2_eval_dir(s["run_id"], suite, int(s["step"]))
            out["arms"][arm][suite] = {"eval_run_id": d.name, "dir": repo_relative(d)}
    cases = {"a_pc_fails_rgb_succeeds": (False, True), "b_both_succeed": (True, True),
             "c_pc_succeeds_rgb_fails": (True, False)}
    for name, (wp, wr) in cases.items():
        sid = next((x for x in states if _ok(pc[x]) == wp and _ok(rgb[x]) == wr), None)
        out[name] = None if sid is None else {
            "state_id": sid, "suite": "ood_cube_shift",
            "categories": {"pc_diffusion": pc[sid]["category"],
                           "rgb_diffusion": rgb[sid]["category"]},
            "kind": "case study" if name.startswith(("a_", "c_")) else "representative"}
    base_states = sorted(set(lab[("pc_diffusion", "robustness_base")])
                         & set(lab[("rgb_diffusion", "robustness_base")]))
    sid = base_states[0]
    out["base_first"] = {"state_id": sid, "suite": "robustness_base",
                         "categories": {a: lab[(a, "robustness_base")][sid]["category"]
                                        for a in ("pc_diffusion", "rgb_diffusion")},
                         "kind": "representative"}
    return out


def denoise_episodes(task: str) -> dict[str, Any]:
    """Rule 4: the validation evaluation of the frozen diffusion checkpoint of ``task``."""
    run, step = DENOISE_RUNS[task]
    eid = f"{run}__val_step{step:07d}__{PLATFORM_ID}__ps0r0"
    man_path = REPO_ROOT / "runs" / eid / "manifest.json"
    man = json.loads(man_path.read_text())
    if man["partition"] != "eval_validation":
        raise ValueError(f"{eid}: partition {man['partition']} is not validation")
    eps = sorted(man["episodes"], key=lambda e: e["state_id"])
    picks = {}
    for kind, want in (("first_success", True), ("first_failure", False)):
        hit = next((e for e in eps if bool(e["success"]) == want), None)
        picks[kind] = None if hit is None else {"state_id": hit["state_id"],
                                                "success": bool(hit["success"])}
    return {"eval_run_id": eid, "train_run_id": run, "step": step,
            "manifest_sha256": sha256_file(man_path),
            "checkpoint": man["checkpoint"], **picks}


def clip(kind: str, task: str, family: str, seed: int, eval_run_id: str, records_dir: str,
         state_id: str, partition: str, roles: list[str]) -> dict[str, Any]:
    """One clip request (merged later by ``id``)."""
    return {"id": f"{task}-{family}-s{seed}-{state_id}", "kind": kind, "task": task,
            "family": family, "seed": seed, "eval_run_id": eval_run_id,
            "records_dir": records_dir, "state_id": state_id, "partition": partition,
            "roles": roles}


def _merge(clips: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for c in clips:
        if c["id"] in out:
            out[c["id"]]["roles"] = sorted(set(out[c["id"]]["roles"]) | set(c["roles"]))
        else:
            out[c["id"]] = {**c, "roles": sorted(c["roles"])}
    return sorted(out.values(), key=lambda c: c["id"])


def _records_dir(eval_run_id: str) -> str:
    man = json.loads((REPO_ROOT / "runs" / eval_run_id / "manifest.json").read_text())
    return man["records_dir"]


def build_selection() -> dict[str, Any]:  # noqa: PLR0912 - one pass per rule
    """Apply D-089 rules 1-5; return the selection document (see the module docstring)."""
    l1r, l1t = _load(L1_RESULTS), _load(L1_TAXONOMY)
    l2r, l2t = _load(L2_RESULTS), _load(L2_TAXONOMY)
    clips: list[dict[str, Any]] = []
    doc: dict[str, Any] = {"schema": SELECTION_SCHEMA, "rules": RULES,
                           "sources": [source(p) for p in (L1_RESULTS, L1_TAXONOMY,
                                                           L2_RESULTS, L2_TAXONOMY)],
                           "gallery": {}, "head_to_head": {}, "denoise": {}}
    for task in ("lift", "can"):
        g = gallery(task, l1t)
        check_against_frozen(task, g, l1r["tasks"][task]["examples"])
        doc["gallery"][task] = g
        picks = [(f"gallery:{k}", p) for fam in FAMILIES
                 for k, p in g["by_family"][fam].items() if p]
        picks += [(f"gallery:category:{c}", p) for c, p in g["by_category"].items() if p]
        for role, p in picks:
            roles = [role, "lab"]
            if (task, p["family"], role) == ("lift", "diffusion_unet", "gallery:first_success"):
                roles.append("hero")
            clips.append(clip("level1_heldout", task, p["family"], p["seed"], p["eval_run_id"],
                              _records_dir(p["eval_run_id"]), p["state_id"],
                              heldout_partition(), roles))
        h = head_to_head(task, l1t)
        doc["head_to_head"][task] = h
        for case in ("a_both_succeed", "b_only_diffusion", "c_only_bc_mlp"):
            if h[case] is None:
                continue
            for fam in h["pair"]:
                eid = h["eval_run_ids"][fam]
                clips.append(clip("level1_heldout", task, fam, 0, eid, _records_dir(eid),
                                  h[case]["state_id"], heldout_partition(),
                                  [f"h2h:{task}:{case}"]))
        d = denoise_episodes(task)
        doc["denoise"][task] = d
        for kind in ("first_success", "first_failure"):
            if d[kind]:
                clips.append(clip("validation_denoising", task, "diffusion_unet", 0,
                                  d["eval_run_id"], _records_dir(d["eval_run_id"]),
                                  d[kind]["state_id"], "eval_validation",
                                  [f"denoise:{kind}"]))
    px = pixels(l2r, l2t)
    doc["pixels"] = px
    for case in ("a_pc_fails_rgb_succeeds", "b_both_succeed", "c_pc_succeeds_rgb_fails",
                 "base_first"):
        if px[case] is None:
            continue
        for arm in ("pc_diffusion", "rgb_diffusion"):
            info = px["arms"][arm][px[case]["suite"]]
            clips.append(clip("level2_placement", "lift", arm, 0, info["eval_run_id"],
                              info["dir"], px[case]["state_id"], "eval_robustness",
                              [f"pixels:{case}"]))
    doc["clips"] = _merge(clips)
    hero = [c["id"] for c in doc["clips"] if "hero" in c["roles"]]
    can_b = [c["id"] for c in doc["clips"] if "h2h:can:b_only_diffusion" in c["roles"]
             and c["family"] == "diffusion_unet"]
    l2_a = [c["id"] for c in doc["clips"] if "pixels:a_pc_fails_rgb_succeeds" in c["roles"]
            and c["family"] == "pc_diffusion"]
    doc["verification_clips"] = hero + can_b + l2_a  # D-089 rule 6
    return doc
