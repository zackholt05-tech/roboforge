"""The Observatory's v1.1 extension section: clip selection and findings (DECISIONS D-134).

Everything here reads **committed, hash-pinned evidence** only:

- X2 (Can MH): the 12 X2-c held-out summaries, each read only after its SHA-256 equals the
  value in X2's pinned ``results.json``;
- X1: the clamped BC summaries pinned in X1's ``results.json`` and the Session 7 robustness
  summaries pinned in the Session 7 analysis, whose own hash X1's file pins.

Suite names, state IDs and record directories come from those summaries; no file of this module
names a held-out suite. No simulator is started and nothing is stepped. The export of the
selected clips (``scripts/export_observatory_ext.py export``) re-derives each clip from its
recorded states on the canonical container, as Session 10 did for held-out records.

Selection rules (D-134, fixed before any export):

- **X2:** seed 0 diffusion against seed 0 of each BC family, on the same start, in the summary's
  state order: (a) first start where only diffusion succeeds; (b) first start where only the BC
  family succeeds; (c) first start where both succeed.
- **X1** (Lift ``ood_arm_offset``, seed 0): diffusion (Session 7), BC-MLP-chunk unclamped
  (Session 7) and BC-MLP-chunk clamped (X1) on the same start: (a) first start where diffusion
  succeeds and both BC-MLP-chunk arms fail; (b) first start where all three succeed; (c) first
  start where the clamped and unclamped outcomes differ.

A case with no matching start is recorded as ``None`` with a reason, never dropped.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from roboforge.addendum import sources as pinned
from roboforge.paths import REPO_ROOT

SELECTION_SCHEMA = "roboforge.observatory_ext_selection/1"
FINDINGS_SCHEMA = "roboforge.observatory_ext_findings/1"
BC = ("bc_mlp", "bc_rnn", "bc_mlp_chunk")
X1_TASK, X1_SET, SEED = "lift", "ood_arm_offset", 0
X2_LABEL = "X2 (Can MH): trained on the robomimic Can MH demonstrations"
RULES = {
    "x2": "D-134: seed 0 diffusion_unet vs seed 0 of each BC family on the same X2 (Can MH) "
          "held-out start, in summary state order: (a) first start only diffusion succeeds; "
          "(b) first start only the BC family succeeds; (c) first start both succeed",
    "x1": "D-134: Lift ood_arm_offset, seed 0; diffusion_unet (Session 7), bc_mlp_chunk "
          "unclamped (Session 7) and bc_mlp_chunk clamped (X1) on the same start: (a) first "
          "start diffusion succeeds and both bc_mlp_chunk arms fail; (b) first start all three "
          "succeed; (c) first start clamped and unclamped bc_mlp_chunk differ",
    "representativeness": "rule-selected cases, labelled as such; not representative samples",
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_pinned(rel: str, sha256: str, root: Path = REPO_ROOT) -> dict[str, Any]:
    """A committed JSON file, refused unless its bytes hash to ``sha256``."""
    p = root / rel
    got = _sha(p)
    if got != sha256:
        raise RuntimeError(f"{rel}: SHA-256 {got} differs from the pinned {sha256}")
    return json.loads(p.read_bytes())


def _outcomes(summary: dict[str, Any]) -> dict[str, bool]:
    return {e["state_id"]: bool(e["success"]) for e in summary["episodes"]}


def _arm(summary: dict[str, Any], summary_path: str, summary_sha: str, study: str,
         arm: str) -> dict[str, Any]:
    family = arm.split("+")[0]
    if summary.get("family", family) != family:
        raise AssertionError(f"{summary_path}: family {summary.get('family')} is not {family}")
    return {"study": study, "arm": arm, "task": summary["task"], "family": family,
            "seed": SEED, "suite": summary["suite"], "partition": summary["partition"],
            "eval_run_id": summary["eval_run_id"], "records_dir": summary["records_dir"],
            "summary": summary_path, "summary_sha256": summary_sha,
            "order": [e["state_id"] for e in summary["episodes"]],
            "outcomes": _outcomes(summary)}


def x2_arms(root: Path = REPO_ROOT) -> dict[str, dict[str, Any]]:
    """Seed-0 X2 evaluations of every family, read through X2's pinned results."""
    x2 = pinned.load("x2", root)
    out = {}
    for e in x2["evaluations"]:
        if e["seed"] == SEED:
            s = read_pinned(e["summary"], e["summary_sha256"], root)
            if s["suite"] != x2["suite"]["name"] or s["eval_run_id"] != e["eval_run_id"]:
                raise AssertionError(f"{e['summary']}: suite or run differs from results.json")
            out[e["family"]] = _arm(s, e["summary"], e["summary_sha256"], "x2", e["family"])
    return out


def x1_arms(root: Path = REPO_ROOT) -> dict[str, dict[str, Any]]:
    """The three X1 arms on Lift ``ood_arm_offset``, seed 0, through their pins."""
    x1 = pinned.load("x1", root)
    a7 = x1["inputs"]["session7_analysis"]
    s7 = read_pinned(a7["path"], a7["sha256"], root)
    out: dict[str, dict[str, Any]] = {}
    for e in s7["tasks"][X1_TASK]["h4"]["evaluations"]:
        if e["seed"] == SEED and e["suite"] == X1_SET and e["family"] in (
                "diffusion_unet", "bc_mlp_chunk"):
            s = read_pinned(e["summary"], e["summary_sha256"], root)
            out[e["family"]] = _arm(s, e["summary"], e["summary_sha256"], "x1", e["family"])
    for e in x1["tasks"][X1_TASK]["evaluations"]:
        if e["seed"] == SEED and e["suite"] == X1_SET and e["family"] == "bc_mlp_chunk":
            s = read_pinned(e["summary"], e["summary_sha256"], root)
            out["bc_mlp_chunk+clamp"] = _arm(s, e["summary"], e["summary_sha256"], "x1",
                                             "bc_mlp_chunk+clamp")
    if sorted(out) != ["bc_mlp_chunk", "bc_mlp_chunk+clamp", "diffusion_unet"]:
        raise AssertionError(f"X1 arms incomplete: {sorted(out)}")
    return out


def _first(order: list[str], pred: Any) -> str | None:
    return next((s for s in order if pred(s)), None)


def _clip_id(arm: dict[str, Any], state_id: str) -> str:
    return f"{arm['study']}-{arm['task']}-{arm['arm'].replace('+', '_')}-s{SEED}-{state_id}"


def build_selection(root: Path = REPO_ROOT) -> dict[str, Any]:
    """The extension clips and their roles; ``cases`` records every rule's result."""
    clips: dict[str, dict[str, Any]] = {}
    cases: list[dict[str, Any]] = []

    def add(arm: dict[str, Any], state_id: str, role: str) -> str:
        cid = _clip_id(arm, state_id)
        c = clips.setdefault(cid, {
            "id": cid, "study": arm["study"], "arm": arm["arm"], "task": arm["task"],
            "family": arm["family"], "seed": SEED, "state_id": state_id,
            "suite": arm["suite"], "partition": arm["partition"],
            "eval_run_id": arm["eval_run_id"], "records_dir": arm["records_dir"],
            "summary": arm["summary"], "summary_sha256": arm["summary_sha256"],
            "success": arm["outcomes"][state_id], "roles": [],
            "display_label": X2_LABEL if arm["study"] == "x2" else None})
        c["roles"].append(role)
        return cid

    x2 = x2_arms(root)
    d = x2["diffusion_unet"]
    for fam in BC:
        b = x2[fam]
        if b["order"] != d["order"]:
            raise AssertionError("X2 summaries list their starts in different orders")
        o, ob = d["outcomes"], b["outcomes"]
        rules = {"a_only_diffusion": lambda s, o=o, ob=ob: o[s] and not ob[s],
                 f"b_only_{fam}": lambda s, o=o, ob=ob: ob[s] and not o[s],
                 "c_both_succeed": lambda s, o=o, ob=ob: o[s] and ob[s]}
        for case, pred in rules.items():
            sid = _first(d["order"], pred)
            role = f"x2:{fam}:{case}"
            if sid is None:
                cases.append({"role": role, "state_id": None,
                              "reason": "no start in the X2 held-out summaries fits this case"})
                continue
            ids = [add(d, sid, role), add(b, sid, role)]
            cases.append({"role": role, "state_id": sid, "clips": ids})

    x1 = x1_arms(root)
    dd, cu, cc = x1["diffusion_unet"], x1["bc_mlp_chunk"], x1["bc_mlp_chunk+clamp"]
    if not (dd["order"] == cu["order"] == cc["order"]):
        raise AssertionError("X1 summaries list their starts in different orders")
    od, ou, oc = dd["outcomes"], cu["outcomes"], cc["outcomes"]
    rules1 = {"a_only_diffusion": lambda s: od[s] and not ou[s] and not oc[s],
              "b_all_succeed": lambda s: od[s] and ou[s] and oc[s],
              "c_clamp_changed_outcome": lambda s: ou[s] != oc[s]}
    for case, pred in rules1.items():
        sid = _first(dd["order"], pred)
        role = f"x1:{case}"
        if sid is None:
            cases.append({"role": role, "state_id": None,
                          "reason": "no start on the X1 set fits this case (seed 0)"})
            continue
        ids = [add(a, sid, role) for a in (dd, cu, cc)]
        cases.append({"role": role, "state_id": sid, "clips": ids})

    def first_diffusion(prefix: str) -> str | None:
        return next((c["clips"][0] for c in cases if c["role"].startswith(prefix)
                     and c.get("clips")), None)

    verification = [v for v in (first_diffusion("x2:bc_mlp:a_"), first_diffusion("x1:a_"))
                    if v]
    strips = {"x2": {fam: {"order": a["order"], "outcomes": [a["outcomes"][s]
                                                             for s in a["order"]]}
                     for fam, a in x2.items()},
              "x1": {arm: {"order": a["order"], "outcomes": [a["outcomes"][s]
                                                             for s in a["order"]]}
                     for arm, a in x1.items()}}
    return {"schema": SELECTION_SCHEMA, "decision": "D-134", "rules": RULES,
            "seed": SEED, "clips": sorted(clips.values(), key=lambda c: c["id"]),
            "cases": cases, "verification_clips": verification, "strips": strips}


def build_findings(root: Path = REPO_ROOT) -> dict[str, Any]:
    """Blocks copied verbatim from the three pinned results; nothing is computed."""
    src = pinned.load_all(root)
    x1, x3, x2 = src["x1"], src["x3"], src["x2"]
    x1_tasks = {t: {"sets": v["sets"], "n_states": v["n_states"],
                    "families": v["bootstrap"]["families"],
                    "x1_differences": v["bootstrap"]["x1_differences"],
                    "x1_contrasts": v["bootstrap"]["x1_contrasts"],
                    "frozen_contrasts": v["bootstrap"]["frozen_contrasts"],
                    "clamp_activity": v["clamp_activity"]}
                for t, v in x1["tasks"].items()}
    return {
        "schema": FINDINGS_SCHEMA, "decision": "D-134",
        "sources": [{"key": s.key, "path": s.path, "sha256": s.sha256, "label": s.label}
                    for s in pinned.SOURCES.values()],
        "x1": {"label": x1["label"], "decision": x1["decision"], "method": x1["method"],
               "readings": x1["readings"], "tasks": x1_tasks},
        "x3": {"label": x3["label"], "decision": x3["decision"], "suite": x3["suite"],
               "seeds": x3["primary"]["seeds"], "families": x3["primary"]["bootstrap"]["families"],
               "frozen_families": x3["frozen_families"], "contrasts": x3["primary"]["contrasts"],
               "engineering": x3["engineering"]},
        "x2": {"study": x2["study"], "label": x2["label"], "decisions": x2["decisions"],
               "suite": x2["suite"], "outcome": x2["design"]["outcome"],
               "families": x2["primary"]["bootstrap"]["families"],
               "frozen_families": x2["frozen_families"], "contrasts": x2["primary"]["contrasts"],
               "censored_episode_length": x2["primary"]["censored_episode_length"],
               "taxonomy": x2["taxonomy"], "engineering": x2["engineering"]},
    }
