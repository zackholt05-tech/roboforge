"""Analyse the Level 2 study exactly as frozen (Level 2 freeze §8; RESEARCH_PROTOCOL §15.4).

Run once, on the host, after every placement and held-out evaluation exists:

    ROBOFORGE_EVIDENCE_SESSION=9 uv run --frozen python scripts/analyze_level2.py

**Reads:**
* the 12 placement reports ``reports/session9/level2_placement/<arm>/<run_id>__<suite>.json``;
* the 3 point-cloud held-out reports ``reports/session9/pc_heldout/<run_id>.json``;
* the 3 frozen RGB diffusion held-out reports ``reports/session8/rgb_heldout/<run_id>.json``,
  reused and hash-checked (:data:`roboforge.eval.pc_heldout.RGB_HELDOUT_REPORTS`);
* their episode records, for the failure taxonomy; the cost ledger; the run manifests.

**Writes** ``reports/session<N>/level2_results/``:
* ``results.json``:
  * per checkpoint and set, k/50 with a Wilson 95 % interval;
  * the placement bootstrap (§9; B = 10,000, seed 20261106). Keys ``<arm>:<set>``; each arm's
    two sets share one seed draw; state index *i* of ``ood_cube_shift`` is paired with index *i*
    of ``robustness_base``. Draw order: states, ``pc_diffusion`` seeds, ``rgb_diffusion`` seeds.
    The primary H5 difference, both drops and the difference of the drops (a contrast);
  * the held-out bootstraps of success and censored episode length (first-success step, else
    400), with the same settings, over the manifest order;
  * engineering status, failure categories, latency, point-cloud throughput, training wall
    time and billed cost, and the hash of every input.
* ``tables.md``: the same numbers as tables;
* ``taxonomy.json``: every episode's label.

Nothing is selected or rerun, and existing results are never overwritten. An arm with an
engineering problem is not bootstrapped; it is reported as engineering-incomplete.
Every result is labelled DP3-inspired.
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from roboforge.compute import cost_log
from roboforge.eval import pc_heldout as ph
from roboforge.eval import taxonomy as tx
from roboforge.eval.statistics import hierarchical_paired_bootstrap, wilson_interval
from roboforge.paths import REPO_ROOT, evidence_reports_dir, repo_relative
from roboforge.provenance import git_revision, sha256_file, write_json
from roboforge.sim import eval_states as es

PLAN = REPO_ROOT / "configs" / "experiments" / "level2_pointcloud.yaml"
ARMS = ("pc_diffusion", "rgb_diffusion")  # bootstrap draw order after the states
ARM_DIR = {"pc_diffusion": "pc", "rgb_diffusion": "rgb"}
BASE, OOD = "robustness_base", "ood_cube_shift"
HORIZON = 400
N_STATES = 50
LABEL = ("Level 2 study, DP3-inspired point clouds vs the frozen RGB diffusion arm, analysed "
         "once as frozen (Level 2 freeze §8, RESEARCH_PROTOCOL §15.4)")
COARSE = ("3 training seeds per arm: the seed-level resampling is coarse; every per-seed value "
          "is listed (RESEARCH_PROTOCOL §9)")
POWER = ("with 3 seeds x 50 starts, binomial noise alone gives about +/-8 points near 85 % "
         "success (an estimate); 'not separated' is not equivalence")


def _index(state_id: str) -> int:
    return int(state_id.rsplit("_", 1)[1])


def _selections(plan: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Per arm, the 3 selected checkpoints in seed order (pc: pinned artifacts; rgb: plan)."""
    pc = []
    for rid in sorted(ph.SELECTIONS, key=lambda r: int(r.split("seed")[1].split("_")[0])):
        sel = ph.load_selection(rid)
        pc.append({"seed": int(sel["seed"]), "run_id": rid, "step": int(sel["selected"]["step"]),
                   "checkpoint_sha256": sel["selected"]["sha256"]})
    rgb = [{"seed": int(r["seed"]), "run_id": r["run_id"], "step": int(r["step"]),
            "checkpoint_sha256": r["checkpoint_sha256"]}
           for r in sorted(plan["arms"]["rgb_diffusion"]["selections"], key=lambda r: r["seed"])]
    return {"pc_diffusion": pc, "rgb_diffusion": rgb}


def _check(s: dict[str, Any], p: Path, want: dict[str, Any], suite_sha: str,
           records_key: str) -> list[str]:
    """Hard checks of one report: anything failing means the input is not the frozen one."""
    checks = {
        "canonical": s["canonical_platform"] is True,
        "selected checkpoint": (s["checkpoint"]["step"] == want["step"] and
                                s["checkpoint"]["sha256"] == want["checkpoint_sha256"]),
        "suite manifest": s["suite_manifest_content_sha256"] == suite_sha,
        "records": sha256_file(REPO_ROOT / s["records_dir"] / "manifest.json") == s[records_key]}
    return [f"{repo_relative(p)}: fails {k}" for k, ok in checks.items() if not ok]


def _engineering(s: dict[str, Any]) -> list[str]:
    return [w for w, cond in (("incomplete", s["n_episodes"] != N_STATES),
                              ("verify_record failed", not s["verify_passed"]),
                              ("episode errors", s["n_errors"] > 0)) if cond]


def _vectors(s: dict[str, Any]) -> tuple[list[float], list[float]]:
    """Success and censored length per state index 0..49."""
    eps = sorted(s["episodes"], key=lambda e: _index(e["state_id"]))
    if [_index(e["state_id"]) for e in eps] != list(range(N_STATES)):
        raise SystemExit(f"{s['eval_run_id']}: states are not indices 0..{N_STATES - 1}")
    if s["episode_rules"]["horizon"] != HORIZON:
        raise SystemExit(f"{s['eval_run_id']}: horizon {s['episode_rules']['horizon']}")
    succ = [float(bool(e["success"])) for e in eps]
    length = [float(e["first_success_step"]) if e["success"] else float(HORIZON) for e in eps]
    return succ, length


def _row(arm: str, sel: dict[str, Any], suite: str, s: dict[str, Any], p: Path,
         problems: list[str]) -> dict[str, Any]:
    lo, hi = wilson_interval(s["successes"], s["n_episodes"])
    return {"arm": arm, "seed": sel["seed"], "train_run_id": sel["run_id"], "step": sel["step"],
            "checkpoint_sha256": sel["checkpoint_sha256"], "suite": suite,
            "eval_run_id": s["eval_run_id"], "successes": s["successes"], "n": s["n_episodes"],
            "wilson95": [lo, hi], "n_errors": s["n_errors"],
            "first_success_step": s["first_success_step"],
            "latency_ms_median_of_episode_medians": s["latency_ms_median_of_episode_medians"],
            "wall_s": s["wall_s"], "engineering_problems": problems,
            "report": repo_relative(p), "report_sha256": sha256_file(p)}


def _training(run_id: str, events: list[dict[str, Any]]) -> dict[str, Any]:
    man_p = REPO_ROOT / "runs" / run_id / "manifest.json"
    man = json.loads(man_p.read_text()) if man_p.exists() else {"segments": []}
    launches = {e["launch_id"] for e in events
                if e["event"] == "launch" and e.get("run_id") == run_id}
    billed = sum(float(e["billed_usd"]) for e in events
                 if e["event"] == "billing" and e.get("launch_id") in launches)
    wall = sum(float(g.get("train_wall_s") or 0) for g in man["segments"])
    steps = sum(int(g.get("end_step") or 0) - int(g.get("start_step") or g.get("resumed_from")
                                                    or 0) for g in man["segments"])
    return {"segments": len(man["segments"]), "train_wall_s": wall, "launches": len(launches),
            "billed_usd": round(billed, 4),
            "ms_per_step": round(1000 * wall / steps, 2) if steps > 0 else None}


def _placement(plan: dict[str, Any], sels: dict[str, list[dict[str, Any]]],
               labels: list[dict[str, Any]]) -> dict[str, Any]:
    root = evidence_reports_dir("level2_placement")
    shas = {BASE: plan["suites"]["placement_base"]["manifest_content_sha256"],
            OOD: plan["suites"]["placement_ood"]["manifest_content_sha256"]}
    for suite, sha in shas.items():
        if es.read_manifest("lift", suite)["content_sha256"] != sha:
            raise SystemExit(f"{suite}: manifest differs from the plan")
    rows, mats, eng = [], {}, {}
    for arm in ARMS:
        for suite in (BASE, OOD):
            m = []
            for sel in sels[arm]:
                p = root / ARM_DIR[arm] / f"{sel['run_id']}__{suite}.json"
                if not p.exists():
                    raise SystemExit(f"{repo_relative(p)} is missing: all 12 placement reports "
                                     "first")
                s = json.loads(p.read_text())
                bad = _check(s, p, sel, shas[suite], "episodes_sha256")
                if bad:
                    raise SystemExit("; ".join(bad))
                # placement reports keep their episode rows in the hash-checked records manifest
                rec = json.loads((REPO_ROOT / s["records_dir"] / "manifest.json").read_text())
                s = s | {"episodes": rec["episodes"]}
                problems = _engineering(s)
                eng.setdefault(arm, []).extend(f"{sel['run_id']} {suite}: {w}" for w in problems)
                rows.append(_row(arm, sel, suite, s, p, problems))
                m.append(_vectors(s)[0])
                labels.extend({"arm": arm, "suite": suite, "train_run_id": sel["run_id"],
                               "seed": sel["seed"], **x}
                              for x in tx.classify_records(REPO_ROOT / s["records_dir"]))
            mats[f"{arm}:{suite}"] = np.array(m)
    out: dict[str, Any] = {"evaluations": rows,
                           "engineering": {a: eng.get(a, []) for a in ARMS}}
    if any(eng.get(a) for a in ARMS):
        out["bootstrap"] = {"status": "not computed: engineering-incomplete arm"}
        return out
    comps = [(f"pc_diffusion:{OOD}", f"rgb_diffusion:{OOD}"),
             (f"pc_diffusion:{BASE}", f"pc_diffusion:{OOD}"),
             (f"rgb_diffusion:{BASE}", f"rgb_diffusion:{OOD}"),
             (f"pc_diffusion:{BASE}", f"rgb_diffusion:{BASE}")]
    contrasts = [("drop pc_diffusion - drop rgb_diffusion",
                  {f"pc_diffusion:{BASE}": 1, f"pc_diffusion:{OOD}": -1,
                   f"rgb_diffusion:{BASE}": -1, f"rgb_diffusion:{OOD}": 1})]
    order = [f"{a}:{s}" for a in ARMS for s in (BASE, OOD)]
    boot = hierarchical_paired_bootstrap(mats, comps, order=order,
                                         groups={k: k.split(":")[0] for k in order},
                                         contrasts=contrasts)
    kinds = ["PRIMARY H5: between-arm difference in OOD success",
             "drop (base - OOD), pc_diffusion", "drop (base - OOD), rgb_diffusion",
             "between-arm difference in base success (context)"]
    for d, k in zip(boot["differences"], kinds, strict=True):
        d["kind"] = k
    out["bootstrap"] = boot
    return out


def _heldout(sels: dict[str, list[dict[str, Any]]], labels: list[dict[str, Any]]
             ) -> dict[str, Any]:
    order = [s["state_id"] for s in es.read_manifest("lift", es.LOCKED_SUITE)["states"]]
    if es.read_manifest("lift", es.LOCKED_SUITE)["content_sha256"] != ph.HELDOUT_MANIFEST_SHA256:
        raise SystemExit("held-out manifest differs from the Level 2 freeze")
    rows, succ, length, eng = [], {}, {}, {}
    for arm in ARMS:
        ms, ml = [], []
        for sel in sels[arm]:
            if arm == "pc_diffusion":
                p = evidence_reports_dir("pc_heldout") / f"{sel['run_id']}.json"
            else:
                p = REPO_ROOT / ph.RGB_HELDOUT_DIR_REL / f"{sel['run_id']}.json"
                if sha256_file(p) != ph.RGB_HELDOUT_REPORTS[sel["run_id"]]:
                    raise SystemExit(f"{repo_relative(p)}: not the frozen RGB held-out report")
            if not p.exists():
                raise SystemExit(f"{repo_relative(p)} is missing: all 3 held-out reports first")
            s = json.loads(p.read_text())
            bad = _check(s, p, sel, ph.HELDOUT_MANIFEST_SHA256, "records_manifest_sha256")
            if s["suite"] != es.LOCKED_SUITE or s.get("rehearsal") is not False:
                bad.append(f"{repo_relative(p)}: not a held-out test report")
            if arm == "pc_diffusion" and (s["schema"] != ph.SCHEMA or
                                          s["gate"]["open"] is not True):
                bad.append(f"{repo_relative(p)}: not written through the open Level 2 gate")
            if bad:
                raise SystemExit("; ".join(bad))
            problems = _engineering(s)
            eng.setdefault(arm, []).extend(f"{sel['run_id']}: {w}" for w in problems)
            rows.append(_row(arm, sel, es.LOCKED_SUITE, s, p, problems))
            by = {e["state_id"]: e for e in s["episodes"]}
            ms.append([float(bool(by[i]["success"])) for i in order])
            ml.append([float(by[i]["first_success_step"]) if by[i]["success"] else float(HORIZON)
                       for i in order])
            if arm == "pc_diffusion":
                labels.extend({"arm": arm, "suite": "heldout", "train_run_id": sel["run_id"],
                               "seed": sel["seed"], **x}
                              for x in tx.classify_records(REPO_ROOT / s["records_dir"]))
        succ[arm], length[arm] = np.array(ms), np.array(ml)
    out: dict[str, Any] = {"evaluations": rows, "state_order": "held-out manifest order",
                           "engineering": {a: eng.get(a, []) for a in ARMS},
                           "rgb_reused": "the frozen RGB study's held-out reports, never rerun"}
    if any(eng.get(a) for a in ARMS):
        out["success"] = out["censored_episode_length"] = {
            "status": "not computed: engineering-incomplete arm"}
        return out
    comp = [("pc_diffusion", "rgb_diffusion")]
    out["success"] = hierarchical_paired_bootstrap(succ, comp, order=list(ARMS))
    out["censored_episode_length"] = hierarchical_paired_bootstrap(length, comp, order=list(ARMS))
    return out


def analyze() -> dict[str, Any]:
    """Build the frozen Level 2 analysis."""
    out_dir = evidence_reports_dir("level2_results")
    if (out_dir / "results.json").exists():
        raise SystemExit(f"{repo_relative(out_dir / 'results.json')} exists: analysed once")
    gate_open, gate_problems = ph.gate_status()
    if not gate_open:
        raise SystemExit("the Level 2 gate is shut: " + "; ".join(gate_problems))
    plan = yaml.safe_load(PLAN.read_text())
    sels = _selections(plan)
    labels: list[dict[str, Any]] = []
    placement = _placement(plan, sels, labels)
    heldout = _heldout(sels, labels)
    events = cost_log.read_events()
    res = {
        "schema": "roboforge.level2_results/1", "label": LABEL,
        "generated_utc": datetime.now(tz=UTC).isoformat(), "git": git_revision(),
        "level2_freeze": {"path": ph.LEVEL2_FREEZE_REL, "commit": ph.LEVEL2_FREEZE_COMMIT,
                          "sha256": sha256_file(REPO_ROOT / ph.LEVEL2_FREEZE_REL)},
        "decision": ph.DECISION_ID, "approval": "A-04", "task": "lift",
        "plan": {"path": repo_relative(PLAN), "sha256": sha256_file(PLAN)},
        "privilege": "sensor-like: depth + calibration only (D-083); not privileged geometry",
        "selections": sels, "placement": placement, "heldout": heldout,
        "note": COARSE, "power": POWER,
        "taxonomy_by_arm_and_set": {
            f"{a}:{s}": tx.counts([x for x in labels if x["arm"] == a and x["suite"] == s])
            for a in ARMS for s in (BASE, OOD, "heldout")
            if any(x["arm"] == a and x["suite"] == s for x in labels)},
        "taxonomy_sha256": tx.taxonomy_sha256(),
        "training": {r["run_id"]: _training(r["run_id"], events)
                     for r in sels["pc_diffusion"]}}
    write_json(out_dir / "taxonomy.json", {"taxonomy_sha256": tx.taxonomy_sha256(),
                                           "episodes": labels})
    write_json(out_dir / "results.json", res)
    (out_dir / "tables.md").write_text(_tables(res))
    return res


def _pct(x: float) -> str:
    return f"{100 * x:.1f}"


def _boot_lines(b: dict[str, Any], unit: str = "%") -> list[str]:
    if "families" not in b:
        return ["", f"Bootstrap: {b.get('status')}."]
    pct = unit == "%"
    f = _pct if pct else (lambda v: f"{v:.1f}")
    lines = ["", f"| Key | Estimate ({unit}) | 95 % interval | Per seed |", "|---|---|---|---|"]
    for k, v in b["families"].items():
        lines.append(f"| {k} | {f(v['estimate'])} | {f(v['ci95'][0])}–{f(v['ci95'][1])} | "
                     f"{', '.join(f(x) for x in v['per_seed'])} |")
    lines += ["", f"| Difference | Kind | Estimate ({'pp' if pct else unit}) | 95 % CI | "
                  "Outcome |", "|---|---|---|---|---|"]
    sc = 100 if pct else 1
    for d in b["differences"]:
        lines.append(f"| {d['a']} − {d['b']} | {d.get('kind', 'difference')} | "
                     f"{sc * d['difference']:+.1f} | [{sc * d['ci95'][0]:+.1f}, "
                     f"{sc * d['ci95'][1]:+.1f}] | **{d['separation']}** |")
    for c in b.get("contrasts", []):
        lines.append(f"| {c['name']} | difference in drop | {sc * c['estimate']:+.1f} | "
                     f"[{sc * c['ci95'][0]:+.1f}, {sc * c['ci95'][1]:+.1f}] | "
                     f"**{c['separation']}** |")
    return lines


def _tables(res: dict[str, Any]) -> str:
    lines = ["# Level 2 study: results (Lift, DP3-inspired, generated)", "",
             f"Freeze `{res['level2_freeze']['commit']}`, decision {res['decision']}, approval "
             f"{res['approval']}. {res['privilege']}. {res['note']}. Power: {res['power']}.", "",
             "## Placement (primary H5 on ood_cube_shift)", "",
             "| Arm | Seed | Step | Set | k/n | Wilson 95 % | Mean first-success step | "
             "Errors |", "|---|---|---|---|---|---|---|---|"]
    for c in res["placement"]["evaluations"]:
        lo, hi = c["wilson95"]
        lines.append(f"| {c['arm']} | {c['seed']} | {c['step']} | {c['suite']} | "
                     f"{c['successes']}/{c['n']} | {_pct(lo)}–{_pct(hi)} % | "
                     f"{c['first_success_step']['mean']} | {c['n_errors']} |")
    lines += _boot_lines(res["placement"]["bootstrap"])
    lines += ["", "## Held-out (secondary; RGB records reused, never rerun)", "",
              "| Arm | Seed | Step | k/n | Wilson 95 % | Mean first-success step | "
              "Latency ms | Errors |", "|---|---|---|---|---|---|---|---|"]
    for c in res["heldout"]["evaluations"]:
        lo, hi = c["wilson95"]
        lines.append(f"| {c['arm']} | {c['seed']} | {c['step']} | {c['successes']}/{c['n']} | "
                     f"{_pct(lo)}–{_pct(hi)} % | {c['first_success_step']['mean']} | "
                     f"{c['latency_ms_median_of_episode_medians']:.1f} | {c['n_errors']} |")
    lines += ["", "**Success**"] + _boot_lines(res["heldout"]["success"])
    lines += ["", "**Censored episode length (steps; failures = 400)**"]
    lines += _boot_lines(res["heldout"]["censored_episode_length"], unit="steps")
    tax = res["taxonomy_by_arm_and_set"]
    if tax:
        cats = list(next(iter(tax.values())))
        lines += ["", "## Failure taxonomy", "", "| Arm:set | " + " | ".join(cats) + " |",
                  "|---" * (1 + len(cats)) + "|"]
        for k, c in tax.items():
            lines.append(f"| {k} | " + " | ".join(str(c[x]) for x in cats) + " |")
    lines += ["", "## Training (point-cloud arm, L4)", "",
              "| Run | Segments | Train wall h | ms/step | Billed USD |", "|---|---|---|---|---|"]
    for rid, t in res["training"].items():
        lines.append(f"| {rid} | {t['segments']} | {t['train_wall_s'] / 3600:.2f} | "
                     f"{t['ms_per_step']} | {t['billed_usd']:.2f} |")
    return "\n".join(lines) + "\n"


def main() -> int:
    """CLI entrypoint (see the module docstring)."""
    res = analyze()
    b = res["placement"]["bootstrap"]
    print(json.dumps({"primary": b.get("differences", [None])[0] if "differences" in b else b,
                      "heldout_success": res["heldout"]["success"].get("differences")},
                     indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
