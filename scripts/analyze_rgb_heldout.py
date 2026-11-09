"""Analyse the RGB held-out test exactly as frozen (RGB freeze §9; D-081). Run once, on the host.

    ROBOFORGE_EVIDENCE_SESSION=8 uv run --frozen python scripts/analyze_rgb_heldout.py

**Reads:**
* the 6 reports ``reports/session<N>/rgb_heldout/<run_id>.json``;
* their episode records under ``runs/rgb_heldout/``, for the failure taxonomy;
* the training segments in ``reports/session8/rgb_runs/``;
* the cost ledger.

**Writes** ``reports/session<N>/rgb_results/``:
* ``results.json`` holds:
  * per checkpoint, k/50 with a Wilson 95 % interval;
  * per family, the mean with its per-seed values;
  * the §9 hierarchical paired bootstrap of success: B = 10,000, seed 20261106, draw order
    :data:`roboforge.eval.rgb_heldout.FAMILY_ORDER`, states in the held-out manifest order;
  * the difference diffusion − BC: "not separated" when its interval contains 0;
  * the engineering status, failure categories, latency, training wall time and billed cost,
    and the hashes of every input.
* ``tables.md`` holds the same numbers as tables. The Level 1 low-dim Lift results appear
  there only as labelled context.
* ``taxonomy.json`` holds every episode's label and attributes.

Nothing is selected or rerun, and existing results are never overwritten. A family with an
engineering problem is not bootstrapped; it is reported as engineering-incomplete.
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from typing import Any

import numpy as np

from roboforge.compute import cost_log
from roboforge.eval import rgb_heldout as rh
from roboforge.eval import taxonomy as tx
from roboforge.eval.statistics import hierarchical_paired_bootstrap, wilson_interval
from roboforge.paths import REPO_ROOT, evidence_reports_dir, repo_relative
from roboforge.provenance import git_revision, sha256_file, write_json
from roboforge.sim import eval_states as es

COARSE = ("3 training seeds per family: the seed-level resampling is coarse; every per-seed "
          "value is listed (RESEARCH_PROTOCOL §9)")


def _load() -> list[dict[str, Any]]:
    out = []
    for run_id in sorted(rh.SELECTIONS):
        p = evidence_reports_dir("rgb_heldout") / f"{run_id}.json"
        if not p.exists():
            raise SystemExit(f"{repo_relative(p)} is missing: all 6 held-out reports first")
        s = json.loads(p.read_text())
        sel = rh.load_selection(run_id)
        checks = {
            "schema": s["schema"] == rh.SCHEMA, "not a rehearsal": s["rehearsal"] is False,
            "canonical": s["canonical_platform"] is True,
            "held-out suite": s["suite"] == es.LOCKED_SUITE,
            "manifest hash": s["suite_manifest_content_sha256"] == rh.HELDOUT_MANIFEST_SHA256,
            "selected checkpoint": (s["checkpoint"]["step"] == sel["selected"]["step"] and
                                    s["checkpoint"]["sha256"] == sel["selected"]["sha256"]),
            "gate was open": s["gate"]["open"] is True,
            "records": sha256_file(REPO_ROOT / s["records_dir"] / "manifest.json") ==
            s["records_manifest_sha256"]}
        bad = [k for k, ok in checks.items() if not ok]
        if bad:
            raise SystemExit(f"{repo_relative(p)}: fails {bad}")
        out.append({"path": repo_relative(p), "sha256": sha256_file(p), "summary": s})
    return out


def _engineering(r: dict[str, Any], n_states: int) -> list[str]:
    s = r["summary"]
    return [w for w, cond in (("incomplete", s["n_episodes"] != n_states),
                              ("verify_record failed", not s["verify_passed"]),
                              ("episode errors", s["n_errors"] > 0)) if cond]


def _training(run_id: str, events: list[dict[str, Any]]) -> dict[str, Any]:
    man = json.loads((REPO_ROOT / "reports" / "session8" / "rgb_runs" / run_id /
                      "manifest.json").read_text())
    launches = {e["launch_id"] for e in events
                if e["event"] == "launch" and e.get("run_id") == run_id}
    billed = sum(float(e["billed_usd"]) for e in events
                 if e["event"] == "billing" and e.get("launch_id") in launches)
    return {"segments": len(man["segments"]),
            "train_wall_s": sum(float(g.get("train_wall_s") or 0) for g in man["segments"]),
            "launches": len(launches), "billed_usd": round(billed, 4)}


def _level1_context() -> dict[str, Any] | None:
    p = REPO_ROOT / "reports" / "session7" / "results" / "results.json"
    if not p.exists():
        return None
    fam = json.loads(p.read_text())["tasks"]["lift"]["success"]["families"]
    return {"label": "CONTEXT ONLY: Level 1 low-dim policies, same 50 Lift held-out starts; "
                     "different observations, not a declared comparison",
            "path": repo_relative(p), "sha256": sha256_file(p),
            "families": {k: {"estimate": v["estimate"], "ci95": v["ci95"]}
                         for k, v in fam.items() if k in ("bc_mlp_chunk", "diffusion_unet")}}


def analyze() -> dict[str, Any]:
    """Build the frozen analysis from the 6 held-out reports."""
    out_dir = evidence_reports_dir("rgb_results")
    if (out_dir / "results.json").exists():
        raise SystemExit(f"{repo_relative(out_dir / 'results.json')} exists: analysed once")
    loaded = _load()
    order = [s["state_id"] for s in es.read_manifest(rh.TASK, es.LOCKED_SUITE)["states"]]
    events = cost_log.read_events()
    per_ckpt, mats, eng, labels = [], {f: {} for f in rh.FAMILY_ORDER}, {}, []
    for r in loaded:
        s = r["summary"]
        problems = _engineering(r, len(order))
        eng.setdefault(s["family"], []).extend(f"{s['train_run_id']}: {w}" for w in problems)
        by_state = {e["state_id"]: e for e in s["episodes"]}
        mats[s["family"]][int(s["seed"])] = [float(bool(by_state[i]["success"]))
                                             if i in by_state else np.nan for i in order]
        lab = tx.classify_records(REPO_ROOT / s["records_dir"])
        labels.extend({"train_run_id": s["train_run_id"], "family": s["family"],
                       "seed": s["seed"], **x} for x in lab)
        lo, hi = wilson_interval(s["successes"], s["n_episodes"])
        per_ckpt.append({
            "family": s["family"], "seed": s["seed"], "train_run_id": s["train_run_id"],
            "step": s["checkpoint"]["step"], "checkpoint_sha256": s["checkpoint"]["sha256"],
            "eval_run_id": s["eval_run_id"], "successes": s["successes"],
            "n": s["n_episodes"], "wilson95": [lo, hi], "n_errors": s["n_errors"],
            "first_success_step": s["first_success_step"], "stop_reasons": s["stop_reasons"],
            "latency_ms_median_of_episode_medians": s["latency_ms_median_of_episode_medians"],
            "eval_wall_s": s["wall_s"], "engineering_problems": problems,
            "training": _training(s["train_run_id"], events), "report": r["path"],
            "report_sha256": r["sha256"]})
    complete = [f for f in rh.FAMILY_ORDER if not eng.get(f) and sorted(mats[f]) == [0, 1, 2]]
    boot = None
    if complete == list(rh.FAMILY_ORDER):
        outcomes = {f: np.array([mats[f][k] for k in (0, 1, 2)]) for f in rh.FAMILY_ORDER}
        boot = hierarchical_paired_bootstrap(outcomes, [rh.COMPARISON],
                                             order=list(rh.FAMILY_ORDER))
    fam_tax = {f: tx.counts([x for x in labels if x["family"] == f]) for f in rh.FAMILY_ORDER}
    res = {
        "schema": "roboforge.rgb_heldout_results/1",
        "label": "RGB study held-out test, analysed once as frozen (RGB freeze §9, D-081)",
        "generated_utc": datetime.now(tz=UTC).isoformat(), "git": git_revision(),
        "rgb_freeze": {"path": rh.RGB_FREEZE_REL, "commit": rh.RGB_FREEZE_COMMIT,
                       "sha256": sha256_file(REPO_ROOT / rh.RGB_FREEZE_REL)},
        "decision": rh.DECISION_ID, "task": rh.TASK,
        "heldout_manifest_sha256": rh.HELDOUT_MANIFEST_SHA256, "n_states": len(order),
        "state_order": "held-out manifest order",
        "engineering": {"status": "complete" if not any(eng.values()) else "incomplete",
                        "problems": eng,
                        "engineering_reruns": [r["summary"]["engineering_rerun"] for r in loaded
                                               if r["summary"]["engineering_rerun"]]},
        "success": boot if boot is not None else
        {"status": "not computed: engineering-incomplete family", "families": complete},
        "note": COARSE, "per_checkpoint": per_ckpt, "taxonomy_by_family": fam_tax,
        "taxonomy_sha256": tx.taxonomy_sha256(),
        "validation_status": "every run reached 50/50 on validation (E8-11); no family "
                             "failed validation",
        "level1_context": _level1_context()}
    write_json(out_dir / "taxonomy.json", {"taxonomy_sha256": tx.taxonomy_sha256(),
                                           "episodes": labels})
    write_json(out_dir / "results.json", res)
    (out_dir / "tables.md").write_text(_tables(res))
    return res


def _pct(x: float) -> str:
    return f"{100 * x:.1f}"


def _tables(res: dict[str, Any]) -> str:
    lines = ["# RGB study: held-out results (Lift, generated)", "",
             f"Freeze `{res['rgb_freeze']['commit']}`, decision {res['decision']}, "
             f"engineering {res['engineering']['status']}. {res['note']}.", "",
             "| Family | Seed | Step | k/n | Wilson 95 % | Mean first-success step | "
             "Latency ms (median) | Errors |", "|---|---|---|---|---|---|---|---|"]
    for c in res["per_checkpoint"]:
        lo, hi = c["wilson95"]
        lines.append(f"| {c['family']} | {c['seed']} | {c['step']} | {c['successes']}/{c['n']}"
                     f" | {_pct(lo)}–{_pct(hi)} % | {c['first_success_step']['mean']} | "
                     f"{c['latency_ms_median_of_episode_medians']:.1f} | {c['n_errors']} |")
    s = res["success"]
    if "families" in s and isinstance(s["families"], dict):
        lines += ["", "| Family | Success | 95 % interval | Per seed |", "|---|---|---|---|"]
        for f, v in s["families"].items():
            lines.append(f"| {f} | {_pct(v['estimate'])} % | {_pct(v['ci95'][0])}–"
                         f"{_pct(v['ci95'][1])} % | {', '.join(_pct(x) for x in v['per_seed'])} |")
        for d in s["differences"]:
            lines += ["", f"**{d['a']} − {d['b']}:** {100 * d['difference']:+.1f} points, "
                          f"95 % interval [{100 * d['ci95'][0]:+.1f}, {100 * d['ci95'][1]:+.1f}]"
                          f": **{d['separation']}**."]
    else:
        lines += ["", f"Bootstrap: {s['status']}."]
    lines += ["", "| Family | " + " | ".join(next(iter(res["taxonomy_by_family"].values())))
              + " |", "|---" * (1 + len(next(iter(res["taxonomy_by_family"].values())))) + "|"]
    for f, c in res["taxonomy_by_family"].items():
        lines.append(f"| {f} | " + " | ".join(str(v) for v in c.values()) + " |")
    lines += ["", "| Run | Segments | Train wall h | Billed USD |", "|---|---|---|---|"]
    for c in res["per_checkpoint"]:
        t = c["training"]
        lines.append(f"| {c['train_run_id']} | {t['segments']} | "
                     f"{t['train_wall_s'] / 3600:.2f} | {t['billed_usd']:.2f} |")
    ctx = res["level1_context"]
    if ctx:
        lines += ["", f"*{ctx['label']}:* " + "; ".join(
            f"{k} {_pct(v['estimate'])} % [{_pct(v['ci95'][0])}, {_pct(v['ci95'][1])}]"
            for k, v in ctx["families"].items()) + "."]
    return "\n".join(lines) + "\n"


def main() -> int:
    """CLI entrypoint (see the module docstring)."""
    res = analyze()
    s = res["success"]
    print(json.dumps({"engineering": res["engineering"]["status"],
                      "differences": s.get("differences"),
                      "families": {f: v["estimate"] for f, v in s["families"].items()}
                      if isinstance(s.get("families"), dict) else s}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
