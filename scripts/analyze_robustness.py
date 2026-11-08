"""Analyse the preregistered robustness (H4) and ablation (A1-A3) evaluations (§14.3-14.5).

    ROBOFORGE_EVIDENCE_SESSION=7 uv run --frozen python scripts/analyze_robustness.py \
        --config lift=reports/session6/lift_test_configuration.json \
        --config can=reports/session7/can_test_configuration.json \
        [--a3-summary reports/session7/suite/ablation_a3_can.json]

Validation regime only (fresh robustness starts; nothing here selects). Reads the summaries
in ``reports/session<N>/robustness/evals/`` and their episode records, and writes
``reports/session<N>/robustness/``: ``analysis.json``, generated ``tables.md`` and
``taxonomy.json``.

* **H4** per task: success per family on every set (mean over the 3 seeds, per-seed values);
  the paired drop base − OOD per family; between-family differences in OOD success and in
  drop for the declared pairs (H2, H2-c, H3). One §9 bootstrap per task (B = 10,000, seed
  20261106): the 50 base indices are drawn once and shared by every family and set (OOD state
  ``i`` is paired with base state ``i``); each family's seeds are drawn once and shared by its
  sets (seed groups), so a drop is paired by seed and state.
* **A1 / A2** per task (diffusion): primary (EMA, DDIM-10) − variant on ``robustness_base``,
  paired by seed and state. Bias disclosure: the checkpoints were selected with the primary
  arm.
* **A3** (Can): T_o = 2 (the main diffusion runs) − T_o = 1 (the A3 runs) on
  ``robustness_base``; different training runs, so seeds are drawn independently per arm.
* The failure taxonomy (§14.5) labels every episode; counts per evaluation and per family/set.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from roboforge.eval import taxonomy as tx
from roboforge.eval.statistics import hierarchical_paired_bootstrap, wilson_interval
from roboforge.paths import REPO_ROOT, evidence_reports_dir, repo_relative
from roboforge.provenance import git_revision, sha256_file
from roboforge.sim import robustness as rb

FAMILIES = ("bc_mlp", "bc_rnn", "bc_mlp_chunk", "diffusion_unet")
PAIRS = {"H2": [("diffusion_unet", "bc_mlp"), ("diffusion_unet", "bc_rnn")],
         "H2c": [("diffusion_unet", "bc_mlp_chunk")], "H3": [("bc_rnn", "bc_mlp")]}
VARIANTS = {"A1": "trained-weights", "A2": "ddpm100"}


def _index(state_id: str) -> int:
    return int(state_id.rsplit("_", 1)[1])


class Evals:
    """Every robustness summary, by (train_run_id, step, suite, variant)."""

    def __init__(self, directory: Path) -> None:
        self.dir = directory
        self.by_key: dict[tuple, tuple[dict[str, Any], Path]] = {}
        for p in sorted(directory.glob("*.json")):
            s = json.loads(p.read_text())
            key = (s["train_run_id"], int(s["checkpoint"]["step"]), s["suite"],
                   s.get("inference_variant"))
            if key in self.by_key:
                raise ValueError(f"two summaries for {key}")
            if not s["canonical_platform"]:
                raise ValueError(f"{p}: not canonical")
            self.by_key[key] = (s, p)
        self.labels: dict[str, list[dict[str, Any]]] = {}

    def get(self, run: str, step: int, suite: str, variant: str | None = None):
        return self.by_key.get((run, int(step), suite, variant))

    def row(self, s: dict[str, Any], p: Path) -> dict[str, Any]:
        labs = self.labels.get(s["eval_run_id"])
        if labs is None:
            labs = tx.classify_records(REPO_ROOT / s["records_dir"])
            if sum(x["category"] == "success" for x in labs) != s["successes"]:
                raise ValueError(f"{p}: taxonomy and summary disagree on successes")
            self.labels[s["eval_run_id"]] = labs
        lo, hi = wilson_interval(s["successes"], s["n_episodes"])
        return {"eval_run_id": s["eval_run_id"], "summary": repo_relative(p),
                "summary_sha256": sha256_file(p), "suite": s["suite"],
                "variant": s.get("inference_variant"), "successes": s["successes"],
                "n_episodes": s["n_episodes"], "wilson95": [lo, hi],
                "first_success_step_mean": s["first_success_step"]["mean"],
                "n_errors": s["n_errors"], "verify_passed": s["verify_passed"],
                "motion_clip_fraction": float(np.mean(s["clip_fraction_per_dim"][:-1])),
                "gripper_clip_fraction": float(s["clip_fraction_per_dim"][-1]),
                "wall_s": s["wall_s"], "taxonomy": tx.counts(labs)}


def _vector(s: dict[str, Any], n: int) -> tuple[list[int], list[float]]:
    eps = sorted(s["episodes"], key=lambda e: _index(e["state_id"]))
    idx = [_index(e["state_id"]) for e in eps]
    if idx != list(range(n)):
        raise ValueError(f"{s['eval_run_id']}: states are not indices 0..{n - 1}")
    return idx, [float(e["success"]) for e in eps]


def _checkpoints(config: Path) -> dict[str, list[dict[str, Any]]]:
    rec = json.loads(config.read_text())
    return {fam: [{"family": fam, "seed": s["seed"], "train_run_id": s["train_run_id"],
                   "step": int(s["selected_step"])} for s in rec["families"][fam]["seeds"]]
            for fam in FAMILIES}


def _matrix(ev: Evals, runs: list[dict], suite: str, variant: str | None, n: int,
            rows: list) -> np.ndarray | None:
    mats = []
    for r in runs:
        got = ev.get(r["train_run_id"], r["step"], suite, variant)
        if got is None:
            return None
        s, p = got
        rows.append({**r, **ev.row(s, p)})
        mats.append(_vector(s, n)[1])
    return np.array(mats)


def h4(ev: Evals, task: str, ck: dict[str, list[dict]], n: int) -> dict[str, Any]:
    """§14.3 for one task."""
    sets = rb.sets_for(task)
    base, oods = sets[0], sets[1:]
    rows: list[dict[str, Any]] = []
    out: dict[str, np.ndarray] = {}
    missing = []
    for fam in FAMILIES:
        for s in sets:
            m = _matrix(ev, ck[fam], s, None, n, rows)
            if m is None:
                missing.append(f"{fam}:{s}")
            else:
                out[f"{fam}:{s}"] = m
    if missing:
        return {"status": "INCOMPLETE", "missing": missing, "evaluations": rows}
    comps = [(f"{f}:{base}", f"{f}:{o}") for o in oods for f in FAMILIES]
    comps += [(f"{a}:{o}", f"{b}:{o}") for o in oods for k in PAIRS for a, b in PAIRS[k]]
    contrasts = [(f"drop {a} - drop {b} on {o}",
                  {f"{a}:{base}": 1, f"{a}:{o}": -1, f"{b}:{base}": -1, f"{b}:{o}": 1})
                 for o in oods for k in PAIRS for a, b in PAIRS[k]]
    boot = hierarchical_paired_bootstrap(out, comps, groups={k: k.split(":")[0] for k in out},
                                         contrasts=contrasts)
    for d in boot["differences"]:
        d["kind"] = ("drop (base - OOD), same family" if d["a"].split(":")[0] ==
                     d["b"].split(":")[0] else "between-family difference in OOD success")
    return {"status": "complete", "sets": sets, "n_states": n, "evaluations": rows,
            "bootstrap": boot,
            "taxonomy": {k: _sum_tax([r for r in rows if f"{r['family']}:{r['suite']}" == k])
                         for k in out}}


def _sum_tax(rows: list[dict[str, Any]]) -> dict[str, int]:
    cats = tx.taxonomy_config()["categories"]
    return {c: sum(r["taxonomy"][c] for r in rows) for c in cats}


def ablation_variant(ev: Evals, ck: dict[str, list[dict]], aid: str, n: int) -> dict[str, Any]:
    """A1 or A2 for one task: primary − variant on the base starts (diffusion)."""
    runs = ck["diffusion_unet"]
    rows: list[dict[str, Any]] = []
    prim = _matrix(ev, runs, rb.base_name(), None, n, rows)
    var = _matrix(ev, runs, rb.base_name(), VARIANTS[aid], n, rows)
    if prim is None or var is None:
        return {"status": "INCOMPLETE", "evaluations": rows}
    boot = hierarchical_paired_bootstrap({"primary": prim, VARIANTS[aid]: var},
                                         [("primary", VARIANTS[aid])],
                                         groups={"primary": "diffusion_unet",
                                                 VARIANTS[aid]: "diffusion_unet"})
    return {"status": "complete", "arms": {"primary": "EMA weights, DDIM-10",
                                           VARIANTS[aid]: VARIANTS[aid]},
            "bias_disclosure": "checkpoints were selected with the primary arm (§14.4)",
            "evaluations": rows, "bootstrap": boot,
            "taxonomy": {arm: _sum_tax([r for r in rows if r["variant"] == v])
                         for arm, v in (("primary", None), (VARIANTS[aid], VARIANTS[aid]))}}


def ablation_a3(ev: Evals, ck: dict[str, list[dict]], a3_summary: Path, n: int) -> dict[str, Any]:
    """A3 (Can): T_o = 2 (main runs) − T_o = 1 (A3 runs) on the base starts."""
    s = json.loads(a3_summary.read_text())
    a3_runs = [{"family": "diffusion_unet", "arm": "T_o=1", "seed": r["seed"],
                "train_run_id": r["run_id"], "step": r["selection"]["step"]}
               for fam in s["families"] for r in sorted(fam["runs"], key=lambda r: r["seed"])]
    rows: list[dict[str, Any]] = []
    two = _matrix(ev, ck["diffusion_unet"], rb.base_name(), None, n, rows)
    one = _matrix(ev, a3_runs, rb.base_name(), None, n, rows)
    if two is None or one is None:
        return {"status": "INCOMPLETE", "evaluations": rows}
    boot = hierarchical_paired_bootstrap({"T_o=2": two, "T_o=1": one}, [("T_o=2", "T_o=1")])
    params = {}
    for arm, runs in (("T_o=2", ck["diffusion_unet"]), ("T_o=1", a3_runs)):
        m = json.loads((REPO_ROOT / "runs" / runs[0]["train_run_id"] / "manifest.json")
                       .read_text())
        params[arm] = m["param_count"]
    return {"status": "complete", "a3_summary": repo_relative(a3_summary),
            "a3_summary_sha256": sha256_file(a3_summary), "param_count": params,
            "a3_validation_selection": [
                {"run_id": r["run_id"], "seed": r["seed"], **r["selection"]}
                for fam in s["families"] for r in fam["runs"]],
            "note": "A3 validation numbers are development numbers (§14.4)",
            "evaluations": rows, "bootstrap": boot}


def _ci(ci: list[float]) -> str:
    return f"[{100 * ci[0]:+.1f}, {100 * ci[1]:+.1f}]"


def tables(res: dict[str, Any]) -> str:
    """Generated Markdown tables (never typed by hand)."""
    out = ["# Robustness and ablations (validation regime, fresh starts; RESEARCH_PROTOCOL §14)",
           "", f"Generated by `scripts/analyze_robustness.py`, git "
           f"`{(res['git']['sha'] or '')[:7]}` dirty={res['git']['dirty']}. Nothing here "
           "selected a model. 95 % intervals: §9 hierarchical paired bootstrap "
           "(B = 10,000, seed 20261106). Percentages of 50 starts × 3 seeds.", ""]
    for task, t in res["tasks"].items():
        h = t["h4"]
        out += [f"## {task}: H4", "", f"Status: **{h['status']}**", ""]
        if h["status"] == "complete":
            fam = h["bootstrap"]["families"]
            out += ["| Family | " + " | ".join(h["sets"]) + " |",
                    "|---" * (len(h["sets"]) + 1) + "|"]
            for f in FAMILIES:
                cells = []
                for s in h["sets"]:
                    v = fam[f"{f}:{s}"]
                    cells.append(f"{100 * v['estimate']:.1f} [{100 * v['ci95'][0]:.1f}, "
                                 f"{100 * v['ci95'][1]:.1f}] ("
                                 + ", ".join(f"{100 * x:.0f}" for x in v["per_seed"]) + ")")
                out.append(f"| {f} | " + " | ".join(cells) + " |")
            out += ["", "| Difference | Kind | Estimate (pp) | 95 % CI | Outcome |",
                    "|---|---|---|---|---|"]
            for d in h["bootstrap"]["differences"]:
                out.append(f"| {d['a']} − {d['b']} | {d['kind']} | {100 * d['difference']:+.1f}"
                           f" | {_ci(d['ci95'])} | {d['separation']} |")
            for c in h["bootstrap"]["contrasts"]:
                out.append(f"| {c['name']} | difference in drop | {100 * c['estimate']:+.1f} | "
                           f"{_ci(c['ci95'])} | {c['separation']} |")
            cats = tx.taxonomy_config()["categories"]
            out += ["", "| Family:set | " + " | ".join(cats) + " |",
                    "|---" * (len(cats) + 1) + "|"]
            for k, cnt in h["taxonomy"].items():
                out.append(f"| {k} | " + " | ".join(str(cnt[c]) for c in cats) + " |")
        for aid in ("A1", "A2", "A3"):
            a = t.get(aid)
            if a is None:
                continue
            out += ["", f"## {task}: {aid}", "", f"Status: **{a['status']}**", ""]
            if a["status"] != "complete":
                continue
            out += ["| Arm | Success % | 95 % CI | Per-seed % |", "|---|---|---|---|"]
            for arm, v in a["bootstrap"]["families"].items():
                out.append(f"| {arm} | {100 * v['estimate']:.1f} | [{100 * v['ci95'][0]:.1f}, "
                           f"{100 * v['ci95'][1]:.1f}] | "
                           + ", ".join(f"{100 * x:.0f}" for x in v["per_seed"]) + " |")
            d = a["bootstrap"]["differences"][0]
            out += ["", f"{d['a']} − {d['b']}: {100 * d['difference']:+.1f} pp, 95 % CI "
                    f"{_ci(d['ci95'])}: **{d['separation']}**."]
            if "bias_disclosure" in a:
                out.append(f"Bias disclosure: {a['bias_disclosure']}.")
            if "param_count" in a:
                out.append(f"Parameters: {a['param_count']}.")
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--config", action="append", required=True, metavar="TASK=PATH")
    p.add_argument("--a3-summary", type=Path, default=None)
    a = p.parse_args(argv)
    configs = {k: REPO_ROOT / v for k, v in (c.split("=", 1) for c in a.config)}
    out_dir = evidence_reports_dir("robustness")
    ev = Evals(out_dir / "evals")
    n = int(rb.robustness_config()["base"]["n_states"])
    res: dict[str, Any] = {"schema": "roboforge.robustness_analysis/1",
                           "label": "VALIDATION REGIME (fresh robustness starts); no selection",
                           "git": git_revision(), "generated_utc": datetime.now(tz=UTC)
                           .isoformat(), "robustness_config_sha256": rb.config_sha256(),
                           "taxonomy_sha256": tx.taxonomy_sha256(),
                           "configs": {k: {"path": repo_relative(v), "sha256": sha256_file(v)}
                                       for k, v in configs.items()}, "tasks": {}}
    for task, cfg in configs.items():
        ck = _checkpoints(cfg)
        t = {"h4": h4(ev, task, ck, n), "A1": ablation_variant(ev, ck, "A1", n),
             "A2": ablation_variant(ev, ck, "A2", n)}
        if task == "can" and a.a3_summary is not None:
            t["A3"] = ablation_a3(ev, ck, REPO_ROOT / a.a3_summary, n)
        res["tasks"][task] = t
        print(task, {k: v["status"] for k, v in t.items()})
    (out_dir / "analysis.json").write_text(json.dumps(res, indent=1, allow_nan=False) + "\n")
    (out_dir / "tables.md").write_text(tables(res))
    (out_dir / "taxonomy.json").write_text(json.dumps(
        {"taxonomy_sha256": tx.taxonomy_sha256(), "labels": ev.labels}, indent=1) + "\n")
    print(f"-> {repo_relative(out_dir)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
