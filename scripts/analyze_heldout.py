"""Analyse the held-out test exactly as frozen (RESEARCH_PROTOCOL §9, §14.6; D-067).

    ROBOFORGE_EVIDENCE_SESSION=7 uv run --frozen python scripts/analyze_heldout.py \
        --latency lift=reports/session6/cost_latency/linux-aarch64-container-1f685638f28c.json \
        --latency can=reports/session7/cost_latency/linux-aarch64-container-1f685638f28c.json

Reads the committed freeze, every held-out summary it names (``reports/session<N>/heldout/``)
and their episode records, and writes to ``reports/session<N>/results/``:

* ``results.json``: per task, the per-checkpoint k/n with Wilson intervals, the §9 bootstrap
  of success and of the censored episode length for every family and declared comparison
  (separated / not separated), the engineering status, failure-category counts, the frozen
  example episodes, and training cost and latency; with the hashes of every input;
* ``tables.md``: the same numbers as generated tables;
* ``taxonomy.json``: every episode's label and attributes.

Nothing is selected or re-run; existing results are never overwritten. ``--rehearsal-dir``
with ``--draft-freeze`` analyses a pre-freeze rehearsal instead (labelled REHEARSAL).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from roboforge.eval import freeze as fz
from roboforge.eval import taxonomy as tx
from roboforge.eval.heldout import eval_id
from roboforge.eval.statistics import hierarchical_paired_bootstrap, wilson_interval
from roboforge.paths import REPO_ROOT, evidence_reports_dir, repo_relative
from roboforge.provenance import git_revision, sha256_file

PLATFORM = "linux-aarch64-container-1f685638f28c"
ORDER = [*fz.FAMILY_ORDER, fz.BASELINE]


def _load(frozen: dict, task: str, sdir: Path, rehearsal: bool) -> list[tuple[dict, dict, Path]]:
    out = []
    for e in fz.entries(frozen, task):
        p = sdir / f"{eval_id(e, PLATFORM, rehearsal=rehearsal)}.json"
        if not p.exists():
            out.append((e, None, p))
            continue
        s = json.loads(p.read_text())
        if s["frozen_entry"] != e:
            raise ValueError(f"{p}: frozen entry differs")
        if not rehearsal and s["suite_manifest_content_sha256"] != frozen["tasks"][task][
                "heldout_manifest_sha256"]:
            raise ValueError(f"{p}: held-out manifest hash differs from the freeze")
        if e["family"] != fz.BASELINE and s["checkpoint"]["sha256"] != e["checkpoint_sha256"]:
            raise ValueError(f"{p}: checkpoint hash differs from the freeze")
        out.append((e, s, p))
    return out


def _engineering(loaded: list, n_states: int, rehearsal: bool) -> dict[str, Any]:
    """Engineering status: missing summaries and structural issues (short run, failed audit,
    non-canonical platform) make a family unavailable; in-episode errors are outcomes."""
    missing = [repo_relative(p) for _e, s, p in loaded if s is None]
    bad, unavailable = [], set()
    for e, s, p in loaded:
        if s is None:
            unavailable.add(e["family"])
            continue
        why = [w for w, cond in (("n_episodes", s["n_episodes"] != n_states),
                                 ("verify_record", not s["verify_passed"]),
                                 ("non-canonical", not s["canonical_platform"] and not rehearsal),
                                 ("episode errors", s["n_errors"] > 0)) if cond]
        if why:
            bad.append({"summary": repo_relative(p), "issues": why})
        if set(why) - {"episode errors"}:
            unavailable.add(e["family"])
    return {"complete": not unavailable, "unavailable_families": sorted(unavailable),
            "missing_summaries": missing, "issues": bad,
            "engineering_reruns": [s["engineering_rerun"] for _, s, _ in loaded
                                   if s and s.get("engineering_rerun")]}


def _matrices(loaded: list, horizon: int,
              families: list[str] | None = None) -> tuple[dict, dict, list[str]]:
    families = ORDER if families is None else families
    states = None
    succ: dict[str, list] = {f: [] for f in families}
    length: dict[str, list] = {f: [] for f in families}
    for e, s, _ in loaded:
        if e["family"] not in families:
            continue
        eps = sorted(s["episodes"], key=lambda r: r["state_id"])
        ids = [r["state_id"] for r in eps]
        if states is None:
            states = ids
        if ids != states:
            raise ValueError("summaries cover different states: pairing broken")
        succ[e["family"]].append([float(r["success"]) for r in eps])
        length[e["family"]].append([float(r["first_success_step"] if r["success"] else horizon)
                                    for r in eps])
    return ({f: np.array(v) for f, v in succ.items()},
            {f: np.array(v) for f, v in length.items()}, states)


def _examples(labels: dict[tuple, list], categories: list[str]) -> dict[str, Any]:
    """The frozen example-selection rule (freeze ``example_rule``): per family, the first
    success and the first failure (sorted state_id) of the lowest seed that has one; per
    failure category, the first episode in (family order, seed, state_id)."""
    def ordered(fams):
        for fam in fams:
            for k in sorted((k for k in labels if k[0] == fam), key=lambda k: k[1]):
                for x in sorted(labels[k], key=lambda x: x["state_id"]):
                    yield k, x

    def pick(it):
        return next(({"family": k[0], "seed": k[1], "eval_run_id": k[2],
                      "state_id": x["state_id"], "category": x["category"]} for k, x in it),
                    None)

    out: dict[str, Any] = {"by_family": {}, "by_category": {}}
    for fam in fz.FAMILY_ORDER:
        out["by_family"][fam] = {
            "first_success": pick((k, x) for k, x in ordered([fam])
                                  if x["category"] == "success"),
            "first_failure": pick((k, x) for k, x in ordered([fam])
                                  if x["category"] != "success")}
    for cat in categories:
        if cat != "success":
            out["by_category"][cat] = pick((k, x) for k, x in ordered(fz.FAMILY_ORDER)
                                           if x["category"] == cat)
    return out


def _cost(frozen: dict, task: str, loaded: list, latency: Path | None) -> dict[str, Any]:
    lat_rows = {}
    if latency is not None:
        for r in json.loads(latency.read_text())["rows"]:
            lat_rows[r["train_run_id"]] = r["inference"]
    out = {}
    for fam in fz.FAMILY_ORDER:
        runs = frozen["tasks"][task]["families"][fam]["runs"]
        mans = [json.loads((REPO_ROOT / "runs" / r["train_run_id"] / "manifest.json")
                           .read_text()) for r in runs]
        inf = lat_rows.get(runs[0]["train_run_id"])
        evals = [s for e, s, _ in loaded if s and e["family"] == fam]
        out[fam] = {"param_count": mans[0]["param_count"],
                    "training_wall_h_per_run": [m["timing"]["wall_s"] / 3600 for m in mans],
                    "training_cpu_h_per_run": [m["timing"]["cpu_s"] / 3600 for m in mans],
                    "training_note": "local CPU, 1 torch thread; wall times contended (KI-035)",
                    "heldout_eval_wall_s": [s["wall_s"] for s in evals],
                    "latency_ms_median": (inf["observe_plus_act_ms"]["median"] if inf
                                          else None),
                    "latency_ms_p95": inf["observe_plus_act_ms"].get("p95") if inf else None,
                    "latency_source": repo_relative(latency) if inf else None,
                    "cost_usd": 0.0}
    return out


def _bootstrap(frozen: dict, mats: dict, hyps: list[str], avail: list[str]) -> dict[str, Any]:
    """The frozen bootstrap over the available families; comparisons with an unavailable
    family are listed as not computed (freeze §4, engineering-incomplete)."""
    b, comps = frozen["analysis"]["bootstrap"], frozen["analysis"]["comparisons"]
    pairs = [(tuple(c), k) for k in hyps for c in comps[k]]
    ok = [c for c, _ in pairs if c[0] in avail and c[1] in avail]
    order = [f for f in b["draw_order"][1:] if f in avail]
    boot = hierarchical_paired_bootstrap({f: mats[f] for f in order}, ok, b=int(b["B"]),
                                         seed=int(b["seed"]), order=order)
    for d in boot["differences"]:
        d["hypothesis"] = next(k for c, k in pairs if c == (d["a"], d["b"]))
    boot["not_computed"] = [{"hypothesis": k, "a": c[0], "b": c[1],
                             "status": "ENGINEERING-INCOMPLETE"} for c, k in pairs
                            if c not in ok]
    return boot


def analyze_task(frozen: dict, task: str, sdir: Path, rehearsal: bool,
                 latency: Path | None) -> tuple[dict, dict]:
    """Everything the freeze prescribes for one task; also returns the taxonomy labels."""
    t = frozen["tasks"][task]
    horizon = int(frozen["rollouts"]["horizon"])
    loaded = _load(frozen, task, sdir, rehearsal)
    eng = _engineering(loaded, int(t["n_states"]), rehearsal)
    avail = [f for f in ORDER if f not in eng["unavailable_families"]]
    res: dict[str, Any] = {
        "status": "complete" if eng["complete"] else
        f"ENGINEERING-INCOMPLETE ({', '.join(eng['unavailable_families'])})",
        "heldout_manifest_sha256": t["heldout_manifest_sha256"], "engineering": eng,
        "available_families": avail}
    if avail:
        succ, length, states = _matrices(loaded, horizon, avail)
        res["n_states"] = len(states)
        res["success"] = _bootstrap(frozen, succ, ["H1", "H2", "H2c", "H3"], avail)
        res["censored_episode_length"] = _bootstrap(
            frozen, length, list(frozen["analysis"]["secondary"]["comparisons"]), avail)
    per_ckpt, labels_all = [], {}
    for e, s, p in loaded:
        if s is None:
            continue
        labs = tx.classify_records(REPO_ROOT / s["records_dir"])
        if sum(x["category"] == "success" for x in labs) != s["successes"]:
            raise ValueError(f"{p}: taxonomy and summary disagree on successes")
        key = (e["family"], e.get("seed", 0), s["eval_run_id"])
        labels_all[key] = labs
        lo, hi = wilson_interval(s["successes"], s["n_episodes"])
        per_ckpt.append({"family": e["family"], "seed": e.get("seed"),
                         "train_run_id": e.get("train_run_id"), "step": e.get("step"),
                         "eval_run_id": s["eval_run_id"], "summary": repo_relative(p),
                         "summary_sha256": sha256_file(p), "successes": s["successes"],
                         "n_episodes": s["n_episodes"], "wilson95": [lo, hi],
                         "first_success_step_mean": s["first_success_step"]["mean"],
                         "n_errors": s["n_errors"], "stop_reasons": s["stop_reasons"],
                         "motion_clip_fraction": float(np.mean(s["clip_fraction_per_dim"][:-1])),
                         "gripper_clip_fraction": float(s["clip_fraction_per_dim"][-1]),
                         "wall_s": s["wall_s"], "taxonomy": tx.counts(labs)})
    res["per_checkpoint"] = per_ckpt
    res["taxonomy_by_family"] = {
        f: {c: sum(r["taxonomy"][c] for r in per_ckpt if r["family"] == f)
            for c in tx.taxonomy_config()["categories"]} for f in ORDER}
    res["examples"] = _examples(labels_all, tx.taxonomy_config()["categories"])
    res["cost_latency"] = _cost(frozen, task, loaded, latency)
    return res, labels_all


def _fmt_ci(ci: list[float], pct: bool = True) -> str:
    return (f"[{100 * ci[0]:.1f}, {100 * ci[1]:.1f}]" if pct else f"[{ci[0]:.1f}, {ci[1]:.1f}]")


def tables(results: dict, label: str) -> str:
    """Markdown tables generated from ``results`` (never typed by hand)."""
    out = [f"# {label}", "", f"Generated by `scripts/analyze_heldout.py` from "
           f"`{results['freeze']['path']}` (sha256 `{results['freeze']['sha256'][:12]}…`), git "
           f"`{(results['git']['sha'] or '')[:7]}` dirty={results['git']['dirty']}. "
           "Tasks are reported separately; no pooled number.", ""]
    for task, r in results["tasks"].items():
        out += [f"## {task}", "", f"Status: **{r['status']}**", ""]
        if "success" not in r:
            continue
        for d in r["success"]["not_computed"]:
            out.append(f"- {d['hypothesis']} {d['a']} − {d['b']}: not computed "
                       f"({d['status']})")
        out += ["### Success per checkpoint (one rollout per start)", "",
                "| Family | Seed | Step | Successes | Wilson 95 % | Mean first-success step | "
                "Errors |", "|---|---|---|---|---|---|---|"]
        for c in r["per_checkpoint"]:
            fs = c["first_success_step_mean"]
            out.append(f"| {c['family']} | {c['seed'] if c['seed'] is not None else '-'} | "
                       f"{c['step'] or '-'} | {c['successes']}/{c['n_episodes']} | "
                       f"{_fmt_ci(c['wilson95'])} | {'-' if fs is None else f'{fs:.1f}'} | "
                       f"{c['n_errors']} |")
        out += ["", "### Family success (mean over seeds × states; §9 bootstrap 95 %)", "",
                "| Family | Success % | 95 % CI | Per-seed % |", "|---|---|---|---|"]
        for f, v in r["success"]["families"].items():
            out.append(f"| {f} | {100 * v['estimate']:.1f} | {_fmt_ci(v['ci95'])} | "
                       + ", ".join(f"{100 * x:.0f}" for x in v["per_seed"]) + " |")
        out += ["", "### Declared paired differences in success (percentage points)", "",
                "| Hypothesis | A − B | Difference | 95 % CI | Outcome |", "|---|---|---|---|---|"]
        for d in r["success"]["differences"]:
            out.append(f"| {d['hypothesis']} | {d['a']} − {d['b']} | "
                       f"{100 * d['difference']:+.1f} | {_fmt_ci(d['ci95'])} | "
                       f"{d['separation']} |")
        out += ["", "### Secondary: censored episode length (steps; failures = 400)", "",
                "| Family | Mean steps | 95 % CI |", "|---|---|---|"]
        for f, v in r["censored_episode_length"]["families"].items():
            out.append(f"| {f} | {v['estimate']:.1f} | {_fmt_ci(v['ci95'], False)} |")
        out += ["", "| Hypothesis | A − B | Difference (steps) | 95 % CI | Outcome |",
                "|---|---|---|---|---|"]
        for d in r["censored_episode_length"]["differences"]:
            out.append(f"| {d['hypothesis']} | {d['a']} − {d['b']} | {d['difference']:+.1f} | "
                       f"{_fmt_ci(d['ci95'], False)} | {d['separation']} |")
        cats = tx.taxonomy_config()["categories"]
        out += ["", "### Failure taxonomy (episodes, all seeds)", "",
                "| Family | " + " | ".join(cats) + " |", "|---" * (len(cats) + 1) + "|"]
        for f, cnt in r["taxonomy_by_family"].items():
            out.append(f"| {f} | " + " | ".join(str(cnt[c]) for c in cats) + " |")
        out += ["", "### Cost and latency (canonical container latency; training on local CPU)",
                "", "| Family | Params | Training CPU-h per run | Latency median ms | Cost |",
                "|---|---|---|---|---|"]
        for f, c in r["cost_latency"].items():
            lat = "-" if c["latency_ms_median"] is None else f"{c['latency_ms_median']:.2f}"
            cpu = ", ".join(f"{x:.2f}" for x in c["training_cpu_h_per_run"])
            out.append(f"| {f} | {c['param_count']:,} | {cpu} | {lat} | $0 |")
        out.append("")
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--latency", action="append", default=[], metavar="TASK=PATH")
    p.add_argument("--rehearsal-dir", type=Path, default=None)
    p.add_argument("--draft-freeze", type=Path, default=None)
    a = p.parse_args(argv)
    rehearsal = a.rehearsal_dir is not None
    if rehearsal != (a.draft_freeze is not None):
        p.error("--rehearsal-dir and --draft-freeze go together")
    frozen = fz.parse_freeze(a.draft_freeze.read_text()) if rehearsal else fz.load_freeze()
    if not rehearsal and fz.code_problems(frozen):
        raise SystemExit(f"frozen code changed: {fz.code_problems(frozen)}")
    sdir = a.rehearsal_dir if rehearsal else evidence_reports_dir("heldout")
    out_dir = (a.rehearsal_dir / "results") if rehearsal else evidence_reports_dir("results")
    if not rehearsal and out_dir.exists():
        raise SystemExit(f"{repo_relative(out_dir)} exists: the held-out analysis runs once")
    lat = {k: REPO_ROOT / v for k, v in (x.split("=", 1) for x in a.latency)}
    missing_lat = [repo_relative(v) for v in lat.values() if not v.exists()]
    if missing_lat:
        raise SystemExit(f"latency reports missing: {missing_lat}")
    freeze_path = a.draft_freeze if rehearsal else REPO_ROOT / fz.FREEZE_REL
    label = ("REHEARSAL on a non-locked suite: NOT test results" if rehearsal
             else "HELD-OUT TEST RESULTS (frozen protocol, run once)")
    results = {"schema": "roboforge.heldout_results/1", "label": label,
               "freeze": {"path": repo_relative(freeze_path), "sha256": sha256_file(freeze_path)},
               "git": git_revision(), "generated_utc": datetime.now(tz=UTC).isoformat(),
               "tasks": {}}
    tax = {}
    for task in frozen["tasks"]:
        r, labs = analyze_task(frozen, task, sdir, rehearsal, lat.get(task))
        results["tasks"][task] = r
        tax[task] = [{"family": k[0], "seed": k[1], "eval_run_id": k[2], "labels": v}
                     for k, v in labs.items()]
    # Everything is computed before anything is written: a failure leaves no partial output.
    texts = {"results.json": json.dumps(results, indent=1, allow_nan=False) + "\n",
             "tables.md": tables(results, label),
             "taxonomy.json": json.dumps({"taxonomy_sha256": tx.taxonomy_sha256(), "tasks": tax},
                                         indent=1, allow_nan=False) + "\n"}
    out_dir.mkdir(parents=True, exist_ok=rehearsal)
    for name, text in texts.items():
        with (out_dir / name).open("w" if rehearsal else "x") as f:
            f.write(text)
    for task, r in results["tasks"].items():
        print(task, r["status"])
    print(f"-> {repo_relative(out_dir)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
