"""X1: analyse the clamped BC evaluations against the frozen Session 7 robustness results (D-107).

    ROBOFORGE_EVIDENCE_SESSION=13 uv run --frozen python scripts/analyze_x1_clamp.py

**Exploratory, after the freeze. Run once.** It refuses to write over an existing
``results.json``. Outputs, in ``reports/session<N>/x1_clamp/``:
- ``results.json``: every input hash, the bootstrap, and the preregistered readings;
- ``tables.md``: generated;
- ``taxonomy.json``: the §14.5 labels of every clamped episode.

**Inputs, read through a hash chain.**
- ``reports/session7/robustness/analysis.json``, the frozen H4 analysis. Its SHA-256 is pinned
  below and also in ``paper/generated/index.json``.
- The 60 Session 7 H4 summaries and the two test configuration records, each read only after
  its SHA-256 equals the value that analysis recorded.
- The 45 X1 summaries in ``reports/session<N>/x1_clamp/evals/``. Each must be clamped,
  canonical, complete (n = 50), audited (``verify_record``) and from one clean commit that
  contains the preregistration. Its checkpoint and suite hashes must equal the frozen ones.

**Method** (``docs/extensions/X1_PREREGISTRATION.md``). One §9 hierarchical paired bootstrap per
task, B = 10,000, seed 20261106:
- The keys are the frozen ``family:set`` outcome matrices, in the frozen order, followed by
  ``<bc family>+clamp:set``.
- Seed groups map every key to its family, so the clamped arm of a family shares that family's
  seed draw. The random draws are therefore exactly those of the frozen analysis.
- The script **asserts** that every frozen family estimate, difference and contrast equals
  ``analysis.json`` exactly before it reports anything new.
- The clamped arms ride on the same replicates. This pairs clamped − unclamped by seed and by
  start, and diffusion − clamped BC by start.
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from roboforge.eval import taxonomy as tx
from roboforge.eval.statistics import hierarchical_paired_bootstrap, wilson_interval
from roboforge.exploratory.clamp import LABEL, PREREGISTRATION, SCHEMA
from roboforge.paths import REPO_ROOT, evidence_reports_dir, repo_relative
from roboforge.provenance import git_revision, sha256_file
from roboforge.sim import robustness as rb

SESSION7_ANALYSIS = ("reports/session7/robustness/analysis.json",
                     "8cca66a8a8a8cecf0b734355167936e81983c821c625a23456523365409c9451")
FAMILIES = ("bc_mlp", "bc_rnn", "bc_mlp_chunk", "diffusion_unet")
BC = FAMILIES[:3]
DIFF = "diffusion_unet"
PAIRS = {"H2": [("diffusion_unet", "bc_mlp"), ("diffusion_unet", "bc_rnn")],
         "H2c": [("diffusion_unet", "bc_mlp_chunk")], "H3": [("bc_rnn", "bc_mlp")]}
CL = "+clamp"
PRIMARY = {"task": "lift", "set": "ood_arm_offset",
           "name": "drop diffusion_unet - drop bc_mlp_chunk+clamp on ood_arm_offset",
           "frozen_name": "drop diffusion_unet - drop bc_mlp_chunk on ood_arm_offset"}
SUPPORT = "drop bc_mlp_chunk - drop bc_mlp_chunk+clamp on ood_arm_offset"
BASE_LEG = ("bc_mlp_chunk+clamp:robustness_base", "bc_mlp_chunk:robustness_base")
TAIL = " Exploratory, after the freeze; the frozen numbers stand."
# The readings, verbatim from the preregistration (§6), keyed by the interval's verdict.
READINGS = {
    "primary": {
        "separated (-)": "R1. The bound does not explain the frozen H4 gap: with the same "
                         "action bound, BC-MLP-chunk still degrades more than diffusion under "
                         "the arm offset on Lift.",
        "not separated": "R2. With the same action bound, the drop difference between "
                         "diffusion and BC-MLP-chunk under the arm offset on Lift is not "
                         "separated: X1 cannot attribute the frozen H4 advantage to the "
                         "generative model rather than to the bound. This is not evidence "
                         "that the two drops are equal.",
        "separated (+)": "R3. With the same action bound, BC-MLP-chunk degrades less than "
                         "diffusion under the arm offset on Lift: the bound accounts for more "
                         "than the frozen H4 gap."},
    "support": {
        "separated (+)": "The clamp reduced BC-MLP-chunk's drop under the arm offset on Lift.",
        "not separated": "No separated change in BC-MLP-chunk's drop under the arm offset on "
                         "Lift.",
        "separated (-)": "The clamp increased BC-MLP-chunk's drop under the arm offset on "
                         "Lift."},
    "base": {
        "separated (-)": "The bound is not free in distribution: it lowers BC-MLP-chunk's "
                         "success on fresh in-distribution starts. H2-c parity stands as "
                         "measured, with BC unclamped.",
        "not separated": "No separated effect of the bound on BC-MLP-chunk's in-distribution "
                         "success.",
        "separated (+)": "The bound raises BC-MLP-chunk's in-distribution success: H2-c was "
                         "measured with BC-MLP-chunk unclamped, a disadvantage the bound "
                         "removes on these starts."},
}


def pinned(rel: str, sha: str) -> Path:
    """The path of ``rel`` after checking its SHA-256 (refuse on mismatch)."""
    got = sha256_file(REPO_ROOT / rel)
    if got != sha:
        raise SystemExit(f"{rel}: SHA-256 {got} != recorded {sha}")
    return REPO_ROOT / rel


def refuse_existing(out_dir: Path) -> None:
    """Run once: refuse if the results file already exists."""
    if (out_dir / "results.json").exists():
        raise SystemExit(f"{repo_relative(out_dir / 'results.json')} exists: analysed once")


def _index(state_id: str) -> int:
    return int(state_id.rsplit("_", 1)[1])


def vector(s: dict[str, Any], n: int) -> list[float]:
    """Success per start, ordered by start index 0..n-1."""
    eps = sorted(s["episodes"], key=lambda e: _index(e["state_id"]))
    if [_index(e["state_id"]) for e in eps] != list(range(n)):
        raise SystemExit(f"{s['eval_run_id']}: states are not indices 0..{n - 1}")
    return [float(e["success"]) for e in eps]


def _commit_has_prereg(sha: str) -> bool:
    r = subprocess.run(["git", "cat-file", "-e", f"{sha}:{PREREGISTRATION}"], cwd=REPO_ROOT,
                       capture_output=True)
    return r.returncode == 0


def load_x1(directory: Path, n: int) -> tuple[dict[tuple, dict[str, Any]], dict[str, Any]]:
    """Every X1 summary by (train_run_id, step, suite), each checked; plus their common git."""
    out, shas = {}, set()
    for p in sorted(directory.glob("*.json")):
        s = json.loads(p.read_text())
        problems = []
        if s.get("schema") != SCHEMA or s.get("label") != LABEL:
            problems.append("schema/label")
        if not s["x1_clamp"]["enabled"]:
            problems.append("clamp disabled")
        if not s["canonical_platform"] or s["partial"] or s["n_episodes"] != n:
            problems.append("not canonical and complete")
        if not s["verify_passed"]:
            problems.append("verify_record failed")
        if s["git"]["dirty"]:
            problems.append("dirty tree")
        if problems:
            raise SystemExit(f"{repo_relative(p)}: {problems}")
        key = (s["train_run_id"], int(s["checkpoint"]["step"]), s["suite"])
        if key in out:
            raise SystemExit(f"two X1 summaries for {key}")
        shas.add(s["git"]["sha"])
        out[key] = {"summary": s, "path": repo_relative(p), "sha256": sha256_file(p)}
    if len(shas) != 1:
        raise SystemExit(f"X1 summaries come from {len(shas)} commits: {sorted(shas)}")
    (sha,) = shas
    if not _commit_has_prereg(sha):
        raise SystemExit(f"commit {sha} does not contain {PREREGISTRATION}")
    return out, {"sha": sha, "contains_preregistration": True}


def _clamp_activity(entries: list[dict[str, Any]]) -> dict[str, Any]:
    dec = sum(e["summary"]["x1_clamp"]["counts"]["n_actions_decoded"] for e in entries)
    acts = sum(e["summary"]["x1_clamp"]["counts"]["n_actions_clamped"] for e in entries)
    per = np.sum([e["summary"]["x1_clamp"]["counts"]["entries_clamped_per_dim"]
                  for e in entries], axis=0)
    return {"n_actions_decoded": int(dec), "n_actions_clamped": int(acts),
            "fraction_actions_clamped": acts / dec if dec else 0.0,
            "fraction_entries_clamped_per_dim": (per / dec).tolist() if dec else None}


def frozen_outcomes(task: str, a7: dict[str, Any], n: int) -> dict[str, Any]:
    """The frozen H4 outcome matrices of ``task`` in the frozen key order, read through the
    hash chain (test configuration record and every summary checked against ``a7``)."""
    h7 = a7["tasks"][task]["h4"]
    sets = h7["sets"]
    if sets != rb.sets_for(task):
        raise SystemExit(f"{task}: frozen sets {sets} != {rb.sets_for(task)}")
    cfg = a7["configs"][task]
    rec = json.loads(pinned(cfg["path"], cfg["sha256"]).read_text())
    rows7 = {(r["train_run_id"], int(r["step"]), r["suite"]): r for r in h7["evaluations"]}
    outcomes: dict[str, np.ndarray] = {}
    groups: dict[str, str] = {}
    summaries: dict[tuple, dict[str, Any]] = {}
    for fam in FAMILIES:  # the frozen keys, in the frozen insertion order
        for s in sets:
            mat = []
            for sd in rec["families"][fam]["seeds"]:
                key = (sd["train_run_id"], int(sd["selected_step"]), s)
                r = rows7[key]
                summaries[key] = json.loads(pinned(r["summary"], r["summary_sha256"]).read_text())
                mat.append(vector(summaries[key], n))
            outcomes[f"{fam}:{s}"], groups[f"{fam}:{s}"] = np.array(mat), fam
    return {"h7": h7, "sets": sets, "rec": rec, "rows7": rows7, "summaries": summaries,
            "outcomes": outcomes, "groups": groups}


def comparisons(sets: list[str]) -> tuple[list, list, int, int]:
    """(differences, contrasts, n frozen differences, n frozen contrasts): the frozen ones
    first, built exactly as the frozen analysis built them, then X1's."""
    base, oods = sets[0], sets[1:]
    comps = [(f"{f}:{base}", f"{f}:{o}") for o in oods for f in FAMILIES]
    comps += [(f"{a}:{o}", f"{b}:{o}") for o in oods for k in PAIRS for a, b in PAIRS[k]]
    contrasts = [(f"drop {a} - drop {b} on {o}",
                  {f"{a}:{base}": 1, f"{a}:{o}": -1, f"{b}:{base}": -1, f"{b}:{o}": 1})
                 for o in oods for k in PAIRS for a, b in PAIRS[k]]
    n_c, n_k = len(comps), len(contrasts)
    # X1: clamped - unclamped; clamped drops; diffusion - clamped BC; difference-of-drop contrasts.
    comps += [(f"{f}{CL}:{s}", f"{f}:{s}") for f in BC for s in sets]
    comps += [(f"{f}{CL}:{base}", f"{f}{CL}:{o}") for f in BC for o in oods]
    comps += [(f"{DIFF}:{s}", f"{f}{CL}:{s}") for f in BC for s in sets]
    contrasts += [(f"drop {DIFF} - drop {f}{CL} on {o}",
                   {f"{DIFF}:{base}": 1, f"{DIFF}:{o}": -1, f"{f}{CL}:{base}": -1,
                    f"{f}{CL}:{o}": 1}) for f in BC for o in oods]
    contrasts += [(f"drop {f} - drop {f}{CL} on {o}",
                   {f"{f}:{base}": 1, f"{f}:{o}": -1, f"{f}{CL}:{base}": -1, f"{f}{CL}:{o}": 1})
                  for f in BC for o in oods]
    return comps, contrasts, n_c, n_k


def frozen_mismatches(boot: dict[str, Any], fb: dict[str, Any], n_c: int, n_k: int) -> list:
    """Every frozen estimate, difference or contrast that ``boot`` does not reproduce exactly."""
    mism = []
    if (len(fb["differences"]), len(fb["contrasts"])) != (n_c, n_k):
        mism.append("number of frozen comparisons")
    for k, v in fb["families"].items():
        got = boot["families"][k]
        if any(got[f] != v[f] for f in ("estimate", "ci95", "per_seed")):
            mism.append(k)
    for g, d in zip(boot["differences"][:n_c], fb["differences"], strict=False):
        if any(g[f] != d[f] for f in ("a", "b", "difference", "ci95", "separation")):
            mism.append(f"{d['a']} - {d['b']}")
    for g, c in zip(boot["contrasts"][:n_k], fb["contrasts"], strict=False):
        if any(g[f] != c[f] for f in ("name", "estimate", "ci95", "separation")):
            mism.append(c["name"])
    return mism


def task_analysis(task: str, a7: dict[str, Any], x1: dict[tuple, dict[str, Any]],
                  n: int, labels: dict[str, Any]) -> dict[str, Any]:
    """The X1 bootstrap of one task, after asserting the frozen part reproduces exactly."""
    fz = frozen_outcomes(task, a7, n)
    h7, sets, rec, rows7 = fz["h7"], fz["sets"], fz["rec"], fz["rows7"]
    outcomes, groups = dict(fz["outcomes"]), dict(fz["groups"])
    evals: list[dict[str, Any]] = []
    for fam in BC:  # the clamped arms, same seed order, grouped with their family
        for s in sets:
            mat = []
            for sd in rec["families"][fam]["seeds"]:
                key = (sd["train_run_id"], int(sd["selected_step"]), s)
                if key not in x1:
                    raise SystemExit(f"missing X1 evaluation {key}")
                e = x1[key]
                summ = e["summary"]
                s7 = fz["summaries"][key]
                if (summ["checkpoint"]["sha256"] != sd["checkpoint_sha256"]
                        or summ["family"] != fam
                        or summ["suite_manifest_content_sha256"]
                        != s7["suite_manifest_content_sha256"]):
                    raise SystemExit(f"{e['path']}: checkpoint/family/suite differs from frozen")
                labs = tx.classify_records(REPO_ROOT / summ["records_dir"])
                if sum(x["category"] == "success" for x in labs) != summ["successes"]:
                    raise SystemExit(f"{e['path']}: taxonomy and summary disagree on successes")
                labels[summ["eval_run_id"]] = labs
                mat.append(vector(summ, n))
                lo, hi = wilson_interval(summ["successes"], summ["n_episodes"])
                evals.append({"family": fam, "seed": sd["seed"], "suite": s,
                              "train_run_id": sd["train_run_id"],
                              "step": int(sd["selected_step"]),
                              "eval_run_id": summ["eval_run_id"], "summary": e["path"],
                              "summary_sha256": e["sha256"],
                              "checkpoint_sha256": summ["checkpoint"]["sha256"],
                              "successes": summ["successes"], "n_episodes": summ["n_episodes"],
                              "wilson95": [lo, hi], "n_errors": summ["n_errors"],
                              "verify_passed": summ["verify_passed"],
                              "frozen_successes": rows7[key]["successes"],
                              "clamp": _clamp_activity([e]), "wall_s": summ["wall_s"],
                              "taxonomy": tx.counts(labs)})
            outcomes[f"{fam}{CL}:{s}"], groups[f"{fam}{CL}:{s}"] = np.array(mat), fam
    comps, contrasts, n_c, n_k = comparisons(sets)
    boot = hierarchical_paired_bootstrap(outcomes, comps, groups=groups, contrasts=contrasts)
    fb = h7["bootstrap"]
    mism = frozen_mismatches(boot, fb, n_c, n_k)
    if mism:  # the method and the hash chain, checked in one step
        raise SystemExit(f"{task}: the frozen bootstrap does not reproduce exactly: {mism}")
    x1_diffs = boot["differences"][n_c:]
    for d in x1_diffs:
        fa, fb_ = d["a"].split(":")[0], d["b"].split(":")[0]
        d["kind"] = ("clamped - unclamped, same checkpoints" if fa == fb_ + CL else
                     "drop (base - OOD), clamped" if fa == fb_ else
                     "diffusion - clamped BC")
    activity = {f"{f}{CL}:{s}": _clamp_activity(
        [x1[(sd["train_run_id"], int(sd["selected_step"]), s)]
         for sd in rec["families"][f]["seeds"]]) for f in BC for s in sets}
    cats = tx.taxonomy_config()["categories"]
    tax = {k: {c: sum(r["taxonomy"][c] for r in evals if f"{r['family']}{CL}:{r['suite']}" == k)
               for c in cats} for k in activity}
    return {"sets": sets, "n_states": n, "evaluations": evals,
            "frozen_reproduced": {"families": len(fb["families"]), "differences": n_c,
                                  "contrasts": n_k, "exact": True},
            "bootstrap": {**{k: v for k, v in boot.items()
                             if k not in ("differences", "contrasts")},
                          "frozen_differences": boot["differences"][:n_c],
                          "frozen_contrasts": boot["contrasts"][:n_k],
                          "x1_differences": x1_diffs,
                          "x1_contrasts": boot["contrasts"][n_k:]},
            "clamp_activity": activity, "frozen_taxonomy": h7["taxonomy"],
            "clamped_taxonomy": tax}


def readings(res: dict[str, Any]) -> dict[str, Any]:
    """The preregistered readings, chosen by the verdicts (no other interpretation here)."""
    lift = res["tasks"]["lift"]["bootstrap"]
    prim = next(c for c in lift["x1_contrasts"] if c["name"] == PRIMARY["name"])
    froz = next(c for c in lift["frozen_contrasts"] if c["name"] == PRIMARY["frozen_name"])
    sup = next(c for c in lift["x1_contrasts"] if c["name"] == SUPPORT)
    out = {"primary": {**prim, "frozen": froz,
                       "reading": READINGS["primary"][prim["separation"]] + TAIL},
           "support": {**sup, "reading": READINGS["support"][sup["separation"]] + TAIL},
           "base": {}}
    for task, t in res["tasks"].items():
        d = next(x for x in t["bootstrap"]["x1_differences"]
                 if (x["a"], x["b"]) == BASE_LEG)
        out["base"][task] = {**d, "reading": READINGS["base"][d["separation"]] + TAIL}
    return out


def _pct(x: float) -> str:
    return f"{100 * x:.1f}"


def _ci(ci: list[float]) -> str:
    return f"[{100 * ci[0]:+.1f}, {100 * ci[1]:+.1f}]"


def tables(res: dict[str, Any]) -> str:
    """Generated Markdown (never typed by hand)."""
    r = res["readings"]
    p = r["primary"]
    out = ["# X1: BC with the diffusion policy's action bound (exploratory, after the freeze)", "",
           f"Generated by `scripts/analyze_x1_clamp.py`, git `{res['git']['sha'][:7]}` "
           f"dirty={res['git']['dirty']}; evaluations from `{res['x1_git']['sha'][:7]}`. "
           "Validation regime (Session 7 robustness starts); nothing here selects a model or "
           "revisits the held-out test. 95 % intervals: §9 hierarchical paired bootstrap "
           "(B = 10,000, seed 20261106), the frozen draws reproduced exactly. Percentages of "
           "50 starts × 3 seeds. `+clamp` = the same frozen BC checkpoint with every action "
           "clipped to [−1, 1] in normalized space (the training range).", "",
           "## Primary (preregistered)", "",
           f"Lift, `ood_arm_offset`: drop(diffusion) − drop(clamped BC-MLP-chunk) = "
           f"**{100 * p['estimate']:+.1f} pp {_ci(p['ci95'])}, {p['separation']}** "
           f"(frozen, unclamped: {100 * p['frozen']['estimate']:+.1f} pp "
           f"{_ci(p['frozen']['ci95'])}, {p['frozen']['separation']}).", "",
           f"> {p['reading']}", "",
           f"Supporting: drop(BC-MLP-chunk) − drop(clamped BC-MLP-chunk) = "
           f"{100 * r['support']['estimate']:+.1f} pp {_ci(r['support']['ci95'])}, "
           f"{r['support']['separation']}. {r['support']['reading']}", ""]
    for task, b in r["base"].items():
        out += [f"Base leg, {task}: clamped − unclamped BC-MLP-chunk on `robustness_base` = "
                f"{100 * b['difference']:+.1f} pp {_ci(b['ci95'])}, {b['separation']}. "
                f"{b['reading']}", ""]
    for task, t in res["tasks"].items():
        fam = t["bootstrap"]["families"]
        out += [f"## {task}: success by arm and set", "",
                "| Arm | " + " | ".join(t["sets"]) + " |", "|---" * (len(t["sets"]) + 1) + "|"]
        for arm in [*BC, *(f"{f}{CL}" for f in BC), DIFF]:
            cells = []
            for s in t["sets"]:
                v = fam[f"{arm}:{s}"]
                cells.append(f"{_pct(v['estimate'])} [{_pct(v['ci95'][0])}, "
                             f"{_pct(v['ci95'][1])}] ("
                             + ", ".join(f"{100 * x:.0f}" for x in v["per_seed"]) + ")")
            out.append(f"| {arm} | " + " | ".join(cells) + " |")
        out += ["", "| Difference | Kind | Estimate (pp) | 95 % CI | Outcome |",
                "|---|---|---|---|---|"]
        for d in t["bootstrap"]["x1_differences"]:
            out.append(f"| {d['a']} − {d['b']} | {d['kind']} | {100 * d['difference']:+.1f} | "
                       f"{_ci(d['ci95'])} | {d['separation']} |")
        for c in t["bootstrap"]["x1_contrasts"]:
            out.append(f"| {c['name']} | difference in drop | {100 * c['estimate']:+.1f} | "
                       f"{_ci(c['ci95'])} | {c['separation']} |")
        out += ["", "Frozen contrasts (reproduced exactly from `analysis.json`, for reference):",
                "", "| Contrast | Estimate (pp) | 95 % CI | Outcome |", "|---|---|---|---|"]
        for c in t["bootstrap"]["frozen_contrasts"]:
            out.append(f"| {c['name']} | {100 * c['estimate']:+.1f} | {_ci(c['ci95'])} | "
                       f"{c['separation']} |")
        out += ["", "Clamp activity (fraction of decoded actions with at least one clamped "
                "entry; per dimension, fraction of entries clamped):", "",
                "| Arm:set | Actions clamped | Per dimension |", "|---|---|---|"]
        for k, v in t["clamp_activity"].items():
            per = ", ".join(f"{x:.3f}" for x in v["fraction_entries_clamped_per_dim"])
            out.append(f"| {k} | {v['fraction_actions_clamped']:.3f} | {per} |")
        cats = tx.taxonomy_config()["categories"]
        out += ["", "| Arm:set (clamped) | " + " | ".join(cats) + " |",
                "|---" * (len(cats) + 1) + "|"]
        for k, cnt in t["clamped_taxonomy"].items():
            out.append(f"| {k} | " + " | ".join(str(cnt[c]) for c in cats) + " |")
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    out_dir = evidence_reports_dir("x1_clamp")
    refuse_existing(out_dir)
    a7_path = pinned(*SESSION7_ANALYSIS)
    a7 = json.loads(a7_path.read_text())
    n = int(rb.robustness_config()["base"]["n_states"])
    x1, x1_git = load_x1(out_dir / "evals", n)
    if len(x1) != 45:
        raise SystemExit(f"expected 45 X1 summaries, found {len(x1)}")
    labels: dict[str, Any] = {}
    prereg = REPO_ROOT / PREREGISTRATION
    res: dict[str, Any] = {
        "schema": "roboforge.x1_clamp_analysis/1", "label": LABEL,
        "decision": "D-107", "preregistration": {"path": PREREGISTRATION,
                                                "sha256": sha256_file(prereg)},
        "git": git_revision(), "x1_git": x1_git,
        "generated_utc": datetime.now(tz=UTC).isoformat(),
        "inputs": {"session7_analysis": {"path": SESSION7_ANALYSIS[0],
                                         "sha256": SESSION7_ANALYSIS[1]},
                   "test_configurations": a7["configs"],
                   "robustness_config_sha256": rb.config_sha256(),
                   "taxonomy_sha256": tx.taxonomy_sha256(),
                   "x1_summaries": {e["path"]: e["sha256"] for e in x1.values()}},
        "method": "RESEARCH_PROTOCOL §9 hierarchical paired bootstrap, B = 10,000, seed "
                  "20261106; frozen keys first in the frozen order, clamped keys grouped with "
                  "their family; frozen part asserted equal to the Session 7 analysis",
        "tasks": {}}
    for task in a7["tasks"]:
        res["tasks"][task] = task_analysis(task, a7, x1, n, labels)
        print(task, "frozen bootstrap reproduced exactly")
    res["readings"] = readings(res)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "results.json").open("x") as fh:
        fh.write(json.dumps(res, indent=1, allow_nan=False) + "\n")
    (out_dir / "tables.md").write_text(tables(res))
    (out_dir / "taxonomy.json").write_text(json.dumps(
        {"taxonomy_sha256": tx.taxonomy_sha256(), "labels": labels}, indent=1) + "\n")
    print(res["readings"]["primary"]["reading"])
    print(f"-> {repo_relative(out_dir)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
