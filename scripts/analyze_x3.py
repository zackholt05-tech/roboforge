"""X3: the one analysis of the 24 evaluations on the X3 held-out starts (DECISIONS D-111).

    ROBOFORGE_EVIDENCE_SESSION=14 uv run --frozen python scripts/analyze_x3.py

**Separate replication.** It is reported beside the frozen Level 1 result, never instead of
it, and the two are never pooled. Everything here is fixed by
``docs/extensions/X3_PREREGISTRATION.md`` §5.

**Inputs, through a hash chain.**
- The 24 X3 summaries (``reports/session<N>/x3/evals/``). Each must be canonical, complete
  (n = 100), audited and from one clean commit that contains the preregistration, with the X3
  suite's content hash and the checkpoint the job list expects (``roboforge.replication.x3_jobs``).
- The frozen result ``reports/session7/results/results.json``, read only after its SHA-256
  equals the pinned value.

**Statistics.** The frozen §9 hierarchical paired bootstrap, unchanged (B = 10,000, seed
20261106, draw order states, bc_mlp, bc_rnn, bc_mlp_chunk, diffusion_unet), on 6 seeds × 100
starts per family; the D-095 Welch interval on the per-seed rates; the verdict, relation and
readings of the preregistration. Secondary: the censored episode length, and the contrasts on
seeds 0–2 and on seeds 3–5 alone. Descriptive: per-seed rates, the new runs' validation
selections, the frozen failure taxonomy, wall time.

**Outputs** (``reports/session<N>/x3/``, written once): ``results.json``, ``tables.md``
(generated) and ``taxonomy.json``. The script refuses to run if ``results.json`` exists.
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
from roboforge.eval.freeze import FAMILY_ORDER
from roboforge.eval.statistics import (
    BOOTSTRAP_SEED,
    hierarchical_paired_bootstrap,
    separation,
    wilson_interval,
)
from roboforge.paper.sensitivity import welch
from roboforge.paths import REPO_ROOT, evidence_reports_dir, repo_relative
from roboforge.provenance import git_revision, sha256_file
from roboforge.replication import x3_states
from roboforge.replication.x3 import PREREGISTRATION, SCHEMA
from roboforge.replication.x3_jobs import (
    PLATFORM,
    X3GateError,
    check_gate,
    failed_training_runs,
    jobs,
    validation_summary,
)

B = 10_000
HORIZON = 400
N_STATES = 100
SEEDS = (0, 1, 2, 3, 4, 5)
SUBSETS = {"seeds 0-2 (frozen checkpoints, new starts)": (0, 1, 2),
           "seeds 3-5 (new checkpoints, new starts)": (3, 4, 5)}
FROZEN_RESULTS = ("reports/session7/results/results.json",
                  "b8b097c38765c588711615917e3cfd5be6847e63083ddebe92f0f1488672408c")
CONTRASTS = (("H2", "diffusion_unet", "bc_mlp"), ("H2", "diffusion_unet", "bc_rnn"),
             ("H2c", "diffusion_unet", "bc_mlp_chunk"), ("H3", "bc_rnn", "bc_mlp"))
NAMES = {"bc_mlp": "BC-MLP", "bc_rnn": "BC-RNN", "bc_mlp_chunk": "BC-MLP-chunk",
         "diffusion_unet": "diffusion"}
HYPOTHESIS_LABEL = {"H2": "H2", "H2c": "H2-c", "H3": "H3"}
# The readings of docs/extensions/X3_PREREGISTRATION.md §5, verbatim (a unit test keeps them
# equal to the preregistration's text).
READINGS = {
    "verdict": {
        "robust": "{name}: separated ({sign}) under both the bootstrap and the seed-level "
                  "interval, with 6 seeds on 100 fresh starts. In this replication the "
                  "difference does not rest on 3 seeds or on the 50 frozen starts.",
        "bootstrap only": "{name}: separated ({sign}) under the bootstrap but not under the "
                          "seed-level interval. With 6 seeds the verdict still depends on the "
                          "interval method.",
        "seed-level only": "{name}: separated ({sign}) under the seed-level interval but not "
                           "under the bootstrap. The verdict depends on the interval method.",
        "not separated": "{name}: not separated under either interval, with 6 seeds on 100 "
                         "fresh starts. This replication gives no evidence of a difference; "
                         "that is not evidence of equality.",
        "conflicting": "{name}: the bootstrap and the seed-level interval separate on "
                       "opposite sides; no verdict.",
    },
    "relation": {
        "holds, robust": "The frozen separation holds in this replication, now under both "
                         "interval methods.",
        "holds, bootstrap only": "The frozen separation holds in this replication, still "
                                 "under the bootstrap only.",
        "not reproduced": "The frozen bootstrap separation is not reproduced in this "
                          "replication.",
        "reversed": "This replication separates in the opposite direction to the frozen "
                    "result.",
        "unchanged, not separated": "As in the frozen result, not separated.",
        "new separation": "Unlike the frozen result, this replication separates ({sign}) "
                          "{method}.",
        "no relation": "No relation to the frozen result can be stated.",
    },
    "closing": "Separate replication; the frozen result stands as reported.",
}
METHOD_PHRASE = {"robust": "under both interval methods",
                 "bootstrap only": "under the bootstrap only",
                 "seed-level only": "under the seed-level interval only"}


def _sign(sep: str) -> str | None:
    return {"separated (+)": "+", "separated (-)": "-"}.get(sep)


def welch_separation(ci: tuple[float, float] | None) -> str:
    """The frozen separation rule on a Welch interval ("undefined" without seed variance)."""
    return "undefined (no seed variance)" if ci is None else separation(list(ci))


def verdict(boot_sep: str, seed_sep: str) -> tuple[str, str | None]:
    """(verdict, sign) of one contrast from its two separations (preregistration §5)."""
    b, s = _sign(boot_sep), _sign(seed_sep)
    if b and s:
        return ("robust", b) if b == s else ("conflicting", None)
    if b:
        return "bootstrap only", b
    if s:
        return "seed-level only", s
    return "not separated", None


def relation(frozen_sep: str, v: str, sign: str | None, boot_sep: str) -> str:
    """Relation of the replication to the frozen bootstrap verdict (preregistration §5)."""
    if v == "conflicting":
        return "no relation"
    f, b = _sign(frozen_sep), _sign(boot_sep)
    if f:
        if b is None:
            return "not reproduced"
        if b != f:
            return "reversed"
        return "holds, robust" if v == "robust" else "holds, bootstrap only"
    return "unchanged, not separated" if v == "not separated" else "new separation"


def reading(name: str, v: str, sign: str | None, rel: str) -> str:
    """The three preregistered sentences for one contrast."""
    first = READINGS["verdict"][v].format(name=name, sign=sign)
    second = READINGS["relation"][rel].format(sign=sign, method=METHOD_PHRASE.get(v, ""))
    return f"{first} {second} {READINGS['closing']}"


def contrast_name(h: str, a: str, b: str) -> str:
    return f"{HYPOTHESIS_LABEL[h]} {NAMES[a]} − {NAMES[b]}"


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True,
                          check=True).stdout.strip()


def commit_contains_preregistration(sha: str) -> bool:
    """True if ``sha`` has the preregistration, byte-identical to the committed file."""
    try:
        blob = _git("rev-parse", f"{sha}:{PREREGISTRATION}")
    except subprocess.CalledProcessError:
        return False
    return blob == _git("rev-parse", f"HEAD:{PREREGISTRATION}")


def load_frozen() -> dict[str, Any]:
    """The frozen Can success analysis, read only after its SHA-256 matches the pinned one."""
    rel, want = FROZEN_RESULTS
    got = sha256_file(REPO_ROOT / rel)
    if got != want:
        raise SystemExit(f"{rel}: SHA-256 {got} != pinned {want}")
    return json.loads((REPO_ROOT / rel).read_text())["tasks"][x3_states.suite_task()]


def check_summary(s: dict[str, Any], j: dict[str, Any], gate: dict[str, Any]) -> list[str]:
    """Why one X3 summary is not a complete, canonical, audited evaluation of job ``j``."""
    problems = []
    if s["schema"] != SCHEMA or s["label"] != x3_states.LABEL:
        problems.append("schema/label")
    if not s["canonical_platform"] or s["platform"]["platform_id"] != PLATFORM:
        problems.append("platform")
    if s["partial"] or s["n_episodes"] != N_STATES or len(s["episodes"]) != N_STATES:
        problems.append("not complete")
    if not s["verify_passed"]:
        problems.append("audit failed")
    if s["suite"] != gate["suite"] or s["suite_manifest_content_sha256"] != gate[
            "content_sha256"]:
        problems.append("suite")
    if (s["checkpoint"]["sha256"] != j["checkpoint_sha256"]
            or s["train_run_id"] != j["train_run_id"] or s["family"] != j["family"]
            or int(s["seed"]) != j["seed"] or int(s["checkpoint"]["step"]) != j["step"]):
        problems.append("checkpoint")
    if s["git"]["dirty"] is not False:
        problems.append("dirty tree")
    return problems


def load_evaluations(reports: Path, gate: dict[str, Any]) -> tuple[list[dict], dict]:
    """The X3 summaries of every job, and the engineering problems of the others.

    Returns ``(complete, incomplete)``: ``incomplete`` maps an evaluation ID to its problems
    (a missing summary included). The complete summaries must come from one clean commit that
    contains the committed preregistration."""
    out, incomplete, shas = [], {}, set()
    evals_dir = reports / "x3" / "evals"
    for j in jobs(reports):
        path = evals_dir / f"{j['eval_run_id']}.json"
        if not path.exists():
            incomplete[j["eval_run_id"]] = {"job": j, "problems": ["no summary"]}
            continue
        s = json.loads(path.read_text())
        problems = check_summary(s, j, gate)
        if problems:
            incomplete[j["eval_run_id"]] = {"job": j, "problems": problems,
                                            "path": repo_relative(path)}
            continue
        shas.add(s["git"]["sha"])
        out.append({"job": j, "summary": s, "path": repo_relative(path),
                    "sha256": sha256_file(path)})
    if len(shas) > 1:
        raise SystemExit(f"the X3 summaries come from {len(shas)} commits, not one: {shas}")
    if shas and not commit_contains_preregistration(next(iter(shas))):
        raise SystemExit(f"commit {shas} does not contain the committed {PREREGISTRATION}")
    return out, incomplete


def matrices(evals: list[dict[str, Any]], state_ids: list[str],
             families: list[str]) -> tuple[dict, dict, dict]:
    """Per family: success and censored-length matrices (seeds x 100 starts, seeds ascending,
    starts in manifest order), and the seeds each row is."""
    succ, length, seeds = {}, {}, {}
    for fam in families:
        rows = sorted((e for e in evals if e["job"]["family"] == fam),
                      key=lambda e: e["job"]["seed"])
        rows_s, rows_l = [], []
        for e in rows:
            by_id = {ep["state_id"]: ep for ep in e["summary"]["episodes"]}
            if sorted(by_id) != sorted(state_ids):
                raise SystemExit(f"{e['path']}: episodes do not cover the X3 starts exactly")
            eps = [by_id[sid] for sid in state_ids]
            rows_s.append([1.0 if ep["success"] else 0.0 for ep in eps])
            rows_l.append([float(ep["first_success_step"]) if ep["success"] else float(HORIZON)
                           for ep in eps])
        succ[fam], length[fam] = np.array(rows_s), np.array(rows_l)
        seeds[fam] = [e["job"]["seed"] for e in rows]
    return succ, length, seeds


def analyse(mats: dict[str, np.ndarray], seeds_of: dict[str, list[int]],
            frozen: dict[str, Any] | None, seeds: tuple[int, ...] = SEEDS) -> dict[str, Any]:
    """The bootstrap, the seed-level intervals, verdicts and readings for one seed subset.

    Families are those of ``mats`` (complete families only), in the frozen draw order; a
    contrast with a family that is missing is reported as not computed."""
    order = [f for f in FAMILY_ORDER if f in mats]
    sub = {f: mats[f][[i for i, s in enumerate(seeds_of[f]) if s in seeds]] for f in order}
    order = [f for f in order if len(sub[f])]
    comps = [(a, b) for _, a, b in CONTRASTS if a in order and b in order]
    boot = hierarchical_paired_bootstrap({f: sub[f] for f in order}, comps, b=B,
                                         seed=BOOTSTRAP_SEED, order=order)
    diffs = {(d["a"], d["b"]): d for d in boot["differences"]}
    rows = []
    for h, a, b in CONTRASTS:
        name = contrast_name(h, a, b)
        if (a, b) not in diffs:
            rows.append({"hypothesis": h, "a": a, "b": b, "name": name,
                         "status": "not computed (ENGINEERING-INCOMPLETE family)"})
            continue
        d = diffs[(a, b)]
        ci, df = welch(boot["families"][a]["per_seed"], boot["families"][b]["per_seed"])
        seed_sep = welch_separation(ci)
        row = {"hypothesis": h, "a": a, "b": b, "name": name, "status": "computed",
               "n_seeds": {a: boot["families"][a]["n_seeds"], b: boot["families"][b]["n_seeds"]},
               "difference": d["difference"], "bootstrap_ci95": d["ci95"],
               "bootstrap_separation": d["separation"],
               "seed_level_ci95": None if ci is None else list(ci), "seed_level_df": df,
               "seed_level_separation": seed_sep}
        if frozen is not None:
            v, sign = verdict(d["separation"], seed_sep)
            fd = next(x for x in frozen["differences"] if (x["a"], x["b"]) == (a, b))
            fci, fdf = welch(frozen["families"][a]["per_seed"], frozen["families"][b]["per_seed"])
            rel = relation(fd["separation"], v, sign, d["separation"])
            row.update(verdict=v, sign=sign, relation=rel,
                       reading=reading(name, v, sign, rel),
                       frozen={"difference": fd["difference"], "bootstrap_ci95": fd["ci95"],
                               "bootstrap_separation": fd["separation"],
                               "seed_level_ci95": None if fci is None else list(fci),
                               "seed_level_df": fdf,
                               "seed_level_separation": welch_separation(fci),
                               "n_seeds": frozen["families"][a]["n_seeds"],
                               "n_states": frozen["families"][a]["n_states"]})
        rows.append(row)
    return {"seeds": list(seeds), "families": order, "bootstrap": boot, "contrasts": rows}


def _pp(x: float) -> str:
    return f"{100 * x:+.1f}"


def _ci(ci: list[float] | None) -> str:
    return "undefined" if ci is None else f"[{100 * ci[0]:+.1f}, {100 * ci[1]:+.1f}]"


def tables(res: dict[str, Any]) -> str:
    """The generated ``tables.md``."""
    p = res["primary"]
    lines = ["# X3 results (separate replication)", "",
             "Generated by `scripts/analyze_x3.py` from the 24 X3 summaries; every number below "
             "comes from `results.json`. The frozen Level 1 result stands as reported; this "
             "replication is shown beside it and never pooled with it.", "",
             "## Declared contrasts (Can): frozen beside the replication", "",
             "| Contrast | Frozen (3 seeds × 50 frozen starts): difference, bootstrap 95 %, "
             "verdict "
             "| Frozen seed-level 95 % | X3 (6 seeds × 100 X3 starts): difference | X3 bootstrap "
             "95 % | X3 seed-level 95 % (df) | X3 verdict | Relation |",
             "|---|---|---|---|---|---|---|---|"]
    for r in p["contrasts"]:
        if r["status"] != "computed":
            lines.append(f"| {r['name']} | | | | | | {r['status']} | |")
            continue
        f = r["frozen"]
        df = "" if r["seed_level_df"] is None else f" ({r['seed_level_df']:.1f})"
        v = r["verdict"] + (f" ({r['sign']})" if r["sign"] else "")
        lines.append(f"| {r['name']} | {_pp(f['difference'])} pp {_ci(f['bootstrap_ci95'])}, "
                     f"{f['bootstrap_separation']} | {_ci(f['seed_level_ci95'])} | "
                     f"{_pp(r['difference'])} pp | {_ci(r['bootstrap_ci95'])} | "
                     f"{_ci(r['seed_level_ci95'])}{df} | {v} | {r['relation']} |")
    lines += ["", "## Readings (preregistered, verbatim)", ""]
    lines += [f"- {r['reading']}" for r in p["contrasts"] if r["status"] == "computed"]
    eng = res["engineering"]
    lines += ["", "## Engineering status", "",
              "Families: " + ", ".join(f"{NAMES[f]} {s}" for f, s in eng["families"].items())
              + f". Failed training runs: {len(eng['failed_training_runs'])}. Incomplete "
              f"evaluations: {len(eng['incomplete'])}. Episodes with an engineering error "
              f"(counted as failures): {eng['n_errors_total']}. Seeds per family: "
              + ", ".join(f"{NAMES[f]} {len(s)}" for f, s in res["seeds_per_family"].items())
              + ". New-seed validation and selection: the suite stopped at its configuration "
              "choice (KI-062); `make x3-validate` finished them with the same commands and "
              "rule (D-112)."]
    lines += ["", "## Family success on the X3 starts (6 seeds × 100)", "",
              "| Family | Success % | Bootstrap 95 % | Per-seed % (seeds 0–5) "
              "| Frozen (3 × 50) % |",
              "|---|---|---|---|---|"]
    for fam in p["families"]:
        fb = p["bootstrap"]["families"][fam]
        fz = res["frozen_families"][fam]
        per = ", ".join(f"{100 * v:.0f}" for v in fb["per_seed"])
        lines.append(f"| {NAMES[fam]} | {100 * fb['estimate']:.1f} | "
                     f"[{100 * fb['ci95'][0]:.1f}, {100 * fb['ci95'][1]:.1f}] | {per} | "
                     f"{100 * fz['estimate']:.1f} |")
    lines += ["", "## Secondary (declared; no readings)", ""]
    for key, block in res["secondary"]["subsets"].items():
        lines += [f"**{key}.**", "", "| Contrast | Difference | Bootstrap 95 % | Seed-level 95 % |",
                  "|---|---|---|---|"]
        lines += [f"| {r['name']} | {_pp(r['difference'])} pp | {_ci(r['bootstrap_ci95'])} "
                  f"{r['bootstrap_separation']} | {_ci(r['seed_level_ci95'])} "
                  f"{r['seed_level_separation']} |" for r in block["contrasts"]
                  if r["status"] == "computed"]
        lines.append("")
    lines += ["**Censored episode length (steps; first success, else 400), 6 seeds.**", "",
              "| Contrast | Difference | Bootstrap 95 % |", "|---|---|---|"]
    hyp = {(a, b): h for h, a, b in CONTRASTS}
    for d in res["secondary"]["censored_episode_length"]["differences"]:
        name = contrast_name(hyp[(d["a"], d["b"])], d["a"], d["b"])
        lines.append(f"| {name} | {d['difference']:+.1f} | [{d['ci95'][0]:+.1f}, "
                     f"{d['ci95'][1]:+.1f}] {d['separation']} |")
    lines += ["", "## Evaluations", "",
              "| Family | Seed | Training run | Step | Successes / 100 | Wilson 95 % | Errors |",
              "|---|---|---|---|---|---|---|"]
    for e in res["evaluations"]:
        lines.append(f"| {NAMES[e['family']]} | {e['seed']} | `{e['train_run_id']}` | "
                     f"{e['step']} | {e['successes']} | [{100 * e['wilson95'][0]:.1f}, "
                     f"{100 * e['wilson95'][1]:.1f}] | {e['n_errors']} |")
    lines += ["", "## New runs: validation selections (development numbers)", "",
              "| Training run | Selected step | Validation successes / 50 |", "|---|---|---|"]
    for s in res["new_selections"]:
        lines.append(f"| `{s['train_run_id']}` | {s['step']} | {s['successes']} |")
    lines += ["", "## Failure taxonomy on the X3 starts (episodes, 6 seeds × 100)", "",
              "| Family | " + " | ".join(res["taxonomy_categories"]) + " |",
              "|---|" + "---|" * len(res["taxonomy_categories"])]
    for fam in FAMILY_ORDER:
        c = res["taxonomy"][fam]
        if c is None:
            continue
        lines.append(f"| {NAMES[fam]} | " + " | ".join(str(c[k]) for k in
                                                     res["taxonomy_categories"]) + " |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    del argv
    reports = evidence_reports_dir()
    out_dir = reports / "x3"
    if (out_dir / "results.json").exists():
        raise SystemExit(f"{repo_relative(out_dir / 'results.json')} exists: the X3 analysis "
                         "runs once")
    try:
        gate = check_gate()
        evals, incomplete = load_evaluations(reports, gate)
        failed_runs = failed_training_runs()
        vpath = reports / "suite" / "x3_validation.json"
        validation_summary(reports)  # committed, complete, the 12 X3 runs (D-112)
    except X3GateError as e:
        raise SystemExit(f"X3 gate: {e}") from e
    frozen = load_frozen()
    task, name = x3_states.suite_task(), x3_states.suite_name()
    from roboforge.sim.eval_states import read_manifest

    state_ids = [s["state_id"] for s in read_manifest(task, name)["states"]]
    bad = {v["job"]["family"] for v in incomplete.values()}
    families = [f for f in FAMILY_ORDER if f not in bad]
    engineering = {f: ("ENGINEERING-INCOMPLETE" if f in bad else "complete")
                   for f in FAMILY_ORDER}
    succ, length, seeds_of = matrices(evals, state_ids, families)
    primary = analyse(succ, seeds_of, frozen["success"])
    subsets = {k: analyse(succ, seeds_of, None, seeds) for k, seeds in SUBSETS.items()}
    cel_comps = [(a, b) for _, a, b in CONTRASTS if a in families and b in families]
    cel = hierarchical_paired_bootstrap(length, cel_comps, b=B, seed=BOOTSTRAP_SEED,
                                        order=families)
    labels, tax, rows = {}, {f: None for f in FAMILY_ORDER}, []
    cats = tx.taxonomy_config()["categories"]
    for e in evals:
        s, j = e["summary"], e["job"]
        labs = tx.classify_records(REPO_ROOT / s["records_dir"])
        if sum(x["category"] == "success" for x in labs) != s["successes"]:
            raise SystemExit(f"{e['path']}: taxonomy and summary disagree on successes")
        labels[s["eval_run_id"]] = labs
        c = tx.counts(labs)
        tax[j["family"]] = c if tax[j["family"]] is None else {
            k: tax[j["family"]][k] + c[k] for k in cats}
        lo, hi = wilson_interval(s["successes"], s["n_episodes"])
        rows.append({"family": j["family"], "seed": j["seed"],
                     "train_run_id": j["train_run_id"], "step": j["step"],
                     "checkpoint_sha256": j["checkpoint_sha256"], "source": j["source"],
                     "eval_run_id": s["eval_run_id"], "summary": e["path"],
                     "summary_sha256": e["sha256"], "successes": s["successes"],
                     "n_episodes": s["n_episodes"], "wilson95": [lo, hi],
                     "n_errors": s["n_errors"], "wall_s": s["wall_s"],
                     "taxonomy": c})
    selections = []
    for e in evals:
        j = e["job"]
        if j["seed"] >= 3:
            rec = json.loads((REPO_ROOT / j["source"]).read_text())
            selections.append({"train_run_id": j["train_run_id"], "step": j["step"],
                               "successes": rec["chosen"]["successes"],
                               "first_success_step_mean": rec["chosen"][
                                   "first_success_step_mean"],
                               "record": j["source"], "record_sha256": j["source_sha256"]})
    res = {"schema": "roboforge.x3_results/1", "label": x3_states.LABEL,
           "decision": x3_states.DECISION, "preregistration": PREREGISTRATION,
           "preregistration_sha256": gate["preregistration_sha256"],
           "suite": {"task": task, "name": name, "content_sha256": gate["content_sha256"],
                     "n_states": len(state_ids)},
           "frozen_source": {"path": FROZEN_RESULTS[0], "sha256": FROZEN_RESULTS[1]},
           "frozen_families": frozen["success"]["families"],
           "primary": primary,
           "secondary": {"subsets": subsets, "censored_episode_length": cel},
           "engineering": {"families": engineering,
                           "incomplete": {k: {"problems": v["problems"],
                                              "path": v.get("path"),
                                              "family": v["job"]["family"],
                                              "seed": v["job"]["seed"]}
                                          for k, v in incomplete.items()},
                           "failed_training_runs": failed_runs,
                           "n_errors_total": sum(r["n_errors"] for r in rows),
                           "validation_and_selection": {
                               "decision": "D-112", "known_issue": "KI-062",
                               "summary": repo_relative(vpath),
                               "summary_sha256": sha256_file(vpath)}},
           "seeds_per_family": seeds_of,
           "evaluations": rows, "new_selections": selections,
           "taxonomy_categories": list(cats), "taxonomy": tax,
           "taxonomy_sha256": tx.taxonomy_sha256(),
           "wall_s_total": sum(r["wall_s"] for r in rows),
           "evaluations_git": evals[0]["summary"]["git"], "git": git_revision(),
           "generated_utc": datetime.now(tz=UTC).isoformat()}
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "results.json").open("x") as fh:
        fh.write(json.dumps(res, indent=1, allow_nan=False) + "\n")
    with (out_dir / "tables.md").open("x") as fh:
        fh.write(tables(res))
    with (out_dir / "taxonomy.json").open("x") as fh:
        fh.write(json.dumps({"taxonomy_sha256": tx.taxonomy_sha256(), "labels": labels},
                            indent=1) + "\n")
    for r in primary["contrasts"]:
        print(r["reading"] if r["status"] == "computed" else f"{r['name']}: {r['status']}")
    print(f"-> {repo_relative(out_dir)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
