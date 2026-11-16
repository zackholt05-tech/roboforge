"""X2-c: the one preregistered analysis of the 12 X2 held-out evaluations (DECISIONS D-130).

    ROBOFORGE_EVIDENCE_SESSION=17 uv run --frozen python scripts/analyze_x2c.py

**Label.** "X2 (Can MH)": reported beside the frozen Level 1 Can PH result, never instead of
it, never pooled with it; within-operator claims only. The statistics, contrasts, verdicts
and relation are ``docs/extensions/X2_PREREGISTRATION.md`` §7 with the D-130 clarifications
(:mod:`roboforge.x2.analysis`).

**Inputs, through a hash chain.**
- The X2 freeze, committed unmodified; every file it pins unchanged; its analysis design
  equal to the code's.
- The 12 summaries (``reports/session<N>/x2/heldout/evals/``). Each must be canonical,
  complete (n = 100), audited, on the frozen suite and checkpoint, from a clean commit whose
  X2 freeze and pinned files equal the freeze's; a second attempt needs its recorded decision.
- The frozen Level 1 Can PH result and the X2-b latency benchmark, read only after their
  SHA-256 equals the pinned value.

**Outputs** (``reports/session<N>/x2/``, written once, exclusive create): ``results.json``,
``tables.md`` (generated), ``taxonomy.json`` and ``analysis_record.json``. The script refuses
to run if ``results.json`` exists.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from roboforge.eval.freeze import FAMILY_ORDER
from roboforge.eval.statistics import wilson_interval
from roboforge.paths import REPO_ROOT, evidence_session, repo_relative
from roboforge.provenance import git_revision, sha256_file
from roboforge.x2 import analysis as xa
from roboforge.x2 import heldout as xh
from roboforge.x2 import heldout_gate as xg
from roboforge.x2.data import DATASET_ID

RESULTS_SCHEMA = "roboforge.x2c_results/1"


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True,
                          check=True).stdout.strip()


def commit_matches_freeze(sha: str, freeze: dict[str, Any], blob: str) -> bool:
    """True if commit ``sha`` holds this X2 freeze (same blob) and every file it pins with the
    pinned SHA-256 (D-130: a rerun may come from a later commit that only adds records)."""
    try:
        if _git("rev-parse", f"{sha}:{xh.FREEZE_REL}") != blob:
            return False
        for rel, want in freeze["code_sha256"].items():
            data = subprocess.run(["git", "show", f"{sha}:{rel}"], cwd=REPO_ROOT,
                                  capture_output=True, check=True).stdout
            if hashlib.sha256(data).hexdigest() != want:
                return False
    except subprocess.CalledProcessError:
        return False
    return True


def check_summary(s: dict[str, Any], job: dict[str, Any], freeze: dict[str, Any], blob: str,
                  *, decision_ok: Callable[[str], bool]) -> list[str]:
    """Why one summary is not a complete, canonical, audited evaluation of frozen ``job``."""
    problems = []
    if (s.get("schema"), s.get("label"), s.get("study")) != (xh.SCHEMA, xh.LABEL, xh.STUDY):
        problems.append("schema/label")
    if not s["canonical_platform"] or s["platform"]["platform_id"] != freeze[
            "evaluation_platform"]["platform_id"]:
        problems.append("platform")
    if s["partial"] or s["n_episodes"] != xh.N_STATES or len(s["episodes"]) != xh.N_STATES:
        problems.append("not complete")
    if not s["verify_passed"]:
        problems.append("audit failed")
    if (s["suite"], s["suite_manifest_content_sha256"]) != (
            freeze["suite"]["name"], freeze["suite"]["manifest_content_sha256"]):
        problems.append("suite")
    if (s["checkpoint"]["sha256"], s["train_run_id"], s["family"], int(s["seed"]),
            int(s["checkpoint"]["step"])) != (job["checkpoint_sha256"], job["train_run_id"],
                                              job["family"], job["seed"], job["step"]):
        problems.append("checkpoint")
    if s["eval_run_id"] != job["eval_run_id"] or s["dataset_id"] != DATASET_ID:
        problems.append("evaluation ID or dataset")
    if int(s["episode_rules"]["horizon"]) != xh.HORIZON:
        problems.append("horizon")
    if s["git"]["dirty"] is not False:
        problems.append("dirty tree")
    if (s.get("freeze") or {}).get("git_blob") != blob:
        problems.append("freeze blob")
    attempt, dec = int(s.get("attempt", 0)), s.get("rerun_decision")
    if not (attempt == 1 and dec is None) and not (attempt == 2 and dec and decision_ok(dec)):
        problems.append("attempt")
    return problems


def load_evaluations(summaries_dir: Path, jobs: list[dict[str, Any]], freeze: dict[str, Any],
                     blob: str, *, commit_ok: Callable[[str], bool],
                     decision_ok: Callable[[str], bool]) -> tuple[list[dict], dict]:
    """``(complete, incomplete)``: the summary of every frozen job, and the engineering
    problems of the others (a missing summary included)."""
    out, incomplete = [], {}
    for j in jobs:
        path = summaries_dir / f"{j['eval_run_id']}.json"
        if not path.exists():
            incomplete[j["eval_run_id"]] = {"job": j, "problems": ["no summary"]}
            continue
        s = json.loads(path.read_text())
        problems = check_summary(s, j, freeze, blob, decision_ok=decision_ok)
        if not problems and not commit_ok(s["git"]["sha"]):
            problems.append("commit does not hold the X2 freeze and its pinned files")
        if problems:
            incomplete[j["eval_run_id"]] = {"job": j, "problems": problems,
                                            "path": repo_relative(path)}
            continue
        out.append({"job": j, "summary": s, "path": repo_relative(path),
                    "sha256": sha256_file(path)})
    return out, incomplete


def build_results(evals: list[dict[str, Any]], incomplete: dict[str, Any],
                  state_ids: list[str], frozen: dict[str, Any], latency: dict[str, Any],
                  freeze: dict[str, Any], classify: Callable[[Path], list[dict[str, Any]]],
                  taxonomy_categories: list[str]) -> tuple[dict[str, Any], dict[str, Any]]:
    """``(results, labels)`` from the checked evaluations (no file is read but records)."""
    bad = {v["job"]["family"] for v in incomplete.values()}
    families = [f for f in FAMILY_ORDER if f not in bad]
    succ, length = xa.matrices(evals, state_ids, families)
    primary = xa.analyse(succ, length, frozen)
    lat = {r["checkpoint_sha256"]: r["observe_plus_act_ms"]["median"] for r in latency["rows"]}
    val = {(fam, int(r["seed"])): int(r["validation_successes"])
           for fam, f in freeze["families"].items() for r in f["runs"]}
    labels, tax, rows = {}, {}, []
    for e in sorted(evals, key=lambda e: (FAMILY_ORDER.index(e["job"]["family"]),
                                          e["job"]["seed"])):
        s, j = e["summary"], e["job"]
        labs = classify(REPO_ROOT / s["records_dir"])
        if sum(x["category"] == "success" for x in labs) != s["successes"] or len(labs) != len(
                state_ids):
            raise SystemExit(f"{e['path']}: taxonomy and summary disagree")
        labels[s["eval_run_id"]] = labs
        c = {k: sum(x["category"] == k for x in labs) for k in taxonomy_categories}
        if sum(c.values()) != len(labs):
            raise SystemExit(f"{e['path']}: labels outside the taxonomy")
        tax[j["family"]] = c if j["family"] not in tax else {
            k: tax[j["family"]][k] + c[k] for k in taxonomy_categories}
        lo, hi = wilson_interval(s["successes"], s["n_episodes"])
        rows.append({"family": j["family"], "seed": j["seed"],
                     "train_run_id": j["train_run_id"], "step": j["step"],
                     "checkpoint_sha256": j["checkpoint_sha256"],
                     "eval_run_id": s["eval_run_id"], "summary": e["path"],
                     "summary_sha256": e["sha256"], "successes": s["successes"],
                     "n_episodes": s["n_episodes"], "wilson95": [lo, hi],
                     "n_errors": s["n_errors"], "wall_s": s["wall_s"],
                     "attempt": s["attempt"], "rerun_decision": s["rerun_decision"],
                     "validation_successes": val[(j["family"], j["seed"])],
                     "latency_ms_median": lat.get(j["checkpoint_sha256"]), "taxonomy": c})
    engineering = {"families": {f: ("ENGINEERING-INCOMPLETE" if f in bad else "complete")
                                for f in FAMILY_ORDER},
                   "incomplete": {k: {"problems": v["problems"], "path": v.get("path"),
                                      "family": v["job"]["family"], "seed": v["job"]["seed"]}
                                  for k, v in incomplete.items()},
                   "n_errors_total": sum(r["n_errors"] for r in rows),
                   "n_attempt2": sum(r["attempt"] == 2 for r in rows),
                   "rule": xa.design()["engineering_incomplete"]}
    res = {"schema": RESULTS_SCHEMA, "study": xh.STUDY, "label": xh.LABEL,
           "decisions": [*freeze["decisions"]], "design": xa.design(),
           "primary": primary, "frozen_families": frozen["families"],
           "engineering": engineering, "evaluations": rows,
           "taxonomy_categories": list(taxonomy_categories), "taxonomy": tax,
           "wall_s_total": sum(r["wall_s"] for r in rows),
           "evaluations_git": sorted({e["summary"]["git"]["sha"] for e in evals})}
    return res, labels


def write_once(path: Path, text: str) -> None:
    """Exclusive create: an output is written once."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as fh:
        fh.write(text)


def _pinned_json(rel: str, want: str) -> dict[str, Any]:
    got = sha256_file(REPO_ROOT / rel)
    if got != want:
        raise SystemExit(f"{rel}: SHA-256 {got} != pinned {want}")
    return json.loads((REPO_ROOT / rel).read_text())


def run(out_dir: Path, summaries_dir: Path, jobs: list[dict[str, Any]],
        freeze: dict[str, Any], blob: str, frozen: dict[str, Any], latency: dict[str, Any],
        state_ids: list[str], *, classify: Callable[[Path], list[dict[str, Any]]],
        categories: list[str], taxonomy_sha256: str, commit_ok: Callable[[str], bool],
        decision_ok: Callable[[str], bool]) -> dict[str, Any]:
    """Load and check the 12 summaries, compute everything and write the four outputs once."""
    if (out_dir / "results.json").exists():
        raise SystemExit(f"{repo_relative(out_dir / 'results.json')} exists: the X2-c analysis "
                         "runs once")
    evals, incomplete = load_evaluations(summaries_dir, jobs, freeze, blob,
                                         commit_ok=commit_ok, decision_ok=decision_ok)
    res, labels = build_results(evals, incomplete, state_ids, frozen, latency, freeze,
                                classify, categories)
    res.update({"preregistration": {"path": xh.PREREGISTRATION,
                                    "sha256": freeze["preregistration"]["sha256"]},
                "freeze": {"path": xh.FREEZE_REL, "git_blob": blob},
                "test_configuration": freeze["test_configuration"],
                "suite": {**freeze["suite"], "state_order": "manifest order"},
                "frozen_source": {"path": xa.FROZEN_RESULTS[0], "sha256": xa.FROZEN_RESULTS[1]},
                "latency_source": {"path": xa.LATENCY[0], "sha256": xa.LATENCY[1]},
                "clarifications": freeze["clarifications"],
                "taxonomy_sha256": taxonomy_sha256, "git": git_revision(),
                "generated_utc": datetime.now(tz=UTC).isoformat()})
    results_text = json.dumps(res, indent=1, allow_nan=False) + "\n"
    tables_text = xa.tables(res)
    tax_text = json.dumps({"taxonomy_sha256": taxonomy_sha256, "labels": labels},
                          indent=1) + "\n"
    write_once(out_dir / "results.json", results_text)
    write_once(out_dir / "tables.md", tables_text)
    write_once(out_dir / "taxonomy.json", tax_text)
    record = {"schema": "roboforge.x2c_analysis_record/1", "study": xh.STUDY,
              "command": "ROBOFORGE_EVIDENCE_SESSION=17 uv run --frozen python "
                         "scripts/analyze_x2c.py (make x2c-analyze)",
              "git": res["git"], "generated_utc": res["generated_utc"],
              "freeze": res["freeze"],
              "inputs": [{"path": e["summary"], "sha256": e["summary_sha256"]}
                         for e in res["evaluations"]]
              + [res["frozen_source"], res["latency_source"],
                 {"path": xh.TEST_CONFIGURATION, "sha256": xh.TEST_CONFIGURATION_SHA256}],
              "code_sha256": freeze["code_sha256"],
              "outputs": {name: hashlib.sha256(text.encode()).hexdigest() for name, text in
                          (("results.json", results_text), ("tables.md", tables_text),
                           ("taxonomy.json", tax_text))}}
    write_once(out_dir / "analysis_record.json", json.dumps(record, indent=1) + "\n")
    return res


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint: the gate, then :func:`run` on the committed inputs."""
    del argv
    from roboforge.eval import taxonomy as tx
    from roboforge.sim.eval_states import read_manifest
    from roboforge.x2.host import tree_clean

    session = evidence_session()
    if not tree_clean():
        raise SystemExit("the working tree is dirty: the X2-c analysis runs from a clean commit")
    try:
        freeze = xh.load_x2_freeze()
        blob = xh.freeze_blob()
        inputs = xg.check_inputs(freeze)
    except (xh.X2FreezeError, xg.X2cGateError) as e:
        raise SystemExit(f"X2-c analysis gate: {e}") from e
    if freeze["analysis"] != xa.design():
        raise SystemExit("the X2 freeze's analysis design differs from roboforge.x2.analysis")
    out_dir = REPO_ROOT / freeze["outputs"]["analysis"]
    if repo_relative(out_dir) != f"reports/session{session}/x2":
        raise SystemExit(f"session {session}: the X2 freeze writes the analysis to {out_dir}")
    jobs = inputs["jobs"]
    foreign = xg.foreign_outputs(xg.existing_heldout_outputs(), jobs, session)
    if foreign:
        raise SystemExit(f"held-out outputs that belong to no frozen job: {foreign[:5]}")
    frozen = _pinned_json(*xa.FROZEN_RESULTS)["tasks"][freeze["suite"]["task"]]["success"]
    latency = _pinned_json(*xa.LATENCY)
    state_ids = [s["state_id"] for s in read_manifest(freeze["suite"]["task"],
                                                      freeze["suite"]["name"])["states"]]
    res = run(out_dir, REPO_ROOT / freeze["outputs"]["summaries"], jobs, freeze, blob, frozen,
              latency, state_ids, classify=tx.classify_records,
              categories=list(tx.taxonomy_config()["categories"]),
              taxonomy_sha256=tx.taxonomy_sha256(),
              commit_ok=lambda sha: commit_matches_freeze(sha, freeze, blob),
              decision_ok=xg.decision_recorded)
    for r in res["primary"]["contrasts"]:
        print(xa.statement(r))
    print(f"-> {repo_relative(out_dir)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
