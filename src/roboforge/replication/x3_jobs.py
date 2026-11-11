"""X3 jobs on the host: the 24 checkpoints and the gate (DECISIONS D-111).

**Separate replication.** Host side only (it asks git). Used by ``scripts/run_x3.py`` to plan
and gate the 24 evaluations, and by ``scripts/analyze_x3.py`` to check every summary against
the checkpoint it should hold.

- Seeds 0–2: the frozen Can checkpoints of the committed ``docs/FINAL_TEST_FREEZE.md``.
- Seeds 3–5: the RESEARCH_PROTOCOL §7 selections of the 12 X3 training runs of
  ``configs/experiments/x3_can.yaml``, each from its committed selection record, with the
  chosen SHA-256 checked against the training manifest and the committed validation summary
  (``reports/session14/suite/x3_validation.json``, ``scripts/x3_validate.py``, D-112).
- The gate: ``docs/extensions/X3_PREREGISTRATION.md`` is committed unmodified and names the X3
  suite's content hash; the suite's files are committed unmodified and agree with each other.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from roboforge.eval.freeze import FAMILY_ORDER, load_freeze
from roboforge.paths import REPO_ROOT, repo_relative
from roboforge.provenance import sha256_file, tracked_and_unmodified
from roboforge.replication import x3_states
from roboforge.replication import x3_validation as xv
from roboforge.replication.x3 import PREREGISTRATION, x3_eval_run_id
from roboforge.sim import eval_states as es
from roboforge.train.run import read_manifest as read_train_manifest
from roboforge.train.suite import load_plan


class X3GateError(RuntimeError):
    """A precondition of the X3 evaluations does not hold."""


PLATFORM = "linux-aarch64-container-1f685638f28c"
PLAN = "configs/experiments/x3_can.yaml"
SELECTION_RULE = ("highest validation success count over all 50 validation starts "
                  "(canonical platform); ties -> lower mean first-success step over "
                  "successful episodes; remaining ties -> earlier checkpoint "
                  "(RESEARCH_PROTOCOL §7)")
RUNS_DIR = REPO_ROOT / "runs"
# Launch order: diffusion first (its evaluations are the longest), then the frozen family order.
LAUNCH_ORDER = ("diffusion_unet", "bc_mlp", "bc_rnn", "bc_mlp_chunk")


def frozen_checkpoints() -> list[dict[str, Any]]:
    """The 12 frozen Can checkpoints (seeds 0-2) of the committed freeze."""
    t = load_freeze()["tasks"][x3_states.suite_task()]["families"]
    return [{"family": fam, "seed": int(r["seed"]), "train_run_id": r["train_run_id"],
             "step": int(r["step"]), "checkpoint_sha256": r["checkpoint_sha256"],
             "source": "docs/FINAL_TEST_FREEZE.md"}
            for fam in FAMILY_ORDER for r in t[fam]["runs"]]


def new_run_ids() -> list[tuple[str, int, str]]:
    """(family, seed, run ID) of the 12 X3 training runs, resolved from the plan."""
    plan = load_plan(REPO_ROOT / PLAN)
    out = []
    for f in plan.families:
        for seed in (plan.tuning_seed, *plan.replicate_seeds):
            out.append((f.name, seed, plan.spec(f.name, 1, seed).run_id))
    return out


def validation_summary(reports: Path) -> dict[str, Any]:
    """The committed outcome of the X3 validation and selection (``scripts/x3_validate.py``,
    D-112): every run selected or failed in training.

    Required before any X3 job is planned, so that a run still training is never mistaken
    for a failed one."""
    path = reports / "suite" / xv.SUMMARY_NAME
    rel = repo_relative(path)
    ok, why = tracked_and_unmodified(rel)
    if not ok:
        raise X3GateError(f"X3 validation summary not usable: {why}")
    s = json.loads(path.read_text())
    if (s.get("schema") != xv.SCHEMA or s["plan"] != PLAN or not s["complete"]
            or s["plan_sha256"] != sha256_file(REPO_ROOT / PLAN)):
        raise X3GateError(f"{rel}: not the complete validation summary of {PLAN}")
    if sorted(r["run_id"] for r in s["runs"]) != sorted(rid for _, _, rid in new_run_ids()):
        raise X3GateError(f"{rel}: not the 12 X3 runs")
    return s


def failed_training_runs() -> list[dict[str, Any]]:
    """X3 runs whose training did not complete (no manifest, or a status other than
    ``complete``). Preregistration §2: such a run is reported and not replaced."""
    out = []
    for fam, seed, rid in new_run_ids():
        path = RUNS_DIR / rid / "manifest.json"
        status = json.loads(path.read_text()).get("status") if path.exists() else None
        if status != "complete":
            out.append({"family": fam, "seed": int(seed), "train_run_id": rid,
                        "status": status})
    return out


def new_checkpoints(reports: Path) -> list[dict[str, Any]]:
    """The §7 selections of the completed X3 runs, each from its committed selection record."""
    failed = {r["train_run_id"] for r in failed_training_runs()}
    out = []
    for fam, seed, rid in new_run_ids():
        if rid in failed:
            continue
        rec_path = reports / "selection" / f"{rid}.json"
        rel = repo_relative(rec_path)
        ok, why = tracked_and_unmodified(rel)
        if not ok:
            raise X3GateError(f"selection record not usable: {why}")
        rec = json.loads(rec_path.read_text())
        if rec["train_run_id"] != rid or rec["rule"] != SELECTION_RULE:
            raise X3GateError(f"{rel}: not the §7 selection of {rid}")
        ch = rec["chosen"]
        tm = read_train_manifest(RUNS_DIR / rid)
        ck = {c["step"]: c["sha256"] for c in tm["checkpoints"]}
        if ck.get(int(ch["step"])) != ch["sha256"]:
            raise X3GateError(f"{rid}: chosen checkpoint does not match the training manifest")
        (row,) = [r for r in validation_summary(reports)["runs"] if r["run_id"] == rid]
        if (row["selection"] is None or row["selection"]["sha256"] != ch["sha256"]
                or row["selection_record_sha256"] != sha256_file(rec_path)):
            raise X3GateError(f"{rid}: selection differs from the validation summary")
        out.append({"family": fam, "seed": int(seed), "train_run_id": rid,
                    "step": int(ch["step"]), "checkpoint_sha256": ch["sha256"],
                    "source": rel, "source_sha256": sha256_file(rec_path)})
    return out


def check_gate() -> dict[str, Any]:
    """The preregistration and suite gate; returns the suite's content hash and blob."""
    ok, why = tracked_and_unmodified(PREREGISTRATION)
    if not ok:
        raise X3GateError(f"preregistration not usable: {why}")
    task, name = x3_states.suite_task(), x3_states.suite_name()
    m = es.read_manifest(task, name)
    if m["content_sha256"] not in (REPO_ROOT / PREREGISTRATION).read_text():
        raise X3GateError(f"{PREREGISTRATION} does not name {task}/{name} content_sha256")
    for rel in (repo_relative(es.manifest_path(task, name)),
                repo_relative(es.data_path(task, name))):
        ok, why = tracked_and_unmodified(rel)
        if not ok:
            raise X3GateError(f"X3 suite not usable: {why}")
    if not es.verify_data_file(task, name):
        raise X3GateError(f"{task}/{name}: data file does not match its manifest")
    return {"suite": name, "content_sha256": m["content_sha256"],
            "preregistration_sha256": sha256_file(REPO_ROOT / PREREGISTRATION)}


def jobs(reports: Path) -> list[dict[str, Any]]:
    """Every X3 evaluation, in launch order: 24, less one per failed training run. Every
    failed run must be recorded as failed (a final ``failed`` manifest) by the committed
    validation summary."""
    reported = set(validation_summary(reports)["failed"])
    unreported = [r["train_run_id"] for r in failed_training_runs()
                  if r["train_run_id"] not in reported or r["status"] != "failed"]
    if unreported:
        raise X3GateError(f"runs without a complete training manifest that the validation "
                          f"summary does not record as failed: {unreported}")
    allc = frozen_checkpoints() + new_checkpoints(reports)
    out = []
    for fam in LAUNCH_ORDER:
        for c in sorted((c for c in allc if c["family"] == fam), key=lambda c: c["seed"]):
            out.append({**c, "eval_run_id": x3_eval_run_id(
                c["train_run_id"], c["step"], PLATFORM,
                stochastic=fam == "diffusion_unet")})
    failed = {(r["family"], r["seed"]) for r in failed_training_runs()}
    expected = sorted((f, s) for f in FAMILY_ORDER for s in range(6) if (f, s) not in failed)
    if sorted((j["family"], j["seed"]) for j in out) != expected:
        raise X3GateError("the jobs are not the 4 families x seeds 0-5 (less failed runs)")
    if len({j["eval_run_id"] for j in out}) != len(out):
        raise X3GateError("two jobs share an evaluation ID")
    return out
