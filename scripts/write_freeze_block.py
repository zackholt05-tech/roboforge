"""Generate the machine-readable block of ``docs/FINAL_TEST_FREEZE.md`` (D-067).

    uv run --frozen python scripts/write_freeze_block.py \
        --config lift=reports/session6/lift_test_configuration.json \
        --config can=reports/session7/can_test_configuration.json --out <block.md>

Every value comes from a committed record or file, never typed by hand: the per-task test
configuration (chosen trials, §7-selected checkpoints and their SHA-256, the held-out
manifest hash), the rollout rule, the §9/§14.6 analysis, the taxonomy hash and the SHA-256
of every file that runs or analyses the test. The block is pasted verbatim into the freeze
document, between its markers. ``--document DATE`` writes the whole document: generated
prose and tables (what runs, schedules, configurations, the analysis, §14.6 verbatim, the
frozen files) followed by the block.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

from roboforge.eval import freeze as fz
from roboforge.eval.statistics import B_DEFAULT, BOOTSTRAP_SEED, LEVEL
from roboforge.eval.taxonomy import taxonomy_sha256
from roboforge.paths import REPO_ROOT, repo_relative
from roboforge.provenance import sha256_file

CODE = [
    "src/roboforge/eval/heldout.py", "src/roboforge/eval/freeze.py",
    "src/roboforge/eval/validation.py", "src/roboforge/eval/learned.py",
    "src/roboforge/eval/rollout.py", "src/roboforge/eval/policies.py",
    "src/roboforge/eval/statistics.py", "src/roboforge/eval/taxonomy.py",
    "src/roboforge/sim/env_adapter.py", "src/roboforge/sim/eval_states.py",
    "src/roboforge/sim/snapshot.py", "src/roboforge/sim/state_replay.py",
    "src/roboforge/data/features.py", "src/roboforge/data/normalization.py",
    "src/roboforge/provenance.py", "scripts/eval_heldout.py", "scripts/run_heldout.py",
    "scripts/analyze_heldout.py", "infra/containers/run_sim_cpu.sh",
    "configs/eval/platform.yaml", "configs/eval/suites.yaml",
    "configs/eval/failure_taxonomy.yaml",
]
COMPARISONS = {
    "H1": [[f, fz.BASELINE] for f in fz.FAMILY_ORDER],
    "H2": [["diffusion_unet", "bc_mlp"], ["diffusion_unet", "bc_rnn"]],
    "H2c": [["diffusion_unet", "bc_mlp_chunk"]],
    "H3": [["bc_rnn", "bc_mlp"]],
}


def block(configs: dict[str, Path]) -> dict:
    """The freeze block as a dict (validated by :func:`roboforge.eval.freeze.parse_freeze`)."""
    plat = yaml.safe_load((REPO_ROOT / "configs/eval/platform.yaml").read_text())
    tasks = {}
    for task, path in configs.items():
        rec = json.loads(path.read_text())
        assert rec["task"] == task
        fams = {}
        for fam in fz.FAMILY_ORDER:
            f = rec["families"][fam]
            fams[fam] = {"chosen_trial": f["chosen_trial"], "overrides": f["overrides"],
                         "runs": [{"seed": s["seed"], "train_run_id": s["train_run_id"],
                                   "step": s["selected_step"],
                                   "checkpoint_sha256": s["checkpoint_sha256"],
                                   "config_sha256": s["config_sha256"]}
                                  for s in f["seeds"]]}
        tasks[task] = {"heldout_manifest_sha256": rec["held_out_schedule"]
                       ["manifest_content_sha256"],
                       "n_states": rec["held_out_schedule"]["n_states"],
                       "test_configuration": {"path": repo_relative(path),
                                              "sha256": sha256_file(path)},
                       "families": fams,
                       "baseline": {"policy": fz.BASELINE, "policy_seed": 0}}
    return {
        "schema": fz.SCHEMA,
        "evaluation_platform": {"platform_id": plat["platform_id"], "image_id": plat["image_id"],
                                "runner": "infra/containers/run_sim_cpu.sh"},
        "rollouts": {"repeats": 1, "policy_seed": 0, "horizon": 400,
                     "termination": "first_success", "per_state": "one rollout per (checkpoint, "
                     "state); stochastic noise seed from (policy_seed, state_id, repeat 0)"},
        "tasks": tasks,
        "analysis": {
            "primary_metric": "success before the 400-step horizon (first-success rule); "
                              "k/n per checkpoint with a Wilson 95 % interval",
            "bootstrap": {"method": "hierarchical paired bootstrap (RESEARCH_PROTOCOL §9)",
                          "B": B_DEFAULT, "seed": BOOTSTRAP_SEED, "level": LEVEL,
                          "interval": "percentile, numpy linear interpolation",
                          "draw_order": ["states", *fz.FAMILY_ORDER, fz.BASELINE],
                          "module": "roboforge.eval.statistics.hierarchical_paired_bootstrap"},
            "comparisons": COMPARISONS,
            "secondary": {"name": "censored_episode_length",
                          "definition": "first-success step if successful, else 400",
                          "comparisons": ["H2", "H2c", "H3"]},
            "separation": "separated when the 95 % interval of a paired difference excludes 0; "
                          "otherwise not separated (reportable, not evidence of equality)",
            "aggregation": "per task only; no pooled Lift + Can number",
            "descriptive_only": ["mean first-success step over successful episodes",
                                 "per-seed values", "failure-category counts",
                                 "latency and training cost"]},
        "taxonomy": {"config": "configs/eval/failure_taxonomy.yaml",
                     "sha256": taxonomy_sha256()},
        "example_rule": "per task and family, among the frozen checkpoints ordered by seed: the "
                        "first success and the first failure (sorted state_id) of the lowest "
                        "seed that has one; plus, per failure category, the first episode in "
                        "(family order, seed, state_id). Rendered only from recorded states, "
                        "labelled illustration",
        "code_sha256": {p: sha256_file(REPO_ROOT / p) for p in CODE},
    }


def protocol_14_6() -> str:
    """RESEARCH_PROTOCOL §14.6, verbatim (the freeze repeats it)."""
    text = (REPO_ROOT / "docs" / "RESEARCH_PROTOCOL.md").read_text()
    start = text.index("### 14.6 ")
    rest = text[start:]
    nxt = [i for i in (rest.find("\n## ", 1), rest.find("\n### ", 1)) if i > 0]
    return rest[:min(nxt)].strip() if nxt else rest.strip()


def document(data: dict, configs: dict[str, Path], frozen_on: str) -> str:
    """The whole freeze document: generated prose and tables, then the block."""
    an, ro = data["analysis"], data["rollouts"]
    lines = [
        "# Final-test freeze: Level 1 core (Lift and Can)", "",
        f"Frozen on {frozen_on}, before any held-out rollout. Binding: RESEARCH_PROTOCOL §9, "
        "§10 and §14.6; DECISIONS D-067. Committing this file opens the lock on the held-out "
        "suites (`src/roboforge/sim/eval_states.py`, `lock_status`); nothing in it changes "
        "after the first held-out rollout. Every table and value below is generated by "
        "`scripts/write_freeze_block.py --document` from committed records; where prose and "
        "the machine-readable block differ, the block is authoritative.", "",
        "## 1. What runs", "",
        "- Per task, 12 checkpoints (4 families × training seeds 0, 1, 2), each the §7-selected "
        "checkpoint of the family's §6.3-chosen configuration, plus the non-learned "
        "`random_uniform` baseline (policy seed 0): 13 evaluations per task, 26 in all, each "
        "over every held-out start of its task, **once**.",
        f"- Platform: `{data['evaluation_platform']['platform_id']}` (image "
        f"`{data['evaluation_platform']['image_id']}`), through "
        f"`{data['evaluation_platform']['runner']}` with `ROBOFORGE_EVIDENCE_SESSION`; at most "
        "4 containers at a time (Gate C, D-051); `make eval-final` (`scripts/run_heldout.py`).",
        f"- Rollouts: R = {ro['repeats']} (one rollout per checkpoint and start), policy seed "
        f"{ro['policy_seed']}, first success or {ro['horizon']} steps, resets only from the "
        "stored snapshots. Stochastic policies (diffusion) draw their noise from "
        "`SeedSequence([policy_seed, sha256(state_id)[:8], repeat])` with repeat 0.", "",
        "## 2. Held-out schedules (by manifest content hash)", "",
        "| Task | Held-out manifest content SHA-256 | Starts | Test configuration record "
        "(SHA-256) |", "|---|---|---|---|"]
    for task, t in data["tasks"].items():
        tc = t["test_configuration"]
        lines.append(f"| {task} | `{t['heldout_manifest_sha256']}` | {t['n_states']} | "
                     f"`{tc['path']}` (`{tc['sha256'][:16]}…`) |")
    lines += ["", "## 3. Configurations and checkpoints", "",
              "Lift is exactly `reports/session6/lift_test_configuration.json` (D-058; Lift "
              "selections are final). Can comes from the Session 7 main suite "
              "(`reports/session7/suite/level1_core_can.json`).", ""]
    for task, t in data["tasks"].items():
        rec = json.loads(configs[task].read_text())
        lines += [f"### {task}", "",
                  "| Family | Chosen trial (overrides) | Seed | Training run | Step | "
                  "Checkpoint SHA-256 | Validation k/50 |", "|---|---|---|---|---|---|---|"]
        for fam, f in t["families"].items():
            vals = {s["seed"]: s["validation"]["successes"] for s in rec["families"][fam]["seeds"]}
            ov = ", ".join(f["overrides"]) or "base"
            for r in f["runs"]:
                lines.append(f"| {fam} | {f['chosen_trial']} ({ov}) | {r['seed']} | "
                             f"`{r['train_run_id']}` | {r['step']} | "
                             f"`{r['checkpoint_sha256']}` | {vals[r['seed']]} |")
        b = t["baseline"]
        lines += [f"| {b['policy']} | — | policy seed {b['policy_seed']} | — | — | — | — |", ""]
    bs = an["bootstrap"]
    lines += ["## 4. Analysis, fixed now", "",
              f"- **Primary metric:** {an['primary_metric']}.",
              f"- **Uncertainty:** {bs['method']}, B = {bs['B']:,}, seed {bs['seed']}, "
              f"{int(100 * bs['level'])} % {bs['interval']} intervals; draw order "
              f"{', '.join(bs['draw_order'])} (`{bs['module']}`). States are resampled once per "
              "replicate and shared by every family (pairing by start); each family's seeds "
              "are resampled independently. Time steps are never units of analysis.",
              "- **Declared comparisons, per task** (every other difference is exploratory; no "
              "p-values, no multiplicity correction):", ""]
    lines += ["| Hypothesis | Paired difference (A − B) |", "|---|---|"]
    for k, pairs in an["comparisons"].items():
        lines += [f"| {k} | {a} − {b} |" for a, b in pairs]
    sec = an["secondary"]
    lines += ["", f"- **Separated / not separated:** {an['separation']}.",
              f"- **Secondary outcome:** {sec['name']}, {sec['definition']}, for "
              f"{', '.join(sec['comparisons'])}, with the same bootstrap.",
              f"- **Aggregation:** {an['aggregation']}. Descriptive only: "
              f"{'; '.join(an['descriptive_only'])}.",
              "- **Three kinds of negative result, kept apart.** *Engineering-incomplete*: an "
              "evaluation with no summary, fewer than all starts, a failed `verify_record` "
              "audit or a non-canonical platform (rerun once only for a fault outside the "
              "policy before the summary, §14.6), and episodes labelled `engineering_error` "
              "(counted as failures in the primary metric, reported separately). *Task "
              "failure*: an episode that reached the horizon without success, labelled by the "
              "taxonomy. *Statistical uncertainty*: a declared difference whose interval "
              "contains 0 (not separated).",
              f"- **Failure taxonomy:** `{data['taxonomy']['config']}` (SHA-256 "
              f"`{data['taxonomy']['sha256']}`), RESEARCH_PROTOCOL §14.5, applied to every "
              "held-out episode.",
              f"- **Example episodes:** {data['example_rule']}.",
              "- **Outputs:** per-evaluation summaries `reports/session7/heldout/`, records "
              "`runs/<eval_run_id>/`; analysis `reports/session7/results/` (`make "
              "analyze-final`, `scripts/analyze_heldout.py`), written once.", "",
              "## 5. RESEARCH_PROTOCOL §14.6, verbatim", "", protocol_14_6(), "",
              "## 6. Frozen code", "",
              "`roboforge.eval.freeze.code_problems` must be empty before every held-out run "
              "and the analysis; the SHA-256 of each file is in the block.", ""]
    lines += [f"- `{rel}`" for rel in data["code_sha256"]]
    lines += ["", "## 7. Machine-readable block", ""]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--config", action="append", required=True, metavar="TASK=PATH")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--document", metavar="YYYY-MM-DD", default=None,
                   help="write the whole freeze document (frozen on this date), not only the block")
    a = p.parse_args(argv)
    configs = {k: REPO_ROOT / v for k, v in (c.split("=", 1) for c in a.config)}
    data = block(configs)
    text = (f"{fz.BEGIN}\n```yaml\n" + yaml.safe_dump(data, sort_keys=False, width=100)
            + f"```\n{fz.END}\n")
    if a.document:
        text = document(data, configs, a.document) + text
    if fz.parse_freeze(text) != data:  # round-trip validation
        raise SystemExit("the freeze block does not round-trip")
    a.out.write_text(text)
    print(f"freeze {'document' if a.document else 'block'} for {sorted(configs)} -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
