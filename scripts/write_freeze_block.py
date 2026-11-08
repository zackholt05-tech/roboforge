"""Generate the machine-readable block of ``docs/FINAL_TEST_FREEZE.md`` (D-067).

    uv run --frozen python scripts/write_freeze_block.py \
        --config lift=reports/session6/lift_test_configuration.json \
        --config can=reports/session7/can_test_configuration.json --out <block.md>

Every value comes from a committed record or file, never typed by hand: the per-task test
configuration (chosen trials, §7-selected checkpoints and their SHA-256, the held-out
manifest hash), the rollout rule, the §9/§14.6 analysis, the taxonomy hash and the SHA-256
of every file that runs or analyses the test. The block is pasted verbatim into the freeze
document, between its markers.
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


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--config", action="append", required=True, metavar="TASK=PATH")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args(argv)
    configs = {k: REPO_ROOT / v for k, v in (c.split("=", 1) for c in a.config)}
    data = block(configs)
    text = (f"{fz.BEGIN}\n```yaml\n" + yaml.safe_dump(data, sort_keys=False, width=100)
            + f"```\n{fz.END}\n")
    fz.parse_freeze(text)  # round-trip validation
    a.out.write_text(text)
    print(f"freeze block for {sorted(configs)} -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
