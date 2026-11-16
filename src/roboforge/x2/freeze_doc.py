"""The X2 freeze document, generated from committed files (DECISIONS D-130).

``docs/extensions/X2_FREEZE.md`` follows the Level 1 template (``docs/FINAL_TEST_FREEZE.md``):
prose sections generated from the same values, a verbatim copy of the X2-b test
configuration, and one machine-readable YAML block (schema ``roboforge.x2_freeze/1``) that is
authoritative. :func:`build_block` assembles the block; :func:`render` writes the document;
:func:`roboforge.x2.heldout.parse_x2_freeze` reads it back and checks the copy and the block
against each other. Nothing here reads a held-out start.
"""

from __future__ import annotations

import json
from typing import Any

import yaml

from roboforge.eval.freeze import FAMILY_ORDER
from roboforge.paths import REPO_ROOT
from roboforge.provenance import sha256_file
from roboforge.x2 import analysis as xa
from roboforge.x2 import heldout as xh
from roboforge.x2 import states as x2s

# Every file that runs or analyses the X2 held-out test (the Level 1 frozen code it reuses is
# pinned by docs/FINAL_TEST_FREEZE.md and checked by `make freeze-check`).
CODE_FILES = (
    "src/roboforge/x2/heldout.py",
    "src/roboforge/x2/heldout_gate.py",
    "src/roboforge/x2/analysis.py",
    "src/roboforge/x2/freeze_doc.py",
    "scripts/x2c_eval.py",
    "scripts/x2c_run.py",
    "scripts/x2c_check.py",
    "scripts/write_x2_freeze.py",
    "scripts/analyze_x2c.py",
    "src/roboforge/x2/__init__.py",
    "src/roboforge/x2/evaluate.py",
    "src/roboforge/x2/binding.py",
    "src/roboforge/x2/data.py",
    "src/roboforge/x2/states.py",
    "src/roboforge/x2/host.py",
    "src/roboforge/paper/sensitivity.py",
    "configs/extensions/x2_suites.yaml",
    "data/manifests/tasks/can_mh.json",
    "data/splits/can_mh.json",
    "data/normalization/can_mh_lowdim_v1.json",
    "infra/containers/run_sim_cpu.sh",
    "configs/eval/platform.yaml",
    "configs/eval/failure_taxonomy.yaml",
)
DECISIONS = tuple(f"D-{n}" for n in range(116, 131))
LIMITS = {"max_containers": 4, "swap_pages_per_s": 3000.0, "min_free_gib": 4.0,
          "hard_stop_free_gib": 2.0, "sample_every_s": 60}
CLARIFICATIONS = (
    "Relation to Level 1 Can PH compares the frozen bootstrap separation with the X2 bootstrap "
    "separation: same direction, reversed, not reproduced, new separation; 'neither "
    "separated' when neither is separated (no preregistered label); 'no relation' when the "
    "X2 verdict is conflicting (D-130).",
    "The Level 1 side shows the frozen bootstrap difference, interval and verdict, and the "
    "D-095 seed-level interval recomputed from its per-seed values (descriptive, 400-step "
    "scale, never pooled).",
    "No preregistered readings exist for X2: one generated sentence per contrast states its "
    "difference, verdict and relation in the preregistered terms only.",
)
ENGINEERING_RULE = (
    "docs/FINAL_TEST_FREEZE.md §4: an evaluation with no summary, fewer than all starts, a "
    "failed verify_record audit or a non-canonical platform is engineering-incomplete; it is "
    "rerun once only for a fault outside the policy before its summary is written, under a "
    "recorded decision (attempt 2); still incomplete, its family is ENGINEERING-INCOMPLETE, "
    "left out of the bootstrap (the draw order skips it), and its contrasts are not computed. "
    "Episodes labelled engineering_error count as failures and are reported separately.")


def code_hashes() -> dict[str, str]:
    """SHA-256 of every pinned file, as it is on disk now."""
    return {rel: sha256_file(REPO_ROOT / rel) for rel in CODE_FILES}


def build_block(test_configuration: dict[str, Any], suite_manifest: dict[str, Any],
                platform: dict[str, Any], preregistration_sha256: str,
                commit_order: dict[str, str], code_sha256: dict[str, str]) -> dict[str, Any]:
    """The machine-readable block of the X2 freeze."""
    fams = {}
    for fam in FAMILY_ORDER:
        f = test_configuration["families"][fam]
        fams[fam] = {"chosen_trial": int(f["chosen_trial"]), "overrides": list(f["overrides"]),
                     "config_sha256": f["config_sha256"],
                     "runs": [{"seed": int(r["seed"]), "train_run_id": r["train_run_id"],
                               "step": int(r["selected_step"]),
                               "checkpoint_sha256": r["checkpoint_sha256"],
                               "validation_successes": int(r["validation_successes"])}
                              for r in sorted(f["runs"], key=lambda r: int(r["seed"]))]}
    summaries = f"reports/session17/{'/'.join(xh.SUMMARY_PARTS)}"
    return {
        "schema": xh.FREEZE_SCHEMA, "study": xh.STUDY, "label": xh.LABEL,
        "decisions": list(DECISIONS),
        "preregistration": {"path": xh.PREREGISTRATION, "sha256": preregistration_sha256},
        "test_configuration": {"path": xh.TEST_CONFIGURATION,
                               "sha256": xh.TEST_CONFIGURATION_SHA256},
        "evaluation_platform": {"platform_id": platform["platform_id"],
                                "image_id": platform["image_id"],
                                "runner": "infra/containers/run_sim_cpu.sh"},
        "suite": {"task": x2s.suite_task(), "name": x2s.HELDOUT_SUITE,
                  "n_states": int(suite_manifest["n_states"]),
                  "manifest_content_sha256": suite_manifest["content_sha256"],
                  "partition": suite_manifest["partition"], "decision": "D-123",
                  "seeds": "16,000,000 + i, i = 0..99"},
        "rollouts": {"repeats": 1, "policy_seed": xh.POLICY_SEED, "horizon": xh.HORIZON,
                     "termination": "first_success", "verify_record_episodes": 3,
                     "per_state": "one rollout per (checkpoint, state); diffusion noise seed "
                                  "from (policy_seed, state_id, repeat 0)"},
        "families": fams,
        "analysis": xa.design(),
        "clarifications": list(CLARIFICATIONS),
        "engineering_rule": ENGINEERING_RULE,
        "limits": dict(LIMITS),
        "outputs": {"records": f"runs/{xh.RUNS_SUBDIR[False]}",
                    "records_attempt2": f"runs/{xh.RUNS_SUBDIR[False]}/{xh.ATTEMPT2_SUBDIR}",
                    "summaries": summaries, "analysis": "reports/session17/x2",
                    "logs": "logs/session17/x2/heldout"},
        "commit_order": dict(commit_order),
        "code_sha256": dict(code_sha256),
    }


def _table(block: dict[str, Any]) -> list[str]:
    lines = ["| Family | Chosen trial (overrides) | Seed | Training run | Step | Checkpoint "
             "SHA-256 | X2 validation k/50 |", "|---|---|---|---|---|---|---|"]
    for fam, f in block["families"].items():
        trial = f"{f['chosen_trial']} ({', '.join(f['overrides']) or 'base configuration'})"
        for r in f["runs"]:
            lines.append(f"| {fam} | {trial} | {r['seed']} | `{r['train_run_id']}` | "
                         f"{r['step']} | `{r['checkpoint_sha256']}` | "
                         f"{r['validation_successes']} |")
    return lines


def render(block: dict[str, Any], test_configuration_text: str, frozen_on: str) -> str:
    """The whole freeze document."""
    s, a = block["suite"], block["analysis"]
    contrasts = "; ".join(f"{c['hypothesis']} {c['a']} − {c['b']}" for c in a["contrasts"])
    lines = [
        "# X2 freeze: Can MH, the X2 held-out suite (X2-c)", "",
        f"Frozen on {frozen_on}, before any X2 held-out rollout. Binding: "
        "`docs/extensions/X2_PREREGISTRATION.md` §7 and DECISIONS D-130. Label: "
        f"{block['label']}. Every table "
        "and value below is generated by `scripts/write_x2_freeze.py` from committed files; "
        "where prose and the machine-readable block differ, the block is authoritative. "
        "Nothing in this file changes after its commit.", "",
        "## 1. What runs", "",
        "- The 12 checkpoints of the X2 test configuration (4 families × training seeds 0, 1, "
        "2; each the frozen §7 selection of the family's §6.3-chosen trial on the 50 X2 "
        "validation starts), each **once** on every X2 held-out start. No random baseline.",
        f"- Platform: `{block['evaluation_platform']['platform_id']}` (image "
        f"`{block['evaluation_platform']['image_id']}`), through "
        "`infra/containers/run_sim_cpu.sh` with `ROBOFORGE_EVIDENCE_SESSION=17`; at most "
        f"{block['limits']['max_containers']} containers, the D-125 host limits; "
        "`make x2c-run` (`scripts/x2c_run.py`, `scripts/x2c_eval.py`), from the clean commit "
        "that contains this file.",
        "- Rollouts: R = 1, policy seed 0, first success or 500 steps (D-120), resets only "
        "from the stored snapshots, the first 3 episodes re-simulated; diffusion draws its "
        "noise from `SeedSequence([policy_seed, sha256(state_id)[:8], repeat])` with repeat 0.",
        "", "## 2. The X2 held-out suite", "",
        "| Task | Suite | Starts | Seeds | Manifest content SHA-256 |", "|---|---|---|---|---|",
        f"| {s['task']} | `{s['name']}` | {s['n_states']} | {s['seeds']} | "
        f"`{s['manifest_content_sha256']}` |", "",
        "## 3. Configurations and checkpoints", "",
        f"Exactly `{block['test_configuration']['path']}` (SHA-256 "
        f"`{block['test_configuration']['sha256']}`), copied verbatim in §8.", "",
        *_table(block), "",
        "## 4. Analysis, fixed (preregistration §7)", "",
        f"- Outcome matrices: per family 3 seeds × {s['n_states']} starts (manifest order); 1 "
        "for success before the 500-step horizon, else 0; engineering errors are failures.",
        f"- Primary interval: the frozen hierarchical paired bootstrap, B = "
        f"{a['bootstrap']['B']:,}, seed {a['bootstrap']['seed']}, 95 % percentile; draw order "
        f"{', '.join(a['bootstrap']['draw_order'])}.",
        "- Seed-level interval: the D-095 Welch t 95 % interval on the 3 + 3 per-seed rates; "
        f"\"{a['seed_level']['undefined']}\" when both families have zero seed variance.",
        f"- Contrasts (A − B): {contrasts}. No p-values, no multiplicity correction.",
        f"- Verdicts: {', '.join(a['verdicts'])}. Relation to Level 1 Can PH "
        f"(`{xa.FROZEN_RESULTS[0]}`, SHA-256 `{xa.FROZEN_RESULTS[1]}`): "
        f"{', '.join(a['relation']['labels'])}.",
        f"- Secondary (no readings): censored episode length "
        f"({a['secondary']['censored_episode_length']}); "
        f"{'; '.join(a['secondary']['descriptive'])}.",
        "- Clarifications beyond the preregistration's text (D-130, fixed before any held-out "
        "data):", *[f"  - {c}" for c in block["clarifications"]], "",
        "## 5. Engineering incompleteness and reruns", "", block["engineering_rule"], "",
        "## 6. Pinned files", "",
        "Every file that runs or analyses the X2 held-out test; the evaluator and the "
        "analysis refuse to run if one differs (`roboforge.x2.heldout.code_problems`). The "
        "Level 1 frozen code they reuse stays pinned by `docs/FINAL_TEST_FREEZE.md`.", "",
        *[f"- `{rel}`" for rel in block["code_sha256"]], "",
        "## 7. Commit order", "",
        *[f"- {k}: {v}" for k, v in block["commit_order"].items()], "",
        "## 8. The X2 test configuration, verbatim", "",
        xh.TC_BEGIN, "```json", test_configuration_text.rstrip("\n"), "```", xh.TC_END, "",
        "## 9. Machine-readable block", "",
        xh.BEGIN, "```yaml",
        yaml.safe_dump(block, sort_keys=False, width=100, allow_unicode=True).rstrip("\n"),
        "```", xh.END, ""]
    return "\n".join(lines)


def load_test_configuration() -> tuple[str, dict[str, Any]]:
    """The committed X2-b test configuration (text and parsed), checked by its SHA-256."""
    path = REPO_ROOT / xh.TEST_CONFIGURATION
    if sha256_file(path) != xh.TEST_CONFIGURATION_SHA256:
        raise xh.X2FreezeError(f"{xh.TEST_CONFIGURATION}: SHA-256 differs from the pin")
    text = path.read_text()
    return text, json.loads(text)
