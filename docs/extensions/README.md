# Post-release extensions (after v1.0.0)

The owner asked on 2026-11-10 to continue after the v1.0.0 release with three follow-up studies
(DECISIONS D-105). Each one answers a question the frozen results left open. None of them
changes a frozen study, its freeze, its results or the published v1.0.0 paper, README and
Observatory.

| ID | Study | Question it answers | Training | Paid compute | Sessions (estimate) |
|---|---|---|---|---|---|
| X1 | [Evaluation-only action-clamp test](X1_action_clamp.md) | Is the diffusion policy's robustness advantage (and its parity with BC-MLP-chunk) due to the generative model or to its action bound (KI-052)? | none | none | 1 (Session 13) |
| X2 | [A harder or multimodal task](X2_harder_task.md) | Does the generative head help where the data are multimodal or the task leaves headroom (the frozen tasks sit near the ceiling)? | yes | none planned (local CPU, as Level 1) | 3 or more |
| X3 | [More training seeds](X3_more_seeds.md) | Do the separations that are fragile at the seed level (KI-053) hold with more seeds, on fresh held-out starts? | yes | none planned (local CPU, as Level 1) | 1 to 2 |

Order: X1 first (cheapest, no training, resolves a named confound). X2 and X3 are independent;
the owner picks which runs next at the end of Session 13.

## Rules shared by every extension

1. **Frozen means frozen.** Level 1, the RGB study and Level 2 stay as tested. Never rerun
   `make eval-final`, `make analyze-final`, `make rgb-heldout`, `scripts/analyze_rgb_heldout.py`,
   `make pc-heldout`, `make pc-placement`, `make level2-analyze` or
   `scripts/analyze_robustness.py`. Never edit `reports/session6`–`reports/session10` results,
   the freezes, `LEVEL1_COMPLETE` or the executed outcome of `LEVEL2_GO_NO_GO`. Never prune or
   move frozen runs. `make freeze-check` passes before and after every code change, and the
   frozen files it lists are never edited; new behavior goes into new files.
2. **Decide, then preregister, then run.** Each extension records its design in
   `docs/DECISIONS.md` and commits a preregistration (`docs/extensions/X<n>_PREREGISTRATION.md`)
   before its first reported rollout: suites and their hashes, checkpoints, contrasts,
   interval method and seed, and what each outcome would mean.
3. **Labels.** X1 results are "exploratory, after the freeze". X2 is a separate study with its
   own freeze and held-out suite. X3 is a "separate replication". None of them replaces or
   revises a frozen number.
4. **Platform.** Every reported rollout runs on the canonical container
   (`infra/containers/run_sim_cpu.sh`, image pinned in `configs/eval/platform.yaml`) from a clean
   tree, with `ROBOFORGE_EVIDENCE_SESSION` set to the session number. Never `uv sync` or
   `uv run` inside the image. Never run the full test suite alongside a container job (KI-043).
5. **New held-out starts use fresh seed blocks.** Never put the string `final_test` in `src/`
   or `scripts/*.py` (the research-integrity lock test forbids it); name new suites after the
   extension (for example `x3_heldout`).
6. **Compute and disk.** Local CPU and the local container cost $0. Any Modal job needs a new
   approval row in `docs/EXPERIMENT_LEDGER.md` with a cost cap first. The disk is the binding
   local limit (about 20 GB free on 2026-11-10; KI-008): measure before training.
7. **Estimates** are labelled as estimates and computed from measured times in this repository.
8. **Identity and publication.** Commits as `Zackary Holt <zackholt05@gmail.com>`, no AI
   co-author trailers. Nothing is pushed, tagged, released or deployed without asking the
   owner first; a later v1.1 (README, paper addendum, Observatory) is the owner's decision.

## Exact prompt for Session 13 (X1)

```text
Implement RoboForge 3D SESSION 13 ONLY: post-release extension X1, the evaluation-only
action-clamp test of KNOWN_ISSUES KI-052, as specified in docs/extensions/X1_action_clamp.md,
under DECISIONS D-105 and the rules of MASTER_SPEC.md.

Begin by reading all of MASTER_SPEC.md, docs/extensions/README.md,
docs/extensions/X1_action_clamp.md, docs/STATUS.md, docs/HANDOFF.md, docs/DECISIONS.md
(especially D-063, D-064, D-096, D-105), docs/KNOWN_ISSUES.md (KI-043, KI-052, KI-061),
docs/EXPERIMENT_LEDGER.md (E7-09), docs/RESEARCH_PROTOCOL.md §14, docs/FINAL_TEST_FREEZE.md and
docs/AUDIT.md (A-6), and inspect the actual repository and test state. Do not blindly trust
handoff claims; confirm every prerequisite with executable checks.

Steps (details and acceptance criteria in X1_action_clamp.md):
1. Record decisions before acting: Makefile defaults PREFIX ?= s13 and evidence session 13;
   the X1 design (clamp range, wrapper in new files, the 45 evaluations, output locations).
2. Engineering checks before any reported rollout: the clamp-disabled wrapper reproduces
   recorded Session 7 BC robustness summaries; the clamp applied to a diffusion checkpoint
   changes nothing; new outputs never touch existing runs/ directories.
3. Write and commit docs/extensions/X1_PREREGISTRATION.md before the first reported rollout.
4. Run the 45 clamped BC evaluations once on the canonical container.
5. Analyze with a new script that reads the Session 7 summaries through their recorded
   SHA-256; write reports/session13/x1_clamp/ (results, tables, analysis record).
6. Update KI-052 with the outcome, labelled "exploratory, after the freeze".

Binding boundaries: Level 1, the RGB study and Level 2 are frozen. Never rerun make
eval-final, make analyze-final, make rgb-heldout, scripts/analyze_rgb_heldout.py, make
pc-heldout, make pc-placement, make level2-analyze or scripts/analyze_robustness.py. Never edit
reports/session6-10 results, the freezes, LEVEL1_COMPLETE or LEVEL2_GO_NO_GO's executed
outcome. Never prune or move frozen runs. Never edit a file that make freeze-check pins; run
make freeze-check before and after touching any code, and tests/research_integrity/ on new
code. No training. No held-out (final test) rollout: X1 runs on the Session 7 robustness sets
only, unless the owner chooses otherwise and a decision is recorded first. No paid or remote
compute, push, tag, release or deployment without explicit owner authorization. No changes to
data, eval suites, uv.lock or Python dependencies without a decision. Never uv sync or uv run
inside the image. Use EVIDENCE_SESSION=13; every container job runs via run_sim_cpu.sh with
ROBOFORGE_EVIDENCE_SESSION=13, from a clean tree. Never run the full test suite alongside a
container job (KI-043). Label every estimate as an estimate.

At the SESSION 13 STOP boundary: update STATUS, HANDOFF, DECISIONS, KNOWN_ISSUES and
EXPERIMENT_LEDGER; run make lint, the full suite alone, make freeze-check,
tests/research_integrity, make release-check; commit verified work as Zackary Holt
<zackholt05@gmail.com> with no AI co-author trailers; inspect git log --format=fuller -n 5;
ask the owner whether X2 or X3 runs next and write the exact prompt for that session; STOP. Do
not proceed automatically.
```

The prompts for X2 and X3 are written at the STOP of the session before them, from the
template in each extension's file, so that they carry that session's verified state.
