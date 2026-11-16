# Post-release extensions (after v1.0.0)

The owner asked on 2026-11-10 to continue after the v1.0.0 release with three follow-up studies
(DECISIONS D-105). Each one answers a question the frozen results left open. None of them
changes a frozen study, its freeze, its results or the published v1.0.0 paper, README and
Observatory.

| ID | Study | Question it answers | Training | Paid compute | Sessions (estimate) |
|---|---|---|---|---|---|
| X1 | [Evaluation-only action-clamp test](X1_action_clamp.md) | Is the diffusion policy's robustness advantage (and its parity with BC-MLP-chunk) due to the generative model or to its action bound (KI-052)? | none | none | 1 (Session 13) |
| X2 | [A harder or multimodal task](X2_harder_task.md) | Does the generative head help where the data are multimodal or the task leaves headroom (the frozen tasks sit near the ceiling)? | yes | none planned (local CPU, as Level 1) | 3 or more |
| X3 | [More training seeds](X3_more_seeds.md) | Do the separations that are fragile at the seed level (KI-053) hold with more seeds, on fresh held-out starts? | yes | none planned (local CPU, as Level 1) | 1 to 2 (Session 14) |

Order: X1 first (cheapest, no training, resolves a named confound). X2 and X3 are independent;
the owner picks which runs next at the end of Session 13.

**Status (2026-11-10).** X1 ran in Session 13 (PASS; exploratory, after the freeze):
- preregistration `X1_PREREGISTRATION.md` (`ab02469`);
- results in `reports/session13/x1_clamp/tables.md`.
With the diffusion sampler's bound on its executed actions, BC's outcomes are unchanged and the
H4 gap stays separated. The diffusion side of KI-052 (the sampler clips at every step; a
`clip_sample=False` arm) was not tested.

**Status (2026-11-11).** X3 ran in Session 14 (PASS; separate replication):
- preregistration `X3_PREREGISTRATION.md` (`fb9f05e`); held-out suite `can/x3_heldout`
  (100 starts, content `9c199e19…`);
- results in `reports/session14/x3/tables.md` (`335c27c`).
With 6 seeds per family on 100 fresh Can starts, H2 diffusion − BC-MLP (+22.7 pp) and H3 BC-RNN
− BC-MLP (+23.8 pp) separate under both the bootstrap and the seed-level interval; H2 diffusion
− BC-RNN and H2-c stay not separated. The frozen result stands as reported. The owner chose X2
for Session 15 (D-115). The exact prompt is in `docs/HANDOFF.md` (# Session 14 Handoff §6).

**Status (2026-11-12).** X2 phase X2-a ran in Session 15 (PASS; data and contract, development
numbers only). The owner chose option A, Can MH (D-116):
- data: the Can MH files in a new manifest, hash-verified; split by robomimic's
  operator-stratified masks (D-119), so every X2 claim is within-operator; horizon 500 (D-120);
  the validated Can contract applies (`reports/session15/x2/contract.json`);
- suites `can/x2_validation` (50 starts, `47b889cd…`) and `can/x2_heldout` (100 starts,
  `40b8228c…`), committed before any rollout (`4459c00`); no X2 held-out start was stepped;
- headroom pilot (D-124; `reports/session15/x2/pilot/tables.md`): best of five checkpoints on
  the 50 validation starts, diffusion 48, BC-MLP-chunk 48, BC-RNN 34, BC-MLP 6. Reading:
  **headroom**, by one family; the top two are at the 50-start ceiling.

**Status (2026-11-16).** X2 phase X2-b ran in Session 16 (PASS; development numbers on the 50
validation starts only), under `docs/extensions/X2_PREREGISTRATION.md` (D-125 to D-128). Level
1's four trials per family were searched with the frozen budget. The four pilot runs counted as
seed-0 trials. 20 new runs were trained and 460 new evaluations run, with none failed.
- **Chosen by the frozen §6.3 rule:** diffusion trial 2 (48), BC-RNN trial 4 (36),
  BC-MLP-chunk trial 2 (49), BC-MLP trial 1 (16).
- **X2 test configuration** (frozen §7 per run, seeds 0–2):
  `reports/session16/x2/suite/x2_test_configuration.json`; tables in
  `reports/session16/x2/suite/tables.md`.
- **Next:** X2-c commits `docs/extensions/X2_FREEZE.md`, then runs the 12 checkpoints once on
  `can/x2_heldout` and the preregistered analysis once. The exact prompt is in
  `docs/HANDOFF.md` (# Session 16 Handoff §6).

**Status (2026-11-16).** X2 phase X2-c ran in Session 17 (PASS; X2 is done), under D-130.
`docs/extensions/X2_FREEZE.md` was committed alone (`3a64dbb`) before any held-out rollout. The
12 checkpoints ran once on the 100 `can/x2_heldout` starts (12/12 complete, 0 errors, isolation
PASS), and the preregistered §7 analysis ran once (`reports/session17/x2/`, `3df25ff`).
Label X2 (Can MH): beside the frozen Level 1 Can PH result, never instead of it, never pooled;
within-operator only.

| Contrast | X2 (Can MH), 3 seeds × 100 starts, 500 steps | Verdict | Relation to Level 1 Can PH |
|---|---|---|---|
| H2 diffusion − BC-MLP | +63.3 pp [+53.0, +73.0] | robust (+) | same direction |
| H2 diffusion − BC-RNN | +27.3 pp [+16.7, +37.7] | robust (+) | new separation |
| H2-c diffusion − BC-MLP-chunk | +2.0 pp [−6.3, +11.0] | not separated | neither separated |
| H3 BC-RNN − BC-MLP | +36.0 pp [+23.7, +48.0] | robust (+) | same direction |

Family success on the held-out starts: diffusion 93.3 %, BC-MLP-chunk 91.3 %, BC-RNN 66.0 %,
BC-MLP 30.0 %. All three extensions are done. Whether X1, X2 and X3 are published (v1.1) is
the owner's decision (D-105).

**Status (2026-11-16).** Session 18, v1.1 built and verified locally; nothing published (D-131).
The owner chose to publish X1, X2 and X3 as v1.1 and to update the Observatory first.
- The addendum is `paper/addendum/addendum.pdf` (D-133). It is generated from the three pinned
  results; `paper/main.pdf` is unchanged.
- The README has a generated section on the three studies, with a chart.
- The Observatory has a section "After v1.0.0: three follow-up studies" (D-134). It shows X1 and
  X2 (Can MH) recorded episodes, re-rendered from their recorded states, and the three studies'
  results under their own labels.
- KI-065: X2 records carry the frozen "PH demonstrations" label. The site shows the Can MH
  label instead.
- Session 19 publishes: push, CI, tag `v1.1.0`, the release, and the Pages redeploy.

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
