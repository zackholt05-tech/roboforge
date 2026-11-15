# X2 preregistration: the Level 1 comparison on Can mixed-human (MH) data

**Labels.** Everything X2-b produces is a development number on the 50 `can/x2_validation`
starts, used only to select checkpoints and configurations, never a result. The X2-c numbers
(held-out, once) are labelled "X2 (Can MH)", reported beside the frozen Level 1 Can PH result,
never instead of it, and never pooled with it.

This file was written and committed on 2026-11-14 (Session 16, EST), before the first X2-b
training run, after the X2-a evidence (`065efbd`). The commit order shows it. It governs X2-b
and fixes the X2-c analysis, under DECISIONS D-105, D-115 to D-127. Level 1, the RGB study,
Level 2, X1, X3 and the X2-a evidence stay frozen.

## 1. Question and scope

Does the Level 1 ranking of the four families hold on Can when the demonstrations come from
six operators of mixed skill (robomimic Can MH; `docs/extensions/X2_harder_task.md`), where the
frozen Can PH study sits near its ceiling?

- **Within-operator only.** The split uses robomimic's operator-stratified masks (D-119): every
  operator contributes 45 training and 5 validation demonstrations. X2 makes no claim about
  operators never seen in training (MASTER_SPEC §3.3).
- **Same task and contract as Can PH**, horizon 500 (D-120), the X2 suites (D-123), the frozen
  trainer, evaluator and statistics bound to the X2 data in-process (D-121, KI-064).

## 2. Search space and budget (X2-b)

The plan is `configs/experiments/x2_can_mh.yaml`, a frozen-schema suite plan resolved by the
frozen `roboforge.train.suite.load_plan`. Its tuning trials and budget equal Level 1's on Can
(D-055, D-061; test `tests/research_integrity/test_x2b_scope.py`), so the per-family tuning
budget is equal:

| Trial | Change from the base configuration |
|---|---|
| 1 | none (`configs/models/<family>.yaml` at the main budget) |
| 2 | cosine learning-rate schedule |
| 3 | constant learning rate 3e-4 |
| 4 | AdamW weight decay 0.1 |

- Families: diffusion_unet, bc_rnn, bc_mlp_chunk, bc_mlp. Architecture, horizons, parameter
  counts, batch and budget never change.
- Budget per run: 100,000 steps, batch 100, checkpoints every 5,000 (K = 20), 1 torch thread,
  local CPU (macOS host, KI-061). Tuning seed 0 (KI-062); the chosen trial then trains seeds 1
  and 2. Run prefix `s16x2`; runs under `runs/x2/`.
- A run that fails (non-finite loss) counts against its family's 4 trials, is never retried and
  ranks last (frozen §6.3 rule).

## 3. The X2-a pilot runs count as seed-0 trials (D-125)

D-124 allowed counting the pilot runs only if this file says so. It says so:

| Family | Pilot run (X2-a, from clean `fe9df74`) | Is the seed-0 run of |
|---|---|---|
| diffusion_unet | `s15x2_diffusion_unet_cos_can_seed0_99251da3` | trial 2 |
| bc_rnn | `s15x2_bc_rnn_cos_can_seed0_5ff5ce91` | trial 2 |
| bc_mlp_chunk | `s15x2_bc_mlp_chunk_wd1e1_can_seed0_0c2f9d4b` | trial 4 |
| bc_mlp | `s15x2_bc_mlp_cos_can_seed0_ae8152e3` | trial 2 |

- **Mapping rule:** a planned seed-0 run whose resolved config SHA-256 equals a pilot run's is
  that run (`roboforge.x2.suite.planned_run`); its training manifest must record that config,
  task `can` and seed 0 (`check_pilot_mapping`). The config hashes equal the frozen Level 1
  ones; training is deterministic on CPU with one thread, and the data, binding and trainer are
  unchanged since the pilot, so retraining would reproduce the same run.
- **Evaluations from two sessions feed one selection.** Each pilot run's checkpoints 20k, 40k,
  60k, 80k and 100k were evaluated in X2-a (`reports/session15/x2/pilot/evals/`, from
  `fe9df74`). Its other 15 are evaluated in X2-b (`reports/session16/x2/suite/evals/`). Both
  come from the same evaluator (`src/roboforge/x2/evaluate.py`, unchanged since `fe9df74`), so
  every summary's `decisions` field reads D-121, D-123, D-124. No step is evaluated twice: the
  loader raises on a duplicate (`load_summaries`), and the frozen §7 rule raises on one too.
- The pilot numbers were known when this file was written (best of five: diffusion 48,
  BC-MLP-chunk 48, BC-RNN 34, BC-MLP 6). The search space and budget are Level 1's, fixed
  before X2 existed, so nothing here is tuned to them.

## 4. Evaluations and selection (X2-b)

- **Every checkpoint of every run, once,** on all 50 `x2_validation` starts. Canonical container
  (`linux-aarch64-container-1f685638f28c`), policy seed 0, R = 1, 500-step episodes, the frozen
  episode rule and audit, through `scripts/x2b_eval.py` (the X2-a evaluator; it admits only the
  X2 validation suite). 20 per run, 480 in all (460 new).
- **No automatic rerun.** An evaluation that fails before writing its summary is rerun once,
  only for a fault outside the policy, as a recorded decision (the frozen engineering rule).
- **Checkpoint per run:** the frozen §7 rule, `roboforge.eval.selection.select_checkpoint`,
  unchanged (highest success count; ties to the lower mean first-success step, then the earlier
  checkpoint).
- **Configuration per family:** the frozen §6.3 rule, `roboforge.eval.tuning.choose_configuration`,
  unchanged, over the four seed-0 trials (highest selected success count; ties to the lower
  mean first-success step, then the lower trial). Choice records:
  `reports/session16/x2/suite/choices/<family>.json`.
- **X2 test configuration** (for X2-c): per family the chosen trial and the §7-selected
  checkpoint of seeds 0, 1 and 2, with SHA-256 and validation successes:
  `reports/session16/x2/suite/x2_test_configuration.json`. X2-c's `docs/extensions/X2_FREEZE.md`
  copies it before any held-out rollout.
- **No X2 held-out start is stepped in X2-b.**

## 5. Host limits (`configs/extensions/x2b.yaml`; KI-035, KI-008, D-125)

At most 4 trainings; trainings plus evaluation containers at most 5 (1 container beside 4
trainings, 2 beside 3, up to 4); trainings launch first, diffusion first. One launch per host
sample (60 s), each after a fresh sample, starting from a baseline sample. No launch while the
last swap rate exceeds 3,000 pages/s, below 4 GiB free disk or on a dirty tree. Replicate
trainings launch only when free disk covers the remaining estimate plus the 4 GiB floor. Below
2 GiB the trainings stop with a resume checkpoint. These limits change only wall time, never a
run, a selection or a choice.

## 6. Latency benchmark (X2-b)

`scripts/x2b_benchmark.py` on the canonical container: the 12 selected checkpoints, 1 torch
thread, observations of a 500-step no-op episode from X2 validation start 0, the frozen timing
functions of `scripts/benchmark_policies.py` (observe + act per decision, forward only; first
pass warm-up; 3 timed passes). A measurement only; no policy acts in the simulator.

## 7. X2-c analysis, fixed now (`docs/extensions/X2_FREEZE.md` adds only checkpoints and hashes)

- **Evaluations:** the 12 selected checkpoints, each once, on all 100 `can/x2_heldout` starts
  (D-123: seeds 16,000,000 + i, content `40b8228c…`), canonical container, policy seed 0, R = 1,
  500-step episodes. From one clean commit containing `X2_FREEZE.md`.
- **Outcome matrices:** per family, 3 seeds (rows 0–2) × 100 starts (the manifest's state
  order); 1 for success before the horizon, else 0; engineering errors count as failures and
  are reported separately.
- **Primary interval:** the frozen §9 hierarchical paired bootstrap,
  `roboforge.eval.statistics.hierarchical_paired_bootstrap`, unchanged: B = 10,000, seed
  20261106, 95 % percentile intervals; draw order states, bc_mlp, bc_rnn, bc_mlp_chunk,
  diffusion_unet; states shared by every family, each family's seeds resampled independently.
- **Seed-level interval (co-primary for robustness):** the D-095 Welch t 95 % interval on the
  3 + 3 per-seed success rates (`roboforge.paper.sensitivity.welch`); "undefined (no seed
  variance)" when both families have zero seed variance.
- **Declared contrasts (A − B):** H2 diffusion_unet − bc_mlp; H2 diffusion_unet − bc_rnn; H2c
  diffusion_unet − bc_mlp_chunk; H3 bc_rnn − bc_mlp. No p-values, no multiplicity correction,
  every interval reported.
- **Verdict per contrast,** as in X3: *robust (±)* when both intervals exclude 0 on the same
  side; *bootstrap only (±)*; *seed-level only (±)*; *not separated* (not evidence of
  equality); *conflicting* (no verdict).
- **Relation to Level 1 Can PH** (frozen bootstrap verdicts, `reports/session7/results/`):
  reported side by side per contrast as "same direction", "new separation", "not reproduced"
  or "reversed"; a different dataset, so never a replication and never pooled.
- **Secondary (no readings):** censored episode length (first-success step, else 500) for the
  4 contrasts with the same bootstrap; per-seed success rates; family means with bootstrap
  intervals; the frozen failure taxonomy on every X2-c episode; the X2-b validation selections
  and the latency benchmark (descriptive).
- **Engineering incompleteness:** the frozen rule (`docs/FINAL_TEST_FREEZE.md` §4); an
  ENGINEERING-INCOMPLETE family is left out of the bootstrap and its contrasts are not computed.

## 8. Biases and limits, stated now

- **Ceiling:** on validation the X2-a pilot's best checkpoints of diffusion and BC-MLP-chunk
  reached 48/50, so H2c may not separate even if the families differ; X2's headroom lies
  mainly with the single-step families.
- **Selection optimism** stays in validation (max over 20 checkpoints on the same 50 starts;
  BC-MLP-chunk varied 37–48 across pilot checkpoints). The 100 held-out starts are unbiased
  for the selected checkpoints.
- **Three seeds per family** (the frozen design); the seed-level interval has few degrees of
  freedom (KI-053).
- **The single-step stall** (LEDGER E15-06: early BC-MLP and BC-RNN checkpoints run every
  episode to the horizon) is a property of the policies on this data, not a pipeline fault; its cause is open.
- **R = 1:** KI-028's start/seed confound for diffusion remains.
- **Scope:** Can, privileged low-dimensional state, one simulator, one dataset (MH), within
  operator.

## 9. Estimates (labelled estimates, from the X2-a measured times)

Training 30.5 training-hours summed (20 new runs; diffusion 4.62 h each); evaluations about 64
container-hours (460 at the pilot mean of 503 s; biased low); wall time about 23 h with the
limits of §5; disk about 18–23 GiB (5.6 GiB of runs and 12–17 GiB of evaluation records);
$0. X2-c: 12 evaluations of 100 starts, about 2–3 container-hours.
