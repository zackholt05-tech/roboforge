<!-- parsable handoff header (values below are real, filled at the end of Session 6) -->
```yaml
project: roboforge-3d
current_session: 6
session_status: PASS
next_session: 7
last_verified_commit: 4721c01   # lint clean, 295 tests; the suite ran at 6dce01b (same code apart from the figure scripts); later commits are docs/evidence only
protocol_commits: {v1: 8b08b95, diffusion_amendment: 3aed462, main_budget_amendment: ec9626b, lift_search_space: 6dce01b}
python: 3.11.15                       # development (macOS arm64); canonical platform uses 3.11.18
sim_engine_version: mujoco 3.2.7
robosuite_revision: 51cc01785bab80ffeed20da15e67d7dd4140e76a   # v1.5.1
robomimic_revision: d309eaecc18acf4152a830a895a6984b8ac71b05
diffusers: 0.11.1
torch: 2.15.0 (macOS) / 2.15.0+cu132 (container and Modal)
evaluation_platform: {platform_id: linux-aarch64-container-1f685638f28c, image_id: "sha256:1f685638f28c720fbec7b084b8e5c2d68073d8f637450d5f6e0a34f0d123071c", runner: infra/containers/run_sim_cpu.sh}
dataset_and_specs: unchanged since Session 2   # git diff d956292 -- data/ uv.lock pyproject.toml configs/eval empty
main_suite:                           # RESEARCH_PROTOCOL §6.3 + §13 (2026-11-07), D-051, D-053..D-056
  families: [bc_mlp, bc_rnn, bc_mlp_chunk, diffusion_unet]   # 4: BC-MLP-chunk is the H2-c control
  tasks: {lift: "done (Session 6)", can: "Session 7"}
  seeds: [0, 1, 2]
  budget: "100,000 steps, batch 100, checkpoints every 5,000 (K = 20), every checkpoint evaluated on 50 validation starts"
  tuning: "<= 4 configurations per family per task at the main budget, seed 0; Lift: 4/4 used per family; Can: 0/4"
  compute: "local CPU only, 1 torch thread, <= 8 concurrent training runs; evaluation on the canonical container, <= 4 in parallel; $0"
  lift_suite: {plan: configs/experiments/level1_core_lift.yaml, plan_sha256: d3de6cbcb14bcda15023a6526c6e3ba8cafb0b37d8692c5f8b6fcba871cfad3f, summary: reports/session6/suite/level1_core_lift.json, runs: 24, evaluations: 480, failed: 0}
baseline_configs:                     # configs/models/*.yaml still carry the 30k pilot budget; main-budget runs pass --set (run_suite does)
  bc_mlp:         {horizons: "T_o=1 T_p=1 T_a=1", model: "MLP 2x1024 ReLU", params: 1077255}
  bc_rnn:         {horizons: "L=10 T_p=1 T_a=1", model: "LSTM 2x296, sliding window", params: 1080703}
  bc_mlp_chunk:   {horizons: "T_o=2 T_p=16 T_a=8", model: "MLP 2x1024 ReLU", params: 1204336}
  diffusion_unet: {horizons: "T_o=2 T_p=16 T_a=8", model: "flatten encoder + ConditionalUnet1D (64,128)", params: 1092551, sampler: "DDIM-10", ema: "power 0.75"}
lift_chosen_configurations:           # §6.3 choice on Lift (reports/session6/tuning/*_lift.json); validation-only
  bc_mlp: {trial: 4, overrides: "optim.weight_decay=0.1"}
  bc_rnn: {trial: 4, overrides: "optim.weight_decay=0.1"}
  bc_mlp_chunk: {trial: 2, overrides: "optim.schedule=cosine"}
  diffusion_unet: {trial: 3, overrides: "optim.lr=3.0e-4"}
lift_test_configuration:              # D-058; reports/session6/lift_test_configuration.json (sha256 38ac3c29...); recorded, NOT frozen
  held_out_schedule_manifest_sha256: e02498651b9d5bb88b6aec0870c5e3277cadaaeaf0e325abf86c79e89b1f22da   # n = 50; hash only, never loaded
  validation_suite_manifest_sha256: 0b1368fae0d5e967955ad7b5c3a6191001bbad2758becad537412b45e71c8785
  bc_mlp:   # chosen trial 4 (optim.weight_decay=0.1)
    s6_bc_mlp_wd1e1_lift_seed0_c39e4980: {seed: 0, selected_step: 15000, validation: 49/50, checkpoint_sha256: e68bc5e05c5b5f78c0e11c33401ed6feee361be44a629ff1fac394949cd7ef54}
    s6_bc_mlp_wd1e1_lift_seed1_628a1e7a: {seed: 1, selected_step: 5000, validation: 48/50, checkpoint_sha256: 5454cf8081b88319bcb91e35d8e69eb8fe0dbef9973c21454b690309dd9995ea}
    s6_bc_mlp_wd1e1_lift_seed2_67c90df6: {seed: 2, selected_step: 5000, validation: 49/50, checkpoint_sha256: 81a69c4c31d1446e5915977831338244e71dda7721697ef62230488548025d1e}
  bc_rnn:   # chosen trial 4 (optim.weight_decay=0.1)
    s6_bc_rnn_wd1e1_lift_seed0_166e9b0a: {seed: 0, selected_step: 15000, validation: 50/50, checkpoint_sha256: fb71d7c60f72ab68c0906d08837d51bbeeb06fbc253de0d5544c255d2d767b1b}
    s6_bc_rnn_wd1e1_lift_seed1_5221b93e: {seed: 1, selected_step: 10000, validation: 50/50, checkpoint_sha256: 30269f1f3b179640687e2c969e8c05095e1c73d1ddbcebbcd30ee5f627511508}
    s6_bc_rnn_wd1e1_lift_seed2_3368e1df: {seed: 2, selected_step: 5000, validation: 50/50, checkpoint_sha256: 1334302f18c9b6a82fa99e676965f5cb54896220b6c8a47b505e727b902dc9b9}
  bc_mlp_chunk:   # chosen trial 2 (optim.schedule=cosine)
    s6_bc_mlp_chunk_cos_lift_seed0_445dbbb5: {seed: 0, selected_step: 15000, validation: 50/50, checkpoint_sha256: c72c4d182450e63456ce53d1db6f35a95531ecab79181602956e6feaaeb03d54}
    s6_bc_mlp_chunk_cos_lift_seed1_c5955411: {seed: 1, selected_step: 5000, validation: 49/50, checkpoint_sha256: 9a46001eab065a990e80270f37e622a4c86664aec75517ac5acbab018dc491bb}
    s6_bc_mlp_chunk_cos_lift_seed2_98c39779: {seed: 2, selected_step: 10000, validation: 50/50, checkpoint_sha256: b7c9aa48bbd063f4466cc9f855542fa7408d181a13a7366aed6c75485ab2fe21}
  diffusion_unet:   # chosen trial 3 (optim.lr=3.0e-4)
    s6_diffusion_unet_lr3e4_lift_seed0_2f635398: {seed: 0, selected_step: 100000, validation: 50/50, checkpoint_sha256: 7e48766542774f52a5c9e73ca2a2776d95981188ed30d65188746479c55f0f38}
    s6_diffusion_unet_lr3e4_lift_seed1_50ae45de: {seed: 1, selected_step: 95000, validation: 50/50, checkpoint_sha256: f26868ea80e307200e92de962493797add56e97d0a1c38519383de64baeb81f7}
    s6_diffusion_unet_lr3e4_lift_seed2_427ac4f5: {seed: 2, selected_step: 80000, validation: 50/50, checkpoint_sha256: b52c30177478e6c8fed6eabd220c08fb839974550da925cb751b341a537f5587}
engineering_runs_session6:            # never trials, never in selection (LEDGER E6-02, E6-03)
  s6dev: [s6dev_bc_mlp_lift_seed0_5b350dd9, s6dev_bc_mlp_lr3e4_lift_seed0_01e97f6f, s6dev_bc_mlp_lr3e4_lift_seed1_6fa904b6, s6dev_diffusion_unet_lift_seed0_979d3226, s6dev_diffusion_unet_lr3e4_lift_seed0_885cf2b0, s6dev_diffusion_unet_lr3e4_lift_seed1_dbba935c]   # 200 steps, dirty 273f86a
  s6dev2: [s6dev2_bc_rnn_lift_seed0_fc3051a1]   # 2,000 steps, stop/resume drill
pilot_runs:                           # Sessions 3-4, 30k, development records; trial 1 re-trained at 100k reproduces them (E6-07)
  s3_bc_mlp_lift_seed0_dbf312cd:          {selected_step: 5000,  checkpoint_sha256: 30289109273d433a364aa5a7e1a4fb41e90fc58d804fcc094936b4582b4559db}
  s3_bc_rnn_lift_seed0_2e7cffe3:          {selected_step: 15000, checkpoint_sha256: 3dbb53ce2a711d89eeb0b5844ae69efb6a45f1da0de21a3fc2cce1472a3da756}
  s3_bc_mlp_chunk_lift_seed0_bb8b4dd6:    {selected_step: 10000, checkpoint_sha256: a763c83730aeb1dbc9d07fbcc76ede2c3b68e63487fc7e3175395f9774ab83a7}
  s4_diffusion_unet_lift_seed0_336e15e9:  {selected_step: 15000, checkpoint_sha256: f768cbe32b2664e97d483d1d135863dedcc3646fe2bdda6c0c641d69658459fc}
modal: {gate_a: done, approval: "A-01 ($2.00 cap) closed", billed_usd: 0.148, used_in_session6: none, egl_vs_osmesa: PENDING (KI-033)}
final_test_locked_until: docs/FINAL_TEST_FREEZE.md   # does not exist; no held-out state loaded
next_required_owner_approval: none for the local Session 7 path; any Modal use needs a new approval row
```

# Session 6 Handoff

Do not trust this file blindly: re-run the checks in "Exact starting instructions".

## 1. Session completed
Session 6 (MASTER_SPEC: Level 1 Lift primary benchmark and checkpoint selection). Status
**PASS**. The full Lift main suite ran from the clean commit `6dce01b` on local CPU and the
canonical container at $0: 24 training runs of 100k steps, 480 checkpoint evaluations on the
50 validation starts, §7 selection, §6.3 choice, then seeds 1–2. No run failed. Every number
below is **validation-only**: the same 50 starts selected the checkpoints and the
configurations. No held-out test state was loaded, named or reset to.

## 2. Completed work
- **Search space** for trials 2–4 (identical for every family: cosine schedule, lr 3e-4,
  weight decay 0.1; D-055) written in the ledger and committed with the plan before the first
  run (`6dce01b`).
- **Main suite** (`scripts/run_suite.py`, D-054): training, canonical evaluation of every
  checkpoint, selection, choice and replicates, end to end in 9.80 h (E6-04–E6-06).
- **Reproduction check:** trial 1 matches the Session 3–4 pilots by content through 30k, and
  its 24 evaluations reproduce the pilots episode for episode (E6-07).
- **Latency** of the 4 chosen configurations on the canonical container, run alone (E6-08).
- **Diagnostics:** per-start failures, action diagnostics, contamination check (E6-09–E6-11).
- **Lift test configuration recorded, not frozen** (D-058, E6-12): 12 checkpoints and the
  held-out schedule by its manifest hash only.
- **Validation-only figures, table and episode montages** (`make figures`, D-057, E6-13).

## 3. Files created or changed
- **New code:** `scripts/plot_validation.py`, `scripts/render_episodes.py`,
  `tests/unit/test_validation_figures.py`; `Makefile` target `figures` (`1280b5d`; footer
  wrapping and latency-figure layout fixed in `4721c01`). The
  suite code (`scripts/run_suite.py`, `src/roboforge/train/suite.py`,
  `src/roboforge/eval/tuning.py`, `configs/experiments/level1_core_lift.yaml`) was committed
  in `6dce01b` before the suite.
- **Docs:** STATUS, HANDOFF, DECISIONS (D-057–D-059), KNOWN_ISSUES (KI-034, KI-035; updates
  to KI-008, KI-021, KI-022, KI-024, KI-028), EXPERIMENT_LEDGER (E6-04–E6-14), COMPUTE_PLAN.
- **Evidence:** `reports/session6/` (suite summary, 480 validation summaries, 24 selections,
  4 tuning records, latency, diagnostics, test configuration, figures),
  `logs/session6/` (suite events, analysis scripts and logs, final checks). Checkpoints and
  episode records stay in `runs/s6_*` (gitignored, 8.7 GB).

## 4. Executed commands and results
See EXPERIMENT_LEDGER E6-01–E6-14. Summary:
- prerequisites all PASS at `273f86a`;
- the suite at `6dce01b`: exit 0, 24/24 runs complete, 480/480 evaluations, 0 errors;
- latency, diagnostics, contamination check, test-configuration record;
- figures at `4721c01`;
- final `make lint` clean and `make test` **295 passed** at `4721c01`.

## 5. Key findings (validation-only development numbers, not held-out results)
- **Selected checkpoints of the chosen configurations, seeds 0 / 1 / 2** (successes of 50):

  | Family | Chosen trial | Seed 0 | Seed 1 | Seed 2 | Mean first-success step (seeds 0–2) |
  |---|---|---|---|---|---|
  | BC-MLP | 4 (weight decay 0.1) | 49 (15k) | 48 (5k) | 49 (5k) | 49.69, 49.56, 45.00 |
  | BC-RNN | 4 (weight decay 0.1) | 50 (15k) | 50 (10k) | 50 (5k) | 43.10, 43.48, 42.62 |
  | BC-MLP-chunk | 2 (cosine) | 50 (15k) | 49 (5k) | 50 (10k) | 44.62, 43.27, 43.40 |
  | Diffusion U-Net | 3 (lr 3e-4) | 50 (100k) | 50 (95k) | 50 (80k) | 45.54, 44.80, 45.10 |

- **Lift validation is saturated (KI-034):** 12 of 16 seed-0 trials and 8 of the 12 chosen
  checkpoints score 50/50 (Wilson [0.929, 1.000]). Three of four choices fell to the
  first-success tiebreak by 0.12–0.68 steps. Success cannot separate BC-RNN, BC-MLP-chunk
  and diffusion here; BC-MLP stays below (47–49/50).
- **Where policies fail** (all 20 checkpoints of the 12 chosen runs, post-hoc categories):
  BC-MLP 439 of 3,000 episodes, mostly missed grasp then drift with saturated motion
  actions (KI-021); BC-MLP-chunk 257, the same mode; BC-RNN 146, mostly missed grasp but
  stays near; diffusion 18, mostly stays near. Start `lift_validation_0029` is hard for the
  two T_a = 8 families (KI-028).
- **When they peak:** BC selects at 5k–15k in 17 of 18 runs, while offline validation MSE
  rises from the first checkpoint (KI-024). Diffusion selects at 35k–100k with success flat
  at 49–50/50; its chosen seed-0 run selected the final checkpoint (100k).
- **Convergence:** training loss flattened in every chosen run (90–100k mean within 1–5 %
  of 80–90k, after a 70–80 % fall over the first half).
- **Latency** (canonical container, median per decision): BC-MLP 0.91 ms, BC-MLP-chunk
  1.66 ms, BC-RNN 9.05 ms, diffusion DDIM-10 36.95 ms.
- **Compute:** 33.25 CPU core-hours of training, 26.92 container-hours of evaluation,
  9.80 h elapsed, **$0**. Host contention inflated wall times but not results (KI-035).

## 6. Open decisions (Session 7)
- **Can search space:** trials 2–4 must be in the ledger before the first Can run. Reusing
  the Lift trials (D-055) is the default; any other choice needs a recorded reason.
- **Freeze contents** (`docs/FINAL_TEST_FREEZE.md`): rollout counts R (KI-028: R = 1 unless
  amended before the freeze), the §9 paired bootstrap, aggregation, failure taxonomy,
  example rule, and how a saturated Lift comparison is reported (KI-034: "not separated" is
  a valid outcome; secondary outcomes declared in advance).
- **Robustness and ablations** (OOD initial-state sets, ≥ 2 ablations) preregistered on
  validation before their runs.
- **Optional:** a Modal workspace budget; an EGL vs OSMesa probe (KI-001).

## 7. Decisions
D-053–D-056 (before the suite) and D-057–D-059 (DECISIONS, Session 6).

## 8. Outstanding issues (none blocks Session 7)
- **KI-034:** Lift validation saturated; plan the paired test accordingly.
- **KI-035:** wall times of Session 6 are contended; use the Session 5 profiles for
  projections.
- **KI-028, KI-024, KI-021, KI-023, KI-011, KI-008:** carried over (updated in Session 6).
- **KI-033 / KI-001:** EGL vs OSMesa on Modal pending; not needed for Level 1.
- **KI-029, KI-020, KI-019, KI-018:** the rules still apply.
- **Makefile defaults** are `EVIDENCE_SESSION` 6 and `PREFIX` s6. Session 7 must pass
  `EVIDENCE_SESSION=7 PREFIX=s7`, or bump the defaults as a recorded change.

## 9. Budget
- **Spent in Session 6:** $0 (local CPU and the canonical container only; no Modal call).
- **All-in to date:** $0.148 on Modal (Gate A, A-01 closed). Month-to-date Modal usage,
  including another app, $5.42 of the $30 credit (as of Session 5; not re-queried, since
  Session 6 made no Modal call).
- **Approved:** local CPU only, **$0** (Gate C). Any Modal use needs a new approval row with
  a cap.

## 10. Exact starting instructions for Session 7
Paste into a fresh Claude Code session:

```text
Implement RoboForge 3D SESSION 7 ONLY from MASTER_SPEC.md (Level 1 Can, paired held-out
testing, robustness, scientific freeze). Begin by reading all of MASTER_SPEC.md,
docs/STATUS.md, docs/HANDOFF.md, docs/DECISIONS.md (through D-059), docs/KNOWN_ISSUES.md
(especially KI-021, KI-024, KI-028, KI-034, KI-035), docs/EXPERIMENT_LEDGER.md (Session 6),
docs/RESEARCH_PROTOCOL.md (v1 + the 2026-11-06 diffusion amendment + the 2026-11-07
main-budget amendment, all binding; §5, §7, §8, §9, §10 matter most),
docs/SIMULATOR_CONTRACT.md, docs/DATA_DICTIONARY.md and docs/COMPUTE_PLAN.md, and inspect
the actual repository/test state. Do not blindly trust prior handoff claims: first run
`make setup`, `make download`, `make check-data`, `make build-data`,
`make check-eval-states`, confirm `git status --short data/` is empty, then run
`make lint`, `make test`, `make smoke-sim SMOKE_OUT=logs/session7/prereq`,
`make train-smoke SMOKE_DIR=logs/tmp/s7_smoke` and, on the canonical container,
`ROBOFORGE_EVIDENCE_SESSION=7 make in-sim CMD="python scripts/smoke_sim.py --task lift --out-dir logs/session7/prereq/container"`
and the same with `--task can`, and confirm they pass; confirm
`docker image inspect -f '{{.Id}}' roboforge-sim-cpu:session0` equals the image_id in
configs/eval/platform.yaml; check that the 12 Session 6 Lift runs listed in
docs/HANDOFF.md (`lift_test_configuration`) exist under runs/ and that each selected
checkpoint's file SHA-256 matches the HANDOFF value and
reports/session6/lift_test_configuration.json (report if missing rather than assuming); run
git log --format=fuller -n 5 to confirm the verified commits and the identity
Zackary Holt <zackholt05@gmail.com>. The Makefile's EVIDENCE_SESSION falls back to 6 and
PREFIX to s6: pass EVIDENCE_SESSION=7 and PREFIX=s7 (or bump the defaults as a recorded
change) so Session 7 never writes into Session 6 evidence.

Session 7 goal (MASTER_SPEC Session 7), in this order:
(1) Can main suite, validation only: Can's data, split and evaluation suites were built and
checked in Sessions 0-2 (open-loop replay 190/200, smoke PASS), but no Can policy has been
trained yet; if a blocker appears, record it precisely and continue with the honest Lift
result.
Train the same 4 families (BC-MLP, BC-RNN, BC-MLP-chunk as the H2-c control,
diffusion_unet) on Can PH with the Session 6 protocol unchanged: 100,000 steps, batch 100,
checkpoints every 5,000 (K = 20), every checkpoint on all 50 Can validation starts on the
canonical container, §7 selection, at most 4 configurations per family on Can at seed 0
(trial 1 = the base config at the main budget), then seeds 1 and 2 of the §6.3 choice.
Write the Can search space (trials 2-4) in docs/EXPERIMENT_LEDGER.md and commit it BEFORE
the first Can tuning run; reusing the Session 6 Lift trials (D-055: cosine, lr 3e-4,
weight decay 0.1) is the default, any other choice needs a recorded reason. Use a new plan
file (e.g. configs/experiments/level1_core_can.yaml, with task: can and prefix: s7) with
scripts/run_suite.py (`make suite SUITE_PLAN=... EVIDENCE_SESSION=7`, D-054), launched
from a clean committed tree, and never edit tracked files while it runs (its clean-tree
check runs at every launch). Generate the same validation-only figures for Can with
`make figures SUITE_SUMMARY=<Can suite summary> LATENCY=<Can latency json> EVIDENCE_SESSION=7`
(D-057; the scripts take the task from the summary but have only been run on Lift).
Lift selections are final: do not retrain, retune or reselect any Lift run.
(2) Robustness and ablations, preregistered and run on VALIDATION only: write the OOD
initial-state perturbation sets (physically valid, implemented through stored snapshots,
validated, frozen before the robustness runs; H4), the failure taxonomy (replacing the
post-hoc diagnose_actions categories) and at least 2 controlled ablations (candidates from
RESEARCH_PROTOCOL: diffusion T_o = 1 vs 2 for H3, DDIM-10 vs DDPM-100, EMA vs no EMA) into
the protocol as a dated amendment, committed before any of those runs. If the budget forces
a narrower scope, record that.
(3) Freeze: write docs/FINAL_TEST_FREEZE.md and lock it with a Git commit BEFORE any
held-out rollout. It must contain the selected configurations and checkpoint hashes (Lift
exactly as recorded in reports/session6/lift_test_configuration.json, D-058; Can from
step 1), the held-out initial-state schedules by suite manifest hash (Lift
e02498651b9d5bb88b6aec0870c5e3277cadaaeaf0e325abf86c79e89b1f22da), rollout counts (R; any
change from R = 1 needs an amendment written before the freeze, KI-028), the §9
hierarchical paired bootstrap (B = 10,000, seed 20261106), the task-metric aggregation,
the failure taxonomy and the visual example-selection rule. Lift validation is saturated
(KI-034: most selected checkpoints are 50/50, Wilson [0.929, 1.000]): preregister the
paired analysis so that "not separated" is a reportable outcome, and declare in advance
any secondary outcome (e.g. mean first-success step) used in the comparison.
(4) Execute the final paired test ONCE under the frozen protocol, for all core models and
tasks that passed feasibility, on the canonical container; no tuning, reselection or rerun
on the test. Generate immutable machine-readable results, CI and pairwise-difference
tables, failure analyses and full cost/latency results; distinguish engineering-incomplete
cases, task failure and statistical uncertainty.
(5) Write docs/LEVEL1_COMPLETE.md (the Level 1 research freeze): actual executed results,
omissions and reasons; record exploratory follow-ups separately.

Compute (Gate C, D-051): local CPU only, $0 — 1 torch thread per run, at most 8
concurrent training runs, at most 4 evaluation containers in parallel; no Modal resource of
any kind (any Modal use needs a new owner approval row in docs/EXPERIMENT_LEDGER.md first).
Can episodes run up to 400 steps and Can evaluation dominates the projected time
(COMPUTE_PLAN §4); if the host is shared again, wall times inflate (KI-035) but results do
not. Track failed runs too. Measure latency on the canonical container with nothing else
running.

Keep the Session 1–6 contracts (DECISIONS D-010, D-017..D-059): training data only from
SequenceDataset.from_committed; policy observations only from RoboForgeEnv; outputs decoded
with ActionCodec; learned policies implement roboforge.eval.policies.Policy and run with
run_episode + verify_record; stochastic policies seeded by the protocol's noise-seed rule
(policy_seed 0, R = 1 unless amended before the freeze); resets only from stored
snapshots; first success or 400 steps; model and checkpoint selection only on the
eval_validation suite; the held-out suites are read only through
src/roboforge/sim/eval_states.py after the lock check, and only after
docs/FINAL_TEST_FREEZE.md is committed (RESEARCH_PROTOCOL §10; the integrity test
restricts which files may name them); every reported, selection or test rollout on the
canonical container via infra/containers/run_sim_cpu.sh with ROBOFORGE_EVIDENCE_SESSION=7
(never uv sync/uv run inside the image, KI-020). Native macOS rollouts are
development-only; never call mj_forward on live evaluation data (KI-019); never use
robosuite transform helpers for precision-critical values (KI-018); checkpoints and
anything a container rollout reads live under runs/<run_id>/; every run writes an immutable
manifest with git SHA + dirty flag (launch reported runs from a clean committed tree; do not
edit tracked files while they run), environment, platform, data/spec hashes, full config,
seeds, parameter count, checkpoint hashes and wall/CPU time; compare checkpoints by
content, not file hash (KI-029); report motion-dimension clipping separately from the
gripper (KI-022); offline validation MSE never selects (KI-024). Use the Make variable
TRAIN_RUN (not RUN) for run directories. No paid or remote compute and no remote
publication without explicit owner approval; do not change committed data artifacts, eval
suites, uv.lock or dependencies without a recorded decision.

Use the current verified simulator/data/checkpoint/manifest contracts. Run relevant tests
and record commands, real results and provenance. Never leak held-out test states into model
selection; never invent results or costs (label estimates as estimates). Use Zackary Holt
as the sole named RoboForge project author with the verified local Git identity; no AI
co-author trailers, while preserving external scientific and license credit (Diffusion
Policy, robomimic, diffusers). At the SESSION 7 STOP boundary (a HARD SCIENTIFIC BOUNDARY:
later RGB/point-cloud results may never modify core test selection, labels or findings),
update docs/STATUS.md, docs/HANDOFF.md, docs/DECISIONS.md, docs/KNOWN_ISSUES.md,
docs/EXPERIMENT_LEDGER.md (all runs, including failed ones, the tuning-budget accounting
and the all-in cost record) and docs/COMPUTE_PLAN.md if compute changed; commit verified
code only, inspect commit metadata (git log --format=fuller -n 5), write the exact prompt
for SESSION 8, and STOP. Do NOT proceed automatically.
```
