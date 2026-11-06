<!-- parsable handoff header (values below are real, filled at the end of Session 3) -->
```yaml
project: roboforge-3d
current_session: 3
session_status: PASS
next_session: 4
last_verified_commit: a37b92d   # lint clean, 231 tests (code unchanged since then except docs)
protocol_commit: 8b08b95        # docs/RESEARCH_PROTOCOL.md v1, before any reported run
evidence_commits:
  training_and_validation: 969e14fa320d4e0f7295b0d5634c3fe8a9139672   # all runs + 18 canonical evals, dirty=false
  diagnostics_cost_equivalence: a37b92d   # Makefile + diagnose_actions.py changed only (git diff 969e14f a37b92d --stat)
python: 3.11.15                       # development (macOS arm64); canonical platform uses 3.11.18
sim_engine_version: mujoco 3.2.7
robosuite_revision: 51cc01785bab80ffeed20da15e67d7dd4140e76a   # v1.5.1
robomimic_revision: d309eaecc18acf4152a830a895a6984b8ac71b05
torch: 2.15.0 (macOS) / 2.15.0+cu132 (container, CPU path)
evaluation_platform: {platform_id: linux-aarch64-container-1f685638f28c, image_id: "sha256:1f685638f28c720fbec7b084b8e5c2d68073d8f637450d5f6e0a34f0d123071c", runner: infra/containers/run_sim_cpu.sh}
dataset_and_specs: unchanged since Session 2   # hashes in data/manifests/tasks/*.json; git diff 0322a15 -- data/ uv.lock empty
baseline_configs:                     # configs/models/*.yaml (DEFAULT, D-033/D-034)
  bc_mlp:       {horizons: "T_o=1 T_p=1 T_a=1", model: "MLP 2x1024 ReLU", params: 1077255}
  bc_rnn:       {horizons: "L=10 T_p=1 T_a=1", model: "LSTM 2x296, sliding window", params: 1080703}
  bc_mlp_chunk: {horizons: "T_o=2 T_p=16 T_a=8", model: "MLP 2x1024 ReLU", params: 1204336}
  training: "AdamW 1e-4, batch 100, masked MSE, constant LR, no clip/EMA, 1 thread CPU; pilot 30k steps, ckpt every 5k"
pilot_runs:                           # runs/<id>/ (gitignored); Lift, seed 0, tuning trial 1 each
  s3_bc_mlp_lift_seed0_dbf312cd:       {selected_step: 5000,  validation: 47/50, checkpoint_sha256: 30289109273d433a364aa5a7e1a4fb41e90fc58d804fcc094936b4582b4559db}
  s3_bc_rnn_lift_seed0_2e7cffe3:       {selected_step: 15000, validation: 50/50, checkpoint_sha256: 3dbb53ce2a711d89eeb0b5844ae69efb6a45f1da0de21a3fc2cce1472a3da756}
  s3_bc_mlp_chunk_lift_seed0_bb8b4dd6: {selected_step: 10000, validation: 50/50, checkpoint_sha256: a763c83730aeb1dbc9d07fbcc76ede2c3b68e63487fc7e3175395f9774ab83a7}
final_test_locked_until: docs/FINAL_TEST_FREEZE.md   # does not exist
training_spend_usd: 0
next_required_owner_approval: none for Session 4 (local CPU); paid compute needs approval (Session 5)
```

# Session 3 Handoff

Do not trust this file blindly: re-run the checks in "Exact starting instructions".

## 1. Session completed
Session 3 (research protocol and deterministic/recurrent baselines): **PASS**. Session 4
can start.

## 2. Completed work
- **Research protocol v1** (`docs/RESEARCH_PROTOCOL.md`, committed at `8b08b95` before any
  reported run). It fixes:
  - the hypothesis → comparison table: H2 is primary, and H2-c is the action-chunking control;
  - observation privilege (identical 19-D privileged state for every family);
  - the paired schedule, the success rule and the checkpoint-selection rule (§7: highest
    validation success, then lower mean first-success step, then the earlier checkpoint);
  - matched training and tuning budgets (§6.3: ≤ 4 configurations per family per task, with
    Session 3 pilots as trial 1);
  - the hierarchical paired bootstrap (§9: B = 10,000, seed 20261106).

  One accounting amendment (§13) counts the BC-MLP-chunk pilot as its trial 1.
- **Models** (`src/roboforge/models/bc.py`, D-033):
  - one interface, `forward(obs_history, obs_valid) → normalized chunk`;
  - BC-MLP;
  - BC-RNN: an LSTM re-run from a zero state over the valid steps of the last L = 10
    observations at every decision, which is a recorded deviation from robomimic;
  - BC-MLP-chunk, the output-horizon control.
- **Training stack** (`src/roboforge/train/`, D-034, D-035):
  - strict hashed configs with `--set` overrides;
  - in-memory windows from `SequenceDataset.from_committed`, plus a seeded epoch-stream
    sampler;
  - a trainer with gradient-norm logging, where non-finite values raise;
  - `weights_only` checkpoints with every RNG, so resuming is bitwise;
  - immutable run manifests (exclusive create, also on failure) with a strict reload check.
- **Closed loop** (`src/roboforge/eval/`, D-032, D-036, D-037):
  - `LearnedPolicy`, which uses the same window arithmetic as training, float32 casting, a
    contract check, history cleared per episode and 1 torch thread;
  - the rollout `observe` hook;
  - validation-only evaluation (`scripts/eval_policy.py`, `make eval`), which verifies
    checkpoint hashes against the manifest and labels native platforms DEVELOPMENT ONLY;
  - the selection rule (`scripts/select_checkpoint.py`).
- **Evidence tools:**
  - `scripts/{overfit_check,train_smoke,param_counts,benchmark_policies,diagnose_actions,check_checkpoint_equivalence}.py`;
  - Make targets `train-smoke`, `train`, `eval`, `select`, `param-counts`, `overfit-check`,
    `benchmark-policies`, `diagnose-actions` and `check-ckpt-equivalence`;
  - the training-run variable is `TRAIN_RUN`, because `RUN` is the uv prefix.

## 3. Files created or changed
- **New:**
  - `src/roboforge/models/bc.py`, `src/roboforge/train/{config,data,loop,checkpoint,run}.py`,
    `src/roboforge/eval/{learned,validation,selection}.py`;
  - `configs/models/{bc_mlp,bc_rnn,bc_mlp_chunk}.yaml`;
  - `scripts/{train,eval_policy,select_checkpoint,check_checkpoint_equivalence,param_counts,benchmark_policies,diagnose_actions,overfit_check,train_smoke}.py`;
  - tests: `tests/unit/{test_bc_models,test_train_loop,test_selection}.py`,
    `tests/integration/test_train_real_data.py`, `tests/simulator/test_learned_policy.py`;
  - evidence: `reports/session3/`, `logs/session3/`.
- **Changed:**
  - `src/roboforge/eval/rollout.py` (the `observe` hook only) and `eval/policies.py` (docs);
  - package docstrings, `Makefile`, `README.md`;
  - docs: RESEARCH_PROTOCOL, DECISIONS, KNOWN_ISSUES, EXPERIMENT_LEDGER, STATUS, HANDOFF.
- **Unchanged:** `uv.lock`, `pyproject.toml`, `data/`, `configs/{eval,data,simulator}`,
  `infra/`, `src/roboforge/{sim,data}/` (`git diff 0322a15 -- …` is empty).

## 4. Executed commands and results
| Command (commit) | Platform | Result |
|---|---|---|
| prerequisites (`0322a15`): setup, download, check-data, build-data, check-eval-states, `git status --short data/`, lint, test, smoke-sim, container smoke Lift/Can, image ID | macOS + container | all PASS; 185 tests; smoke 33/33, 17/17, 16/16 (`logs/session3/prereq/`) |
| KI-007 probe (Write/Edit in `src/roboforge/train/`) | macOS | not blocked; deleted |
| `make lint`, `make test` (`969e14f`, again at `a37b92d`) | macOS | clean; **231 passed** |
| `make train-smoke` (`969e14f`) | macOS | BC-MLP and BC-RNN PASS (dev-only rollouts) |
| `make overfit-check` (`969e14f`) | macOS | MSE ×1.7e-6 to ×0.015 (`reports/session3/overfit_lift.json`) |
| `make param-counts` (`969e14f`) | macOS | `reports/session3/param_counts.json` |
| `scripts/train.py --config configs/models/<m>.yaml --task lift --seed 0 --prefix s3` × 3 (`969e14f`, concurrently) | macOS CPU | 3 runs complete, reload checks PASS (`logs/session3/train/`) |
| `make eval RUN=runs/<id>` × 3 (`969e14f`; now `TRAIN_RUN=`) | container | 18 evaluations × 50 starts, 0 errors, 54/54 audits PASS (`reports/session3/validation/`) |
| `scripts/select_checkpoint.py` / `make select TRAIN_RUN=…` | macOS | `reports/session3/selection/` |
| `scripts/check_checkpoint_equivalence.py` (last and selected checkpoints) | container + macOS | all PASS, including the Session 2 sanity regression 4/4 (`reports/session3/checkpoint_equivalence/`) |
| `make benchmark-policies RUNS=…` and the macOS equivalent (`a37b92d`) | container + macOS | `reports/session3/cost_latency/` |
| `scripts/diagnose_actions.py --train-runs …` (`a37b92d`) | macOS | `reports/session3/diagnostics/actions.json` |
| contamination check | macOS | 900/900 records `eval_validation`; locked suite named only in the 3 allowed files (`logs/session3/contamination_check.log`) |

## 5. Key findings (development, validation-only, one seed)
- **Lift is not saturated for one-step BC.**
  - BC-MLP: 41–47/50 across checkpoints.
  - BC-RNN: 48–50/50; 50/50 at 5k–15k.
  - BC-MLP-chunk: 46–50/50.

  These pilots do not test any hypothesis.
- **Every failure is a grasp failure** (KI-021).
  - In the typical BC-MLP failure, the gripper closes without the cube, and the arm then drifts
    out of the workspace (z up to 2.19 m) on saturated actions and never recovers.
  - Motion clipping covers 50–78 % of failed-episode steps, against ≤ 1.3 % in successes.
- **No action convention problem:**
  - the controller axes and signs are verified;
  - the gripper sign agrees with the demos 92–96 % of the time (teacher-forced);
  - gripper magnitude clipping is harmless, because robosuite uses the sign only (KI-022).
- **Offline validation MSE rises from the first checkpoint** while closed-loop success
  barely moves (KI-024). Offline loss must not select, which §7 already requires.
- **Cost:**
  - 30k steps take 146 s (BC-MLP), 983 s (BC-RNN) and 155 s (BC-MLP-chunk) on local CPU,
    3 concurrent runs.
  - Container inference is 13–17× slower than native: 0.91 / 8.6 / 1.26 ms per decision
    (KI-023).
  - A 50-start Lift validation evaluation took 43–101 s of wall time in the container
    (`wall_s` in the 18 summaries). Those runs were not isolated: up to three containers and
    the BC-RNN training ran at the same time.

## 6. Known action/controller quirks
- The OSC_POSE delta is in the world frame. Rotation commands also translate the grip-site
  position, because the site is offset from the hand body
  (`reports/session3/diagnostics/actions.json` → `controller_axes`).
- The gripper is binary by sign (`PandaGripper.format_action`), so its clip rate must be
  reported separately from the motion dimensions.
- The normalized rotation dimensions are stretched up to 7.4× (D-020). A decoded rotation
  beyond ±1 is more than twice the demos' extreme values, and it occurs only after drift.

## 7. Open tuning decisions (Session 6, within RESEARCH_PROTOCOL §6.3)
- **The search space per family**, at most 3 more configurations each on Lift. It must be
  written in the ledger before the first tuning run. Candidates:
  - weight decay, dropout or observation noise against drift (KI-021);
  - learning-rate schedule;
  - MLP depth or width;
  - LSTM vs GRU.
- **The main budget** (100k steps, checkpoints every 10k) is to be confirmed in Session 5
  from measured diffusion cost, and stays identical across families.
- **Can pilots** have not been run (trial counts on Can are 0).
- **GMM-BC** is optional and not implemented.

## 8. Decisions
- Research protocol v1.
- D-032 (`observe` hook), D-033 (BC families), D-034 (training defaults), D-035
  (checkpoints/manifests/artifacts), D-036 (inference contract), D-037 (validation-only
  entry points), D-038 (evidence location).

## 9. Outstanding issues (none blocks Session 4)
- KI-021, KI-024: compounding error and offline/closed-loop mismatch; tuning topics.
- KI-023: measure diffusion latency in the container.
- KI-011: Lift starts are narrow, so spatial claims need H4 sets.
- KI-001: x86_64 and EGL are unverified.
- KI-008: 55 GB free.
- KI-020, KI-019, KI-018: the rules still apply.
- **Session-3 paths are hardcoded in the tooling.** Fix this before Session 4 writes evidence:
  - The Makefile has `EVIDENCE_SESSION ?= 3`. Its `eval`, `select`, `benchmark-policies` and
    `check-ckpt-equivalence` targets therefore mount and write `reports/session3`, whatever
    `ROBOFORGE_EVIDENCE_SESSION` is set to. Pass `EVIDENCE_SESSION=4` explicitly, or make
    its default follow `ROBOFORGE_EVIDENCE_SESSION`.
  - The scripts' `--reports-dir` and `--out-dir` defaults point to `reports/session3`; pass
    `reports/session4`.
  - `scripts/check_checkpoint_equivalence.py` hardcodes `OUT` and `REFERENCE` to
    `reports/session3/checkpoint_equivalence/`. A session-4 container does not mount that
    directory, so the report would be lost and check E would fail. Parameterize the path
    (a recorded change) and copy the Session 2 sanity reference into the session-4 evidence
    first.
- **The BC stack assumes a deterministic forward pass plus masked action MSE.** This applies
  to `Trainer.step`, `offline_metrics`, `reload_check`, `LearnedPolicy.act` and the
  `FAMILIES` registry. Diffusion needs a family-specific loss and a seeded sampling path.

## 10. Budget
$0.00 spent; local CPU only. Session 4 needs no approval for local CPU. Any paid compute
needs explicit owner approval and a cost bound first (Session 5).

## 11. Exact starting instructions for Session 4
Paste into a fresh Claude Code session:

```text
Implement RoboForge 3D SESSION 4 ONLY from MASTER_SPEC.md. Begin by reading all of
MASTER_SPEC.md, docs/STATUS.md, docs/HANDOFF.md, docs/DECISIONS.md, docs/KNOWN_ISSUES.md,
docs/EXPERIMENT_LEDGER.md, docs/RESEARCH_PROTOCOL.md (v1, binding), docs/SIMULATOR_CONTRACT.md,
docs/DATA_DICTIONARY.md and docs/COMPUTE_PLAN.md, and inspect the actual repository/test
state. Do not blindly trust prior handoff claims: first run `make setup`, `make download`,
`make check-data`, `make build-data`, `make check-eval-states`, confirm
`git status --short data/` is empty, then run `make lint`, `make test`,
`make smoke-sim SMOKE_OUT=logs/session4/prereq`, `make train-smoke SMOKE_DIR=logs/tmp/s4_smoke`
and, on the canonical container,
`ROBOFORGE_EVIDENCE_SESSION=4 make in-sim CMD="python scripts/smoke_sim.py --task lift --out-dir logs/session4/prereq/container"`
and the same with `--task can`, and confirm they pass; confirm
`docker image inspect -f '{{.Id}}' roboforge-sim-cpu:session0` equals the image_id in
configs/eval/platform.yaml; check whether the Session 3 pilot runs exist under runs/
(gitignored; their manifests, checkpoint hashes and selections are in reports/session3 and
docs/HANDOFF.md; report if they are missing rather than assuming); run
git log --format=fuller -n 5 to confirm the verified commits and identity
Zackary Holt <zackholt05@gmail.com>.

Session 4 goal (MASTER_SPEC Session 4): a real conditional diffusion action policy with
truthful denoising instrumentation, not a renamed deterministic head. BEFORE the first
reported diffusion run, fill every RESEARCH_PROTOCOL item marked TO FINALIZE (Session 4) by a
dated §13 amendment: architecture (observation encoder, 1D temporal U-Net widths/stages),
exact parameter count and the §6.1 capacity rule (if > 2x the 1.08 M BC budget, add
capacity-matched BC-MLP and BC-RNN using reports/session3/param_counts.json tiers),
noise schedule, training and inference denoising steps, sampler (DDPM/DDIM), prediction
target, EMA, horizons (default T_o=2, T_p=16, T_a=8) and the stochastic noise-seed rule
(policy_seed, state_id, repeat) with the repeat count R. Implement and test: observation
conditioning, the U-Net with timestep embedding, DDPM/DDIM training and sampling, masked
normalized action-chunk loss, optional EMA, receding-horizon control and per-episode reset,
integrated with the Session 3 stack (src/roboforge/train trainer/checkpoint/manifest,
configs/models/<name>.yaml, LearnedPolicy extended for seeded stochastic sampling via the
observe hook, ActionCodec decoding, validation-only scripts/eval_policy.py, selection rule).
Compare scheduler equations, tensor shapes, seeded reproducibility and a fixed synthetic
case against robomimic's reference (the pinned d309eae diffusion_policy and diffusers
0.11.1 schedulers), and mark any adapted external component with its license. Show a
tiny-batch overfit on real Lift windows with finite gradients; checkpoint reload
equivalence including identical seeded samples and identical closed-loop physics_sha256 on
one platform. Build export_denoising that saves actual intermediate action tensors A_{t_k}
at selected scheduler timesteps with the conditioning observation, normalized and
denormalized values, noise seed and scheduler coefficients, labeled as proposals. Run a
small Lift validation pilot on the canonical container (same pilot budget as Session 3:
30k steps, checkpoints every 5k, if local CPU cost allows; measure and record it, and if it
does not fit, record the measured cost and propose a reduced, family-identical budget as a
protocol amendment before running); measure diffusion inference latency per decision in
the container (KI-023) and debug normalization/replanning before any mass training. The
diffusion pilot counts as diffusion's tuning trial 1 (§6.3). Do not retune or rerun the
BC pilots.

Keep the Session 1–3 contracts (DECISIONS D-010, D-017..D-038): training data only from
SequenceDataset.from_committed; policy observations only from RoboForgeEnv; outputs decoded
with ActionCodec; learned policies implement roboforge.eval.policies.Policy and run with
run_episode + verify_record; resets only from stored snapshots; first success or 400 steps;
model and checkpoint selection only on the eval_validation suite; never load, name,
summarize or reset to the held-out test states and do not create docs/FINAL_TEST_FREEZE.md;
every reported or selection rollout on the canonical container via
infra/containers/run_sim_cpu.sh with ROBOFORGE_EVIDENCE_SESSION=4 (never uv sync/uv run
inside the image, KI-020). The Session 3 tooling hardcodes session-3 evidence paths
(HANDOFF §9): pass EVIDENCE_SESSION=4 to make targets and --reports-dir/--out-dir
reports/session4 to scripts, or fix the defaults as a recorded change, and parameterize
scripts/check_checkpoint_equivalence.py's output and sanity-reference paths before using it
under session 4. The BC trainer and LearnedPolicy assume a deterministic forward pass plus
masked action MSE, so diffusion needs a family-specific loss and sampling path; native macOS rollouts are development-only; never call
mj_forward on live evaluation data (KI-019); never use robosuite transform helpers for
precision-critical values (KI-018); checkpoints and anything a container rollout reads
live under runs/<run_id>/; every run writes an immutable manifest with git SHA + dirty flag
(launch reported runs from a clean committed tree; do not edit tracked files while they
start), environment, platform, data/spec hashes, full config, seeds, parameter count,
checkpoint hashes and wall/CPU time; report motion-dimension clipping separately from the
gripper (KI-022). Use the Make variable TRAIN_RUN (not RUN) for run directories. Train on
local CPU only; no paid or remote compute (Modal is Session 5) and no remote publication
without explicit owner approval; do not change committed data artifacts, eval suites,
uv.lock or dependencies without a recorded decision (an image rebuild is a new canonical
platform under D-017).

Use the current verified simulator/data/checkpoint/manifest contracts. Run relevant tests
and record commands, real results and provenance. Never leak held-out test states into model
selection; never invent results. Use Zackary Holt as the sole named RoboForge project
author with the verified local Git identity; no AI co-author trailers, while preserving
external scientific and license credit (Diffusion Policy, robomimic, diffusers). At the
SESSION 4 STOP boundary, update docs/STATUS.md, docs/HANDOFF.md, docs/DECISIONS.md,
docs/KNOWN_ISSUES.md and docs/EXPERIMENT_LEDGER.md (exact architecture, parameter count,
scheduler, horizons, validated smoke-run artifacts), commit verified code only, inspect
commit metadata (git log --format=fuller -n 5), write the exact prompt for SESSION 5, and
STOP. Do NOT proceed automatically.
```
