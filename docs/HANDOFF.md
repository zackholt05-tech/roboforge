<!-- parsable handoff header (values below are real, filled at the end of Session 4) -->
```yaml
project: roboforge-3d
current_session: 4
session_status: PASS
next_session: 5
last_verified_commit: 183b56c   # lint clean, 257 tests; later commits are docs/evidence only
protocol_commits: {v1: 8b08b95, diffusion_amendment: 3aed462}   # amendment before any reported diffusion run
evidence_commits:
  diffusion_training: 3aed4629bbc5089e92c2902f5f8cf753a97c41f1   # pilot run, dirty=false
  validation_selection_equivalence_latency_export_diagnostics: 183b56cb69a3f15b5ca961071fe536858f297ab5   # dirty=false; differs from 3aed462 only in scripts/diagnose_actions.py
python: 3.11.15                       # development (macOS arm64); canonical platform uses 3.11.18
sim_engine_version: mujoco 3.2.7
robosuite_revision: 51cc01785bab80ffeed20da15e67d7dd4140e76a   # v1.5.1
robomimic_revision: d309eaecc18acf4152a830a895a6984b8ac71b05
diffusers: 0.11.1
torch: 2.15.0 (macOS) / 2.15.0+cu132 (container, CPU path)
evaluation_platform: {platform_id: linux-aarch64-container-1f685638f28c, image_id: "sha256:1f685638f28c720fbec7b084b8e5c2d68073d8f637450d5f6e0a34f0d123071c", runner: infra/containers/run_sim_cpu.sh}
dataset_and_specs: unchanged since Session 2   # git diff 71ab833 -- data/ uv.lock pyproject.toml configs/eval empty
baseline_configs:                     # configs/models/*.yaml (DEFAULT, D-033/D-034/D-040)
  bc_mlp:         {horizons: "T_o=1 T_p=1 T_a=1", model: "MLP 2x1024 ReLU", params: 1077255}
  bc_rnn:         {horizons: "L=10 T_p=1 T_a=1", model: "LSTM 2x296, sliding window", params: 1080703}
  bc_mlp_chunk:   {horizons: "T_o=2 T_p=16 T_a=8", model: "MLP 2x1024 ReLU", params: 1204336}
  diffusion_unet: {horizons: "T_o=2 T_p=16 T_a=8", model: "flatten encoder (0) + ConditionalUnet1D down (64,128), kernel 5, GN 8, step embed 64", params: 1092551,
                   schedule: "DDPM forward, K=100, squaredcos_cap_v2, epsilon-prediction", sampler: "DDIM-10, eta=0, clip_sample, set_alpha_to_one",
                   ema: "diffusers EMAModel power 0.75, max 0.9999; EMA weights evaluated", noise_seed_rule: "SeedSequence([policy_seed, u64le(sha256(state_id)[:8]), repeat]); policy_seed 0; R=1"}
  training: "AdamW 1e-4, batch 100, masked (denoising) MSE, constant LR, no clip, 1 thread CPU; pilot 30k steps, ckpt every 5k"
pilot_runs:                           # runs/<id>/ (gitignored); Lift, seed 0, tuning trial 1 each
  s3_bc_mlp_lift_seed0_dbf312cd:          {selected_step: 5000,  validation: 47/50, checkpoint_sha256: 30289109273d433a364aa5a7e1a4fb41e90fc58d804fcc094936b4582b4559db}
  s3_bc_rnn_lift_seed0_2e7cffe3:          {selected_step: 15000, validation: 50/50, checkpoint_sha256: 3dbb53ce2a711d89eeb0b5844ae69efb6a45f1da0de21a3fc2cce1472a3da756}
  s3_bc_mlp_chunk_lift_seed0_bb8b4dd6:    {selected_step: 10000, validation: 50/50, checkpoint_sha256: a763c83730aeb1dbc9d07fbcc76ede2c3b68e63487fc7e3175395f9774ab83a7}
  s4_diffusion_unet_lift_seed0_336e15e9:  {selected_step: 15000, validation: 50/50, checkpoint_sha256: f768cbe32b2664e97d483d1d135863dedcc3646fe2bdda6c0c641d69658459fc}
final_test_locked_until: docs/FINAL_TEST_FREEZE.md   # does not exist
training_spend_usd: 0
next_required_owner_approval: any paid Modal job (Gate A smoke first), with a cost bound
```

# Session 4 Handoff

Do not trust this file blindly: re-run the checks in "Exact starting instructions".

## 1. Session completed
Session 4 (conditional diffusion action policy and truthful denoising instrumentation):
**PASS**. Session 5 can start.

## 2. Completed work
- **Protocol amendment first.** RESEARCH_PROTOCOL §6.4 + §13 were committed at `3aed462`,
  before the first reported diffusion run. They fix:
  - the architecture and its exact parameter count;
  - the capacity rule, which is not triggered;
  - the schedule, sampler, prediction target and EMA;
  - the horizons, the noise-seed rule with R = 1, and tuning trial 1.
- **Diffusion family `diffusion_unet`** (D-040):
  - `src/roboforge/models/{diffusion,unet1d,noise_schedule}.py`, `src/roboforge/train/ema.py`,
    `configs/models/diffusion_unet.yaml`;
  - the U-Net is adapted from robomimic `d309eae` (MIT, credited in `THIRD_PARTY_NOTICES.md`);
  - the schedulers re-implement the diffusers 0.11.1 equations (Apache-2.0 headers recorded);
  - all of it is **bitwise equal** to the references on macOS and the container.
- **Stack integration:**
  - a model registry and family-specific loss kinds (D-042);
  - the trainer's own noise generator plus EMA, both checkpointed, so resuming is bitwise;
  - a reload check that compares seeded samples;
  - `LearnedPolicy` seeded per episode by `(policy_seed, state_id, repeat)`;
  - `policy_episode` in each record, and evaluation IDs that end in `__ps0r0` (D-041).
- **Evidence tooling** (D-039):
  - `ROBOFORGE_EVIDENCE_SESSION` drives every default output directory, and the container
    runner forwards it;
  - `check_checkpoint_equivalence.py` has `--out-dir` and check F;
  - new `compare_diffusion_reference.py` and `export_denoising.py` (D-043);
  - `benchmark_policies.py` gains `--max-decisions` and `--diffusion-samplers`;
  - `diagnose_actions.py` is diffusion-aware, with receding-horizon checks.
- **Pilot:**
  - 30k-step Lift training (57 min, local CPU);
  - 6 canonical validation evaluations, then selection (step 15000);
  - equivalence checks, latency, denoising exports, diagnostics and the contamination check.

## 3. Files created or changed
- **New:**
  - `src/roboforge/models/{diffusion,unet1d,noise_schedule,losses,registry,reference}.py`,
    `src/roboforge/train/ema.py`, `src/roboforge/viz/denoising.py`;
  - `configs/models/diffusion_unet.yaml`;
  - `scripts/{compare_diffusion_reference,export_denoising}.py`;
  - tests: `tests/unit/{test_diffusion_reference,test_diffusion_policy}.py`,
    `tests/simulator/test_diffusion_closed_loop.py`;
  - evidence: `reports/session4/`, `logs/session4/`.
- **Changed:**
  - `src/roboforge/{paths.py,train/{config,loop,run}.py,eval/{learned,rollout,validation}.py}`;
  - `src/roboforge/models/__init__.py`, `src/roboforge/viz/__init__.py`;
  - `scripts/{benchmark_policies,check_checkpoint_equivalence,diagnose_actions,eval_policy,overfit_check,param_counts,select_checkpoint,train_smoke}.py`;
  - `tests/integration/test_train_real_data.py`;
  - `Makefile`, `infra/containers/run_sim_cpu.sh` (forwards the evidence session),
    `THIRD_PARTY_NOTICES.md`;
  - docs: RESEARCH_PROTOCOL, DECISIONS, KNOWN_ISSUES, EXPERIMENT_LEDGER, STATUS, HANDOFF,
    COMPUTE_PLAN; `README.md`.
- **Unchanged:** `uv.lock`, `pyproject.toml`, `data/`, `configs/{eval,data,simulator}`,
  `src/roboforge/{sim,data}/`, the container image.

## 4. Executed commands and results
| Command (commit) | Platform | Result |
|---|---|---|
| Prerequisites (`71ab833`): setup, download, check-data, build-data, check-eval-states, `git status --short data/`, lint, test, smoke-sim, train-smoke, container smoke Lift/Can, image ID, Session 3 runs | macOS + container | all PASS; 231 tests; smoke 33/33, 17/17, 16/16 (`logs/session4/prereq/`) |
| `scripts/compare_diffusion_reference.py` (`22adb8b` macOS, `4c68ca1` container) | both | bitwise, PASS (`reports/session4/reference/`) |
| `make param-counts`, `make overfit-check` (`22adb8b`) | macOS | 1,092,551 params; sampled-chunk MSE ×0.014 / ×0.062 |
| `check_checkpoint_equivalence.py` on the 3 Session 3 selected checkpoints (`4c68ca1`) | container | A–E PASS (BC regression under the new code) |
| `benchmark_policies.py` on a 200-step dev checkpoint (`22adb8b`) | both | `reports/session4/latency_pre_amendment/` (basis of DDIM-10) |
| `make train MODEL=diffusion_unet TASK=lift SEED=0 PREFIX=s4` (`3aed462`) | macOS CPU | `complete`, reload PASS, 3,428 s |
| `make eval TRAIN_RUN=runs/s4_diffusion_unet_lift_seed0_336e15e9 EVIDENCE_SESSION=4` (`183b56c`) | container | 6 × 50 starts: 49/50/50/49/49/49; 0 errors; 18/18 audits |
| `make select TRAIN_RUN=…` (`183b56c`) | macOS | step 15000 |
| `ROBOFORGE_EVIDENCE_SESSION=4 infra/containers/run_sim_cpu.sh python scripts/check_checkpoint_equivalence.py --train-run … --step {15000,30000}`, plus macOS 30000 | container + macOS | all PASS (A–F) |
| `benchmark_policies.py --train-runs … --max-decisions 100 --repeats 1 --diffusion-samplers ddpm:100` (`183b56c`) | container + macOS | DDIM-10 36.3 / 9.9 ms; DDPM-100 649 / 99.5 ms |
| `make export-denoising EVAL_RUN=runs/<eval id>` for steps 15000 and 30000 | container | 9 decisions, bitwise replay |
| `make diagnose-actions RUNS=runs/s4_…` | macOS | axes PASS; receding horizon PASS 300/300; clip 0.0 |
| `logs/session4/contamination_check.py` | macOS | 300/300 `eval_validation`; locked suite named only in the 3 allowed files |
| `make lint`, `make test`, `make train-smoke` (`183b56c`) | macOS | clean; **257 passed**; train-smoke PASS for BC-MLP, BC-RNN and diffusion |

## 5. Key findings (development, validation-only, one seed)
- **The diffusion policy controls Lift.**
  - 49–50/50 at every checkpoint; 50/50 at 10k and 15k.
  - Its mean first-success step falls from 82 to about 48 over training. BC-RNN's selected
    checkpoint needed 43.2.
  - These pilots do not test any hypothesis.
- **Every diffusion failure is `lift_validation_0029`** under the same noise seed (KI-028).
  With R = 1, the start and the seed are confounded.
- **Normalization and replanning are correct.** Across all 300 episodes:
  - raw actions lie within [−1, 1] (`clip_sample`);
  - executed actions equal the codec decode;
  - the chunk offsets and decision counts follow T_a = 8;
  - 0 clipping in any dimension. The BC clipping problem (KI-021/KI-022) does not occur.
- **Offline metrics disagree with each other and with the closed loop** (KI-024 again):
  - validation noise-prediction MSE is lowest at 15k, then rises;
  - the sampled-chunk MSE keeps falling to 30k;
  - success is flat.
- **Teacher-forced gripper sign agreement** of one sample is 0.83–0.89, lower than BC's
  0.92–0.96. Gripper sign changes per episode drop from 5.9 at 5k to 1.7–3.0 later.
- **Cost:**
  - Training: 30k steps take 3,428 s on local CPU, 1 thread (8.75 steps/s). That is about
    23× BC-MLP and 3.5× BC-RNN.
  - Evaluation: a 50-start Lift validation evaluation takes 60–88 s in the container.
  - DDIM-10 takes 36 ms per decision in the container. DDPM-100 takes 649 ms there, which
    would make every evaluation many times longer.

## 6. Denoising instrumentation (MASTER_SPEC §5.5, §8.4)
- `reports/session4/denoising/<eval_run_id>/` holds `index.json` plus one `.json` and one
  `.npz` per decision.
- Each decision stores:
  - per reverse step `t_k` (90 … 0): `A_{t_k}`, ε̂, x̂₀ before and after the clip, the step
    output and every coefficient, normalized and denormalized;
  - the conditioning window, the noise seed and the final decoded chunk.
- Every exported decision re-ran the sampler with the episode's own seed and reproduced the
  recorded actions **bitwise**. The arrays are labeled **proposals**: only the first 8
  actions of each final chunk were executed.
- The 15k export has no failure, because that evaluation scored 50/50. The 30k export
  includes failure `lift_validation_0029`.
- The targets were pre-registered in `logs/session4/denoising/export_plan.txt` before any
  validation result existed.

## 7. Open decisions
- **Session 5 (Modal):**
  - choose the GPU from a short profile;
  - project the main-suite cost;
  - no paid job without owner approval (Gates A–C).
- The main suite has **4 families** (BC-MLP, BC-RNN, diffusion, and the BC-MLP-chunk H2-c
  control), against MASTER_SPEC's "3 algorithms". The cost projection must use 4, or the
  protocol must be amended before Session 6.
- **The main budget** (100k steps, checkpoints every 10k, RESEARCH_PROTOCOL §6.3) is still to
  be confirmed in Session 5 from these measurements:
  - diffusion 8.75 steps/s on local CPU, so 100k steps ≈ 3.2 h per run locally;
  - 60–88 s per diffusion validation evaluation in the container.
- **Tuning (Session 6):**
  - each family has 3 more trials on Lift, with its search space written in the ledger first;
  - diffusion candidates: U-Net width, DDIM steps, learning rate / schedule
    (robomimic uses cosine with warm-up), weight decay.
- **R > 1** on the validation suite (KI-028), only by an amendment before the freeze.
- The Can pilots have not run.

## 8. Decisions
- Research protocol amendment 2026-11-06 (diffusion items).
- D-039 (evidence session), D-040 (diffusion family), D-041 (stochastic inference),
  D-042 (registry, checkpoint keys), D-043 (denoising export), D-044 (evidence location).

## 9. Outstanding issues (none blocks Session 5)
- **KI-028:** the diffusion failures are confounded with one seed.
- **KI-024:** offline metrics do not select.
- **KI-021:** BC drift.
- **KI-023:** container slowdown; diffusion latency is measured.
- **KI-025:** robomimic ignores `algo.unet.*`.
- **KI-026:** NNPACK, mitigated.
- **KI-027:** batch-size rounding.
- **KI-011:** Lift starts are narrow.
- **KI-001:** x86_64 and EGL are unverified. **Session 5 must verify both** for Modal.
- **KI-008:** disk space.
- **KI-020, KI-019, KI-018:** the rules still apply.
- **Session 5 must set `ROBOFORGE_EVIDENCE_SESSION=5` for every evidence command.** The
  Makefile's `EVIDENCE_SESSION` falls back to 4, and `PREFIX` defaults to `s4`. Either pass
  `EVIDENCE_SESSION=5 PREFIX=s5`, or bump the defaults as a recorded change.

## 10. Budget
$0.00 spent; local CPU only. Any Modal job needs explicit owner approval and a cost bound
first. Gate A is a GPU smoke test of at most 15 minutes, approved separately
(COMPUTE_PLAN §5).

## 11. Exact starting instructions for Session 5
Paste into a fresh Claude Code session:

```text
Implement RoboForge 3D SESSION 5 ONLY from MASTER_SPEC.md. Begin by reading all of
MASTER_SPEC.md, docs/STATUS.md, docs/HANDOFF.md, docs/DECISIONS.md, docs/KNOWN_ISSUES.md,
docs/EXPERIMENT_LEDGER.md, docs/RESEARCH_PROTOCOL.md (v1 + the 2026-11-06 diffusion
amendment, binding), docs/SIMULATOR_CONTRACT.md, docs/DATA_DICTIONARY.md and
docs/COMPUTE_PLAN.md, and inspect the actual repository/test state. Do not blindly trust
prior handoff claims: first run `make setup`, `make download`, `make check-data`,
`make build-data`, `make check-eval-states`, confirm `git status --short data/` is empty,
then run `make lint`, `make test`, `make smoke-sim SMOKE_OUT=logs/session5/prereq`,
`make train-smoke SMOKE_DIR=logs/tmp/s5_smoke` and, on the canonical container,
`ROBOFORGE_EVIDENCE_SESSION=5 make in-sim CMD="python scripts/smoke_sim.py --task lift --out-dir logs/session5/prereq/container"`
and the same with `--task can`, and confirm they pass; confirm
`docker image inspect -f '{{.Id}}' roboforge-sim-cpu:session0` equals the image_id in
configs/eval/platform.yaml; check whether the Session 3 BC pilot runs and the Session 4
diffusion pilot run (runs/s4_diffusion_unet_lift_seed0_336e15e9) exist under runs/
(gitignored; manifests, checkpoint hashes and selections are in reports/session3,
reports/session4 and docs/HANDOFF.md; report if they are missing rather than assuming);
run git log --format=fuller -n 5 to confirm the verified commits and identity
Zackary Holt <zackholt05@gmail.com>.

Session 5 goal (MASTER_SPEC Session 5): make cloud training reliable and economically
bounded, and estimate the full research cost. Build the Modal CLI/SDK entrypoint
(infra/modal_app.py per docs/COMPUTE_PLAN.md §3) with a pinned CUDA image installed from
uv.lock, Volumes for data and runs, safe secrets (none in the repo), a machine-readable
cost/time log, a local dry-run that is the default, and an explicit approved-launch path
(--budget-approved, max_containers=1, retries=0, explicit timeout, refusing to overwrite an
existing run_id). No Modal resource (app deploy, volume, job, image build on Modal) may be
created without explicit owner approval recorded in docs/EXPERIMENT_LEDGER.md; recheck
https://modal.com/pricing before any estimate. Prove resumable, interrupt-safe checkpoints
and hash-validated artifact retrieval, and that a Modal-trained checkpoint is reusable
outside Modal: it must load locally with the same manifest/checkpoint contract (D-035,
D-042) and run through the canonical container's validation path; until a paid run is
approved, prove this with a local simulation of the remote layout (Volume-shaped
directories, interrupted and resumed training, hash-checked download). Verify KI-001 for
the Modal target (linux/x86_64 install of uv.lock; EGL vs OSMesa) only inside an approved
job; otherwise record it as pending. Profile one CPU/parallel rollout pilot locally
(several canonical containers in parallel on validation starts only, measuring wall time
per evaluation and whether rendering or physics is the bottleneck); a GPU training profile
runs only after Gate A approval. Compute the projected cost of the main suite from the
measured numbers (HANDOFF §5, §7: diffusion 8.75 training steps/s on local CPU, 3,428 s per
30k steps; BC 146–983 s per 30k; 60–88 s per 50-start diffusion validation evaluation and
36 ms per DDIM-10 decision in the container): 3 training seeds × the protocol's families
(BC-MLP, BC-RNN, diffusion, plus the BC-MLP-chunk H2-c control: state explicitly whether
the main suite has 3 or 4 families, and amend RESEARCH_PROTOCOL before any main-suite run
if it changes) × two tasks (Lift, Can), plus validation-based checkpoint selection, the
held-out test rollouts, storage, rendering and a failed-run reserve. Present alternative
budgets, including a fully local-CPU path and a reduced-budget path, and confirm or amend
the main training budget (RESEARCH_PROTOCOL §6.3: 100k steps, checkpoints every 10k,
identical across families) as a dated §13 amendment before Session 6. Ask the owner to
approve Gate C before any full suite. Prevent automatic retries and multi-GPU scale-up.

Acceptance: a paid smoke test happens only if explicitly approved; otherwise the working
integration and the local dry-run are present and cloud execution is marked PENDING
AUTHORIZATION. If approved, verify real remote training, checkpoint persistence, resume,
logs and accurate billed-cost accounting.

Keep the Session 1–4 contracts (DECISIONS D-010, D-017..D-044): training data only from
SequenceDataset.from_committed; policy observations only from RoboForgeEnv; outputs decoded
with ActionCodec; learned policies implement roboforge.eval.policies.Policy and run with
run_episode + verify_record; stochastic policies seeded by the protocol's noise-seed rule
(policy_seed 0, R = 1); resets only from stored snapshots; first success or 400 steps;
model and checkpoint selection only on the eval_validation suite; never load, name,
summarize or reset to the held-out test states and do not create docs/FINAL_TEST_FREEZE.md;
every reported or selection rollout on the canonical container via
infra/containers/run_sim_cpu.sh with ROBOFORGE_EVIDENCE_SESSION=5 (never uv sync/uv run
inside the image, KI-020); a Modal image is a different platform and can never produce
canonical evaluation evidence without a recorded decision under D-017. The Makefile's
EVIDENCE_SESSION falls back to 4 and PREFIX to s4: pass EVIDENCE_SESSION=5 and PREFIX=s5 (or
bump the defaults as a recorded change) so Session 5 never writes into Session 4 evidence.
Native macOS rollouts are development-only; never call mj_forward on live evaluation data
(KI-019); never use robosuite transform helpers for precision-critical values (KI-018);
checkpoints and anything a container rollout reads live under runs/<run_id>/; every run
writes an immutable manifest with git SHA + dirty flag (launch reported runs from a clean
committed tree; do not edit tracked files while they run), environment, platform, data/spec
hashes, full config, seeds, parameter count, checkpoint hashes and wall/CPU time; report
motion-dimension clipping separately from the gripper (KI-022). Use the Make variable
TRAIN_RUN (not RUN) for run directories. Do not retune or rerun the Session 3–4 pilots. No
paid or remote compute and no remote publication without explicit owner approval; do not
change committed data artifacts, eval suites, uv.lock or dependencies without a recorded
decision (an image rebuild is a new canonical platform under D-017; the Modal SDK must stay
outside the project venv or be added only by a recorded decision).

Use the current verified simulator/data/checkpoint/manifest contracts. Run relevant tests
and record commands, real results and provenance. Never leak held-out test states into model
selection; never invent results or costs (label estimates as estimates). Use Zackary Holt
as the sole named RoboForge project author with the verified local Git identity; no AI
co-author trailers, while preserving external scientific and license credit (Diffusion
Policy, robomimic, diffusers). At the SESSION 5 STOP boundary, update docs/STATUS.md,
docs/HANDOFF.md, docs/DECISIONS.md, docs/KNOWN_ISSUES.md, docs/EXPERIMENT_LEDGER.md and
docs/COMPUTE_PLAN.md (actual GPU choice and billed/tested runtime, or PENDING AUTHORIZATION;
the cost proposal; the approved limit), commit verified code only, inspect commit metadata
(git log --format=fuller -n 5), write the exact prompt for SESSION 6, and STOP. Session 6
proceeds only after compute authorization or a clearly budget-reduced local path. Do NOT
proceed automatically.
```
