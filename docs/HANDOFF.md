<!-- parsable handoff header (values below are real, filled at the end of Session 5) -->
```yaml
project: roboforge-3d
current_session: 5
session_status: PASS
next_session: 6
last_verified_commit: db10a1b   # lint clean, 281 tests; later commits are docs/evidence only
protocol_commits: {v1: 8b08b95, diffusion_amendment: 3aed462, main_budget_amendment: ec9626b}   # 2026-11-07: 100k steps, checkpoints every 5k
python: 3.11.15                       # development (macOS arm64); canonical platform uses 3.11.18
sim_engine_version: mujoco 3.2.7
robosuite_revision: 51cc01785bab80ffeed20da15e67d7dd4140e76a   # v1.5.1
robomimic_revision: d309eaecc18acf4152a830a895a6984b8ac71b05
diffusers: 0.11.1
torch: 2.15.0 (macOS) / 2.15.0+cu132 (container and Modal)
evaluation_platform: {platform_id: linux-aarch64-container-1f685638f28c, image_id: "sha256:1f685638f28c720fbec7b084b8e5c2d68073d8f637450d5f6e0a34f0d123071c", runner: infra/containers/run_sim_cpu.sh}
dataset_and_specs: unchanged since Session 2   # git diff d956292 -- data/ uv.lock pyproject.toml configs/eval empty
main_suite:                           # RESEARCH_PROTOCOL §6.3 + §13 (2026-11-07), D-051
  families: [bc_mlp, bc_rnn, bc_mlp_chunk, diffusion_unet]   # 4: BC-MLP-chunk is the H2-c control
  tasks: [lift (Session 6), can (Session 7)]
  seeds: [0, 1, 2]
  budget: "100,000 steps, batch 100, checkpoints every 5,000 (K = 20), every checkpoint evaluated on 50 validation starts"
  tuning: "<= 4 configurations per family per task at the main budget, seed 0; trial 1 = pilot configuration re-trained at the main budget (not an extra trial)"
  compute: "local CPU only, 1 torch thread, <= 8 concurrent training runs; evaluation on the canonical container, <= 4 in parallel; $0"
baseline_configs:                     # configs/models/*.yaml still carry the 30k pilot budget
  bc_mlp:         {horizons: "T_o=1 T_p=1 T_a=1", model: "MLP 2x1024 ReLU", params: 1077255}
  bc_rnn:         {horizons: "L=10 T_p=1 T_a=1", model: "LSTM 2x296, sliding window", params: 1080703}
  bc_mlp_chunk:   {horizons: "T_o=2 T_p=16 T_a=8", model: "MLP 2x1024 ReLU", params: 1204336}
  diffusion_unet: {horizons: "T_o=2 T_p=16 T_a=8", model: "flatten encoder + ConditionalUnet1D (64,128)", params: 1092551, sampler: "DDIM-10", ema: "power 0.75"}
pilot_runs:                           # runs/<id>/ (gitignored); Lift, seed 0, tuning trial 1 each; reload equivalence re-passed in Session 5 (E5-14)
  s3_bc_mlp_lift_seed0_dbf312cd:          {selected_step: 5000,  validation: 47/50, checkpoint_sha256: 30289109273d433a364aa5a7e1a4fb41e90fc58d804fcc094936b4582b4559db}
  s3_bc_rnn_lift_seed0_2e7cffe3:          {selected_step: 15000, validation: 50/50, checkpoint_sha256: 3dbb53ce2a711d89eeb0b5844ae69efb6a45f1da0de21a3fc2cce1472a3da756}
  s3_bc_mlp_chunk_lift_seed0_bb8b4dd6:    {selected_step: 10000, validation: 50/50, checkpoint_sha256: a763c83730aeb1dbc9d07fbcc76ede2c3b68e63487fc7e3175395f9774ab83a7}
  s4_diffusion_unet_lift_seed0_336e15e9:  {selected_step: 15000, validation: 50/50, checkpoint_sha256: f768cbe32b2664e97d483d1d135863dedcc3646fe2bdda6c0c641d69658459fc}
engineering_runs_session5:            # never trials, never in selection
  s5modal2_diffusion_unet_lift_seed0_2fda84af: {where: "trained on Modal CPU, downloaded", steps: 2000, final_ckpt_sha256: 724a506edcae332a7e352bcd67d76e31d5ffad055b3eab222b54aaa8bc6382fb}
  s5sim_diffusion_unet_lift_seed0_2fda84af: {where: "local Modal simulation"}
  s5sim2_diffusion_unet_lift_seed0_2fda84af: {where: "local Modal simulation, symlinked runs mount"}
modal: {gate_a: done, approval: "A-01 ($2.00 cap) closed", billed_usd: 0.148, gpu_tested: L4, gpu_used_for_main_suite: none, egl_vs_osmesa: PENDING (KI-033)}
final_test_locked_until: docs/FINAL_TEST_FREEZE.md   # does not exist
next_required_owner_approval: none for the local Session 6 path; any Modal use needs a new approval row
```

# Session 5 Handoff

Do not trust this file blindly: re-run the checks in "Exact starting instructions".

## 1. Session completed
Session 5 (MASTER_SPEC: Modal, costs, pilot benchmark). Status **PASS**. The integration was
tested on Modal under owner approval A-01 ($0.148 billed of $2.00). The main-suite cost was
projected from measured numbers. The owner then fixed the training budget (100k / 5k) and
Gate C (local CPU only, $0).

## 2. Completed work
- **Resumable, interrupt-safe training** (D-046):
  - a launch lock;
  - cadence and resume checkpoints with indexes;
  - a segment journal with heartbeats, orphan quarantine and metric truncation;
  - stops between steps (signals, deadline, `stop_at_step`).

  Resumed runs are content-identical to uninterrupted ones (9 tests; KI-029 explains why
  file hashes can differ).
- **Modal integration** (D-045, D-047, D-049, D-050):
  - `infra/modal_app.py` (`train`, `probe`, `fetch`, `cancel`), a dry run by default;
  - an approval gate against `docs/EXPERIMENT_LEDGER.md` and a cost ledger with billing
    reconciliation;
  - an attempt cap and an attempt-aware upper bound;
  - a preemption drill with a real SIGINT.
- **Gate A on Modal** (E5-04–E5-09):
  - x86_64 install of `uv.lock` and CUDA on L4 verified;
  - CPU and L4 training profiles of all 4 families;
  - real-SIGINT drill → Modal restart → clean stop → resume launch → hash-checked
    download;
  - the Modal-trained checkpoint passes reload equivalence and validation on the canonical
    container (E5-08).
- **Three Modal-specific bugs**, all found and fixed:
  - BC-RNN built a device-less index (KI-030);
  - `simulate_preemption` plus Modal reschedules made a restart loop (KI-031);
  - paths under a symlinked mount were recorded absolute (KI-032).

  The render check also needed the eval suite, which the image lacks (KI-033).
- **Local profiles:** training concurrency (8 processes, 1.24× slowdown); parallel canonical
  containers (1.78× at 4, `physics_sha256` unchanged); render backends in the container.
- **Cost projection** for five variants (COMPUTE_PLAN §4), RESEARCH_PROTOCOL §13 amendment,
  Gate C decision.

## 3. Files created or changed
- **New:**
  - `src/roboforge/train/{journal,stop}.py`;
  - `src/roboforge/compute/{pricing,cost_log,approval,plan,retrieve,remote,profile,estimate}.py`;
  - `infra/modal_app.py`, `configs/compute/modal.yaml`;
  - `scripts/{modal_sim,profile_rollouts,profile_parallel,profile_train_concurrency,render_probe}.py`;
  - tests: `tests/integration/{test_resumable_run,test_remote_job}.py`,
    `tests/unit/test_compute.py`, `tests/simulator/test_render_probe.py`.
- **Changed:**
  - `src/roboforge/train/{loop,run,checkpoint}.py`, `scripts/train.py`;
  - `src/roboforge/models/{bc,diffusion,noise_schedule}.py` (device placement only;
    bitwise unchanged on CPU);
  - `src/roboforge/paths.py` (D-050), `src/roboforge/provenance.py`, `Makefile` (D-048).
- **Docs:** STATUS, HANDOFF, DECISIONS (D-045–D-052), KNOWN_ISSUES (KI-001, KI-029–KI-033),
  EXPERIMENT_LEDGER (E5-01–E5-19, A-01, Gate C), COMPUTE_PLAN, RESEARCH_PROTOCOL (§6.3, §13).
- **Evidence:** `reports/session5/`, `logs/session5/`, `reports/compute/cost_ledger.jsonl`.

## 4. Executed commands and results
See EXPERIMENT_LEDGER E5-01–E5-19. Summary:
- prerequisites all PASS at `d956292`;
- the local simulations pass (`c7f043f`, `feef27e`);
- 6 Modal launches billed $0.148;
- profiles and projection;
- pilot equivalence re-passed;
- contamination check clean;
- final `make lint` clean and `make test` **281 passed** at `db10a1b`.

## 5. Key findings (engineering and cost, not results)
- **Platform:** Modal x86_64 CPU single-thread speed is family-dependent relative to the M1
  Pro.
  - BC-MLP is 3.4× slower on Modal: 60 vs 205 steps/s.
  - Diffusion is about the same: 8.41 pure step on Modal vs 8.75 for the full local run.
  - A full Modal diffusion run, metrics and checkpoints included, reaches 6.16 steps/s.
- **GPU:** L4 trains diffusion at 65 steps/s and BC at 300–660. It is not used: the protocol
  trains on CPU (D-051).
- **Evaluation** is physics-bound; 4 parallel containers give 1.78× on Lift diffusion.
- **Modal retry behaviour:** Modal reschedules an interrupted input even with `retries=0`.
  The attempt cap bounds this (KI-031). Only a workspace budget is a hard spending cap.
- **Cost:** the full suite at the chosen budget costs **$0** locally. It takes about 10 h of
  training plus 22–51 h of evaluation (estimates). Session 6 (Lift) needs about 5 h of
  training core time ÷ 8 plus 5.4–8.2 h of evaluation. The two tuning waves put at least two
  ~3.9 h diffusion runs on the critical path.

## 6. Open decisions
- **Session 6 tuning:**
  - each family has 3 more trials on Lift at the main budget;
  - the search space must be written in the ledger before the family's first tuning run;
  - diffusion candidates: U-Net width, DDIM steps, learning rate / schedule, weight decay.
- **Main-budget runs** need `train.max_steps=100000` and `train.ckpt_every=5000`.
  - Pass `--set` (recorded in each manifest), or commit a config change as a recorded
    decision.
  - `make train` has no `--set` variable yet.
  - The configs still say 30k: that is the pilot budget.
- **R > 1** on validation (KI-028): only by an amendment before the freeze.
- **Optional:** a Modal workspace budget; a future probe for EGL vs OSMesa (KI-001).

## 7. Decisions
RESEARCH_PROTOCOL amendment 2026-11-07 (§6.3, §13), and D-045–D-052 (DECISIONS, Session 5).

## 8. Outstanding issues (none blocks Session 6)
- **KI-033 / KI-001:** EGL vs OSMesa on Modal is pending. Not needed for Level 1.
- **KI-028, KI-024, KI-021, KI-023, KI-011, KI-008:** carried over.
- **KI-029:** compare checkpoints by content (`checkpoint_differences`), not by file hash,
  for equivalence.
- **KI-020, KI-019, KI-018:** the rules still apply.
- **Makefile defaults** are `EVIDENCE_SESSION` 5 and `PREFIX` s5. Session 6 must pass
  `EVIDENCE_SESSION=6 PREFIX=s6`, or bump the defaults as a recorded change.

## 9. Budget
- **Spent:** $0.148 on Modal (Gate A, A-01 closed), billed list price. Month-to-date,
  including another app, $5.42 of the $30 credit.
- **Approved for the main suite:** local CPU only, **$0** (Gate C). Any Modal use needs a new
  approval row with a cap.

## 10. Exact starting instructions for Session 6
Paste into a fresh Claude Code session:

```text
Implement RoboForge 3D SESSION 6 ONLY from MASTER_SPEC.md (Level 1 Lift primary benchmark
and checkpoint selection). Begin by reading all of MASTER_SPEC.md, docs/STATUS.md,
docs/HANDOFF.md, docs/DECISIONS.md (through D-052), docs/KNOWN_ISSUES.md,
docs/EXPERIMENT_LEDGER.md, docs/RESEARCH_PROTOCOL.md (v1 + the 2026-11-06 diffusion
amendment + the 2026-11-07 main-budget amendment, all binding), docs/SIMULATOR_CONTRACT.md,
docs/DATA_DICTIONARY.md and docs/COMPUTE_PLAN.md, and inspect the actual repository/test
state. Do not blindly trust prior handoff claims: first run `make setup`, `make download`,
`make check-data`, `make build-data`, `make check-eval-states`, confirm
`git status --short data/` is empty, then run `make lint`, `make test`,
`make smoke-sim SMOKE_OUT=logs/session6/prereq`, `make train-smoke SMOKE_DIR=logs/tmp/s6_smoke`
and, on the canonical container,
`ROBOFORGE_EVIDENCE_SESSION=6 make in-sim CMD="python scripts/smoke_sim.py --task lift --out-dir logs/session6/prereq/container"`
and the same with `--task can`, and confirm they pass; confirm
`docker image inspect -f '{{.Id}}' roboforge-sim-cpu:session0` equals the image_id in
configs/eval/platform.yaml; check that the four pilot runs in docs/HANDOFF.md exist under
runs/ with matching checkpoint hashes (report if missing rather than assuming); run
git log --format=fuller -n 5 to confirm the verified commits and the identity
Zackary Holt <zackholt05@gmail.com>.

Session 6 goal (MASTER_SPEC Session 6): train the 4 main-suite families (BC-MLP, BC-RNN,
BC-MLP-chunk as the H2-c control, diffusion_unet) on Lift PH with the preapproved seeds and
budget, select checkpoints by the predeclared rule, and report validation-only performance.
Binding budget (RESEARCH_PROTOCOL §6.3, §13 2026-11-07): 100,000 steps, batch 100,
checkpoints every 5,000 (K = 20), identical across families; every checkpoint runs all 50
Lift validation starts on the canonical container; selection by §7 (highest validation
success, then lower mean first-success step, then earlier checkpoint). Tuning: at most 4
configurations per family on Lift, each at the main budget with seed 0; trial 1 is the
pilot configuration re-trained at the main budget (not an extra trial); write each family's
remaining search space (trials 2-4) in docs/EXPERIMENT_LEDGER.md BEFORE its first tuning
run; choose each family's configuration by the §7 ordering of its seed-0 runs' selected
checkpoints, then train that configuration with seeds 1 and 2. The configs still carry the
30k pilot budget: pass the main budget explicitly (scripts/train.py --set
train.max_steps=100000 --set train.ckpt_every=5000, recorded in each manifest) or change
the configs/Makefile as a recorded decision; never change the budget between families.
Compute (Gate C, D-051): local CPU only, $0 — 1 torch thread per run, at most 8 concurrent
training runs, at most 4 evaluation containers in parallel; no Modal resource of any kind
(approval A-01 is closed; any Modal use needs a new owner approval row in
docs/EXPERIMENT_LEDGER.md first). Track failed runs too. Generate initial plots
(training/validation curves, seed spread, action histograms, validation success and
latency), each clearly labelled validation-only, not final test. Inspect real successes
and failures; if every policy performs poorly, diagnose data alignment, controller
semantics and architecture before any freeze; a poor but correctly measured result is
reportable. Record the Lift final-test configuration only when the methodology is stable
(selected configurations and checkpoint hashes; the initial-state schedule only by its
suite manifest hash, never loading or naming the test states); do NOT run the held-out test
and do NOT create docs/FINAL_TEST_FREEZE.md — both belong to Session 7 (MASTER_SPEC).

Keep the Session 1–5 contracts (DECISIONS D-010, D-017..D-052): training data only from
SequenceDataset.from_committed; policy observations only from RoboForgeEnv; outputs decoded
with ActionCodec; learned policies implement roboforge.eval.policies.Policy and run with
run_episode + verify_record; stochastic policies seeded by the protocol's noise-seed rule
(policy_seed 0, R = 1); resets only from stored snapshots; first success or 400 steps;
model and checkpoint selection only on the eval_validation suite; never load, name,
summarize or reset to the held-out test states; every reported or selection rollout on the
canonical container via infra/containers/run_sim_cpu.sh with ROBOFORGE_EVIDENCE_SESSION=6
(never uv sync/uv run inside the image, KI-020). The Makefile's EVIDENCE_SESSION falls back
to 5 and PREFIX to s5: pass EVIDENCE_SESSION=6 and PREFIX=s6 (or bump the defaults as a
recorded change) so Session 6 never writes into Session 5 evidence. Native macOS rollouts
are development-only; never call mj_forward on live evaluation data (KI-019); never use
robosuite transform helpers for precision-critical values (KI-018); checkpoints and
anything a container rollout reads live under runs/<run_id>/; every run writes an immutable
manifest with git SHA + dirty flag (launch reported runs from a clean committed tree; do
not edit tracked files while they run), environment, platform, data/spec hashes, full
config, seeds, parameter count, checkpoint hashes and wall/CPU time; compare checkpoints by
content, not file hash (KI-029); report motion-dimension clipping separately from the
gripper (KI-022). Use the Make variable TRAIN_RUN (not RUN) for run directories. Do not
retune or rerun the Session 3–4 pilots themselves (their 30k runs stay development records;
trial 1 is re-trained at the main budget as a new run). No paid or remote compute and no
remote publication without explicit owner approval; do not change committed data
artifacts, eval suites, uv.lock or dependencies without a recorded decision.

Use the current verified simulator/data/checkpoint/manifest contracts. Run relevant tests
and record commands, real results and provenance. Never leak held-out test states into model
selection; never invent results or costs (label estimates as estimates). Use Zackary Holt
as the sole named RoboForge project author with the verified local Git identity; no AI
co-author trailers, while preserving external scientific and license credit (Diffusion
Policy, robomimic, diffusers). At the SESSION 6 STOP boundary, update docs/STATUS.md,
docs/HANDOFF.md, docs/DECISIONS.md, docs/KNOWN_ISSUES.md, docs/EXPERIMENT_LEDGER.md (all
runs, including failed ones, the tuning-budget accounting and the all-in cost record) and
docs/COMPUTE_PLAN.md if compute changed; report performance as validation-only; commit
verified code only, inspect commit metadata (git log --format=fuller -n 5), write the exact
prompt for SESSION 7, and STOP. Do NOT proceed automatically.
```
