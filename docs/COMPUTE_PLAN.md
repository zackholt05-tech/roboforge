# Compute and Modal Plan (Session 5: implemented, tested on Modal, Gate C decided; Sessions 6–7: Level 1 ran locally at $0; Session 8: Gate D proposal, §6)

**Status (2026-11-07):**
- **Built and tested:** the Modal integration (D-045–D-050).
- **Gate A:** done under owner approval A-01 ($2.00 cap). **$0.148 billed** over 6 launches.
  A-01 is closed.
- **Gate C:** the owner chose **local CPU only, $0**, for the main suite (D-051). No Modal
  resource is used in Sessions 6–7 without a new approval row in `docs/EXPERIMENT_LEDGER.md`.
- **Session 6:** the Lift main suite ran under Gate C: 33.25 CPU core-hours of training,
  26.92 container-hours of evaluation, 9.80 h elapsed, **$0**. No Modal call (§4, "Session 6
  (Lift) actual vs projected").
- **Prices:** fetched from https://modal.com/pricing on **2026-11-06**
  (`configs/compute/modal.yaml`). Recheck before any future launch.
- **Modal client:** 1.6.3, outside the project venv (`/opt/anaconda3`, profile
  `zackholt05-tech`).

## 1. Measured workloads
| Workload | Measurement (Session) | Evidence |
|---|---|---|
| Local CPU training, 1 thread (M1 Pro) | BC-MLP 205 steps/s, BC-RNN 30.5, BC-MLP-chunk 193, diffusion 8.75. These are full 30k-step pilot runs including offline metrics (S3–S4). Concurrent diffusion: 7.5 / 29.2 / 41.8 / 48.5 aggregate steps/s at 1 / 4 / 6 / 8 processes; slowdown at 8 is 1.24 (S5) | pilot manifests, `reports/session5/cost/train_concurrency.json` |
| Modal CPU training, 1 thread, x86_64 | Probe container, pure step: BC-MLP 60.4, BC-RNN 12.3, BC-MLP-chunk 87.0, diffusion 8.41 steps/s. A full diffusion run on `train_cpu` (2 cores), metrics and checkpoints included: **6.16 steps/s**, i.e. 0.733 × pure step (S5) | `reports/session5/modal/probe_result.json`, `runs/s5modal2_*/manifest.json` |
| Modal L4 training | BC-MLP 658, BC-RNN 307, BC-MLP-chunk 664, diffusion **65.2** steps/s, pure step (S5) | `probe_result.json` |
| Closed-loop evaluation, canonical container | 50-start Lift evaluations take 43–101 s (BC) and 60–88 s (diffusion) (S3–S4). Parallel containers on Lift diffusion: 1.43× at 2, **1.78× at 4**. On Can (random policy): 3.38× at 4. `physics_sha256` unchanged (S5). Under the Session 6 suite load (4 containers plus up to 8 training runs and a foreign process): 18–26 control steps/s per container, 1.6–3.1× below S3–S4 (KI-035) | `reports/session5/rollout_profile/`, `reports/session6/validation/` |
| Rendering, 256² | Canonical container: OSMesa and EGL 39–47 ms per frame; macOS CGL 4.6–5.0 ms (S5). Modal GPU EGL vs OSMesa: **PENDING** (KI-001, KI-033) | `logs/session5/modal/render_probe_container_*.log` |
| Storage | 13–18 MB per checkpoint. Full suite at 100k / 5k: ≈ 27.6 GiB of checkpoints and records (estimate) | `suite_projection.json` |

Physics, not rendering or inference, dominates a state-based evaluation. A GPU does not speed
up MuJoCo.

## 2. What ran on Modal (Gate A, approval A-01)
| Launch | App | Outcome | Billed |
|---|---|---|---|
| L4 probe 1 | `ap-LyuJSmb9OLUAKPQ0oGZMQt` | Volume looked up before it was created; no job ran | $0.00001 |
| L4 probe 2 | `ap-YAO3zMqPWKlHWy5cyQ0kT2` | x86_64 install proven; CUDA BC-RNN device bug (KI-030); results lost | $0.0657 |
| CPU smoke 1 | `ap-ObjyqZEiECqNHIh4j3xt7n` | Restart loop, 8 attempts (KI-031); stopped by hand; run abandoned | $0.0252 |
| CPU smoke 2 | `ap-z4RXwqVQfQuJ9dx6EQVd7l` | Real-SIGINT drill at step 376 → Modal restart → clean stop at step 1500 | $0.0101 |
| CPU resume | `ap-sRC21D7MuIIeHq3hZsIeaa` | Completed 2,000 steps; downloaded, hash-verified, validated locally (E5-08) | $0.0032 |
| L4 probe 3 | `ap-WKAnfgdtYEriXZDg6FtoCC` | CUDA and all 8 training profiles measured; render check never rendered (KI-033) | $0.0441 |
| **Total** | | | **$0.148** |

Billed amounts are list cost before credits (`modal billing report`), reconciled per launch in
`reports/compute/cost_ledger.jsonl`. Month-to-date, including another app: $5.42 of the $30
credit.

**GPU choice: L4** was the tested GPU, at $0.000222/s. It is **not used** for the main suite:
- the protocol trains on CPU in float32 (RESEARCH_PROTOCOL §6.1);
- CUDA kernels are not bitwise deterministic;
- local CPU is free and finishes the training in about 10 h.

GPU training would need a dated protocol amendment and a new approval.

## 3. Architecture as built
See DECISIONS D-045–D-050.
- **Entrypoint:** `infra/modal_app.py`, the only file that imports `modal`. It is a dry run by
  default; `--budget-approved A-xx` launches. Subcommands: `train`, `probe`, `fetch`,
  `cancel`.
- **Image:** `debian_slim` 3.11 plus the apt packages of `Dockerfile.sim-cpu`, then
  `uv_sync(frozen)` from `uv.lock`. Only the code directories are mounted. No evaluation
  suite or test state goes to Modal.
- **Volumes:** `roboforge-data` (processed HDF5, hash-checked in every attempt) and
  `roboforge-runs` (mounted at `/app/runs`).
- **Policy:**
  - `max_containers=1`, `retries=0`;
  - preemptible;
  - an explicit timeout, with a job deadline 180 s earlier;
  - at most 4 attempts per launch;
  - one GPU at most.
- **Resumable runs** (D-046): signal or deadline stops write a resume checkpoint, and a
  restart of the same launch resumes from it. Retrieval (`fetch`) verifies every hash before
  and after install, and never overwrites a run.
- **Cost controls:** the approval gate checks committed spend plus an upper bound covering
  4 attempts against the cap, and `record-billing` reconciles each launch. The gate is
  client-side. Only a **Modal workspace budget** is a hard cap: set one before any future
  paid session.

## 4. Main-suite cost proposal (estimates from measured inputs; `make cost-estimate`)
Main suite: 4 families × 3 seeds × Lift and Can. Each family/task gets up to 4 tuning
configurations at seed 0, then seeds 1–2 of the chosen one: 6 runs per family/task, 48 runs
in all. The test evaluations, 25 % failed-run reserve, storage and rendering are included.

| Variant | Runs | Validation evaluations | Local training (8 proc.) | Local evaluation (4 containers) | Modal CPU training | Modal L4 training |
|---|---|---|---|---|---|---|
| **100k steps, checkpoints every 5k (CHOSEN)** | 48 | 960 | 10.1 h | 22.4–50.7 h | $12.35–16.72, 97–132 h sequential | $12.36–16.02, 11–14 h |
| 100k / 10k (previous DEFAULT) | 48 | 480 | 10.1 h | 11.5–26.0 h | $12.35–16.72 | $12.36–16.02 |
| 50k / 5k | 48 | 480 | 5.1 h | 11.5–26.0 h | $6.34–8.53 | $7.34–9.16 |
| 100k / 10k, no further tuning | 24 | 240 | 5.1 h | 6.0–13.6 h | $6.22–8.40 | $6.22–8.05 |
| 50k / 10k, no further tuning | 24 | 120 | 2.5 h | 3.3–7.4 h | $3.21–4.30 | $3.71–4.62 |

- **Session 6 (Lift only), chosen variant:** about 5.1 h of training core time ÷ 8, and
  5.4–8.2 h of evaluation.
  - The two tuning waves (seed-0 configurations, then seeds 1–2 of the chosen one) put at
    least two diffusion runs on the critical path. At 100k steps that is about 3.9 h each
    under load.
  - Evaluation of Can (Session 7) dominates the high end of the range.
- **Local training** time is core-hours ÷ 8 concurrent processes × the measured slowdown.
  The evaluation range spans Can episode lengths of 160–400 steps (no Can policy exists yet)
  and the measured spread of Lift evaluation times.
- **Modal training** assumes the local evaluation either way. The range spans the measured
  pure-step rate and 0.733 × that rate.
- **Rendering:** 40 clips × 400 frames × 2 cameras ≈ 0.4 core-hours with OSMesa (Session 10).
- **Approved limit:** **$0 paid compute** for the main suite (Gate C, D-051). Any change needs
  a new owner approval row with a cap, and, for GPU training, a protocol amendment.

### Session 6 (Lift) actual vs projected
Lift is half of the chosen variant above. The projection includes the 25 % failed-run
reserve; Session 6 had no failed run.

| Quantity | Projected (Session 5, Lift share) | Actual (Session 6) | Source |
|---|---|---|---|
| Training runs / validation evaluations | 24 / 480 | 24 / 480 | `reports/session6/suite/level1_core_lift.json` |
| Training core time | 32.7 core-h (26.2 without the reserve) | **33.25 CPU core-h** (37.54 h summed run wall time) | `logs/session6/analysis/run_table.md` |
| Training on the critical path | ≈ 3.9 h per diffusion run, two waves | 5.10–5.12 h per diffusion trial, then 3.39 h per replicate (8.5 h in all) | run manifests, KI-035 |
| Evaluation | 5.4–8.2 h elapsed at 4 containers (≈ 9.6–14.6 single-container hours, derived as × 1.78) | **26.92 container-hours** summed over up to 4 concurrent, contended containers (mean 202 s per evaluation) | suite events |
| Elapsed, end to end | 10.5–13.3 h (training, then evaluation) | **9.80 h** (06:59–16:47 UTC; the orchestrator overlaps evaluation with training) | suite events |
| Storage | full suite ≈ 20–28 GiB | `runs/s6_*` 8.7 GB; `reports/session6/` 34 MB | `du` |
| Cost | $0 | **$0** | — |

- Contention (the suite's own 10 concurrent jobs plus a foreign process) slowed training by
  about 1.3× on the critical path and evaluation by 1.6–3.1× per container (KI-035). The
  overlap of training and evaluation more than made up for it.
- Container-hours are not comparable one to one with the projection's single-container
  hours. The Session 5 parallel profile predicts about 2.25× per container at 4 in parallel;
  26.92 / 2.25 ≈ 12.0 single-container hours, inside the projected range (derived, not
  measured).
- **For Session 7 (Can):** the projection's Can share is about 17–42 h of evaluation elapsed
  (the full-suite 22.4–50.7 h minus Lift's 5.4–8.2 h; estimate), plus training like Lift's.
  Use the Session 5 profiles; if the host is shared again, scale by the Session 6 factors.

### Session 7 (Can, robustness, ablations, held-out test) actual vs projected
All local, $0, Gate C (D-051); no Modal resource. Measured, from manifests and summaries.

| Quantity | Projected | Actual (Session 7) | Source |
|---|---|---|---|
| Can training runs / validation evaluations | 24 / 480 | 24 / 480 | `reports/session7/suite/level1_core_can.json` |
| Can training core time | like Lift's (≈ 33 core-h) | **34.12 CPU core-h** (37.37 h summed run wall time) | run manifests |
| Can evaluation | 17–42 h elapsed at 4 containers (estimate, §4 above) | **46.36 container-hours**; 16.46 h elapsed for the whole suite, 3.1 h of it the crash gap (KI-036) | suite events, summaries |
| Robustness + A1 + A2 | not in the Session 5 projection (§14 came later) | 72 evaluations, 7.86 container-hours, 2.6 h elapsed | `logs/session7/robustness/events.jsonl` |
| A3 (3 diffusion runs, T_o = 1) | not projected | 12.07 CPU-h training; 63 evaluations, 6.65 container-hours | `reports/session7/suite/ablation_a3_can.json` |
| Rehearsal + held-out test | not projected | 52 evaluations, 3.37 container-hours (61 min + 58 min elapsed) | `logs/session7/rehearsal/`, `logs/session7/heldout/` |
| Session total | — | **46.19 CPU core-h training, 64.25 container-h evaluation, ≈ 24.4 h elapsed** | LEDGER Session 7 |
| Cost | $0 | **$0** | — |

- Can evaluations take longer than Lift's: the mean first-success step is about 110 against
  about 45, and failures run to the 400-step horizon. The Can validation share came out above
  the projected range. That projection was an estimate made from Lift-only profiles.
- **Level 1 total (Sessions 1–7):** $0.148 billed (Session 5 Modal Gate A). Everything else
  was local.
- **For Session 8 (RGB):** any GPU or Modal use needs a new owner approval row with a cap
  first. Image-conditioned training on this CPU host is not projected yet; Session 8 must
  measure a pilot before it asks for a budget.

## 5. Approval gates (MASTER_SPEC §6.3)
- **Gate A:** GPU smoke. **Done** under A-01, $0.148 of $2.00, closed. EGL vs OSMesa on Modal
  is still pending; a fixed probe is ready.
- **Gate B:** pilot. Done locally at $0 in Sessions 3–4: one seed per family on Lift, with
  wall time, evaluation throughput and storage measured.
- **Gate C:** main suite. **Approved as local CPU only, $0** (2026-11-07).
- **Gate D:** each extension, approved separately. Session 8 RGB proposal: §6 (awaiting the
  owner's decision).

No paid job runs without an explicit owner approval recorded in `docs/EXPERIMENT_LEDGER.md`.

## 6. Session 8 Gate D proposal: RGB visual policies (from the measured pilot, 2026-11-09)
**Status: DECIDED 2026-11-08 21:21 EST: option D**, the ≤ $1 L4 probe (approval A-02,
DECISIONS D-076). The probe ran (§7.1). The main plan is proposed in §7. No RGB dataset,
training run or RGB evaluation exists. Every duration below is an **estimate** computed from the measurements in
EXPERIMENT_LEDGER E8-03–E8-05; none is a promise.

### 6.1 Measured inputs (this host, an M1 Pro with 8 performance + 2 efficiency cores)
| Input | Measured | Source |
|---|---|---|
| Render, canonical container (OSMesa), ms per frame | 37–71 (84² and 128² alike: per-frame overhead dominates) | E8-03 |
| Control step in the container: physics only / + both cameras | Lift 5.8 / 103–123 ms; Can 11.6 / 127–149 ms | E8-03 |
| Storage, both cameras, HDF5 gzip-4, whole task | Lift 0.13 GiB (84²), 0.30 (128²); Can 0.51, 1.07 | E8-03 |
| Training step, batch 64, T_o = 2, two cameras, 1 thread alone | compact 84² 0.91–0.98 s; compact 128² 2.06–2.10 s; ResNet-18 84² 4.07–4.14 s; ResNet-18 128² 6.96–7.64 s | E8-05 |
| Same, compact 84², N concurrent 1-thread runs (s/step each; aggregate steps/s) | 3: 1.07 (2.77); 4: 1.37 (2.94); **6: 1.90 (3.16)**; 8: 3.69 (2.18). More threads per run do not help (8 threads: 1.08–1.21 s) | E8-05 |
| Inference per decision, container, 1 thread (compact 84²) | BC-chunk head 11.5 ms; diffusion DDIM-10 92.9 ms (ResNet-18: 41.8 / 109.9 ms) | E8-05 |
| Free disk | 20 GiB (2026-11-09) | `df -h` |

### 6.2 What the budget buys (the design the RGB freeze would fix)
- **Comparison:** visual BC-MLP-chunk vs visual diffusion (Lift; Can optional). Both use the
  same per-camera encoder (architecture and per-seed initialization), the same random crop
  (76 of 84 px), the same 9-D proprioception (`robot0_eef_pos`, `robot0_eef_quat`,
  `robot0_gripper_qpos`, train-only normalization) and no object state; the heads, horizons
  (T_o 2, T_p 16, T_a 8), optimizer and sampler are the Level 1 ones. The two arms differ only
  in the head: deterministic regression vs conditional diffusion.
- **Encoder:** the compact residual trunk + SpatialSoftmax, trained from scratch: 0.63 M
  parameters for two cameras, no download and no licence question. ResNet-18 (22.4 M) is
  4.4× slower per step alone and not affordable on this CPU (§6.4).
- **Resolution:** 84² (robomimic's image default). 128² costs 2.2× per training step and the
  same to render.
- **Budget per run:** batch 64, S steps, a checkpoint every 5,000 steps, every checkpoint on the
  50 validation starts (RESEARCH_PROTOCOL §7 rule). Seeds {0, 1, 2}. One configuration per
  family, fixed in the freeze (no tuning trials).
- **Evaluation:** canonical container, ≤ 4 in parallel, rendering only the frames each decision
  reads (2 per camera per 8 steps); the RGB held-out suite is decided in the freeze.
- **Dataset:** both cameras at 84², rendered from the raw states in the container (D-074).

### 6.3 Options (local CPU is $0; durations are estimates)
Training: 6 runs (2 families × 3 seeds) as 6 concurrent 1-thread processes at the measured
1.90 s/step, plus 10 % for data loading, checkpoints and offline metrics. Evaluation: per 50-start
evaluation, between "every episode succeeds near the demonstrations' length" and "every
episode times out at 400 steps", scaled by the measured 2.25× slowdown per container at 4 in
parallel.

| Option | Scope | Training | Validation + test evaluations | Elapsed (estimate) | Disk | Cost |
|---|---|---|---|---|---|---|
| **A (recommended)** | Lift, compact 84², S = 30k (K = 6) | 17.4 h (104 CPU-h) | 36 + 6 = 42; 0.6–6.2 h | **≈ 18–24 h** | ≈ 2 GiB | $0 |
| B | A + Can (same plan) | + 17.4 h | + 42; + 1.8–7.5 h | ≈ 38–51 h | ≈ 4 GiB | $0 |
| C | Lift, compact 84², S = 20k (K = 4) | 11.7 h | 24 + 6 = 30; 0.4–4.4 h | ≈ 12.5–16.5 h | ≈ 1.5 GiB | $0 |
| D | GPU path for ResNet-18 (Modal L4) | not estimable yet | local container | — | — | probe ≤ $1, then a capped plan |

Dataset build (container, one process): Lift ≈ 19 min, Can ≈ 53 min. A 50-start Lift
evaluation takes about 1.5–2 min if every episode succeeds within 50 steps and 12–16 min if every
episode times out; Can 4.7–5.7 min and 16–19 min.

### 6.4 Why not ResNet-18 on CPU, and what the GPU path would need
- ResNet-18 at 84² takes 4.1 s per step alone; 30k steps would be ≥ 34 h per run with nothing
  else running, and concurrency makes each run slower (compact: 1.9× at 6 processes).
- **Option D needs, in order:** (1) a new approval row for a ≤ $1.00 L4 probe (one function,
  ≤ 15 min, the Gate A pattern) to *measure* visual training steps/s on the GPU; (2) a dated
  RESEARCH_PROTOCOL amendment, because the protocol trains on CPU in float32 and CUDA training is
  not bitwise deterministic; (3) then a capped main plan. List price of the L4 function
  (1 L4, 4 cores, 16 GiB) is $1.12/h, $1.95/h at the highest region multiplier
  (`configs/compute/modal.yaml`, fetched 2026-11-06: recheck before any launch). For scale only:
  6 runs capped at 2 h each would have an upper bound of ≈ $23, more than the remaining monthly
  credit the owner asked to stay within. Not recommended now.

### 6.5 What happens after the decision
1. The decision is recorded as an approval row (A-02) in `docs/EXPERIMENT_LEDGER.md`, with cap
   $0.00 for a local-only choice, so no Modal launch can use it.
2. `docs/RGB_EXPERIMENT_FREEZE.md` is written and committed before the first RGB training run:
   observation spec, encoder, augmentations, budget, selection rule, held-out suite and the
   analysis. Before any RGB held-out rollout, the held-out lock is re-gated on that file with a
   test, as a recorded decision; `make freeze-check` then needs to compare the Level 1 frozen
   hashes with the freeze commit `bf0a0fa` rather than the working tree (frozen modules such as
   `eval_states.py` must change for the re-gate), while `heldout.evaluate_entry` keeps refusing
   any Level 1 rerun.
3. Then the RGB dataset build, training and evaluation under that freeze.

## 7. Session 8 Gate D2 proposal: main RGB training from the L4 probe (2026-11-09)
**Status: PROPOSED, awaiting the owner's decision.** A-02 covers only the probe, which is
done. No main RGB training, local or remote, starts until the owner chooses here and the
choice is recorded as an approval row. Every duration and cost below is an **estimate**
built from the probe's measurements, not a promise.

### 7.1 What the probe measured (E8-06; `reports/session8/modal/README.md`)
| Trunk, 84², batch 64 | Head | L4 default | L4 deterministic | CPU, this host, 1 thread, same step (E8-07) |
|---|---|---|---|---|
| ResNet-18 | BC-MLP-chunk | 54.5 ms | 58.4 ms | 4.02 s |
| ResNet-18 | diffusion | 71.6 ms | 79.8 ms | 4.11 s |
| compact | BC-MLP-chunk | 28.1 ms | not measured | 0.90 s |
| compact | diffusion | 45.8 ms | not measured | 0.98 s |

- **Determinism.** With default kernels, two same-seed runs on the same L4 diverge within
  20 steps. With deterministic kernels they are bitwise equal: parameters, EMA and every
  loss. Every op has a deterministic kernel, and the cost is +7 % (BC) and +12 %
  (diffusion). Not verified: equality across different L4 hosts or drivers.
- **Memory** peaks at 1.2 GiB (84², batch 64), so the L4 has ample room.
- **Probe cost:** $0.029 measured (container time × list price). $0.032 is on Modal's report
  so far.

### 7.2 Proposed design (the RGB freeze would fix it)
§6.2 still applies, with two changes:
- **Encoder:** ResNet-18, GroupNorm, from scratch, at 84² with a random crop of 76. This
  is the robomimic and Diffusion Policy image encoder, and it is affordable on the GPU.
- **Training:** on one Modal L4 with deterministic kernels. Each run is its own sequential
  launch, resumable on preemption like `train_cpu`.

Unchanged: batch 64, S = 30,000 steps, a checkpoint every 5,000 (K = 6), seeds {0, 1, 2},
and one configuration per family. The 6 runs cover 2 families × 3 seeds, on Lift.

Evaluation stays local on the canonical container ($0): 36 validation and 6 test
evaluations of 50 starts each. That takes about 0.7–7 h, from the §6.3 range scaled for
ResNet-18's slower CPU inference (41.8 / 109.9 ms per decision).

Engineering before any launch, all $0, with tests and CPU smokes:
- the RGB dataset build in the container, 84², both cameras (Lift ≈ 19 min);
- an RGB trainer that reuses the Level 1 EMA, loss and checkpoint conventions;
- a `train_gpu` Modal function;
- the RGB closed-loop observation path, with warm-up and only the frames each decision
  reads;
- `docs/RGB_EXPERIMENT_FREEZE.md` and the lock re-gate (§6.5).

**Proposed RESEARCH_PROTOCOL amendment** (committed only if the owner approves a GPU
option):
- RGB-experiment training runs on one NVIDIA L4 in float32 under
  `torch.use_deterministic_algorithms(True)`, with cuDNN deterministic, cuDNN benchmark off,
  and `CUBLAS_WORKSPACE_CONFIG=:4096:8`.
- TF32 stays at PyTorch's defaults: on for convolutions, off for matmuls. Both are recorded
  per run.
- Rollouts, selection and test stay on the canonical CPU container with one torch thread.
- The visual BC vs visual diffusion comparison is made only between runs trained this way.
  It is a separate study from Level 1 and never modifies Level 1.

### 7.3 Options
L4 function at list price: $1.12/h, and $1.95/h at the 1.75× region maximum. Each launch also
carries a build allowance of $0.08.

Expected training time, deterministic L4 at 84² and batch 64, with 15 % for checkpoints, data
and start-up:

| Run | Steps | Expected time | Timeout | Upper bound per launch |
|---|---|---|---|---|
| BC-MLP-chunk | 30k | 34 min | 45 min | $3.43 |
| diffusion | 30k | 46 min | 60 min | $4.54 |

Launches run one after another, so a cap only has to cover the spend already committed plus
one launch's upper bound.

| Option | Scope | GPU time | Expected cost (list / region max) | Proposed cap | Elapsed (estimate) |
|---|---|---|---|---|---|
| **G1 (recommended)** | Lift, ResNet-18 84², 30k, K = 6, 3 seeds × 2 families | 4.0 h | **$4.91 / $8.59** | $10 | engineering, then ≈ 4 h training overlapped with 0.7–7 h evaluation |
| G2 | G1 + Can (same plan) | 7.9 h | $9.82 / $17.18 | $20 | + 4 h training, + Can dataset (≈ 53 min) and its evaluations |
| G3 | Lift, ResNet-18 84², the Level 1 budget (100k steps, batch 100, K = 20) | 17.3 h | $19.81 / $34.67 | $36 | not recommended: may exceed the remaining monthly credit |
| A | local CPU only, compact 84², 30k (§6.3; per-step CPU times re-measured with the same step in E8-07) | — | $0 | $0 | ≈ 18–24 h (§6.3: unchanged, because the same-step CPU times match E8-05 within 2 %) |

**Monthly credit.** Month-to-date Modal usage is $5.45, from `modal billing report`, through
the probe's day. About $24.5 of the $30 credit remains. G1 stays within it even at the
region maximum, and G2 does at list price.

**Why G1.**
- ResNet-18 is the standard visual encoder for both arms. A deterministic L4 step is 52–69×
  faster than one 1-thread step on this CPU: 58–80 ms against 4.0–4.1 s.
- Deterministic CUDA keeps the protocol's bitwise-repeatability property on one GPU type.
- The cost fits the owner's "do not exceed the $30 monthly quota" instruction with a wide
  margin.
- Option A remains the $0 fallback. It uses a much smaller, non-standard encoder and takes
  about a day of this machine.
