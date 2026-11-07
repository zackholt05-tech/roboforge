# Compute and Modal Plan (Session 5: implemented, tested on Modal, Gate C decided)

**Status (2026-11-07):**
- **Built and tested:** the Modal integration (D-045–D-050).
- **Gate A:** done under owner approval A-01 ($2.00 cap). **$0.148 billed** over 6 launches.
  A-01 is closed.
- **Gate C:** the owner chose **local CPU only, $0**, for the main suite (D-051). No Modal
  resource is used in Sessions 6–7 without a new approval row in `docs/EXPERIMENT_LEDGER.md`.
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
| Closed-loop evaluation, canonical container | 50-start Lift evaluations take 43–101 s (BC) and 60–88 s (diffusion) (S3–S4). Parallel containers on Lift diffusion: 1.43× at 2, **1.78× at 4**. On Can (random policy): 3.38× at 4. `physics_sha256` unchanged (S5) | `reports/session5/rollout_profile/` |
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

## 5. Approval gates (MASTER_SPEC §6.3)
- **Gate A:** GPU smoke. **Done** under A-01, $0.148 of $2.00, closed. EGL vs OSMesa on Modal
  is still pending; a fixed probe is ready.
- **Gate B:** pilot. Done locally at $0 in Sessions 3–4: one seed per family on Lift, with
  wall time, evaluation throughput and storage measured.
- **Gate C:** main suite. **Approved as local CPU only, $0** (2026-11-07).
- **Gate D:** each extension, approved separately.

No paid job runs without an explicit owner approval recorded in `docs/EXPERIMENT_LEDGER.md`.
