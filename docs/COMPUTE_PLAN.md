# Compute and Modal Plan — PROPOSAL (implemented in Session 5)

**Nothing in this document has been executed on Modal.** No Modal app, volume or job was
created in Session 0, and resource spend is $0. Modal client 1.6.3 is installed locally,
outside this project's venv, with a profile named `zackholt05-tech`. Prices were fetched from
https://modal.com/pricing on **2026-11-05** and **must be rechecked before any launch**.

## 1. What actually needs compute (measured where possible)
| Workload | Bound by | Session 0 measurement / estimate |
|---|---|---|
| MuJoCo stepping (closed-loop eval) | CPU, single-threaded per env | **Measured:** Lift ≈ 195 control steps/s/core, Can ≈ 163 (macOS arm64). Linux aarch64 Docker VM on the same laptop: Lift 161, Can 71 (replay median). Not representative of Modal x86_64. |
| Policy inference in rollouts | CPU or GPU | **Measured for BC (Session 3, 1 thread, median per decision):** canonical container BC-MLP 0.91 ms, BC-MLP-chunk 1.26 ms, BC-RNN 8.6 ms; macOS 0.07 / 0.075 / 0.66 ms (KI-023). A 50-start Lift validation evaluation took 43–101 s of wall time in the container. Those runs were not isolated: up to 3 containers and a training job ran at the same time. Diffusion with K denoising steps per replan × (400/T_a) replans is **unmeasured**; benchmark it in the container in Session 4 (DDPM-100 vs DDIM-10). |
| Training state-based policies | GPU optional | **Measured for BC (Session 3, local M1 Pro CPU, 1 thread, 3 concurrent runs):** 30,000 steps × batch 100 take 146 s (BC-MLP, 205 steps/s), 983 s (BC-RNN, 30.5 steps/s) and 155 s (BC-MLP-chunk), including offline-loss evaluation and checkpoints. No GPU is needed for BC. Diffusion is unmeasured (Session 4). |
| Rendering (RGB extraction, videos) | GPU (EGL) or CPU (OSMesa) | **Measured:** macOS CGL 7–25 ms per 256² frame; Linux OSMesa (CPU) 88–196 ms. At ~0.1–0.2 s/frame, 2 cameras over all demos is roughly 0.5–1 core-hour (Lift) and 1–2.5 core-hours (Can), an estimate. Not needed for state-based Level 1 eval. |
| Storage | — | Raw datasets: 170 MB (Lift + Can). RGB 84×84 ×2 cameras: ~0.4 GB Lift, ~1 GB Can uncompressed (estimate). |

**Eval sim budget, upper bound** (spec target: 3 seeds × 3 algorithms × 50 states = 450
rollouts per task, each up to 400 steps):
- Lift: 180,000 steps ≈ 15 core-minutes of simulation.
- Can: ≈ 18 core-minutes.
- Validation-based checkpoint selection multiplies this by the number of evaluated
  checkpoints.

The simulation itself is cheap; diffusion inference is the unknown.

## 2. Recommended split (DEFAULT, D-015)
- **GPU (Modal):** PyTorch training only, and only if local benchmarks show CPU/MPS is too
  slow for the approved seed budget. A GPU does **not** speed up MuJoCo physics.
- **CPU:** all closed-loop evaluation and replay. Start locally (10 cores). Move to Modal CPU
  containers (`cpu=N`, `MUJOCO_GL=osmesa` or no rendering) only if wall time requires it.
- **Initial GPU recommendation:**
  - **L4** ($0.000222/s ≈ $0.80/h) or **T4** ($0.000164/s ≈ $0.59/h). Either is ample for
    small 1D U-Nets/MLPs on low-dim data.
  - **A10** ($0.000306/s ≈ $1.10/h) only if memory/throughput profiling shows a need.
  - Not A100/H100 for Level 1.
- CPU: $0.0000131 per physical core per second (≈ $0.047/core-h). Memory ≈ $0.008/GiB-h.
- Volumes: $0.09/GiB-month beyond 1 TiB free.
- Starter plan includes $30/month credits. Credits are **not** a hard spending cap.

## 3. Proposed Modal architecture (to build and test in Session 5)
- `infra/modal_app.py` (single Modal App, e.g. `roboforge`):
  - **Image:**
    - Base `debian_slim(python_version="3.11")` with the apt packages from
      `infra/containers/Dockerfile.sim-cpu` (git, cmake, build-essential, libegl-dev,
      libegl1, libosmesa6, libgl1, libglib2.0-0).
    - Dependencies installed from `uv.lock` (exact pins). The method — e.g. `uv export` →
      `pip_install`, or a uv-sync image step if the pinned Modal version supports it — is to
      be verified.
    - Record the image ID/digest in run manifests.
  - **Volumes:**
    - `roboforge-data`: read-mostly raw datasets, uploaded once and SHA-256-verified inside
      the container before use.
    - `roboforge-runs`: checkpoints, metrics, manifests. Commit periodically and on graceful
      exit; download artifacts and verify their hashes locally.
  - **Functions:**
    - `train(config, run_id, budget_approved: bool)`: `gpu="L4"`, explicit `timeout`,
      `max_containers=1`, `retries=0`. Refuses to start unless `budget_approved` is true and
      `run_id` does not already exist on the volume (idempotent launch).
    - `evaluate(...)`: CPU only, no GPU.
  - **Local entrypoint:** `--config`, `--run-id`, `--dry-run` (default) and an explicit
    `--budget-approved` flag. No resources are allocated at import time or during tests/CI.
- **Secrets:** none required for public data. Use Modal Secrets only if ever needed; never in
  the repo.
- **Cost controls:**
  - Set a Modal **workspace budget**, with owner consent.
  - Per-job timeouts; concurrency 1; no automatic retries.
  - `python -m roboforge.compute.estimate` dry-run estimates. These are estimates, not caps.
  - Log actual billed usage after each job.

## 4. Open questions for Session 5 (must be verified, not assumed)
1. Does the lockfile install on linux/x86_64 in a Modal image? (Only linux/aarch64 was tested.)
2. Does MuJoCo EGL rendering work in Modal GPU containers? Modal documents that the NVIDIA
   driver (610.57.04, CUDA 13.3 driver API) is installed. EGL/graphics capability is
   **not documented**. Fallback: OSMesa on CPU (verified on linux/aarch64, KI-001).
3. Does torch 2.15 from PyPI (CUDA build) run on the Modal driver? Expected yes; verify with
   a ≤15-minute approved smoke job (Gate A).
4. CPU-only torch wheels for eval images, to avoid the 8.7 GB image (KI-010).

## 5. Approval gates (from MASTER_SPEC §6.3)
- **Gate A:** ≤15-min GPU smoke, approved separately.
- **Gate B:** a capped pilot.
- **Gate C:** the full Level 1 suite, against an owner-approved $ budget.
- **Gate D:** each extension.

No paid job runs without an explicit owner approval recorded in `docs/EXPERIMENT_LEDGER.md`.
