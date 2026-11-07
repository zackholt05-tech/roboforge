# RoboForge

**Can a diffusion model teach a robot arm to grasp and place objects?**

RoboForge is a research project by **Zackary Holt** that compares deterministic behavior
cloning, recurrent behavior cloning and a conditional diffusion action policy on simulated
3D robotic manipulation: a 7-DoF Franka Panda arm in MuJoCo via robosuite, trained from
robomimic proficient-human demonstrations, and evaluated by closed-loop simulator rollouts
from paired, held-out initial states.

> **Status: Session 4 (conditional diffusion policy and denoising instrumentation) complete.
> Only short validation pilots of BC-MLP, BC-RNN, BC-MLP-chunk and the diffusion policy have
> been run; no held-out test exists yet.**
> Key findings: Results pending. Validation pilots are development numbers, not results. See
> [docs/STATUS.md](docs/STATUS.md), [docs/RESEARCH_PROTOCOL.md](docs/RESEARCH_PROTOCOL.md),
> [docs/DATA_DICTIONARY.md](docs/DATA_DICTIONARY.md) and
> [reports/feasibility/STACK_REPORT.md](reports/feasibility/STACK_REPORT.md).

## What has been verified (Session 0)

- Pinned stack: Python 3.11.15 · `mujoco==3.2.7` · robosuite v1.5.1 (`51cc017`) ·
  robomimic `d309eae` · torch 2.15.0 (`uv.lock`).
- Official robomimic PH datasets for **Lift** and **Can** downloaded and SHA-256-verified.
- Real 3D simulation: reset, step, success predicate, offscreen render (macOS CGL),
  exact MuJoCo state restore, and open-loop replay of all 400 recorded demonstrations.
- Session 1: one canonical, hash-verified data pipeline for Lift and Can. Typed
  task/action/observation specs, features recomputed from MuJoCo state by one function
  shared with evaluation (fixes robosuite's lagged Can relative pose, KI-003), upstream
  episode splits, train-only normalization, and an audited window sampler
  ([docs/DATA_DICTIONARY.md](docs/DATA_DICTIONARY.md), `reports/data_audit.json`).
- Session 2: the measurement apparatus.
  - `RoboForgeEnv` resets only from exact serialized snapshots and validates and clips
    actions. Success is scored on current kinematics, at first success, with a 400-step
    horizon.
  - Validation and final-test initial-state suites are hashed; the final test stays locked
    until `docs/FINAL_TEST_FREEZE.md` exists.
  - A frame-indexed rollout recorder with a re-simulation audit, plus random and no-op
    sanity policies.
  - Two-process determinism checks, and a MuJoCo → three.js clip export verified with the
    real three.js library.
  - Canonical evaluation runs in a pinned Linux container
    ([docs/SIMULATOR_CONTRACT.md](docs/SIMULATOR_CONTRACT.md), DECISIONS D-017, D-024–D-030).
- Session 3: the research protocol and the baselines.
  - [docs/RESEARCH_PROTOCOL.md](docs/RESEARCH_PROTOCOL.md) was committed before any reported
    run. It fixes the hypotheses, matched budgets, the checkpoint-selection rule and the
    bootstrap plan.
  - A shared trainer with strict configs, `weights_only` checkpoints, bitwise resume and
    immutable run manifests.
  - BC-MLP and BC-RNN (LSTM), capacity-matched at about 1.08 M parameters, plus an
    action-chunking control, BC-MLP-chunk.
  - Validation-only closed-loop evaluation on the pinned container, with checkpoint
    hashes carried into every record (DECISIONS D-032–D-038).
- Session 4: the conditional diffusion action policy.
  - A 1D temporal U-Net (adapted from robomimic, 1.09 M parameters) denoises 16-step action
    chunks (DDPM training, K = 100; DDIM-10 sampling), with EMA. It is bitwise equal to
    diffusers 0.11.1 and robomimic `d309eae` on fixed arrays.
  - Seeded stochastic evaluation: one noise seed per (policy seed, start, repeat).
  - `scripts/export_denoising.py` exports the real intermediate action tensors of recorded
    validation decisions, replayed bitwise and labeled as proposals (DECISIONS D-039–D-044).
- Known issues that shape later sessions are listed in [docs/KNOWN_ISSUES.md](docs/KNOWN_ISSUES.md).

## Quick start

```bash
make setup        # uv sync --frozen (Python 3.11 venv from uv.lock)
make test         # unit + simulator tests; dataset tests skip if data absent
make smoke-sim    # real Lift/Can reset/step/success/render, no dataset needed
make download     # official Lift + Can PH datasets (~170 MB), SHA-256 verified
make check-data   # rebuild the data pipeline in memory; must equal the committed manifests
make build-data   # write the processed Lift/Can features (gitignored); refuses on any mismatch
make verify-data  # HDF5 audit, obs/state alignment, replay of all demos, data determinism check
make audit-data   # reports/data_audit.json, figures, docs/DATA_DICTIONARY.md
make check-eval-states   # regenerate the eval suites in memory; must equal data/eval_states
make replay-demos        # 12 sampled demos/task through RoboForgeEnv: divergence, montage, video
make check-determinism   # two fresh processes, same seed -> bitwise-identical rollouts
make sanity-eval-canonical  # random + no-op policies on the validation suite, pinned container
make export-clip && make web-contract  # one real clip, verified against three.js (node)
make train-smoke         # CPU miniature: overfit checks + 200-step train + 2-start dev eval
make train MODEL=bc_mlp TASK=lift SEED=0        # one local CPU training run -> runs/<run_id>/
make eval TRAIN_RUN=runs/<run_id>               # validation rollouts of every checkpoint (container)
make select TRAIN_RUN=runs/<run_id>             # RESEARCH_PROTOCOL §7 checkpoint selection
```

Requirements: macOS arm64 or Linux x86_64/aarch64, [uv](https://docs.astral.sh/uv/).
Linux headless rendering uses `MUJOCO_GL=osmesa` or `egl` (see
[infra/containers/Dockerfile.sim-cpu](infra/containers/Dockerfile.sim-cpu)). OSMesa is
verified on linux/aarch64; x86_64 and EGL are not yet (KNOWN_ISSUES KI-001).

## Repository layout

```text
configs/simulator/   task contracts generated from dataset metadata (tested for equality)
configs/data/        canonical observation-feature selection, horizons, normalization per task
configs/models/      BC run configs (architecture, horizons, optimizer, step budget)
configs/eval/        evaluation suites (seeds, final-test lock, termination) + canonical platform
data/manifests/      dataset source registry (pinned revision, SHA-256) + tasks/ task manifests
data/splits/         episode-level train/valid split manifests (all IDs, hashed)
data/normalization/  train-only normalization statistics (hashed)
data/eval_states/    validation + final-test initial-state suites (final test locked)
src/roboforge/       data/ (fetch, specs, features, processed, split, normalization, sequence,
                     audit) · sim/ (env adapter, snapshots, eval suites, kinematics, render) ·
                     eval/ (policies, rollout recorder, learned policy, validation, selection)
                     · models/ (BC-MLP, BC-RNN, diffusion U-Net, schedulers, registry) ·
                     train/ (config, trainer, EMA, checkpoints, runs) · viz/ (three.js
                     coordinates, clip, denoising traces) · compute/ (Modal plans,
                     approval gate, cost ledger, retrieval, remote jobs, profiles, estimates)
scripts/             reproducible CLIs used by the Makefile
tests/               unit/ · simulator/ · integration/ · research_integrity/ · web_contract/
                     (pytest markers: sim, data, render, slow, web)
infra/containers/    headless Linux simulation image + run_sim_cpu.sh (canonical platform)
infra/modal_app.py   Modal entrypoint (dry run by default; paid launches need an approval row)
reports/feasibility/ Session 0 evidence (JSON reports, rendered frames)
reports/data_audit*  Session 1 dataset audit (JSON + figures)
reports/session2/    Session 2 evidence (replay, contract, determinism, sanity eval, clip)
reports/session3/    Session 3 evidence (validation pilots, selection, equivalence, diagnostics)
reports/session4/    Session 4 evidence (diffusion reference, pilot, equivalence, latency, denoising)
reports/session5/    Session 5 evidence (Modal probe, simulations, profiles, cost projection)
reports/compute/     cumulative paid-compute cost ledger (launch, result and billing events)
runs/                training and evaluation runs (gitignored; manifests + checkpoints)
docs/                status, handoff, decisions, known issues, research protocol
```

## Authorship and credit

Project author: **Zackary Holt**. RoboForge relies on robomimic, robosuite, MuJoCo and the
Diffusion Policy line of work; datasets, licenses and required citations are listed in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## License

RoboForge is licensed under the [Apache License 2.0](LICENSE); see also [NOTICE](NOTICE).
Third-party software, datasets and assets keep their own licenses
([THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)).
