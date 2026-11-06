# RoboForge

**Can a diffusion model teach a robot arm to grasp and place objects?**

RoboForge is a research project by **Zackary Holt** that compares deterministic behavior
cloning, recurrent behavior cloning and a conditional diffusion action policy on simulated
3D robotic manipulation: a 7-DoF Franka Panda arm in MuJoCo via robosuite, trained from
robomimic proficient-human demonstrations, and evaluated by closed-loop simulator rollouts
from paired, held-out initial states.

> **Status: Session 1 (canonical data pipeline) complete — no models trained, no results yet.**
> Key findings: Results pending. See [docs/STATUS.md](docs/STATUS.md),
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
```

Requirements: macOS arm64 or Linux x86_64/aarch64, [uv](https://docs.astral.sh/uv/).
Linux headless rendering uses `MUJOCO_GL=osmesa` or `egl` (see
[infra/containers/Dockerfile.sim-cpu](infra/containers/Dockerfile.sim-cpu)). OSMesa is
verified on linux/aarch64; x86_64 and EGL are not yet (KNOWN_ISSUES KI-001).

## Repository layout

```text
configs/simulator/   task contracts generated from dataset metadata (tested for equality)
configs/data/        canonical observation-feature selection, horizons, normalization per task
data/manifests/      dataset source registry (pinned revision, SHA-256) + tasks/ task manifests
data/splits/         episode-level train/valid split manifests (all IDs, hashed)
data/normalization/  train-only normalization statistics (hashed)
src/roboforge/       data/ (fetch, specs, features, processed, split, normalization, sequence,
                     audit) · sim/ (env factory, replay, render) ·
                     models/ train/ eval/ viz/ compute/ (boundaries; implemented in later sessions)
scripts/             reproducible CLIs used by the Makefile
tests/               unit/ · simulator/ · integration/ (pytest markers: sim, data)
infra/containers/    headless Linux simulation image
reports/feasibility/ Session 0 evidence (JSON reports, rendered frames)
reports/data_audit*  Session 1 dataset audit (JSON + figures)
docs/                status, handoff, decisions, known issues, protocol drafts
```

## Authorship and credit

Project author: **Zackary Holt**. RoboForge relies on robomimic, robosuite, MuJoCo and the
Diffusion Policy line of work; datasets, licenses and required citations are listed in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## License

RoboForge is licensed under the [Apache License 2.0](LICENSE); see also [NOTICE](NOTICE).
Third-party software, datasets and assets keep their own licenses
([THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)).
