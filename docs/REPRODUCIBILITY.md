# Reproducibility

## Environment
```bash
# prerequisites: uv >= 0.11 (https://docs.astral.sh/uv/), git; macOS arm64 or Linux
git clone <repo> roboforge && cd roboforge
make setup            # == uv sync --frozen  -> .venv with Python 3.11 and exact pins from uv.lock
```
Pinned core: mujoco 3.2.7 · robosuite v1.5.1 (`51cc017`) · robomimic `d309eae` ·
torch 2.15.0 · numpy 2.4.6 · h5py 3.16.0 · diffusers 0.11.1 · transformers 4.41.2.
`egl_probe` installs on Linux only (D-005). On Linux it needs `cmake`, `build-essential` and
`libegl-dev`; for headless rendering also `libosmesa6` (`MUJOCO_GL=osmesa`) or an NVIDIA EGL
driver (`MUJOCO_GL=egl`).

## Session 0 evidence commands
| Command | What it proves | Output |
|---|---|---|
| `make download` | official files fetched; SHA-256 equals manifest (= HF LFS oid) | `data/raw/...` (gitignored) |
| `make smoke-sim` | real 3D reset/step/success/render; seeding and cross-process determinism | `reports/feasibility/smoke_sim_{lift,can}.json`, `frames/smoke_*.png` |
| `make verify-data` | HDF5 structure, low_dim ≡ raw, obs ⇔ state alignment, replay of all demos, rendered frames | `reports/feasibility/data_report_{lift,can}.json`, `frames/*` |
| `make probe-obs` | relative-pose observation correctness (KI-003) | `reports/feasibility/obs_consistency.json` |
| `make test` | unit + simulator + dataset integration tests | pytest summary |
| `make lint` | ruff | — |

Every JSON report embeds `environment_snapshot()`: Python, platform, package versions,
`MUJOCO_GL`, and git SHA plus dirty flag.

## Seeds and determinism
- Simulator resets: seed the **global** `np.random` before `env.reset()`. Restoring
  serialized states is exact (KI-002).
- Random actions in smoke tests: `np.random.default_rng(seed)`.
- Torch: `torch.manual_seed` in tests. Training determinism settings are defined in Session 3.

## What is not reproducible yet
- linux/x86_64 installation and EGL rendering are unverified. linux/aarch64 with OSMesa is
  verified in Docker (KI-001).
- Physics is not bit-identical across platforms: macOS and Linux give the same pass/fail
  outcomes but different trajectories (KI-013). Reproduce rollouts on the platform recorded
  in the run manifest.
- Remote (Modal) execution: none performed.
