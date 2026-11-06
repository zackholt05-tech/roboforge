# Session 0 Feasibility Stack Report

**Decision: GO for Session 1**, Level 1 state-based track, Lift (primary) and Can
(secondary). Conditions are listed in §10.
Date: 2026-11-05. Machine: macOS 26.6.2, Apple Silicon (arm64), 10 cores, 16 GB RAM.
Every number below comes from a file in this directory or in `logs/session0/`. The JSON
reports embed the git SHA, the package versions and the `MUJOCO_GL` they were produced with.

## 1. Environment and version matrix (spec §2.3 item 1)
| Component | Version | How chosen |
|---|---|---|
| Python | 3.11.15 (uv-managed) | D-001 |
| mujoco | 3.2.7 | D-002: 3.17.0 breaks robosuite 1.5.1; 3.3.0 equivalent (`mujoco_version_comparison.json`) |
| robosuite | 1.5.1 @ `51cc01785bab80ffeed20da15e67d7dd4140e76a` | D-003: robomimic dataset docs + `env_version` in data |
| robomimic | 0.5.0+15 @ `d309eaecc18acf4152a830a895a6984b8ac71b05` | D-004 |
| torch / torchvision | 2.15.0 / 0.29.2 (MPS available) | latest compatible; imports verified |
| numpy / h5py / scipy / numba | 2.4.6 / 3.16.0 / 1.17.1 / 0.68.1 | resolver (uv.lock) |
| diffusers / transformers / huggingface-hub | 0.11.1 / 4.41.2 / 0.23.4 | robomimic hard pins |

Lockfile: `uv.lock` (universal: darwin-arm64, linux-x86_64, linux-aarch64).
`import robomimic.algo.diffusion_policy` and the diffusers DDPM/DDIM schedulers work.
The CLIP model is not downloaded at import.

## 2. Datasets (item 2)
- Lift PH and Can PH, `low_dim` + `raw`, downloaded from HF `robomimic/robomimic_datasets`
  @ `74fa018…`. All four SHA-256 hashes equal the HF LFS oids
  (`logs/session0/download_sha256.log`; manifest `data/manifests/robomimic_v1.5_ph.json`).
  License: MIT (dataset card).
- Lift: 200 demos, length 36–64 (mean 48.3, total 9,666). Actions 7-D within [−1, 1],
  gripper ∈ {−1, +1}. States 32-D. Masks train/valid 180/20.
- Can: 200 demos, length 82–151 (mean 116.0, total 23,207). Actions 7-D. States 71-D.
  Masks 180/20.
- `low_dim` ≡ `raw` (states, actions, model XML, masks; max diff 0.0). All 400 recorded
  final states satisfy the task success predicate.
- No pre-rendered image HDF5 exists for v1.5 PH. RGB/depth must be regenerated from raw
  states (states and `model_file` are present → feasible).
- Square PH is available (51/69 MB), not downloaded.

## 3. Simulator reset/step and 3D state (item 3) — `smoke_sim_{lift,can}.json`
All 11 Lift checks and 10 Can checks PASS:
- Same global seed gives an identical reset (diff 0.0). A different seed changes object
  placement (Lift 2.8 mm, Can 15.8 cm).
- Action dim 7, bounds ±1. 20 Hz control, 2 ms physics, 25 substeps.
- 50 random closed-loop steps: sim time +2.5 s, the robot moves, all observations are
  finite, reward 0, no success.
- Lift success predicate flips when the cube is raised 10 cm, with the boundary at
  table + 4 cm ± 1 mm (tests).
- `done` only at the horizon.
- Two fresh processes restoring one serialized state and applying 20 identical actions give
  identical trajectories (max diff 0.0).
- 3D kinematics recorded: joint names/angles, eef pose, object world positions.
- robosuite task constructors reject `seed=` (KI-002).

## 4. Demonstration replay (item 4) — `data_report_{lift,can}.json`
- **State restore:** model XML + `states[0]` read back exactly (max error 0.0, all 400 demos).
- **Observation alignment:** observations recomputed from `states[t]` equal the stored
  `obs[t]` (0.0) for every key and sub-block, except Can's two relative-pose blocks
  (KI-003).
- **Open-loop action replay** (mujoco 3.2.7):
  - Lift 199/200 succeed. Task-object position error median 1.1 mm, p95 3.2 mm. Robot joint
    p95 3.2 mrad.
  - Can 190/200 succeed. Can position error median 6.5 cm, p95 9.8 cm, final median 1.0 cm.
    Robot joint p95 11.7 mrad.
  - Failures are grasp/contact divergences with the robot still tracking.
  - **Controller and action mapping: consistent** (no controller mismatch).
  - Tolerances are declared as regression guards: Lift ≥ 0.98, Can ≥ 0.90 (D-009).

## 5. Headless render (item 5)
- **macOS (verified):** `MUJOCO_GL=cgl`, cameras `agentview`, `frontview`,
  `robot0_eye_in_hand`, 256×256. Frames rendered from the replayed demo at t = 0, mid and end
  (`frames/{lift,can}_demo_0_*.png`, smoke `frames/smoke_*_agentview.png`).
- Frames were **inspected visually**:
  - Lift: cube on the table → cube grasped and lifted.
  - Can: can in the left bin → can in the right goal-bin compartment.
  - Images are upright: the OpenGL convention is flipped in `sim/renderer.py`.
- Render cost: 8–25 ms/frame.
- **Linux headless: NOT VERIFIED** (KI-001). The image built and the lockfile installed on
  linux/aarch64, but running the container failed after Docker storage corruption (host
  disk filled). EGL on Modal is unverified. Not required for state-based Level 1.

## 6. Data loader + one-batch forward pass (item 6)
`roboforge.data.windows.LowDimWindowDataset` produces `obs [T_o, D]`, `obs_valid [T_o]`,
`action [T_p, 7]` and `action_valid [T_p]` within episode bounds. Tests cover
boundaries/padding, no future-observation leakage, and real Lift windows (2×19 obs, 16×7
actions, finite, |a| ≤ 1). A tiny MLP forward pass with masked MSE on one batch is finite.
The forward pass only, no backward: D-014.

## 7. Closed-loop random policy and success signal (item 7)
Covered in §3: random actions give no success. Success comes from robosuite's own
`_check_success`, validated against all recorded final states (400/400 True) and the
boundary tests.

## 8. Provenance and redistribution (item 8)
- Datasets: MIT (HF card). Not redistributed; fetched by script and verified by hash.
- Code: robosuite MIT (includes Apache-2.0 MuJoCo parts), robomimic MIT, MuJoCo Apache-2.0,
  Diffusion Policy and DP3 reference code MIT.
- Assets: Panda and object meshes ship inside robosuite (no separate license file found;
  covered by robosuite's MIT license). Rendered PNGs here are RoboForge renders.
- No pretrained weights are used.
- Citations: `THIRD_PARTY_NOTICES.md` (BibTeX copied from official sources).

## 9. Resource estimates (item 9) — details in `docs/COMPUTE_PLAN.md`
- Disk: 170 MB of raw data. RGB extraction ≈ 0.4–1 GB per task (estimate). **Local disk
  has only 17 GB free** (KI-008).
- Simulation: ≈ 195 (Lift) / 163 (Can) control steps/s/core. The full 450-rollout × 400-step
  eval suite is ≈ 15–18 core-minutes of physics per task, excluding policy inference.
- Training memory/time: unmeasured. Low-dim models are expected to be feasible on
  CPU/MPS; benchmark in Sessions 3–4.
- A Modal GPU smoke test is not needed until Session 5, and only with approval.
  Recommended first GPU: L4/T4.
- Spend: **$0**.

## 10. GO / NO-GO
**GO** for Session 1 (data pipeline), because:
- a real Lift PH file was parsed;
- the matching 3D simulator runs;
- replay and offscreen render were observed and compared;
- the controller/action mapping was inspected;
- the downstream data path is demonstrated.

Conditions carried forward:
1. **Can observations:** the Can observation spec must handle KI-003 before any Can
   training.
2. **Evaluation initial states:** must be serialized MuJoCo states (KI-002).
3. **Plugin hooks:** confirm before Session 3 that the plugin hook no longer blocks
   training code (KI-007).
4. **Linux/EGL:** verify before Sessions 5 and 8 (KI-001).

## Command log index
- `logs/session0/download_sha256.log`
- `verify_data_mujoco3.2.7.log`
- `verify_data_mujoco3.3.0.log`
- `smoke_sim.log`
- `probe_obs_consistency.log`
- `docker_build_sim_cpu.log`
- `docker_linux_aarch64_osmesa.log` (the failure)
- `upstream_download_datasets_dry_run.log`
