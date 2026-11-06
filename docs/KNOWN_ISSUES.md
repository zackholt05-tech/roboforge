# Known Issues

Severity: **BLOCKER** (prevents the next session) · **HIGH** (affects research validity
if ignored) · **MEDIUM** · **LOW**. Every entry cites its evidence.

| ID | Severity | Status | Summary |
|---|---|---|---|
| KI-001 | MEDIUM | OPEN | Linux headless rendering not verified end-to-end |
| KI-002 | HIGH | MITIGATED | robosuite seeding does not control reset randomness |
| KI-003 | HIGH | OPEN (Session 1) | Can relative-pose observations are lagged / wrong at t=0 |
| KI-004 | MEDIUM | ACCEPTED | Open-loop demo replay is not exact |
| KI-005 | HIGH | MITIGATED | Latest mujoco breaks robosuite v1.5.1 |
| KI-006 | LOW | MITIGATED | egl_probe does not build on macOS |
| KI-007 | MEDIUM | PENDING RESTART | Local Claude Code plugin hooks blocked legitimate commands |
| KI-008 | MEDIUM | OPEN | Local disk nearly full; Docker storage corrupted |
| KI-009 | LOW | OPEN | Evidence files contain absolute local paths |
| KI-010 | LOW | OPEN | Docker sim image is 8.7 GB |
| KI-011 | MEDIUM | OPEN | Lift's initial-state distribution is narrow |
| KI-012 | LOW | ACCEPTED | Repackaged v1.5 datasets ≠ CoRL 2021 numbers |

### KI-001 — Linux headless rendering not verified end-to-end
- **Verified:** on linux/aarch64 (Docker Desktop VM, Debian bookworm), `uv sync --frozen`
  installed the full lockfile, built `egl_probe`, and produced image
  `roboforge-sim-cpu:session0` (`logs/session0/docker_build_sim_cpu.log`).
- **Not verified:**
  - Running `smoke_sim.py`/`verify_data.py` inside it with `MUJOCO_GL=osmesa`. After the
    host disk filled, Docker returned `input/output error` on the image blob.
  - linux/x86_64 install (Modal's architecture).
  - EGL on Modal GPUs.
- **Impact:** none for Level 1 state-based training/evaluation (no rendering needed).
  Matters for RGB extraction (Session 8) and video export.
- **Recovery:** restart Docker Desktop (or Troubleshoot → Clean/Purge data, which deletes
  *all* local Docker data). Then:
  `docker build -f infra/containers/Dockerfile.sim-cpu -t roboforge-sim-cpu .`
  `docker run --rm -v "$PWD/data/raw:/app/data/raw:ro" -v "$PWD/out:/out" roboforge-sim-cpu`
  Read the PNG it writes.

### KI-002 — robosuite seeding does not control reset randomness
- robosuite v1.5.1 `Lift`/`PickPlaceCan` constructors raise `TypeError` on `seed=`.
- `UniformRandomSampler` and robot initialization noise use the global `np.random`.
- **Mitigation:** seed `np.random` before `reset()` (verified identical), and restore
  serialized states for evaluation (verified identical across processes).
- Evidence: `reports/feasibility/smoke_sim_{lift,can}.json`
  (`reset_determinism`, `cross_process`). Hidden RNG use elsewhere (e.g. `random.choice`
  for `obj_to_use`) is irrelevant for Can (single-object mode fixes the object), but
  Session 2 must restore states rather than replay seeds.

### KI-003 — Can relative-pose observations are lagged / wrong at t=0
- In robosuite v1.5.1 PickPlace, `Can_to_robot0_eef_pos` and `Can_to_robot0_eef_quat`
  (dims 0–6 of robomimic's 14-D Can `object` obs) are evaluated before `Can_pos`/`Can_quat`
  in the observable order, so they read the cache from the previous update.
- Measured (`reports/feasibility/obs_consistency.json`, 50 demos):
  - Stored dataset value at t>0 equals rel(can pose at t−1, eef pose at t) to 2.2e-16 m and
    7.4e-4 rad. Versus same-step truth: p95 1.6 cm, max 8.4 cm.
  - At t=0 the stored value is wrong by up to 0.86 m.
  - Closed-loop `env.step` observations have the same lag, and up to 0.85 m error at the
    first step after a state restore.
- World-frame `Can_pos`/`Can_quat` are exact.
- Lift's `gripper_to_cube_pos` is evaluated after `cube_pos` and is exact everywhere.
- **Not fixed in robosuite 1.5.2** (no diff in these sensors).
- **Impact:** policies using robomimic's default Can `object` key see a lagged, partially
  corrupted feature in both training and evaluation. Any comparison with published robomimic
  Can numbers inherits this.
- **Action (Session 1):** implement D-010 (one state→feature function for train and eval),
  or drop dims 0–6; apply identically to all models.

### KI-004 — Open-loop demo replay is not exact
State restore is exact (0.0). Replaying recorded actions from `states[0]` diverges because
of contact dynamics, and because MuJoCo warm-start/actuator state is not part of the
flattened state. mujoco 3.2.7, all 200 demos per task:

| Task | Replay success | Failed demos | Task-object max position error (median / p95) |
|---|---|---|---|
| Lift | 199/200 | `demo_177` | 1.1 mm / 3.2 mm |
| Can | 190/200 | `demo_11, 52, 71, 82, 95, 104, 112, 124, 139, 174` | 6.5 cm / 9.8 cm |

All 400 recorded final states satisfy the task predicate. Robot joint tracking stays within
~0.012 rad at p95; failures are grasp/contact outcomes, not a controller mismatch.

Implication: never treat replayed actions as ground truth for observations. Training uses
the recorded `obs` and `actions`; evaluation uses fresh closed-loop rollouts.

### KI-005 — Latest mujoco breaks robosuite v1.5.1
mujoco 3.17.0 raises `AssertionError` in `robosuite/utils/binding_utils.py:523`. Pinned to
3.2.7 (D-002). Never upgrade mujoco without re-running `make verify-data`.

### KI-006 — egl_probe does not build on macOS
Fails with "CMake must be installed". Restricted to Linux via a uv override (D-005).

### KI-007 — Local Claude Code plugin hooks blocked legitimate commands
- The `ml-pipeline` hook blocks any file containing a backward pass.
- `cto-toolkit` hooks blocked shell commands containing the substring "format" (e.g.
  `ruff format`, `docker --format`) and post-write test reviews.
- At the owner's request, the agent added `ML_PIPELINE_ENFORCE=0` under `env` in
  `~/.claude/settings.json`. The owner disabled `cto-toolkit@synced`.
- **Verify after restart:** creating a file with a training step is no longer blocked
  (required in Session 3).

### KI-008 — Local disk nearly full; Docker storage corrupted
- The disk had 55 GB free at session start; it filled during the Docker image export.
- After the owner freed space: 17 GB free.
- Docker's containerd store now returns I/O errors (KI-001).
- RGB extraction (Session 8) needs ~0.5–1.5 GB per task at 84×84. Checkpoints and rollouts
  need more. Monitor `df -h` before heavy steps.

### KI-009 — Evidence files contain absolute local paths
Some files contain `/Users/zackaryholt/...` paths: `reports/feasibility/*.json`
(environment snapshot, file paths) and `logs/session0/*.log`. Scrub or regenerate before a
public release (Session 12).

### KI-010 — Docker sim image is 8.7 GB
The PyPI torch wheels for linux-aarch64 (and x86_64) bundle CUDA libraries. CPU rollout
images should install CPU-only torch, or omit torch for pure-simulation jobs (Session 5).

### KI-011 — Lift's initial-state distribution is narrow
- Cube x/y uniform in ±3 cm, random yaw, z offset 0.01; Panda joint noise is Gaussian with
  σ = 0.02 rad (robosuite defaults).
- Observed: seed 0 vs seed 1 moves the cube 2.8 mm in x/y.
- "In-distribution" Lift test starts are therefore very similar. Spatial OOD claims need
  explicitly widened, physically valid placement ranges (spec §3.5), and must not rely on
  seed changes alone.

### KI-012 — Repackaged v1.5 datasets ≠ CoRL 2021 numbers
The robomimic docs state results may not match the original `mujoco-py`-based datasets.
Never compare RoboForge numbers to the 2021 paper as if they were the same benchmark.
