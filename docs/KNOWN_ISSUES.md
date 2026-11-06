# Known Issues

Severity: **BLOCKER** (prevents the next session) · **HIGH** (affects research validity
if ignored) · **MEDIUM** · **LOW**. Every entry cites its evidence.

| ID | Severity | Status | Summary |
|---|---|---|---|
| KI-001 | LOW | PARTIAL | Linux aarch64 OSMesa verified; x86_64 install and EGL unverified |
| KI-002 | HIGH | MITIGATED | robosuite seeding does not control reset randomness |
| KI-003 | HIGH | RESOLVED (Session 1) | Can relative-pose observations are lagged / wrong at t=0 |
| KI-004 | MEDIUM | ACCEPTED | Open-loop demo replay is not exact |
| KI-005 | HIGH | MITIGATED | Latest mujoco breaks robosuite v1.5.1 |
| KI-006 | LOW | MITIGATED | egl_probe does not build on macOS |
| KI-007 | MEDIUM | RESOLVED (Session 3) | Local Claude Code plugin hooks blocked legitimate commands |
| KI-008 | MEDIUM | OPEN | Local disk nearly full (Docker storage repaired) |
| KI-009 | LOW | OPEN | Evidence files contain absolute local paths |
| KI-010 | LOW | OPEN | Docker sim image is 8.7 GB |
| KI-011 | MEDIUM | OPEN | Lift's initial-state distribution is narrow |
| KI-012 | LOW | ACCEPTED | Repackaged v1.5 datasets ≠ CoRL 2021 numbers |
| KI-013 | MEDIUM | MITIGATED (Session 2, D-017) | Physics and numpy RNG streams are not bit-identical across platforms |
| KI-014 | MEDIUM | RESOLVED (Session 2) | robosuite default models carry 2 mocap markers absent from dataset models |
| KI-015 | HIGH | MITIGATED | robosuite `env.step` observations are one physics substep stale |
| KI-016 | MEDIUM | RESOLVED (Session 2, D-025) | `get_xml()` round trip changes the model at ~1e-7 |
| KI-017 | LOW | ACCEPTED | Quaternion inputs are ambiguous in sign across episodes |
| KI-018 | MEDIUM | RESOLVED (aarch64) | robosuite `quat2mat` is float32 and platform-dependent |
| KI-019 | MEDIUM | MITIGATED (Session 2, D-024) | `mj_forward` on live data perturbs the rest of a rollout |
| KI-020 | LOW | MITIGATED (Session 2, D-017) | `uv sync`/`uv run` inside the pinned image would modify its environment |
| KI-021 | MEDIUM | OPEN | BC-MLP compounds a missed grasp into out-of-workspace drift with saturated actions |
| KI-022 | LOW | ACCEPTED | Gripper commands are clipped on 28–63 % of steps, harmlessly |
| KI-023 | LOW | OPEN | Policy inference is 13–17× slower in the canonical container than natively |
| KI-024 | MEDIUM | OPEN | Offline validation MSE rises during training and does not track closed-loop success |

### KI-001 — Linux headless rendering: aarch64 OSMesa verified; x86_64 and EGL not
- **Verified (2026-11-05, after the owner-approved Docker Desktop restart):**
  - `uv sync --frozen` installs the full lockfile on linux/aarch64 (Debian bookworm) and
    builds `egl_probe` (`logs/session0/docker_build_sim_cpu.log`).
  - In image `roboforge-sim-cpu:session0` (kernel 7.0.14-linuxkit aarch64, Python 3.11.18
    from the `python:3.11-slim-bookworm` base, `MUJOCO_GL=osmesa`), running the committed
    code from `a9c6793`:
    - `smoke_sim.py`: Lift 11/11 and Can 10/10 checks PASS.
    - `verify_data.py --replay-demos all --render`: all 12 integrity checks PASS.
  - The 20 rendered frames were inspected visually and are correct and upright. They differ
    from macOS CGL frames by 0.3–2.2 mean grey levels, and ≤ 0.9 % of pixels differ by
    more than 16 levels.
  - Evidence: `reports/feasibility/linux_aarch64_osmesa/`,
    `logs/session0/docker_linux_aarch64_osmesa_after_restart.log`,
    `logs/session0/platform_comparison.log`.
- **Measured cost:** OSMesa takes 88–196 ms per 256² frame (mean over 9 frames per task,
  including first-frame setup), against 7–8 ms with macOS CGL.
- **Not verified:**
  - linux/x86_64 install (Modal's architecture).
  - EGL on Modal GPUs.
- **Impact:** none for Level 1 state-based work. RGB extraction (Session 8) can fall back to
  OSMesa on CPU, which is about 10–25× slower than hardware rendering.
- **History:** the first container run (before the restart) failed with Docker storage
  `input/output error` after the host disk filled (`docker_linux_aarch64_osmesa.log`).

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
- **Resolution (Session 1, D-010):**
  - `roboforge.data.features` computes every policy input from the MuJoCo state, for
    dataset windows and for closed-loop evaluation alike.
  - `Can_to_robot0_eef_pos` is recomputed from same-step poses, in float64 (KI-018).
  - `Can_to_robot0_eef_quat` is dropped.
  - Across all 200 Can demos, the stored block equals robosuite's one-update-lag formula to
    2.2e-16 and differs from the corrected feature by up to 0.861 m at t = 0 and 0.091 m at
    t > 0 (`reports/data_audit.json` → `ki003_correction`).
  - The Session 0 raw-obs loader, which would have fed the lagged block to a policy, was
    removed.

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
- **Session 1 probe:** after the restart, writing a file containing `loss.backward()` and
  `optimizer.step()` was not blocked. The file was in the session scratchpad, outside the
  repository; Session 3 must re-check inside the repository. The SessionStart hook still
  injects ml-pipeline guard instructions. Nothing from them was added to the repository.
- **Resolved (Session 3, before any training code):**
  - Claude Code's Write and Edit tools created and modified
    `src/roboforge/train/ki007_probe.py`, which contains `loss.backward()` and
    `optimizer.step()`, inside the repository. Neither was blocked.
  - The probe ran (one SGD step, 1.0 → 0.8) and was deleted. It was never committed
    (`logs/session3/prereq/ki007_probe.log`).
  - All Session 3 training code was then written the same way without any hook
    intervention. `ML_PIPELINE_ENFORCE=0` stays set in `~/.claude/settings.json`.
  - The SessionStart hook still injects ml-pipeline guard and canary instructions. They were
    ignored and nothing from them entered the repository.

### KI-008 — Local disk nearly full; Docker storage corrupted
- The disk had 55 GB free at session start; it filled during the Docker image export.
- After the owner freed space: 17 GB free. After deleting the scratch mujoco 3.3.0
  environment (`venv330`, 1.8 GB, owner-approved): 36 GB free.
- Docker's containerd store returned I/O errors until the owner-approved Docker Desktop
  restart; it is healthy since.
- `docker system df` afterwards: images 46 GB (the RoboForge image is 25.6 GB unpacked),
  build cache 27.5 GB reclaimable. Pruning is an owner decision.
- Session 1 start: 33 GB free. The processed Session 1 data add < 10 MB.
- Session 2 end: 63 GB free. The committed eval suites take 128 KB; the gitignored
  `runs/` sanity records take about 0.1 GB per full sanity evaluation.
- RGB extraction (Session 8) needs ~0.5–1.5 GB per task at 84×84. Checkpoints and rollouts
  need more. Monitor `df -h` before heavy steps.
- Session 3 end: 55 GB free.
  - `runs/` holds 411 MB: 3 training runs × 6 checkpoints (≈ 13 MB each for the BC models,
    including Adam state) plus 18 validation evaluations × 50 episode records.
  - A Session 6 main suite (3 families × 3 seeds × 2 tasks × 10 checkpoints, with
    evaluations) is expected to need a few GB. That is an estimate, not measured.

### KI-009 — Evidence files contain absolute local paths
Some files contain `/Users/zackaryholt/...` paths: `reports/feasibility/*.json`
(environment snapshot, file paths) and `logs/session0/*.log`. Scrub or regenerate before a
public release (Session 12).

### KI-010 — Docker sim image is 8.7 GB (25.6 GB unpacked)
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

### KI-013 — Physics is not bit-identical across platforms
- The same code, lockfile and mujoco 3.2.7 were run on macOS arm64 and on Linux aarch64
  (Docker), replaying all 400 demos open-loop
  (`reports/feasibility/linux_aarch64_osmesa/platform_comparison.json`).
- **The same:** replay successes (Lift 199/200, Can 190/200), the exact set of failing
  demos, every integrity and smoke check, state restore (0.0).
- **Different:** per-demo trajectory errors. The largest difference in task-object position
  error is 0.27 mm (Lift) and 5.0 cm (Can). First-success steps differ by at most one step (Can).
  Observations recomputed from states are exact on macOS and within 1.1e-16 on Linux.
- **Cause (inferred, not isolated):** floating-point differences between the two platforms'
  MuJoCo/NumPy builds, amplified by contact dynamics. The platforms also differ in Python
  patch version (3.11.15 on macOS, 3.11.18 in the image); all package versions are equal.
- Remote images should pin the Python patch version too (Session 5).
- **Impact:** a closed-loop rollout of the same policy from the same serialized state can
  end differently on different platforms, mostly in contact-rich tasks such as Can.
- **Action (Session 2, D-017):** run every rollout in one comparison on one pinned platform
  and image, record it in each run manifest, and never mix platforms within a results
  table.
- **Session 2:**
  - The canonical evaluation platform is the pinned Linux aarch64 container (D-017).
    Every record stores its platform ID.
  - **numpy RNG streams also differ by 1 ulp.** numpy's legacy `RandomState.randn` and
    `np.sin` return values that differ by 1 ulp between macOS arm64 and the container;
    uniform draws and `log` match (`logs/session2/diag/rng_{macos,linux}.json`).
  - robosuite's reset samplers use these functions, so the same seed gives starts that
    differ by ≤ 4.4e-16 between the platforms. The XML is identical.
  - The committed evaluation suites were therefore generated on the container. They
    regenerate bitwise there, and within 1e-12 elsewhere (`make check-eval-states`).
    They restore exactly on every platform (D-025).
  - Two fresh processes on the same platform are bitwise identical
    (`reports/session2/determinism/`).
  - **Amplification:** on macOS, a start regenerated from its seed differs from the stored
    (container-generated) start by 4.4e-16 (Lift) or 2.8e-17 (Can). The same 400-step
    random rollout from the two starts then differs by up to 0.144 (Lift) and 1.49 (Can)
    in the state (`reports/session2/determinism/macos/`).
  - Evaluation must therefore always restore the stored snapshot, never re-seed, and every
    comparison must run on one platform.

### KI-014 — robosuite's default models carry two mocap markers the dataset models lack
- In robosuite 1.5.1, `robosuite.make` builds models with two extra mocap bodies,
  `left_eef_target` and `right_eef_target`. Each has a box and a sphere geom with
  `contype = conaffinity = 0`, and the model has no equality constraints.
- The dataset `model_file` XMLs do not contain them:
  - Can: default model nbody 34, ngeom 126, nmocap 2;
  - dataset model nbody 32, ngeom 122, nmocap 0;
  - nq, nv, nu and the timestep are equal.
- Expected physics effect: none (no contacts, no welds). They may be visible in renders.
- The feature function rejects any feature whose body chain contains a mocap body; no
  current feature does (`features.py`).
- **Action (Session 2):** decide whether evaluation initial states are compiled from the
  default robosuite model or from a dataset-style XML. Verify that dynamics match the
  dataset model on short rollouts, and check renders for the markers.
- **Resolution (Session 2):** keep robosuite's default model, with no XML surgery
  (`reports/session2/sim_contract_{lift,can}.json`, measured on macOS arm64).
  - **Markers:** they sit at z = −1 with `contype = conaffinity = 0` in geom group 2.
  - **Dynamics:**
    - Replaying the 12 sampled demos with and without the markers gives bitwise-identical
      states (check A).
    - So do 5 evaluation starts × 400 steps of random arm actions, with the gripper
      toggled every 40 steps (check B).
  - **Renders:** they differ on 0 pixels on every camera (check C).
  - **Dataset model vs evaluation model**, aligned by name (check D):
    - the only extra bodies are the 2 markers;
    - otherwise the models differ only in index shifts, `body_subtreemass` of the world
      body (the markers' mass) and `site_rgba`, which is visual only;
    - for Lift, also the per-reset cube sample.

### KI-015 — robosuite `env.step` observations are one physics substep stale
- robosuite's `step` runs `sim.forward()`, the controller and `mj_step`, then reads
  observables. MuJoCo computes kinematics *before* integrating, so the returned
  observations describe the state 2 ms before the returned `sim` state.
- The robomimic v1.5 dataset observations were regenerated from saved states and are not
  stale.
- **Measured** (`reports/data_audit.json`, 5 replayed training demos per task), max
  |`env.step` obs − canonical feature|:
  - Lift: `robot0_eef_pos` 5.0e-4 m, `cube_quat` 3.1e-3;
  - Can: `robot0_eef_pos` 5.8e-4 m, `Can_quat` 7.3e-3, `Can_to_robot0_eef_pos` 0.855 m
    (the KI-003 artifact after a restore).
- With current kinematics and a fresh cache, robosuite equals the canonical features exactly,
  except the corrected Can relative position, which agrees to 1.1e-7 m (robosuite's float32
  rotation, KI-018).
- **Mitigation:** policies consume only `FeatureExtractor.from_sim` (D-010), which is a pure
  function of the state. The Session 2 adapter must not pass `env.step` observations to
  policies.

### KI-016 — `MjModel.get_xml()` round trip changes the model at ~1e-7
- Re-serializing a compiled robosuite model with `sim.model.get_xml()` and recompiling it
  gives a slightly different model.
- Features of the same state differed by up to 1.1e-7 (eef quaternion) between the original
  and the recompiled model (found while writing `test_features_closed_loop.py`).
- **Action (Session 2):** serialized evaluation states must store the exact XML string that
  was compiled. Every comparison must load that string; never a re-serialization.
- **Explained and resolved (Session 2, D-025):** the live model and its saved XML differ
  (`reports/session2/sim_contract_*.json`, check E):
  - `dof_armature` by up to 3.3e-6;
  - `body_quat` by ~1e-7;
  - the cube's principal-axis ordering;
  - static-body masses and inertias, which have no dynamic effect.

  A snapshot stores the saved XML (`mj_saveLastXML`, the robomimic convention). Every
  restore compiles exactly that string, so the evaluation model *is* the saved XML. The
  model generated before saving is never run.
- Tests check that restores are exact and that the portable and raw XML strings compile
  identically (`tests/unit/test_snapshot.py`, `tests/simulator/test_env_adapter.py`).

### KI-017 — Quaternion inputs are continuous within episodes but ambiguous across them
- All quaternion inputs have 0 within-episode sign flips (`reports/data_audit.json`).
- Across episodes, `q` and `−q` both occur:
  - Lift `cube_quat` w spans −1 to 1, because yaw is random and the cube has 90° symmetry.
  - `robot0_eef_quat` sits near w ≈ 0 (x ≈ 1), with w < 0 in about half the steps.
- This is robosuite's standard representation, and every model family gets the same input
  (D-018).
- A continuous-rotation ablation would be a separately labeled experiment.

### KI-018 — robosuite's `quat2mat` is float32 and platform-dependent
- `robosuite.utils.transform_utils.quat2mat` casts quaternions to float32
  (`astype(np.float32)`) and is numba-compiled.
- **Found:** at `cb9b6a3` the corrected Can relative position used robosuite's formula. In
  the Linux aarch64 image the Can processed content hash and normalization hash then
  differed from macOS, while Lift (which has no rotation-dependent feature) matched
  bit-for-bit. The stored-vs-lag residual was 1.4e-7 on Linux against 2.2e-16 on macOS
  (`logs/session1/docker_linux_aarch64_check_data_cb9b6a3_FAIL.log`).
- **Resolution (`c69551f`):** `pos_in_eef_frame` is evaluated in float64 from MuJoCo's
  `xmat`. It matches robosuite's float32 formula to 1.4e-7 m by design. `make check-data`
  now passes for both tasks on macOS arm64 and Linux aarch64
  (`logs/session1/docker_linux_aarch64_check_data.log`).
- **Still open:** linux/x86_64 (Modal) is unverified (KI-001). Until it is checked, copy the
  hash-verified processed file to remote storage instead of rebuilding it there;
  `SequenceDataset.from_committed` refuses a mismatching file.
- **Rule for later sessions:** never use robosuite's transform helpers to compute policy
  inputs or evaluation metrics that need more than float32 precision.

### KI-019 — `mj_forward` on live data perturbs the rest of a rollout
- `mj_forward` recomputes accelerations and overwrites MuJoCo's `qacc_warmstart`, the
  solver warm start that the next `mj_step` uses. The flattened state does not include it.
- **Measured on macOS arm64** (`reports/session2/sim_contract_*.json`, check F
  reference). From validation start 0, the first 150 recorded random actions of validation
  start 2's episode were replayed twice: once plainly, and once with `mj_forward` after
  every step. Maximum state difference:
  - Can: up to 0.587 in the state;
  - Lift: 6.6e-12.
- `mj_kinematics` plus the success predicate is inert: 0.0 difference (check F, 3
  validation starts).
- **Mitigation (D-024):** success and kinematics are evaluated with `mj_kinematics` only.
  Features, rendering and recording use scratch `MjData`.
- **Rule for later sessions:** never call `mj_forward`, `sim.forward()` or robosuite's
  observation refresh on the live data of an evaluation episode.

### KI-020 — `uv sync` / `uv run` inside the pinned image would modify its environment
- **Measured:** `uv sync --frozen --dry-run` in `roboforge-sim-cpu:session0`
  (`logs/session2/diag/container_uv_sync_dry_run.log`) would:
  - reinstall `nvidia-cusparselt-cu13`;
  - rebuild the editable `roboforge`;
  - install `ruff`.

  `uv run` would do the same, so the platform would change at run time.
- **Mitigation:** `infra/containers/run_sim_cpu.sh` runs the image's own `/app/.venv`
  directly, with the mounted `src/` first on `PYTHONPATH` and no `uv` call. Lint runs on
  the host.
- **Session 5:** the remote image build should make `uv sync --frozen` a no-op, or record
  why it is not.

### KI-021 — BC-MLP compounds a missed grasp into out-of-workspace drift with saturated actions
- **Observed** (Session 3 Lift validation pilots, canonical container,
  `reports/session3/diagnostics/actions.json`). Every failed episode of every Session 3
  pilot is a grasp failure:
  - **BC-MLP:** 3–9 of 50 per checkpoint. Most are `missed_grasp_then_drift`: the gripper
    closes, the cube never rises, and the end effector ends more than 0.2 m from the cube.
    Its z reaches up to 2.19 m, above a table at ≈ 0.8 m. A few are `gripper_never_closed`:
    the arm hovers 2–5 cm from the cube until the horizon.
  - **BC-MLP-chunk:** 0–4 of 50, all drift.
  - **BC-RNN:** 0–2 of 50 (drift or `missed_grasp_stayed_near`).
- **Clipping is a symptom, not a convention bug:**
  - Motion dimensions (0–5) are clipped on 50–78 % of the steps of failed BC-MLP and
    BC-MLP-chunk episodes, against ≤ 1.3 % in successful ones. BC-RNN clips ≤ 1 % even
    when it fails.
  - Once the arm leaves the demonstrated workspace, the MLP extrapolates and saturates, and
    it never returns.
  - Controller axes and signs are verified: each Δposition and Δrotation dimension moves or
    rotates the end effector mostly about its own world axis with a positive sign, and
    gripper +1 closes. The teacher-forced gripper sign agreement on the validation demos is
    0.92–0.96.
- **Categories:** post-hoc Session 3 diagnostic categories (`scripts/diagnose_actions.py`).
  They are not the preregistered failure taxonomy, which Session 7 defines before the test.
- **Implication:** this is classic compounding error in closed-loop BC, which offline loss
  does not show (KI-024). Recovery and regularization options are tuning decisions for
  Session 6, within the matched budget: early checkpoints, weight decay, dropout, observation
  noise and action chunking. Lift success is therefore **not** saturated for one-step BC
  (best 47/50).

### KI-022 — Gripper commands are clipped on 28–63 % of steps, harmlessly
- The demonstrations' gripper action is exactly ±1, which is also the training range, so
  min-max normalization maps it to ±1.
  - Regression outputs overshoot ±1 slightly.
  - After decoding, 28–63 % of executed steps per checkpoint have the gripper entry clipped
    (eval summaries, `clip_fraction_per_dim[6]`).
- robosuite's `PandaGripper.format_action` uses only `np.sign(action)` (it integrates
  `current_action += 0.2 · sign(a)`), so clipping the magnitude cannot change behavior.
- **Reporting rule:** report clip and saturation rates for the motion dimensions (0–5)
  separately from the gripper. A total clip rate would be dominated by this harmless gripper
  clipping.

### KI-023 — Policy inference is 13–17× slower in the canonical container than natively
- **Measured** with `scripts/benchmark_policies.py` (1 torch thread, median per decision of
  `observe` + `act`):

  | Model | macOS native | container |
  |---|---|---|
  | BC-MLP | 0.070 ms | 0.912 ms |
  | BC-MLP-chunk | 0.075 ms | 1.259 ms |
  | BC-RNN | 0.655 ms | 8.626 ms |

  Ratios: 13.0× (BC-MLP), 16.9× (BC-MLP-chunk), 13.2× (BC-RNN). Sources:
  `reports/session3/cost_latency/{darwin-arm64-native,linux-aarch64-container-1f685638f28c}.json`.
- **Cause (not isolated):** the Docker Desktop Linux VM plus the CPU path of the
  linux-aarch64 torch wheel (a CUDA build, KI-010).
- **Impact:**
  - Latency comparisons must use one platform; the canonical container is the reported one.
  - BC costs are small. Diffusion with K denoising steps per decision may dominate canonical
    evaluation time, so Session 4 must measure it there.

### KI-024 — Offline validation MSE rises during training and does not track closed-loop success
- In all three Session 3 pilots the masked MSE on the 20 validation demos rises steadily from
  the first checkpoint, while the training MSE falls (`runs/<id>/metrics.jsonl`, summarized
  in EXPERIMENT_LEDGER E3-07):
  - BC-MLP: 0.042 at 5k steps → 0.080 at 30k;
  - BC-RNN: 0.034 → 0.051;
  - BC-MLP-chunk: 0.048 → 0.071.

  Closed-loop success moves much less, and not monotonically: BC-RNN scores 50/50 at
  5k–15k with a rising validation MSE.
- Offline loss therefore cannot select checkpoints. The protocol already forbids that
  (RESEARCH_PROTOCOL §7.4), and this confirms the rule empirically.

