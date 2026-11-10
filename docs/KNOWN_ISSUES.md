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
| KI-009 | LOW | ACCEPTED (Session 12, D-100) | Evidence files contain absolute local paths |
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
| KI-022 | LOW | ACCEPTED | Gripper commands are clipped on 19–73 % of steps (BC), harmlessly |
| KI-023 | LOW | OPEN | Policy inference is 13–17× slower in the canonical container than natively (diffusion DDIM-10: 3.7×) |
| KI-024 | MEDIUM | OPEN | Offline validation MSE rises during training and does not track closed-loop success |
| KI-025 | LOW | ACCEPTED | robomimic `d309eae` ignores `algo.unet.*` and always builds its 65.8 M U-Net |
| KI-026 | LOW | MITIGATED (Session 4, D-040) | NNPACK makes the 1D U-Net about 3× slower on macOS arm64 |
| KI-027 | LOW | ACCEPTED | Diffusion samples depend on batch size at float-rounding level |
| KI-028 | MEDIUM | OPEN | All diffusion pilot failures are one start, under one noise seed (R = 1) |
| KI-029 | LOW | ACCEPTED | Content-identical checkpoints can have different file SHA-256 |
| KI-030 | MEDIUM | RESOLVED (Session 5) | Gate A probe lost every result to a BC-RNN CUDA device bug |
| KI-031 | MEDIUM | RESOLVED (Session 5) | Modal reschedules an interrupted input despite retries=0 |
| KI-032 | LOW | RESOLVED (Session 5) | Paths under a symlinked mount were recorded absolute |
| KI-033 | LOW | OPEN | The Modal image carries no evaluation suite; EGL vs OSMesa pending |
| KI-034 | MEDIUM | OPEN | Lift validation is saturated: most §7 choices are decided by the first-success tiebreak |
| KI-035 | LOW | ACCEPTED | Host contention during the Session 6 suite inflated wall times (results unaffected) |
| KI-036 | MEDIUM | RESOLVED (Session 7, D-070) | The Can suite orchestrator crashed on a half-written evaluation summary |
| KI-037 | LOW | RESOLVED (Session 7) | The robustness `--check` called equal Can manifests "not bitwise" |
| KI-038 | LOW | RESOLVED (Session 7) | `analyze_robustness.py` read the checkpoint step from the wrong field |
| KI-039 | MEDIUM | RESOLVED (Session 7, D-072) | A simulator test would have built a held-out start once the lock opened |
| KI-040 | MEDIUM | RESOLVED (Session 8) | `make smoke-sim` built Lift held-out start 0 in memory once the lock had opened |
| KI-041 | LOW | MITIGATED (Session 8, D-074) | The first render of a fresh CGL context differs from later renders by a few pixels (macOS only) |
| KI-042 | LOW | OPEN (Session 8) | Frozen `DiffusionPolicy.sample` draws its initial noise on the CPU, so diffusion inference cannot run on CUDA |
| KI-043 | LOW | MITIGATED (Session 8) | The container dataset build and the full test suite run together exhausted the 16 GB host's memory and stalled both |
| KI-044 | LOW | MITIGATED (Session 8) | A Modal preemption restarted one RGB resume launch |
| KI-045 | LOW | MITIGATED (Session 9) | Native macOS (CGL) depth lacks reversed-Z precision |
| KI-046 | MEDIUM | FIXED (Session 9) | robosuite's hidden sites still write depth |
| KI-047 | LOW | MITIGATED (Session 9) | A client network drop stopped the first Level 2 launch |
| KI-048 | LOW | FIXED (Session 9) | The Level 2 gate could not see Session 8 evidence inside the container |
| KI-049 | LOW | DOCUMENTED (Session 9) | One generated training row of the Level 2 analysis was miscomputed |
| KI-050 | LOW | FIXED (Session 10) | The first Observatory export stopped on the random baseline; its reference masks counted sites |
| KI-051 | LOW | FIXED (Session 10) | Observatory selection code named the locked suite; the lock test failed at STOP |
| KI-052 | MEDIUM | DOCUMENTED (Session 11, audit A-6) | The diffusion sampler bounds its actions to the training range; BC heads are not bounded, a confound for H2-c and H4 |
| KI-053 | MEDIUM | DOCUMENTED (Session 11, audit B-1 to B-4, D-095) | Several separations depend on the interval method, the bootstrap seed or ties (3 seeds, 50 binary outcomes) |
| KI-054 | MEDIUM | FIXED by reporting (Session 11, audit C-9, D-096) | Declared secondary metrics were computed per episode but never aggregated by the frozen analyses |
| KI-055 | LOW | FIXED (Session 11, audit B-12, B-13) | `analyze_robustness.py` had no run-once guard; `analyze_level2.py` hashed a report before checking it exists |
| KI-056 | LOW | DOCUMENTED (Session 11, audit C-6) | One RGB run's training hours omit a preempted attempt that was billed |
| KI-057 | LOW | DOCUMENTED (Session 11, audit C-5, C-21) | Governance record defects: A-01 probe launch count exceeded before its amendment; the Level 1 freeze calls a content hash "SHA-256" |
| KI-058 | LOW | FIXED (Session 12) | A fresh clone without datasets failed 9 RGB and point-cloud trainer unit tests that read the processed Lift data without skipping |
| KI-059 | LOW | FIXED (Session 12) | A fresh clone failed 4 held-out gate tests: their "runs/ not present" skip tested the directory, which a clone has (README.md) |
| KI-060 | LOW | ACCEPTED (Session 12) | The locked diffusers 0.11.1 and transformers 4.41.2 (pinned by robomimic) carry published security advisories |

### KI-001 — Linux headless rendering: aarch64 OSMesa verified; x86_64 install and CUDA verified on Modal (Session 5); EGL vs OSMesa on Modal PENDING
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
- **Verified on Modal (Session 5, Gate A, approval A-01):**
  - **linux/x86_64 install:** `Image.uv_sync(frozen=True, --no-dev)` installed the whole
    lockfile. Every job imported torch 2.15.0, mujoco 3.2.7 and robosuite 1.5.1 and ran:
    training on CPU in all launches, the CUDA profiles in the probe.
  - **CUDA:** NVIDIA L4, driver 610.57.04, torch CUDA 13.2. All four families trained on
    the GPU (EXPERIMENT_LEDGER E5-09).
- **Not verified:** EGL vs OSMesa inside a Modal GPU container. The probe's render check
  failed before rendering because the image carries no evaluation suite (KI-033). A fixed
  check (`scripts/render_probe.py`) is ready for a future approved probe. In the canonical
  (CPU) container both OSMesa and EGL render, at 39–47 ms per 256² frame.
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
- **Session 6 end (measured):** 29 GB free. `runs/` holds 9.7 GB, of which the Session 6 Lift
  suite is 8.7 GB (24 runs × 20 checkpoints plus 480 evaluations × 50 episode records);
  `reports/session6/` is 34 MB. Swap was 20.0 of 21.5 GB in use (KI-035). The Can suite
  (Session 7, episodes up to 400 steps) needs a similar or larger amount: check `df -h`
  before it.

### KI-009 — Evidence files contain absolute local paths
Some files contain `/Users/zackaryholt/...` paths: `reports/feasibility/*.json`
(environment snapshot, file paths) and `logs/session0/*.log`. Scrub or regenerate before a
public release (Session 12).
- **Session 12 (owner decision D-100, "Disclose, no rewrite"):** 136 tracked files carry such
  paths (the logs of Sessions 0–9, four `reports/feasibility/` and `reports/session5/` evidence
  files, this file and one test fixture), and every commit in history has them. They reveal only
  the local username; no secret was found (`make release-check`). The evidence bytes stay
  unchanged, since some logs are hash-pinned paper sources and rewriting history would change
  every cited commit SHA. `make release-check` fails if such a path appears in any other file.

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
- **Session 6 (main budget, 100k steps; all 20 checkpoints of the 12 chosen-configuration
  runs, `reports/session6/diagnostics/actions.json`):** the mode persists and remains
  BC-MLP's main loss.
  - **BC-MLP** (chosen trial 4, weight decay 0.1): 439 of 3,000 episodes fail, 349 of them
    `missed_grasp_then_drift` (69 stayed near, 21 gripper never closed). Motion dimensions
    are clipped on 17–78 % of the steps of failed episodes against ≤ 3.3 % in successes, and
    the raw normalized actions reach −28 to +41 in the drift episodes.
  - **BC-MLP-chunk** (chosen trial 2, cosine): 257 failures, 252 of them drift; motion
    clipping 33–89 % in failures; raw actions −51 to +71.
  - **BC-RNN:** 146 failures, mostly `missed_grasp_stayed_near` (97); motion clipping ≤ 4 %.
    **Diffusion:** 18 failures, 16 of them stayed near; no clipping (its actions stay within
    [−1, 1]).
  - Weight decay 0.1 was BC-MLP's chosen trial (49/50 at its selected checkpoint, seed 0;
    48 and 49 at seeds 1–2), but it did **not** remove the mode or measurably reduce it.
    Over all 20 checkpoints, trial 4 (seeds 0–2) fails 439 of 3,000 episodes (14.6 %) and
    trials 1–3 (seed 0) fail 445 of 3,000 (14.8 %). The 49 vs 47–48 at the selected
    checkpoints is a maximum over 20 checkpoints, within episode noise. BC-MLP stays below
    50/50 in every trial and seed (47–49/50).

### KI-022 — Gripper commands are clipped on 19–73 % of steps (BC), harmlessly
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
- **Session 6:** across all 480 evaluations the gripper is clipped on 26–67 % of steps per
  evaluation for BC-MLP, 27–73 % for BC-MLP-chunk and 19–56 % for BC-RNN. Diffusion never
  clips (0 %): its sampled actions stay within [−1, 1]. Motion clipping is reported apart
  (KI-021).

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
- **Diffusion (Session 4)**, final pilot checkpoint, `183b56c`, 1 thread, median of 100
  decisions:

  | Sampler | macOS native | container | ratio |
  |---|---|---|---|
  | DDIM-10 (primary) | 9.88 ms | 36.29 ms | 3.7× |
  | one U-Net forward | 0.97 ms | 3.47 ms | 3.6× |
  | DDPM-100 | 99.50 ms | 649.19 ms | 6.5× |

  Sources: `reports/session4/cost_latency/`. The pre-amendment measurement (untrained,
  `22adb8b`, 60 decisions) gave 42.39 ms and 436.42 ms in the container. Container
  run-to-run variance is therefore large for long sampler loops.
- The slowdown is much smaller than BC's 13–17×. A plausible reason, not isolated, is that
  the U-Net arithmetic outweighs the fixed per-call overhead that dominates the tiny BC
  networks.
- **Canonical evaluation cost:** a 50-start Lift validation evaluation of the diffusion pilot
  took 60–88 s of wall time (E4-10). With DDIM-10 that is comparable to BC.

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
- **Diffusion (Session 4)**, EMA network, 5k → 15k → 30k (E4-09):
  - the validation noise-prediction MSE is lowest at 15k (0.066 → 0.058 → 0.068);
  - the validation sampled-chunk MSE keeps falling (0.125 → 0.100 → 0.087);
  - closed-loop success is flat (49–50/50).

  The two offline metrics disagree with each other, as well as with the closed loop.
- **Session 6 (100k steps, chosen seed-0 runs, valid-demo MSE at 5k → 15k → 50k → 100k):**
  - BC-MLP (weight decay 0.1): 0.042 → 0.054 → 0.074 → 0.093 (lowest at 1k);
  - BC-MLP-chunk (cosine): 0.048 → 0.062 → 0.076 → 0.079 (lowest at 3k);
  - BC-RNN (weight decay 0.1): 0.034 → 0.044 → 0.053 → 0.056 (lowest at 5k);
  - diffusion (lr 3e-4), EMA: noise-prediction MSE 0.058 → 0.066 → 0.137 → 0.173 (lowest
    at 7k), while the sampled-chunk MSE falls 0.108 → 0.090 → 0.069 → 0.065.

  Training loss falls throughout. Closed-loop success peaks at 5k–15k for BC and is flat at
  49–50/50 for diffusion, whose §7 selections fall at 35k–100k. No offline metric entered
  any selection (§7.4).

### KI-025 — robomimic `d309eae` ignores `algo.unet.*` and always builds its 65.8 M U-Net
- `robomimic/algo/diffusion_policy.py` (lines 74–77) constructs
  `ConditionalUnet1D(input_dim, global_cond_dim)` with no other arguments. It therefore
  always builds the default widths (256, 512, 1024) with a 256-D step embedding. That is
  65.8 M parameters for Lift's 38-D condition, whatever `algo.unet.down_dims` or
  `diffusion_step_embed_dim` say in the config.
- **Impact:**
  - A robomimic config with a smaller U-Net does not train a smaller network. Any
    "robomimic-reproduced" diffusion number therefore uses the 65.8 M network.
  - RoboForge's primary configuration (1,092,551 parameters, RESEARCH_PROTOCOL §6.4) is a
    **deliberate deviation** in width, chosen for the capacity match. It does not reproduce
    robomimic's default size.
- **Mitigation in the reference comparison:**
  - robomimic's own `ConditionalUnet1D` is compared with RoboForge's, with identical weights,
    at three widths, including the default (256, 512, 1024);
  - robomimic's sampler and training step are run with a U-Net of the primary widths
    swapped into the algo, with its EMA rebuilt (`roboforge.models.reference.robomimic_algo`).

### KI-026 — NNPACK makes the 1D U-Net about 3× slower on macOS arm64
- Measured at `183b56c` on macOS arm64, 1 torch thread, with the primary config and batch
  100. The median of 20 training steps was:
  - 341.8 ms with torch's default backend (NNPACK enabled);
  - 111.4 ms and 110.6 ms with NNPACK disabled.

  Source: `logs/session4/dev/nnpack_timing.{py,log}`. A 30k-step training run was running
  concurrently.
- **Mitigation:** `roboforge.models.diffusion.unet_backend()` wraps every U-Net call in
  `torch.backends.nnpack.flags(enabled=False)`. The scope is limited to U-Net calls, so BC
  code paths are unaffected. Outputs are bitwise identical to the robomimic and diffusers
  references with the context active (`tests/unit/test_diffusion_reference.py`).
- **Not measured** inside the canonical container: there, the reference report and every
  container run already use the context.

### KI-027 — Diffusion samples depend on batch size at float-rounding level
- Sampling one row alone and sampling it inside a batch of 8, from the same initial noise,
  differ by up to 3.7e-6 in normalized units. This was measured with the primary config,
  untrained, DDIM-10 on macOS (`logs/session4/dev/batch_size_sampling.log`). The unit test
  bounds the difference at 1e-4. The convolution arithmetic depends on the batch shape.
- **Impact:** none on the closed loop, which always samples exactly one row per decision.
  The denoising export replays with batch 1 and is bitwise.
- **Offline metrics** batch the windows, so they are not bitwise comparable with
  per-decision samples. They are never used for selection (§7.4).
- Cross-platform sample equality is **not** claimed. Physics already differs across
  platforms (KI-013). The export is checked only on the recording platform.

### KI-028 — All diffusion pilot failures are one start, under one noise seed (R = 1)
- Four of the six pilot checkpoints fail exactly one episode: 5k, 20k, 25k and 30k. Each
  time it is `lift_validation_0029`, truncated at 400 steps (`reports/session4/validation/`).
  10k and 15k score 50/50.
- In Session 3, the BC families lost `lift_validation_0029` in only 1 of 18 evaluations.
- With R = 1 (RESEARCH_PROTOCOL §6.4), each (checkpoint, start) has a single noise seed. The
  seed is the same for every checkpoint by design (common random numbers). The start's
  difficulty and that seed's draw are therefore **confounded**: these records cannot tell
  whether the start is hard for diffusion or the seed is unlucky.
- The failure modes differ: missed grasp then drift at 5k and 25k; missed grasp, stayed near
  at 20k and 30k (`reports/session4/diagnostics/actions.json`).
- **Not a bug.** The denoising export of the 30k failure replays bitwise (E4-14).
- **Options before the freeze**, each only by an amendment written before the evidence:
  - repeats r ≥ 1 on the validation suite, to estimate per-start variance;
  - a diagnostic-only multi-seed rerun of this start.

  Neither may change selection retroactively.
- **Session 6** (120 evaluations, 6,000 episodes per family, all checkpoints;
  `logs/session6/analysis/per_start_failures.json`): `lift_validation_0029` failed

  | Family | Failures on 0029 (of 120) | Rate on 0029 | Rate over all starts |
  |---|---|---|---|
  | diffusion_unet | 15 | 12.5 % | 0.65 % (39 / 6,000; 7 distinct starts) |
  | BC-MLP-chunk | 43 | 35.8 % | 8.6 % (515 / 6,000) |
  | BC-MLP | 16 | 13.3 % | 14.7 % (884 / 6,000) |
  | BC-RNN | 3 | 2.5 % | 4.1 % (244 / 6,000) |

  - The start is hard for the two T_a = 8 families (about 4× and 19× their base rates), not
    for the single-step ones.
  - BC-MLP-chunk is deterministic, so for it the start itself is hard; no noise seed is
    involved. For diffusion, start and seed remain confounded under R = 1.
  - Descriptive only: never used for selection. The options above still apply only by an
    amendment before the freeze.

### KI-029 — Content-identical checkpoints can have different file SHA-256
- **Observed (Session 5):** a run interrupted at step 30 and resumed wrote a step-40
  checkpoint with a different SHA-256 from the uninterrupted run's step-40 file. Every tensor
  was bitwise equal and every other value equal (`payload_differences` empty).
- **Cause:** the zip member `data.pkl` differed in length (6,935 vs 6,978 bytes). The pickle
  inside `torch.save` memoizes repeated Python objects by identity. In the uninterrupted
  process, values such as the optimizer's `betas` tuple are the same objects as the
  config's. After `load_state_dict` they are separate objects, so they serialize differently.
- **Rule:** resume and transfer equivalence compares **content**
  (`roboforge.train.checkpoint.checkpoint_differences`). The file SHA-256 remains the
  integrity check for storage and download, and it is what manifests and selection record.
- **Impact:** none on results. A resumed run's checkpoint hashes are its own, recorded in its
  manifest like any other run's.

### KI-030 — Gate A probe lost every result to a BC-RNN CUDA device bug (FIXED, Session 5)
- **Observed:** probe launch 2 (L4, app `ap-YAO3zMqPWKlHWy5cyQ0kT2`, billed $0.0657, 184 s of
  L4) failed in the CUDA training profile of `bc_rnn`: `RuntimeError: Expected all tensors
  to be on the same device ... cuda:0 and cpu`. `BCRNN.encode` built its alignment index with
  `torch.arange(horizon)` on the CPU. The probe's sections ran in one call chain, so the one
  exception discarded every earlier result. The container log shows that the CPU profile of
  all four families, both render smokes and the CUDA profile of `bc_mlp` had run.
- **Fix:** the index is created on the mask's device (`af9c93a`). On the CPU it is the same
  tensor, so protocol runs and reference tests are unchanged (full suite passed). A new test
  runs BC-RNN on Apple MPS and compares it with the CPU output (the bug reproduced on MPS).
- **Probe hardening:** every probe section (CUDA info, EGL smoke, OSMesa smoke, each family's
  CPU and CUDA profile) runs separately. A failed section is recorded with its traceback,
  and the status becomes `partial`. Each section is also printed as one JSON line, so `modal
  app logs` keeps the evidence. A render smoke fails its section unless all checks passed.
- **Also fixed:** the launcher read `app.app_id` after the app had stopped. The ledger result
  of launch 2 therefore had no app ID; it was reconciled by hand from the launch log.
- **Earlier launch 1** (`ap-LyuJSmb9OLUAKPQ0oGZMQt`, billed $0.00001) failed before any job
  ran: `app.run()` looked up the Volumes before `_upload_data` created them (fixed in
  2477312).

### KI-031 — Modal reschedules an interrupted input despite retries=0; simulate_preemption bypasses SIGINT handlers (FIXED, Session 5)
- **Observed:** the Gate A training smoke (app `ap-ObjyqZEiECqNHIh4j3xt7n`, run
  `s5modal_diffusion_unet_lift_seed0_2fda84af`) ran 8 attempts in 12.5 min. Each attempt was
  interrupted about 80 s into training and recorded as `aborted` with no resume checkpoint.
  Each time, Modal started the same input again in a new container (AWS, GCP and Azure
  regions). It was stopped by hand with `modal app stop`.
- **Causes:**
  1. `modal.exception.simulate_preemption` does not send SIGINT. Its SIGALRM handler raises
     `KeyboardInterrupt` directly (modal 1.6.3, `exception.py`), so the job's SIGINT handler
     never ran.
  2. When the container exits on `KeyboardInterrupt`, Modal reschedules the input, although
     the function has `retries=0`.
  3. The plan re-armed the drill on every attempt, so the loop could not end on its own.
- **Fixes:**
  - The drill (`remote.arm_preemption_drill`) sends a real SIGINT, and SIGKILL 30 s later as
    in a Modal preemption. It arms on a launch's first attempt only.
  - Each launch has an attempt cap (`policy.max_attempts_per_launch: 4`). Past it, the job
    returns `status: error` without training, so no restart loop can run unbounded.
  - `stop_at_step` can end a launch cleanly, so that a separate resume launch has work to
    do.
  - `infra/modal_app.py cancel` passes `--yes` (it failed without a terminal).
  - The launch upper bound prices all 4 allowed attempts, each held for the full timeout
    (`db10a1b`). Before, one attempt was priced, while smoke 1 ran 8 attempts against a
    $0.11 bound.
- **Retry semantics now:**
  - a function failure is not retried (`retries=0`);
  - an interrupt or preemption reschedule runs at most 4 attempts per launch;
  - `nonpreemptible=True` would avoid preemptions, at 3× the price.

  The gate is client-side. Only a Modal workspace budget is a hard spending cap.
  - `tests/integration/test_remote_job.py` covers the drill, the restart, the cap and the
    paths.
- **What the incident did prove:** every restart verified the data hash and resumed under
  the same launch ID from the indexed step-500 checkpoint. It dropped the metric rows past
  that point and recorded the segment. The checkpoint at step 1000 was renamed into place
  but never indexed, the torn case that orphan quarantine handles.
- **Run status:** `s5modal_diffusion_unet_lift_seed0_2fda84af` is abandoned and never used
  as evidence. Its journal is kept in `logs/session5/modal/abandoned_<run_id>/`; it records
  absolute paths (KI-032).

### KI-032 — Paths under a symlinked mount were recorded absolute (FIXED, Session 5)
- **Observed:** on Modal, `/app/runs` is a symlink to `/__modal/volumes/<id>`.
  `repo_relative` resolved symlinks first, so the journal recorded
  `/__modal/volumes/vo-…/<run_id>/checkpoints/…` instead of `runs/<run_id>/…`. Such paths do
  not resolve after download.
- **Fix:** `repo_relative` tries the lexical (unresolved) path first and falls back to the
  resolved one. The local simulation now mounts its runs directory through a symlink too,
  and the new remote-job test asserts repo-relative paths.

### KI-033 — The Modal image carries no evaluation suite, so the probe's render check never rendered (OPEN: EGL vs OSMesa pending)
- **Observed:** in the Gate A probe retry (app `ap-WKAnfgdtYEriXZDg6FtoCC`) both render
  sections failed before rendering:
  `FileNotFoundError: /app/data/eval_states/lift/validation.json`. `smoke_sim.py` loads
  the validation suite. The image mounts only `src`, `scripts`, `configs` and
  `data/{manifests,splits,normalization}`, which is deliberate: no evaluation state goes to
  Modal.
- **Consequence:** EGL vs OSMesa on a Modal GPU is still **unmeasured** (KI-001). The
  approved retry was spent, and no further launch was made.
- **Fix, ready for a future approved probe:** `scripts/render_probe.py` renders the task
  offscreen with no dataset and no evaluation suite. The probe now uses it.
  - Verified with OSMesa and EGL in the canonical container, and with CGL on macOS.
  - `tests/simulator/test_render_probe.py` runs it from a copy holding only the image's
    code directories.
- **Impact:** none on Level 1. Evaluation and rendering run on the canonical container,
  where OSMesa is verified. RGB extraction (Session 8) can stay local.

### KI-034 — Lift validation is saturated: most §7 choices are decided by the first-success tiebreak
- **Observed** (Session 6 main suite, `reports/session6/{selection,tuning}/`):
  - 12 of the 16 seed-0 selected checkpoints score 50/50; every diffusion, BC-MLP-chunk and
    BC-RNN trial does. Only BC-MLP's four trials stay below 50 (47–49/50).
  - The Wilson 95 % interval of a 50/50 checkpoint is [0.929, 1.000]. On success alone,
    Lift validation cannot separate those three families or their trials.
  - In 3 of 4 families the §6.3 choice was therefore decided by the mean first-success
    step. The chosen trial led the runner-up by only 0.12–0.68 control steps (20 Hz):

    | Family | Trials at the top success count | Chosen | Mean first-success step of the tied trials |
    |---|---|---|---|
    | diffusion_unet | 4 (50/50) | trial 3 | 45.54 vs 45.76, 46.56, 46.92 |
    | bc_mlp_chunk | 4 (50/50) | trial 2 | 44.62 vs 45.30, 45.64, 50.26 |
    | bc_rnn | 4 (50/50) | trial 4 | 43.10 vs 43.22, 43.40, 43.50 |
    | bc_mlp | 1 (49/50) | trial 4 | none tied (47, 48, 47 for trials 1–3) |

  - The **§7 checkpoint choice** within a run is mostly a first-success tiebreak too.
- **This is the predeclared rule working as written**, not a defect. The rule and its
  tiebreaks were fixed before any result (RESEARCH_PROTOCOL §7, §6.3).
- **But:** differences this small in mean first-success step are within the
  episode-to-episode spread. Do not read a chosen trial as better than a tied
  alternative. The choice is a deterministic way to pick one configuration, not evidence of
  a better one.
- **BC selections sit at the earliest checkpoints.** The three BC families select at
  5k–15k in 17 of 18 runs (5k in 8). The exception is BC-MLP trial 3 (lr 3e-4), selected at
  100k. Success declines slowly after the early checkpoints while offline validation MSE
  rises (KI-024). The 5k cadence cannot show whether BC peaks before 5k; finding out would
  need an amendment.
- **Diffusion selects late (35k–100k), with success flat at 49–50/50.** The chosen seed-0
  run (trial 3, lr 3e-4) selected its **final** checkpoint, 100k, so that selection sits at
  the budget boundary. Its seeds 1 and 2 selected 95k and 80k. A longer budget might move
  the selection further; the protocol does not allow testing that in Level 1.
- **Implication for Session 7:**
  - the paired Lift test will likely not separate the 50/50 families on success. Its
    preregistered analysis should be written knowing that, as a paired comparison with CIs
    that can be reported as "not separated", not as a ranking;
  - mean first-success step and per-start failure patterns are secondary descriptors, and
    any use as an outcome must be declared in the freeze;
  - Can (harder) is where success differences are more likely to show.
  - Any change of R (KI-028), or any added validation statistic, needs an amendment written
    before the freeze. It may not change Session 6 selections retroactively.

- **Session 8 (RGB study):** the RGB arms saturate Lift too.
  - Every RGB run reaches 50/50 on validation.
  - Selection falls to the first-success tiebreak (two BC runs select step 5000).
  - Held-out success is 99.3–100 %. The diffusion − BC interval [+0.0, +4.0] is "not
    separated" (E8-15).

### KI-035 — Host contention during the Session 6 suite inflated wall times (results unaffected)
- **Observed:** the suite itself ran up to 10 jobs at once (peak from its event log; limits
  8 training runs, 4 evaluation containers and 10 jobs in total, D-054) on an M1 Pro with 8 performance and 2
  efficiency cores.
  For most of the suite a CPU-bound process from another project also ran, and the machine
  was swapping (20.0 of 21.5 GB of swap in use at the end).
  - **Training:** the four diffusion trials took 5.10–5.12 h of wall time per 100k steps
    (4.4 CPU-h each), against 3.9 h projected from the Session 5 concurrency profile
    (7.5 steps/s alone, 1.24× slowdown at 8 processes). The two diffusion replicates (same code and
    model, trial 3 with seeds 1 and 2; 12:58–16:22 UTC) ran with fewer neighbours: **3.39 h**
    each (3.36 CPU-h), faster than the projection. The foreign process's end time was not
    recorded; it was no longer running at a check at about 15:43 UTC. BC-RNN took
    1.42–1.45 h for the trials and 0.89 h for the replicates.
  - **Evaluation:** per-container throughput, in control steps per second (which removes
    the episode-length effect of failed episodes running to 400 steps), fell 1.6–3.1×
    against the Session 3–4 evaluations:

    | Family | Sessions 3–4 (steps/s, 6 evaluations) | Session 6 (steps/s, 120 evaluations) | Ratio |
    |---|---|---|---|
    | BC-MLP | 56.9 | 18.4 | 3.1× |
    | BC-MLP-chunk | 54.2 | 25.8 | 2.1× |
    | BC-RNN | 35.2 | 21.4 | 1.6× |
    | diffusion_unet | 44.3 | 19.1 | 2.3× |

    The Session 5 parallel profile alone predicts about 4 / 1.78 = 2.25× per container at
    4 parallel containers (E5-10), before any training load. The 480 evaluations took
    26.92 container-hours in all (mean 202 s, median 171 s per 50-start evaluation).
  - The suite's own concurrency and the foreign process were **not separated**. The ratios
    above combine both.
- **Mitigation (scheduling only):** `logs/session6/suite/renice_loop.sh` set every BC
  training process to nice 15 every 60 s, so the diffusion runs on the critical path won CPU
  contention. Each change is logged in `logs/session6/suite/renice.log`. The suite's processes
  had inherited nice 5. One early manual `renice -n 15` is an increment on macOS: it put the
  BC-MLP trials at nice 20 (logged), and the loop uses the absolute form. No run was stopped
  or relaunched.
- **Why results are unaffected:**
  - training is deterministic on CPU with 1 thread, and scheduling does not enter it;
  - trial 1's 5k–30k checkpoints are content-identical to the Session 3–4 pilots
    (`logs/session6/analysis/pilot_prefix_equivalence.json`);
  - their 24 Session 6 evaluations reproduce the pilots' episode for episode, including
    every `physics_sha256` (`logs/session6/analysis/pilot_eval_reproduction.json`);
  - evaluation physics does not depend on wall time.
- **Impact:** only wall-time and throughput numbers. The training steps/s recorded in the
  Session 6 manifests (and repeated in `reports/session6/cost_latency/`) are contended
  numbers, not a clean throughput profile. COMPUTE_PLAN projections for Session 7 should use
  the Session 5 profiles, scaled by the factors above if the host is shared again.

### KI-036 — The Can suite orchestrator crashed on a half-written evaluation summary (FIXED, D-070)
- **Observed:** `scripts/run_suite.py` exited at 2026-11-08 02:14:19 UTC with a
  `JSONDecodeError` in `eval_summary`. It had read an empty summary of one of the four
  BC-MLP trial-3 (`lr3e4`) evaluations, which the containers wrote at 02:14:18–02:16:17 UTC.
  `eval_policy.py` writes the summary with a plain `write_text`, and the container writes it
  through the bind mount, so a poll can see the file before its content.
- **Why it went unnoticed for about 3 h:** only `SuiteError` was caught, so neither
  `suite_error` nor `end` was written, and the watcher waited for those events. The disk
  guard and the renice loop stop when the suite process exits, and both stopped.
- **What ran unsupervised:** the jobs were started in their own sessions and kept running.
  - The six training runs finished with manifest status `complete`: BC-RNN trials 3–4 at
    02:58:59 and 02:59:31 UTC, the four diffusion trials at 04:15:08–04:15:44 UTC.
  - The four `lr3e4` evaluations (steps 5k–20k) finished with audit PASS and 50 episodes each.
  - These 10 jobs have no `finished` event. Their manifests and summaries are the record.
  - Nothing was relaunched or lost. All 44 evaluation summaries parse, and each
    `runs/*__val_*` directory has its summary.
- **Impact:** about 3 h of idle evaluation capacity (02:14 UTC to the restart). No result is
  affected. Without the renice loop, the diffusion runs finished sooner than projected
  because no evaluations competed with them for CPU. Scheduling does not enter the
  deterministic training (KI-035).
- **Fix:** D-070 (reader side only). Watchers now also check that the suite process is alive.

### KI-037 — The robustness `--check` called equal Can manifests "not bitwise" (FIXED, Session 7)
- **Observed:** the first container run of `generate_robustness_states.py --check` (2026-11-08,
  on 5d9b8c0) failed `can/ood_arm_offset` as "not bitwise reproducible on its own platform".
  Its states were identical (max |Δ| 0.0), the model XML equal and the data file intact;
  every Lift set and `can/robustness_base` were bitwise equal.
- **Cause:** a comparison bug, not a reproducibility one. `check_task` compared the committed
  manifest (read from JSON, so lists) with the in-memory one, where each validity record's
  `contact_pairs` entries are tuples. A tuple never equals a list. Only Can's arm-offset
  states record contact pairs (the can resting on `bin1`), so only that set failed. The field
  diff is in `logs/session7/robustness/diff_manifest_can.log`: 50 differences, every one a
  tuple-versus-list `contact_pairs` entry.
- **Fix:** `robustness.same_manifest` compares after the same JSON encoding as the stored file,
  and a unit test covers the tuple case. The committed sets were not regenerated or changed.
- **Impact:** none on any state or result. No robustness evaluation had started.

### KI-038 — `analyze_robustness.py` read the checkpoint step from the wrong field (FIXED, Session 7)

- **Observed (2026-11-08 ~19:20 UTC):** after the robustness, A1 and A2 evaluations ended,
  `scripts/analyze_robustness.py` stopped at its first summary with `KeyError: 'step'`.
  Nothing was written.
- **Cause:** the loader keyed each summary on `s["step"]`. The writer
  (`roboforge.eval.validation`) stores the step as `checkpoint.step`. The script had no test,
  so the mismatch first showed on real summaries.
- **Fix:** read `s["checkpoint"]["step"]`. A unit test (`tests/unit/test_analyze_robustness.py`)
  builds summaries in the writer's layout and checks the keys, the duplicate check and the
  canonical check. It fails on the old line and passes on the fix. The A3 suite was still
  running, so the tracked file stayed unchanged until it exited. The fixed copy was first run
  from the scratchpad to check that the analysis completes, and those outputs were then
  replaced by the run from the committed fix.
- **Impact:** none on any evaluation or state. The analysis reads summaries and never rolls
  out, and the script is not part of the frozen held-out code.

### KI-039 — A simulator test would have built a held-out start once the lock opened (FIXED before the freeze commit, Session 7)
- **Observed (2026-11-08, while reviewing the freeze commit):** the tests that depend on the
  held-out lock were checked before committing `docs/FINAL_TEST_FREEZE.md`.
  `tests/simulator/test_env_adapter.py::test_seed_resets_and_the_final_test_seed_lock` asserted
  that `env.reset(900003)`, a seed in the held-out block, is refused. That holds only while the
  lock is shut.
- **Cause:** the test used the real lock state. With the freeze committed, the lock is open, so
  `make test` would have built a held-out start on macOS (and then failed). That reads the test
  set outside the frozen path.
- **Fix (in the freeze commit `bf0a0fa`, D-072):** the test points the lock at a missing freeze
  through `suites_config`, as the post-freeze lock test does. It asserts that the lock is shut,
  then expects the refusal. Searching `tests/` for `lock_status`, `FinalTestLockedError`,
  `LOCKED_SUITE`, `final_test`, `FINAL_TEST_FREEZE`, `load_freeze` and both held-out manifest
  hashes found no other test that depends on the real lock state. The others use temporary or
  missing freezes, or only check suite names.
- **Impact:** none. The test was changed before the lock opened, and `make test` at `bf0a0fa`
  (372 passed) built no held-out start.

### KI-040 — `make smoke-sim` built Lift held-out start 0 in memory once the lock had opened (FIXED, Session 8)
- **Observed (2026-11-09 00:48 UTC, Session 8 prerequisites):** the native `make smoke-sim
  SMOKE_OUT=logs/session8/prereq` failed one check, `adapter_reserved_test_seed_locked`
  (`logs/session8/prereq/code_checks.log`, `logs/session8/prereq/smoke_sim_lift.json`).
- **Cause:** `scripts/smoke_sim.py` (Session 2) asserted the seed lock by calling
  `env.reset(900000)` and expecting `FinalTestLockedError`. Since the Level 1 freeze was
  committed (`bf0a0fa`) the real lock is open on a host with a clean freeze file, so the call
  succeeded: `RoboForgeEnv.reset` generated the snapshot of seed 900000 (Lift held-out start 0,
  `random.seed`/`np.random.seed` then `env.reset()`) and restored it into the simulator. This is
  the KI-039 defect in a script instead of a test; the Session 7 STOP did not re-run
  `make smoke-sim` after the freeze, so it was not caught there.
- **What happened and what did not:**
  - Native, Lift only: the state existed in memory for the rest of `adapter_smoke`, which then
    closed the env. No policy acted, no step, render, feature, hash or value of that state was
    computed or written; the report records only the boolean check (`false`). The Can native
    run never started (the Make loop stops at the first failure).
  - Container, Lift and Can: the check passed only because the working tree was dirty
    (untracked Session 8 files), so the wrapper attested `ROBOFORGE_GIT_DIRTY=1` and the lock
    stayed shut inside the container. With a clean tree the container would have built the
    held-out starts too.
  - No Level 1 result, selection, label or finding depends on this; the held-out evaluations
    ran in Session 7 from the frozen path only. Nothing of the held-out start entered any
    Session 8 artifact.
- **Fix (Session 8):** the check now points `suites.yaml`'s lock at a missing freeze
  (`docs/NO_SUCH_FREEZE.md`) for the duration of the call and resets only after seeing the lock
  shut, the same rule as the KI-039 test; it also records whether the real lock is open
  (`real_heldout_lock_open`). `tests/simulator/test_smoke_sim_lock.py` replaces the env's
  snapshot generator with one that raises on any held-out seed and asserts the smoke check passes
  without generating anything; the pre-fix script fails that guard (verified on a copy of
  `HEAD:scripts/smoke_sim.py`). A sweep of `scripts/`, `src/` and `tests/` for the held-out
  seed block and suite name found no other caller outside the frozen held-out path, the suite
  module and the tests that point the lock at temporary or missing freezes.
- **Rule for later sessions:** with the Level 1 lock open, no tool may test the lock against the
  real freeze. The lock is re-gated on `docs/RGB_EXPERIMENT_FREEZE.md` (HANDOFF) before any RGB
  held-out work.

### KI-041 — The first render of a fresh CGL context differs from later renders by a few pixels (MITIGATED, Session 8)
- **Observed (macOS arm64, `MUJOCO_GL=cgl`, Session 8):** rendering one state several times with
  a fresh `mujoco.Renderer`, the first frame differs from every later frame on 1–6 pixels by one
  grey level (Can agentview 4–6 px, Can wrist 1–6 px, Lift ≤ 1 px); frames 2, 3, … are bitwise
  equal. The artifact belongs to the GL context (any camera, any state); warming one size's
  context does not warm another's (`reports/session8/rgb_feasibility/darwin-arm64-native/*_invariants.json`,
  `first_render`).
- **Canonical container (OSMesa):** 0 px on every checked context, both tasks
  (`linux-aarch64-container-1f685638f28c/*_invariants.json`).
- **Why it matters:** `ShadowRenderer` creates a context per model and size, and every episode
  restore compiles a new model, so without a warm-up the first frame of each episode (and of each
  dataset demo with a new model) would differ from steady-state frames.
- **Mitigation (D-074):** every kept RGB frame is a steady-state frame: each new context renders
  once and discards it (`roboforge.data.rgb.WARM_UP_RENDERS = 1`). The closed-loop RGB observation
  path must do the same. Native frames are development-only anyway (CGL vs OSMesa differ by
  0.6–3.1 mean grey levels, E8-04).

### KI-042 Frozen `DiffusionPolicy.sample` cannot run on CUDA (LOW, OPEN; Session 8)
- **Observed** (code reading while building the Gate D probe, 2026-11-09):
  `src/roboforge/models/diffusion.py` (frozen at `bf0a0fa`) draws the initial noise of
  `sample` with `torch.randn(shape, generator=generator, dtype=...)`. That call takes no
  device, so it is always on the CPU. Each U-Net call also builds its timestep tensor on the
  CPU.
- **Consequence:** with a CUDA model, sampling mixes devices and fails. Training
  (`training_loss`) is unaffected: it draws noise and timesteps on the data's device, and it
  ran on the L4 (E8-06).
- **Current handling:**
  - No rollout needs CUDA. Every reported rollout runs on the canonical CPU container, and a
    checkpoint trained on a GPU loads there on the CPU.
  - The RGB probe does not time inference on CUDA.
  - Any GPU-side offline metric that samples, such as validation action MSE, must move the
    model to the CPU first, or use a new, non-frozen sampler wrapper. Level 1's file stays
    unchanged.

### KI-043 Concurrent container build and full test suite stall the 16 GB host (LOW, MITIGATED; Session 8)
- **Observed 2026-11-09, 03:10–03:52 UTC.** `make rgb-dataset` (Docker VM up to 7.7 GiB) and
  the full `pytest` suite ran natively at the same time.
- **Effect:**
  - host memory was exhausted: 131 MB free, 6.7 GB in the compressor, heavy swapping, load
    average 35;
  - the build stopped progressing for about 40 min, at 120 of 200 demos. It took 73 min of wall
    time, while the container's monotonic clock counted only 26 min, so the VM was starved;
  - the suite ran 33 min instead of about 7.5.
- **No result was affected.** Rendering is a pure function of the state, and the build
  re-reads and hash-checks its output.
- **Handling:**
  - the test-suite run was stopped (not counted as a test result);
  - the build resumed on its own;
  - from now on the full suite never runs alongside a container job or a large data build.

### KI-044 A Modal preemption restarted one RGB resume launch (LOW, MITIGATED; Session 8)
- **Observed:** Modal preempted diffusion s2's resume launch at about 11:44 UTC on 2026-11-09
  (`Container terminated due to preemption`, `logs/session8/rgb_train/s8_rgb_diffusion_resnet18_lift_seed2_ec213b6a_launch1.log`
  line 98). The functions are `preemptible: true` and cheaper that way.
- **Handled by design:**
  - The function retried as attempt 1 of the same launch. Attempt 0 wrote no status, and its
    `segments.jsonl` line was committed before any work.
  - Attempt 1 resumed from the atomic `resume/latest.pt` at step 45,000 and finished.
  - No cadence checkpoint was orphaned.
  - Resume equals uninterrupted is a tested property (`tests/unit/test_rgb_train.py`).
- **Consequence:** the run has 3 segments instead of 2, and the minutes lost between 45,000
  and the preemption were billed and are included in the $8.06. E8-10 originally said every
  run had 2 segments; the correction is in E8-10.
- **Rule:** long GPU jobs keep the per-attempt marker before any work and an atomic resume
  file. Reports must count segments from `segments.jsonl`, not from launches.

### KI-045 Native macOS (CGL) depth lacks reversed-Z precision (LOW, MITIGATED; Session 9)
- **Observed:** on the native macOS backend MuJoCo warns `ARB_clip_control unavailable while
  mjDEPTH_ZEROFAR requested, depth accuracy will be limited`. CGL's OpenGL lacks the
  extension, so the reversed-Z depth buffer loses precision with distance.
- **Mitigation:** every Level 2 depth, cloud, dataset and evaluation that counts comes from
  the canonical container (OSMesa). There, depth matches exact `mj_ray` to 3.6 µm on 358,383
  interior pixels (`reports/session9/pc_feasibility/linux-aarch64-container-1f685638f28c/`).
  Native depth is development-only, as native rollouts already are (D-017).
- **Rule:** never report or pool native depth or clouds. `make pc-feasibility-native` labels
  its output with the native platform ID.

### KI-046 robosuite's hidden sites still write depth (MEDIUM, FIXED; Session 9)
- **Observed:** robosuite hides visualization sites by setting non-positive alpha, so they are
  invisible in RGB. MuJoCo's depth buffer still records them. In Lift, the
  `gripper0_right_grip_site` sphere and the 10 m `grip_site_cylinder` appeared in the wrist
  camera's depth, about 0.2 m off the true surface on 80 pixels (segmentation type 6, sites 3
  and 7). Found by the depth-vs-ray check before any cloud was built.
- **Fix (D-083):** the depth renderer sets `opt.sitegroup[:] = 0`, and the diagnostic
  segmentation renders with the same options. The invariant `sites_in_depth` and
  `tests/simulator/test_pointcloud_alignment.py` keep it fixed.
- **Rule:** any new depth path must hide sites, or prove they are absent, before its clouds
  are used.

### KI-047 A client network drop stopped the first Level 2 launch (LOW, MITIGATED; Session 9)
- **Observed:**
  - seed 0's first launch waited about 40 min for an L4 (unbilled);
  - it then trained from 16:40:55 UTC;
  - at 16:57:10 UTC the host lost DNS (`ConnectionError: [Errno 8] nodename nor servname
    provided`), and the client recorded `status: error` (`logs/session9/pc_train/queue.jsonl`,
    `launch_records/`);
  - the app is ephemeral, so Modal stopped it about 90 s later, just after the step-25,000
    checkpoint and resume file were written; the run manifest was left `running`.
- **Handled by design:**
  - the atomic `resume/latest.pt` holds step 25,000 (verified);
  - the queue's next launch is a `resume` from it, which A-04 covers;
  - the 16:00 UTC hour was billed at $0.329 and recorded against the launch, because a
    launch without a measured result counts at its $4.54 upper bound.
- **Second loss:** the resume launch that followed queued for an L4 from 17:00 UTC. Its
  container started at 17:22:49 and resumed from step 25,000. About 20 s later the client lost
  its connection (`StreamTerminatedError`), and the app stopped before any step was saved
  ($0.0043 billed).
- **Third launch:** resumed again from the same step-25,000 file and completed 60,000 steps
  (42.2 ms/step). The run therefore has 3 segments; two of them have no end record.
- **Fix:** D-085 makes launches detached and single-flight, so a client loss no longer stops
  the job.
- **Side effect:** the restarted queue numbers launches from 0 again, so it overwrote the
  first launch's streamed log (`*_launch0.log`). Its plan, result, ledger events and the run
  manifest survive. Before any restart, copy the launch logs aside.
- **Rule:** keep the host online for the whole of a launch. A launch that ends `error` on the
  client is checked on the Volume (status, checkpoints, resume step) before any resume.

### KI-048 The Level 2 gate could not see Session 8 evidence inside the container (LOW, FIXED; Session 9)
- **Observed:** the first 3 `make pc-heldout` attempts (from `4ebebc1`, clean) were refused by
  the D-086 gate inside the container. They stopped before any held-out state was read and
  wrote nothing (`logs/session9/pc_heldout/`).
- **Cause:** the frozen wrapper `infra/containers/run_sim_cpu.sh` mounts only
  `reports/session<N>`, here 9. The gate's Session 8 checks therefore could not pass in a
  Session 9 container: the RGB gate (D-081) and the 3 reused RGB held-out reports.
- **Fix (D-086 amendment, `08c3aa8`):**
  - those checks run on the host, where git also verifies the files;
  - `make pc-heldout` runs the full gate on the host first;
  - the container gate checks everything else;
  - a test pins the split and the order.
- **Rule:** a gate that pins an earlier session's evidence must check it where that evidence
  is mounted.

### KI-049 One generated training row of the Level 2 analysis was miscomputed (LOW, DOCUMENTED; Session 9)
- **Observed:** `reports/session9/level2_results/` shows seed 0 at 24.62 ms/step and 0.41 h.
  `_training` read the resume step from the wrong key and counted two unfinished segments
  (KI-047).
- **Correct values:** 42.2 ms/step. The total training time is 0.70 h, including the cut first
  segment; the completed segment alone is 0.41 h, which is what the fixed script reports
  (`reports/session9/level2_results/ERRATA.md`). No success, interval, separation or taxonomy
  count is affected.
- **Handling:** the analysis is never rerun or overwritten. The erratum sits beside it, and
  the script is fixed for the record.

### KI-050 The first Observatory export stopped on the random baseline; its reference masks counted sites (LOW, FIXED; Session 10)
- **Stop:** `require_provenance` read `manifest.get("checkpoint", {})`. The random baseline's
  manifest has `checkpoint: null`, so the call returned `None` and the first export failed at
  clip 19 of 37 (`can-random_uniform-s0-can_final_test_0000`). It had written 18 clips and no
  index. The unit test used `checkpoint: {}` and missed it.
- **Fix:** a missing checkpoint is accepted only when the record says the policy is
  non-learned (`learned: false`). A test covers `checkpoint: null` for both cases.
- **Masks:** the viewer-verification reference masks came from MuJoCo segmentation with sites
  drawn. robosuite's transparent `grip_site_cylinder` (10 m long) covered robot pixels, which
  the mask then counted as background (as KI-046 found for depth). The masks now hide sites
  (D-091), with a render test.
- **Handling:** the partial export is discarded and the export reruns from one clean commit
  (D-091). Before any long container run, a development pass must cover one clip of every
  kind, including the non-learned baseline.
- **Follow-up:** the first mask fix edited the frozen `roboforge.sim.shadow_render`
  (FINAL_TEST_FREEZE), and `make freeze-check` failed at STOP. The file is restored, and the
  fix moved to `roboforge.viz.obs_media` (D-091 amendment). A development re-render showed
  the committed reference frames are byte-identical (E10-08).

### KI-051 Observatory selection code named the locked suite; the lock test failed at STOP (LOW, FIXED; Session 10)
- **Failure:** the first complete full-suite run at STOP gave 491 passed and 1 failed. The
  failing test was `tests/research_integrity/test_final_test_lock.py::test_no_other_code_names_the_locked_suite`.
  `src/roboforge/viz/obs_select.py` (from `d12729f`) wrote the held-out partition label as a
  string literal. Only the suite module and its generator may name the locked suite.
- **Why it was late:** no full-suite run on Session 10 code finished before STOP (one was
  stopped so the final code could be tested instead), so the failure first showed at STOP.
- **Fix (`4348244`):** the label is read from `configs/eval/suites.yaml` through
  `eval_states.LOCKED_SUITE`. The value is unchanged, and `build_selection()` reproduces the
  committed `selection.json` exactly (apart from its git and time stamps), so no evidence is
  regenerated. The frozen test was not edited.
- **Handling:** for a session's own new code, run the research-integrity tests
  (`tests/research_integrity/`) before the first evidence export, not only at STOP.

### KI-052 The diffusion sampler bounds its actions to the training range; BC heads are not (MEDIUM, DOCUMENTED; Session 11)
- **Mechanism:** the samplers' shared `_Sampler._x0` (`src/roboforge/models/noise_schedule.py`) clamps the predicted clean chunk to [−1, 1] in min–max
  normalized space (`clip_sample=True`) and the last DDIM step returns it, so every diffusion
  action lies inside each dimension's training range. BC outputs are clipped only to the
  controller limits (`ActionCodec.decode`). Example: Lift drot_x training range
  [−0.151, 0.119] vs ±0.5 rad for BC.
- **Evidence (audit A-6, scan of `runs/` arrays):** diffusion executes 0.000 out-of-range motion
  actions on every set; BC-MLP-chunk 0.53–0.54 and BC-MLP 0.64–0.71 on the Lift OOD sets, almost
  all in failed episodes. Correlational: the out-of-range actions start after a missed grasp.
- **Effect:** H2-c and the H4 between-family gaps compare the generative head together with this
  bound; neither isolates the generative model. The measured differences stand.
- **Handling:** disclosed in the paper (§4, §6, §7, §9). An evaluation-only clamp test of the BC
  checkpoints would resolve it; it needs its own decision before any rollout (D-096).

### KI-053 Several separations depend on the interval method (MEDIUM, DOCUMENTED; Session 11)
- **Evidence (audit B-1 to B-4):** with 3 training seeds, Can H2 and H3 are separated by the
  preregistered bootstrap but not by Welch, variance-components or Bonferroni intervals; 6 of 41
  robustness verdicts and 3 of 16 Level 1 verdicts change under the seed-level t analysis
  (D-095); two verdicts change with the bootstrap seed (tails 0.0264 and 0.0237); 10 of the 64
  declared success differences have a frozen bound at exactly zero or zero width.
- **Handling:** the frozen verdicts stay the reported result. The paper reports the D-095
  sensitivity analysis beside every declared difference, labelled not preregistered, and
  calls the Can separations fragile.

### KI-054 Declared secondary metrics were never aggregated (MEDIUM, FIXED by reporting; Session 11)
- RESEARCH_PROTOCOL §8 declares gripper sign changes and action smoothness; every held-out
  summary computes them per episode, but no frozen result aggregates them, and
  FINAL_TEST_FREEZE §4 dropped them without an amendment. Plan diversity was measured only for
  the Session 4 pilot; point-cloud throughput is absent from the Level 2 results.
- **Fix (D-096):** the paper's Table 4 reports plain means from the 26 hash-chained held-out
  summaries, labelled "declared secondary, aggregated after the frozen analysis"; the two
  missing items are listed as declared, not reported.

### KI-055 Two analysis scripts lacked safety checks (LOW, FIXED; Session 11)
- `scripts/analyze_robustness.py` could overwrite an existing `analysis.json` in the same
  evidence session; it now refuses (test
  `tests/unit/test_analyze_robustness.py::test_main_refuses_to_overwrite_an_existing_analysis`).
- `scripts/analyze_level2.py` hashed an RGB held-out report before checking it exists; it now
  checks existence first. Neither script is hash-frozen; `make freeze-check` passed before and
  after. No output was regenerated.

### KI-056 One RGB run's training hours omit a preempted attempt (LOW, DOCUMENTED; Session 11)
- RGB diffusion seed 2: segment 2 (launch `116f287e`, attempt 0) was preempted (KI-044) and has
  no `train_wall_s`, so the 1.29 h in `reports/session8/rgb_results/results.json` omits about
  0.1–0.2 h; the billed $1.60 includes it. The "77.9 ms/step" of E8-10 is the steady-state rate;
  segment wall time over steps is 79.5–84.1 ms.
- **Handling:** footnote b of the paper's cost table (generated rule: more segments than
  launches). The frozen results are not edited.

### KI-057 Governance record defects (LOW, DOCUMENTED; Session 11)
- **A-01 launch count (audit C-5):** A-01 authorized one GPU probe; two probe launches ran
  (`5688db17`, failed in 5 s, $0.00001; `4489c09f`, $0.0657) before the 00:20 EST amendment
  authorized a retry. The launch guard checks function and dollar cap, not launch counts
  (D-049). The cap was never at risk.
- **Taxonomy hash wording (audit C-21):** FINAL_TEST_FREEZE gives `5a5d998f…` as the "SHA-256"
  of `configs/eval/failure_taxonomy.yaml`; it is the canonical-JSON content hash from
  `taxonomy_sha256()`. The file bytes hash to `e9aea501…`. Both are consistent; the freeze is
  frozen, so only this note records it.
- **Record order (audit C-20):** D-086's amendment time and E9-11's rehearsal order were wrong;
  both are annotated in place.

### KI-058 Trainer tests needed processed data but did not skip without it (LOW, FIXED; Session 12)
- **Found by** the Session 12 fresh-clone reproduction (D-098): before `make download`, the
  CI test selection failed 9 tests (7 errors, 2 failures) in `tests/unit/test_rgb_train.py`,
  `test_pc_train.py` and `test_rgb_train_plan.py` with `FileNotFoundError` on
  `data/processed/robomimic_v1.5_ph/lift/lowdim_v1.hdf5`. The README promised that dataset
  tests skip with a reason.
- **Fix** (`d9da1dd`): a session-scoped fixture `lift_processed_path` in `tests/conftest.py`
  skips them with the reason. Test infrastructure only; in the working repository, which has the
  data, all 16 tests of these files still run and pass.

### KI-059 Gate tests looked for runs/, which a fresh clone has (LOW, FIXED; Session 12)
- **Found by** the fresh clone: 4 tests of `tests/research_integrity/` failed with
  `FileNotFoundError`. `runs/` is gitignored except `runs/README.md`, so the existing
  `if not runs.is_dir(): pytest.skip(...)` never fired; and the frozen RGB and Level 2 evaluators
  read the training record before the platform check, so the "canonical platform" assertion met
  a missing file.
- **Fix** (`bd3bf6b`): each test skips only its record-dependent part when the specific frozen
  record is absent. The gate-shut and host-refusal assertions still run everywhere; with the
  records present (the working repository) all 45 research-integrity tests run and pass. The
  frozen evaluators are unchanged (`make freeze-check`, both gates open).

### KI-060 Published advisories in two pinned dependencies (LOW, ACCEPTED; Session 12)
- `pip-audit` over the 72 PyPI packages of `uv.lock` (`reports/session12/pip_audit.json`) lists
  29 advisory IDs in **diffusers 0.11.1** (remote code through `from_pretrained`) and
  **transformers 4.41.2** (deserialization of untrusted checkpoints, ReDoS in tokenizers). Both
  versions are pinned by robomimic `d309eae`. RoboForge uses diffusers only for the scheduler
  classes in the reference-equivalence check (`src/roboforge/models/reference.py`, never
  `from_pretrained`) and never imports transformers; no RoboForge path loads a model, tokenizer
  or pipeline from an untrusted source.
- **Not upgraded:** `uv.lock` is the frozen environment record and a hash-pinned paper source
  (D-094); changing it would need a new decision and a re-verification of the reference
  equivalence. Anyone reusing this environment for other work should upgrade.
- robosuite and robomimic are installed from git, so `pip-audit` cannot audit them; `npm audit`
  finds no vulnerability in the site's packages (`reports/session12/npm_audit_*.json`).
