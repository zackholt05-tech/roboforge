# Decisions Log

Each entry: decision, rationale, evidence, and whether it is **FIXED** (settled) or a
**DEFAULT** (a justified starting point a later session may revisit with a recorded reason).

## Session 0 (2026-11-05)

### D-001 Python 3.11 (FIXED)
robomimic `d309eae` hard-pins `transformers==4.41.2`, `diffusers==0.11.1`,
`huggingface_hub==0.23.4`; torch 2.15 needs ≥3.11; Modal supports 3.10–3.15. 3.11 resolved
and imported every pinned package on macOS arm64 and installed via `uv sync --frozen` on
Linux aarch64 (Docker). The same minor version must be used in remote images.

### D-002 `mujoco==3.2.7` (FIXED for Level 1)
- Unpinned resolution picked mujoco 3.17.0, which **breaks robosuite v1.5.1** at env
  construction (`AssertionError` in `binding_utils.get_joint_qpos_addr`).
- 3.2.7 (2025-01-14) is contemporaneous with robosuite v1.5.1 (2025-01-02) and satisfies
  its `mujoco>=3.2.3` bound. 3.2.6 replays bit-identically to 3.2.7 on Lift.
- 3.3.0 is statistically indistinguishable on open-loop replay (Lift 199/200 both; Can
  190/200 vs 193/200; task-object p95 error 0.098 vs 0.096 m):
  `reports/feasibility/mujoco_version_comparison.json`. Neither reproduces the recorded
  trajectories exactly, so the data-generation version cannot be identified from replay.
- Kept 3.2.7 because all canonical evidence uses it; 3.3.0 is a recorded viable alternative.

### D-003 robosuite v1.5.1, commit `51cc01785bab80ffeed20da15e67d7dd4140e76a` (FIXED)
robomimic's v0.1 dataset docs recommend the `v1.5.1` branch, and the low_dim files record
`env_version: 1.5.1`. Installed as a git dependency pinned to the tag commit (PyPI has 1.5.2,
which requires mujoco ≥3.3.0 and was not evaluated).

### D-004 robomimic master `d309eaecc18acf4152a830a895a6984b8ac71b05` (FIXED)
PyPI only has robomimic 0.3.0. The v0.5.0 tag (`ae5799f`) downloads the CLIP ViT-L/14
model at *import time* (`utils/lang_utils.py`) and predates 12 fixes, including diffusion
policy fixes (`2194b50` To==1, `77677b4`) and the dataset Hugging Face repo rename
(`e10526b`). `d309eae` makes CLIP loading lazy. Verified: `robomimic.algo.diffusion_policy`
and the diffusers DDPM/DDIM schedulers import and run on torch 2.15 / numpy 2.4.

### D-005 `egl_probe` restricted to Linux via a uv override (FIXED)
`egl_probe` (robomimic dependency) fails to build on macOS ("CMake must be installed") and
only selects an EGL GPU on Linux. `[tool.uv] override-dependencies` limits it to Linux,
where it built successfully in Docker (needs `cmake`, `build-essential`, `libegl-dev`).

### D-006 Build environments with `robosuite.make` from dataset metadata (FIXED)
`roboforge.sim.factory.make_env` passes the dataset's `env_args.env_kwargs` (controller,
control frequency, robot) verbatim. robomimic's `EnvRobosuite` wrapper is avoided because it
imports `egl_probe` whenever offscreen rendering is on. `configs/simulator/{lift,can}.yaml`
are generated from the HDF5 metadata, and a test asserts they still match it.

### D-007 Dataset source and scope (FIXED)
- Source: Hugging Face `robomimic/robomimic_datasets` at pinned revision `74fa018…`. Every
  file is SHA-256-checked against the HF LFS oid recorded in the manifest.
- Downloaded: Lift and Can, PH, `low_dim` and `raw` (`demo_v15`, which has states and model
  XML for later RGB/depth extraction). Total 170 MB.
- Variant choice: **PH (proficient-human)**, because MASTER_SPEC prescribes it for the
  core benchmark. It is the cleanest single-quality demo set (200 per task), which suits a
  fair algorithm comparison. MH (multi-human) and MG (machine-generated) differ in data
  quality and quantity; they belong only to an explicitly separate dataset-quality study
  (spec §7.4).
- Square PH: metadata recorded but not downloaded. **No pre-rendered `image` HDF5 exists**
  for v1.5 PH (registry link is `None`), so RGB must be regenerated from states
  (`dataset_states_to_obs.py`).

### D-008 Reproducible initial conditions come from serialized MuJoCo states (FIXED)
- In robosuite v1.5.1, `Lift`/`PickPlaceCan` constructors **reject** a `seed` kwarg
  (`TypeError`).
- Object placement and robot initialization noise use the global `np.random`.
- Same global seed gives an identical state (max diff 0.0), and restoring a serialized state
  in two fresh processes gives identical 20-step trajectories (max diff 0.0).
- Evaluation initial states will therefore be generated once (seeded), serialized
  (`model_xml` + flattened state), and restored for every model (Session 2).

### D-009 Replay fidelity is an engineering guard, not a gate on every demo (FIXED)
- Open-loop action replay is chaotic in contact (see KI-004).
- `verify_data.py` gates on integrity: hashes, low_dim ≡ raw, exact state restore, exact
  observation recomputation, recorded final states successful.
- Replay success is gated by declared guards: Lift ≥ 0.98, Can ≥ 0.90. These were set from
  Session 0 evidence as regression checks.
- Object metrics are restricted to the task object, because PickPlace's three inactive
  objects sit at (10, 10, 10) and free-fall.

### D-010 Observation features for Can (DEFAULT for Session 1 — superseded: FIXED in Session 1, see below)
robosuite's `Can_to_robot0_eef_pos/quat` blocks are lagged by one observation update and
wrong at t=0 (KI-003). Default: RoboForge computes all low-dimensional policy inputs from
the MuJoCo state with **one function used for both dataset windows and closed-loop
evaluation**. That function is verified equal to robosuite where robosuite is correct (all of
Lift; the world-frame Can blocks) and gives physically correct relative features otherwise.
The alternative (drop dims 0–6) remains acceptable; Session 1 must choose one, record it,
and apply it identically to all models.

### D-011 Environment management: uv + universal lockfile (FIXED)
`pyproject.toml` + `uv.lock` resolved for darwin-arm64, linux-x86_64 and linux-aarch64.
`make setup` runs `uv sync --frozen`.

### D-012 Git identity (FIXED)
- Repository-local `user.name=Zackary Holt`, `user.email=zackholt05@gmail.com`; global
  config untouched.
- Evidence: `gh api user` shows this address as the public profile email of the active
  GitHub account `zackholt05-tech`, and GitHub only allows a verified address there.
- No commit template, hooks path, template dir or trailer configuration exists globally.
- Commits carry no AI co-author trailers.

### D-013 RoboForge is released under the Apache License 2.0 (FIXED, owner decision 2026-11-05)
- **Decision (Zackary Holt):** RoboForge is to be released under **Apache-2.0**.
- **Scope:** everything authored for RoboForge in this repository. Third-party software,
  the robomimic datasets and robosuite/MuJoCo assets keep their own licenses
  (`THIRD_PARTY_NOTICES.md`). All are compatible: MIT and Apache-2.0 dependencies, and no
  third-party code is vendored.
- **Why Apache-2.0 over MIT** (both were offered): it adds an explicit patent grant and a
  NOTICE mechanism, and matches MuJoCo's license.
- **Applied:**
  - `LICENSE` is the official text from https://www.apache.org/licenses/LICENSE-2.0.txt
    (SHA-256 `cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30`; matches
    the SPDX text).
  - `NOTICE`: "Copyright 2026 Zackary Holt".
  - `pyproject.toml` `license = "Apache-2.0"`, `CITATION.cff` `license: Apache-2.0`, and
    the README.
- Per-file license headers are not required by Apache-2.0 and are not used.
- Choosing the license does **not** authorize publishing the repository; remote
  publication still needs separate owner approval.

### D-014 Session 0 model check is forward-only (FIXED for Session 0)
A plugin hook (ml-pipeline) blocked code containing a backward pass. Session 0 only
requires a one-batch forward pass (§2.3 item 6). At the owner's request, the agent then
added `ML_PIPELINE_ENFORCE=0` to `~/.claude/settings.json` (`env`). The owner disabled
the cto-toolkit plugin. Both take effect after a Claude Code restart.

### D-015 Compute split: GPU for training only; rollouts on CPU (DEFAULT)
State-based Level 1 evaluation needs no rendering, and MuJoCo stepping is CPU-bound
(~160–200 control steps/s/core measured). Training goes on a single Modal GPU only if local
benchmarks show it is needed. See `docs/COMPUTE_PLAN.md`.

### D-016 Repository layout (FIXED)
Session 1 extends it with `configs/data/`, `data/splits/`, `data/normalization/` and
`data/manifests/tasks/` (D-021). The spec's functional boundaries are kept. Packages (`models`, `train`, `eval`, `viz`,
`compute`) exist only as documented boundaries until their session. CLIs live in `scripts/`,
and Make targets exist only where a working command exists (spec §6.5).

### D-017 One evaluation platform per comparison (DEFAULT for Session 2)
Physics differs at floating-point level between macOS arm64 and Linux aarch64 (KI-013).
Default: all rollouts that enter one comparison (every algorithm, seed and initial state)
run on the same platform and container image. The platform is recorded in each run
manifest, and results from different platforms are never pooled in one table. Session 2
chooses the platform; a Linux image is preferred because it matches remote execution.

## Session 1 (2026-11-06)

### D-010 Observation features: one state → feature function (FIXED, Session 1; was DEFAULT)
- **Decision:** every low-dim policy input is computed by `roboforge.data.features`. The
  input is the flattened MuJoCo state `[time, qpos, qvel]` and the compiled model;
  `mj_kinematics` runs on a private scratch `MjData`, and robosuite's own formulas and
  quaternion conventions are applied.
  - The processed-data builder uses it for dataset windows (`states[t]` of every demo).
  - Session 2's simulator adapter must use it after every `env.step`
    (`FeatureExtractor.from_sim`). Policies never consume robosuite's `env.step`
    observations (KI-015).
- **Can:** the relative position `Can_to_robot0_eef_pos` is recomputed from same-step poses.
  The relative quaternion `Can_to_robot0_eef_quat` is **dropped**, because:
  1. it is a deterministic function of `robot0_eef_quat` and `Can_quat`, which are both kept;
  2. robosuite's `mat2quat` forces w ≥ 0, so the block jumps sign 598 times in the stored
     data, and 446 times even when recomputed from same-step poses
     (`reports/data_audit.json` → `quaternions`);
  3. a continuous re-encoding would be a new representation that neither robosuite nor
     robomimic uses.
- **Evidence (all 400 demos, `make check-data` / `reports/data_audit.json`):**
  - Every Lift feature, and Can's robot and world-frame blocks, equal the stored robomimic
    obs exactly (max |Δ| 0.0).
  - The corrected Can block is evaluated in float64 from MuJoCo's rotation matrix
    (`xmat`). It equals an independent float64 evaluation on the stored same-step poses to
    7.2e-16. It matches robosuite's own formula, which uses a float32 rotation matrix
    (KI-018), to 1.4e-7 m, and robosuite's observation with current kinematics and a fresh
    cache to 1.1e-7 m (closed loop, 5 demos).
  - The stored block equals robosuite's one-update-lag formula to 2.2e-16 on macOS
    (1.4e-7 on Linux, again the float32 path) and is wrong by up to 0.861 m at t = 0 and
    0.091 m at t > 0.
  - After `env.step`, live features equal those of the same state restored in a fresh
    simulator, bitwise. Computing them leaves the rollout bitwise unchanged
    (`tests/simulator/test_features_closed_loop.py`).
- **Consequence:** the Can input is 19-D, not robomimic's default 23-D (`object` 14-D). Can
  numbers are therefore not directly comparable with robomimic Can results that used the
  `object` key, which also inherited the lag.

### D-018 Canonical Level 1 observation spec (FIXED for the Level 1 core comparison)
- **Features** (`configs/data/{lift,can}.yaml`): robomimic's standard low-dim key set
  (`robot0_eef_pos` 3, `robot0_eef_quat` 4, `robot0_gripper_qpos` 2, then the `object`
  blocks), with KI-003 handled (D-010). No joint positions or velocities.
  - Lift (19-D): `cube_pos` 3, `cube_quat` 4, `gripper_to_cube_pos` 3.
  - Can (19-D): `Can_to_robot0_eef_pos` 3, `Can_pos` 3, `Can_quat` 4.
- **Why:** this is the input set robomimic and Diffusion Policy use for these tasks, so
  results stay interpretable against reference implementations. It is minimal, and every
  block except the corrected one is robosuite-identical.
- **Quaternions** stay raw `xyzw`, as in robosuite. They are continuous within episodes
  (0 sign flips) but ambiguous across episodes (KI-017).
- **Privilege:** object features are exact simulator state, so the spec is **privileged**.
- **Scope:** the observation spec is the per-step contract only. History length T_o is a
  per-run parameter (D-022), so one-frame BC, BC-RNN and diffusion share one spec hash.
- **Change control:** a different feature set needs a new `converter_version` and gives a
  new spec hash. Any representation ablation is a separately labeled experiment.
- **Hashes:** Lift `dfbd868e…a7721`, Can `71e89629…5b71c` (full values in
  `data/manifests/tasks/*.json`).

### D-019 Episode split: robomimic upstream masks (FIXED)
- **Split:** `mask/train` (180) and `mask/valid` (20), stored with every ID sorted
  numerically in `data/splits/{lift,can}_ph.json` with a self-hash.
- **The two tasks have identical ID lists.** Valid = `demo_0, 5, 14, 21, 27, 32, 36, 44, 56,
  93, 103, 105, 111, 125, 131, 155, 168, 176, 180, 191`; train = the other 180.
- **Why:** this is Session 0's documented default. The masks are disjoint, identical in
  `low_dim` and `raw`, and used by prior work. No hash split was needed.
- **No operator partition is possible:** the raw `user_info` holds one teleoperation
  channel `'0'` in every demo and no identity.
- **Use:** validation demos give offline losses only. There is no demonstration test split;
  final tests use held-out simulator initial states (RESEARCH_PROTOCOL §5).
- Flagged episodes would be listed, never removed. None were flagged.
- **Hashes:** Lift `73097265…9c331e`, Can `743779b0…b042b46f`.

### D-020 Normalization: train-only per-dimension min-max to [−1, 1] (DEFAULT for Sessions 3–4; identical for all families)
- **Method:** Diffusion Policy's `limits` semantics (re-implemented) for observations and
  actions. `range_eps = 1e-4`; dimensions below it get scale 1 (none occur). Statistics come
  from per-step arrays of the 180 training episodes, never from windows.
- **Decoding** model actions: unnormalize, then clip to the action bounds [−1, 1], and
  record clip events (`ActionCodec`).
- **Effects:**
  - The binary gripper maps to itself.
  - Rotation deltas are stretched (Lift scales up to 7.4×).
  - Validation observations reach up to 1.18 in normalized units (Lift eef y). They are not
    clipped.
- **Alternatives:** z-score is implemented and tested for a possible ablation. Changing the
  mode must be decided and recorded before training.
- **Hashes:** Lift `253bfc7e…4812d6`, Can `e0437cd7…b3228d`.

### D-021 Processed dataset and committed data artifacts (FIXED)
- **Processed files:** `$ROBOFORGE_DATA_ROOT/processed/robomimic_v1.5_ph/<task>/lowdim_v1.hdf5`
  (gitignored) hold float32 `obs`, `actions`, `rewards` and uint8 `dones` per demo.
  - Content hash: SHA-256 over (demo, array name, dtype, shape, bytes) in numeric demo
    order. No timestamps.
  - Verification against the dataset runs in float64 *before* the cast. float32 is the
    precision the models consume.
- **Committed artifacts** (no timestamps or absolute paths; self-hashed;
  byte-deterministic):
  - `data/manifests/tasks/<task>_ph.json`: the spec §4.4 task manifest: source, specs plus
    hashes, every episode ID, length and split, and hash references to the split,
    normalization and processed content;
  - `data/splits/<task>_ph.json`;
  - `data/normalization/<task>_ph_lowdim_v1.json`.
- **Determinism checks:**
  - `make check-data` rebuilds in memory and fails on any difference. It passed twice on
    macOS, and a rebuild was byte-identical.
  - `SequenceDataset.from_committed` refuses a processed file whose hash differs.
- **Layout (extends D-016):** `configs/data/`, `data/splits/`, `data/normalization/`,
  `data/manifests/tasks/`.
- **Cross-platform:** `make check-data` passes for both tasks on macOS arm64 and in the
  Linux aarch64 image (`roboforge-sim-cpu:session0`, OSMesa). The processed content,
  split, normalization and task manifests are bit-identical
  (`logs/session1/docker_linux_aarch64_check_data.log`). This needed the float64 Can
  feature (KI-018); linux/x86_64 is unverified (KI-001).

### D-022 Window semantics (FIXED) and default horizons (DEFAULT)
- **Windows:** one per decision step `t` of every split episode.
  - `obs_history[k] = obs[clip(t−T_o+1+k)]`: before the start, the first observation is
    repeated with `obs_valid = False`.
  - `future_action[k] = actions[t+k]`: after the end, the last action is repeated with
    `valid_mask = False`, and losses must mask these steps.
  - Observations at indices > t are never read (tested on synthetic and real data).
- **Default horizons:** T_o = 2, T_p = 16, the MASTER_SPEC §4.3 defaults. They are per-run
  parameters (`SequenceDataset.from_committed(obs_horizon=…, pred_horizon=…)`), recorded in
  each run manifest. T_a is a control-loop setting (Session 2/4).
- **Measured padding:** with T_p = 16, 31 % of Lift training windows and 13 % of Can
  windows contain padded actions, because Lift demos are short (36–64 steps).
- The Session 0 raw-obs loader (`LowDimWindowDataset`) was removed. For Can it would have
  fed the lagged block to a policy.

### D-023 Session 1 evidence location (FIXED)
Session 1 command outputs go under `logs/session1/` (Make variables `VERIFY_OUT` and
`SMOKE_OUT`), so Session 0's evidence in `reports/feasibility/` keeps the commit it was
generated from. Re-rendered `verify_data` frames that are byte-identical to Session 0's
are not committed again; the comparison is recorded in the ledger.
