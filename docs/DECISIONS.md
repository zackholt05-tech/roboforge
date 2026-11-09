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
  - `NOTICE`: "Copyright 2026 Zackary Holt" (ORCID iD 0009-0007-3965-4043 added at the
    owner's request, 2026-11-06).
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

### D-017 One evaluation platform per comparison (DEFAULT for Session 2 — superseded: FIXED in Session 2, see below)
Physics differs at floating-point level between macOS arm64 and Linux aarch64 (KI-013).
Default: all rollouts that enter one comparison (every algorithm, seed and initial state)
run on the same platform and container image. The platform is recorded in each run
manifest, and results from different platforms are never pooled in one table. Session 2
chooses the platform; a Linux image is preferred because it matches remote execution.

## Session 1 (2026-11-06)

### D-010 Observation features: one state → feature function (FIXED, Session 1; was DEFAULT)
- **Decision:** every low-dim policy input is computed by `roboforge.data.features`. The
  input is the flattened MuJoCo state `[time, qpos, qvel]` and the compiled model;
  `mj_kinematics` runs on a private scratch `MjData`. Feature definitions and quaternion
  conventions are robosuite's. `equal` features reproduce robosuite bitwise. The corrected
  Can relative position is evaluated in float64 and agrees with robosuite's float32
  formula to 1.4e-7 m (KI-018).
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
- **Write protection:** `make build-data` writes only the processed file, and only after
  the rebuild equals the committed artifacts. Rewriting those needs
  `scripts/build_data.py --update-committed` and a recorded decision.
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

## Session 2 (2026-11-06)
Evidence: `reports/session2/` and `logs/session2/` (D-031), regenerated from the clean
code commit `283a811`.

### D-017 Evaluation platform: the pinned Linux aarch64 container (FIXED, Session 2; was DEFAULT)
- **Platform:**
  - image `roboforge-sim-cpu:session0`, ID
    `sha256:1f685638f28c720fbec7b084b8e5c2d68073d8f637450d5f6e0a34f0d123071c`;
  - Python 3.11.18, `MUJOCO_GL=osmesa`;
  - host: Apple M1 Pro, Docker Desktop 29.9.2 (linux/arm64 VM).

  Its platform ID is `linux-aarch64-container-1f685638f28c` (`configs/eval/platform.yaml`).
- **How to run:** every rollout that enters a comparison runs through
  `infra/containers/run_sim_cpu.sh`. This covers validation checkpoint selection from
  Session 3 on and the final test. The wrapper:
  - refuses any other image ID;
  - mounts code, configs, docs and the committed data read-only;
  - uses the image's own locked venv, with no `uv sync` at run time (KI-020);
  - passes the host git SHA and dirty flag in.
  - makes only `reports/session<N>`, `logs/session<N>` and `runs/` writable, with N from
    `ROBOFORGE_EVIDENCE_SESSION` (default 2; added in `b739e99`).
- **Recorded:** every record and report stores `roboforge.provenance.evaluation_platform()`.
  Results from different platform IDs are never pooled in one table.
- **Why not native macOS:**
  - It cannot be pinned: an OS update can change libm/Accelerate results.
  - Session 2 measured 1-ulp differences in numpy's legacy `randn` and `np.sin` between the
    two platforms (KI-013).
  - The container also matches remote Linux execution.
- **Development use:** native macOS stays the development platform, and its outputs carry
  `darwin-arm64-native`.
- **Not covered:**
  - The same image on a different host CPU (e.g. x86_64 Modal) is a different platform,
    and is unverified (KI-001).
  - Rebuilding the image changes the platform ID. Every rollout of an affected comparison
    must then be re-run.

### D-024 `RoboForgeEnv` adapter contract (FIXED)
`src/roboforge/sim/env_adapter.py` (MASTER_SPEC §4.1).
- **Construction:**
  - the robosuite env is built from `configs/simulator/<task>.yaml`;
  - specs are loaded hash-checked with `load_specs`;
  - the env's `action_spec` must equal the committed `ActionSpec`;
  - `reward_shaping` is refused.
- **Reset always goes through an exact snapshot.**
  - `reset(InitialState)` compiles the snapshot's XML in a fresh `MjSim`, restores the
    state (`reset_to_state`) and asserts an exact read-back.
  - `reset(seed)` first checks the seed against the reserved blocks (D-025), then
    generates and restores a snapshot.
  - **Hidden state:** a restore resets everything outside the flattened state. That is
    MuJoCo's `qacc_warmstart`, `ctrl` and mocap poses, and robosuite's gripper integrator,
    controller goals and step counter. Two episodes from the same snapshot are therefore
    identical, whatever ran before (check G: 0.0).
- **`step(action)`:**
  - The action must have shape `(ActionSpec.dim,)` and be finite. Otherwise it raises
    `InvalidActionError` and leaves the state untouched; invalid actions are never repaired.
  - Finite actions are clipped to the `ActionSpec` bounds. `clip_mask` reports which
    entries were clipped.
  - It returns `(Observation, reward, terminated, truncated, info)`. The observation is
    `FeatureExtractor.from_sim` of the post-step state (D-010, KI-015); robosuite's
    observation is discarded.
  - Stepping after the episode ends raises `EpisodeDoneError`.
- **Success (D-026):** `mj_kinematics` on the live data, then the task's `_check_success()`.
  - `mj_kinematics` only writes positions derived from `qpos`, so it is inert for the
    dynamics (check F: 0.0).
  - `mj_forward` is not used: it overwrites `qacc_warmstart` and perturbs the rest of
    the rollout (check F reference; KI-019).
  - robosuite's own value, computed on stale kinematics, is recorded as
    `success_robosuite_raw` and is never scored.
- **Rendering:** `ShadowRenderer` renders from a scratch `MjData` built from the state,
  with robosuite's geom groups `(0, 1, 1, 0, 0, 0)`.
  - Measured on macOS (CGL), 256², 3 validation starts per task, every camera
    (`logs/session2/diag/shadow_vs_robosuite_macos.json`): identical to robosuite's
    renderer, except `frontview`, which differs by 1 grey level on 6 px (Lift) and 23 px
    (Can).
  - The unit test at 96² Lift `frontview` is pixel-identical.
  - OSMesa was not compared.
  - Rendering never touches the live data.
- **Kinematics:** `KinematicsExtractor` uses a scratch `MjData` with `mj_kinematics`,
  `mj_comPos` and `mj_camlight`.
  - It returns world poses of every body except the world and mocap bodies, categorized as
    robot, free_object or fixture.
  - It also returns joint `qpos`/`qvel`, site positions and camera extrinsics/intrinsics.
  - Quaternions are wxyz, MuJoCo's convention.

### D-025 Evaluation initial-state suites, XML convention and the final-test lock (FIXED)
`src/roboforge/sim/{snapshot,eval_states}.py`, `configs/eval/suites.yaml`, `data/eval_states/`.
- **XML convention:** a snapshot stores `env.sim.model.get_xml()`, MuJoCo's
  `mj_saveLastXML` of the live model after reset. This is the robomimic dataset
  convention.
  - Asset paths are rewritten to robosuite-relative ones (`portable_xml`), so the string is
    machine-independent.
  - Restores compile it through `env.edit_model_xml`. Capture verifies that the portable
    and raw strings compile to the same model.
  - **Why not robosuite's generator XML (`env.model.get_xml()`):** it is not the model that
    ran. Lift samples the cube size at every reset, and PickPlace edits bin and
    visual-object `body_pos` at runtime.
  - The saved XML is therefore the evaluation model, and every restore compiles exactly
    that string (KI-016 resolved).
- **Suites:**
  - `validation` uses seeds `100000 + i`; `final_test` uses seeds `900000 + i`;
    `i ∈ [0, 50)` for both tasks.
  - Ad-hoc `reset(seed)` may not use the reserved blocks [100000, 200000) and
    [900000, 1000000).
  - Generation seeds `random` and `np.random`, then calls `env.reset()`. It verifies the
    restore and that no start is already a success.
  - Lift has 50 unique XMLs per suite (sampled cube sizes); Can has 1.
  - Content hashes:

    | Task | validation | final_test |
    |---|---|---|
    | Lift | `0b1368fa…` | `e0249865…` |
    | Can | `85767338…` | `5f407748…` |

    The full values are in the manifests and `partitions.json`.
- **Storage:** each suite has an open manifest (`<suite>.json`), which holds hashes only,
  never state values. The states themselves sit in `<suite>.data.jsonl.xz`.
- **Generated on the canonical container (D-017):** `make eval-states` runs there.
  - Regeneration is bitwise there.
  - On macOS it agrees within 4.4e-16, because numpy's legacy `randn` and `np.sin` differ
    by 1 ulp (KI-013).
  - The committed values are canonical, and they restore exactly on every platform.
  - `write_suite` refuses to overwrite a differing suite without `--overwrite`, which
    needs a recorded decision.
- **Final-test lock:** `load_suite(task, "final_test")` and `reset(seed)` with a seed in
  [900000, 1000000) raise `FinalTestLockedError`. They keep raising until
  `docs/FINAL_TEST_FREEZE.md`:
  - exists;
  - is tracked by git and unmodified;
  - names the suite's `content_sha256`.
- **What reads the locked data:**
  - Only `eval_states.py`, `snapshot.py` and `scripts/generate_eval_states.py` may contain
    the suite name (`tests/research_integrity/`).
  - The data file is decompressed in exactly three places, all in `eval_states.py`:
    - its hash check;
    - `load_suite` after the lock check;
    - the generator's reproducibility check. It compares regenerated against stored
      values and reports only the maximum difference.
- **Partition labels** (`snapshot.PARTITIONS`): `demo_train`, `demo_valid`,
  `eval_validation`, `eval_final_test`, `eval_adhoc`.
  - Every snapshot, record and report carries its partition.
  - `data/eval_states/<task>/partitions.json` indexes the demo split and both suites.

### D-026 Success, reward and termination (FIXED)
- **Success:** the task predicate (robosuite `_check_success`) on the post-step state's
  current kinematics (D-024).
- **Reward:** sparse, 1.0 on a success step and 0 otherwise.
- **Scored outcome:** the episode ends (`terminated`) at the **first success**. It is
  `truncated` at the task manifest's `eval_horizon_steps` (400 for both tasks).
  - Primary metric: success before the horizon.
  - Secondary: `first_success_step`.
- **Persistence:** `hold_steps = k > 0` keeps stepping k more control steps after the first
  success and records `success_held_through_hold`. It is a diagnostic only. The default is
  0, and it never changes the scored outcome.
- **Why first success:**
  - it matches robomimic's rollout default, `experiment.rollout.terminate_on_success =
    True` (`robomimic/config/base_config.py`), which counts any success before the horizon;
  - every demonstration's success persists to its end (Session 1);
  - persistence over a fixed hold is available as a declared diagnostic, if a policy turns
    out to knock objects off after success.

### D-027 Sanity policies and non-autonomous instrumentation (FIXED)
`src/roboforge/eval/policies.py`.
- **Policy metadata:** every policy carries a `PolicyInfo` (name, kind, learned,
  autonomous, action space, label), stored in each record.
- **`RandomPolicy`:**
  - i.i.d. uniform actions over the `ActionSpec` bounds; it ignores observations;
  - its per-episode RNG is `default_rng([policy_seed, uint64(sha256(state_id)[:8])])`, so
    streams are reproducible per (seed, state) and distinct across states.
- **`NoOpPolicy`:** zero actions, i.e. a zero OSC delta. Gripper action 0 holds the current
  gripper state.
- **`DemoReplayPolicy`:**
  - it plays a demonstration's recorded actions open-loop;
  - it is labeled `autonomous = False`, with a label that starts "NON-AUTONOMOUS";
  - when it runs out of actions it stops the episode with `stop_reason =
    "policy_exhausted"`;
  - it is instrumentation and never a result.
- **Learned policies (Session 3 on)** implement the same `Policy` protocol. They may return
  action chunks, and normalized outputs are decoded with `ActionCodec`.

### D-028 Rollout record schema `roboforge.rollout/1` and frame convention (FIXED)
`src/roboforge/eval/rollout.py`.
- **Frame convention:** frame k is the state after k executed actions; frame 0 is the
  restored snapshot. Action t is chosen from frame t's observation and produces frame t+1.
  - Per-frame arrays have T+1 rows: `state`, `obs`, `time_s`, `joint_qpos`, `joint_qvel`,
    `body_pos`, `body_quat_wxyz`, `success`, `success_robosuite_raw`.
  - Per-action arrays have T rows: `raw_policy_action`, `executed_action`, `clip_mask`,
    `reward`, `reward_robosuite_raw`, `decision_index`, `chunk_offset`,
    `planning_latency_ms`, `phase`.
  - Optional RGB frames are stored with their `rgb_frame_index`.
- **Files:** `record.json` holds the metadata, outcome and array index. `arrays.npz` holds
  the compressed arrays.
- **Hashes:**
  - `arrays_sha256` hashes (name, dtype, shape, bytes) in sorted order.
  - `physics_sha256` is the same hash without the wall-clock `planning_latency_ms`. Two
    runs of the same episode on one platform must give the same `physics_sha256`.
- **Metadata:** platform (D-017), git, packages, simulator constants, specs and hashes, the
  initial-state reference (ID, partition, hashes), policy info, body and joint names with
  categories, and the frame convention.
- **Outcome fields:**
  - `success`, `first_success_step` and robosuite's raw counterpart;
  - `steps`, `terminated`, `truncated`;
  - `stop_reason`: `terminated_success`, `truncated_horizon`, `policy_exhausted` or
    `error`;
  - error type and message;
  - the hold diagnostic;
  - clip counts.
- **Policy errors:** an invalid policy output ends the episode as a recorded failure
  (`stop_reason = "error"`), never a crash or a silent repair.
- **Audit:** `verify_record` re-simulates the executed actions from the snapshot (states
  and observations) and re-evaluates success on the recorded states.

### D-029 Demonstration-replay sample (FIXED)
- **Sampling rule** (`replay_compare.select_demos`, declared before the first run):
  - sort the 200 demos by (length, index) into 10 bins of 20;
  - draw one demo per bin with `default_rng(0)`;
  - add the lowest-index Session 0 replay failure (Lift `demo_177`, Can `demo_11`);
  - add validation-split demos (lowest index first) until there are ≥ 2.
  - Result: 12 demos per task.
- **Gates:**
  - each replay through `RoboForgeEnv` + recorder must be **bitwise** equal to a bare
    robosuite loop;
  - it must match Session 0's legacy metrics on the same platform;
  - dataset actions must cause zero clip events.
- **Divergence from the recorded states is reported, not gated (KI-004).**

### D-030 MuJoCo → three.js conversion (FIXED)
`src/roboforge/viz/coords.py`. MuJoCo is z-up; three.js is y-up. Both are right-handed.
- **Positions:** `p_three = C p_mj`, with `C = [[1,0,0],[0,0,1],[0,−1,0]]`, i.e.
  (x, y, z) → (x, z, −y).
- **Rotations:** `q_three = q_C ⊗ q_mj`, with `q_C = (cos(−π/4), sin(−π/4), 0, 0)` (wxyz).
  three.js stores xyzw.
  - This equals one root `Group` with `rotation.x = −π/2` and untouched MuJoCo poses
    inside.
- **Cameras:** MuJoCo cameras look along −z with +y up, like three.js, so they convert like
  bodies. `fov` = `cam_fovy` (vertical, degrees).
- **Verification** (`scripts/export_clip.py`, `tests/web_contract/`): one real Lift demo
  replay, checked three ways.
  - Re-derived forward kinematics equals MuJoCo's world poses.
  - three.js-convention projection equals MuJoCo's pinhole projection.
  - Projected geom vertices fall inside MuJoCo segmentation masks.

  The real three.js library (npm `three@0.187.0`, run in node) reproduces the conversion
  and projection.
- **Clip format:** `roboforge.web_clip/1`. Its metadata keeps the record's provenance
  (`require_provenance`).

### D-031 Session 2 evidence location (FIXED)
- Session 2 command outputs go under `logs/session2/` and `reports/session2/` (Make
  variable `S2_OUT`).
- Per-episode rollout records go under `runs/<run_id>/`. They are gitignored and can be
  regenerated; summaries and hashes are committed in the reports.
- The evidence was regenerated from the clean code commit `283a811`. Each JSON carries its
  git SHA, dirty flag and platform.

## Session 3 (2026-11-06)
Evidence: `reports/session3/` and `logs/session3/` (D-038). The protocol was committed first
(`8b08b95`), then the code (`969e14f`). Every training run and validation evaluation ran from
the clean `969e14f`. Diagnostics, cost and some equivalence reports ran from the clean
`a37b92d`, which changes only the Makefile and `scripts/diagnose_actions.py`.

### Research protocol v1 (FIXED for model development)
`docs/RESEARCH_PROTOCOL.md` was finalized and committed at `8b08b95`, before any reported
training or evaluation run. It fixes:
- each hypothesis's comparison;
- primary configurations and matched budgets;
- the checkpoint-selection rule (§7);
- the paired schedule and the bootstrap plan (§9).

Diffusion items are TO FINALIZE (Session 4), by dated amendment only.

### D-032 Rollout `observe` hook (FIXED)
- `run_episode` calls `policy.observe(obs)`, if the policy defines it:
  - with frame 0's observation after the reset;
  - with every new frame's observation right after each executed action, including actions
    inside a chunk.
- An exception in the hook is a recorded error.
- **Why:** a chunked policy with an observation history (T_o = 2, T_a = 8) must see the frames
  it does not decide at. The Session 2 runner passed only the decision frame.
- **Physics unaffected:**
  - the hook only feeds the policy;
  - sanity policies do not define it;
  - re-running the Session 2 random and no-op episodes on the canonical container reproduces
    their recorded `physics_sha256` (`scripts/check_checkpoint_equivalence.py`, check E).

### D-033 BC families and primary configurations (DEFAULT, Session 3; tuning per RESEARCH_PROTOCOL §6.3)
`src/roboforge/models/bc.py`, `configs/models/{bc_mlp,bc_rnn,bc_mlp_chunk}.yaml`.
- **One interface:** `forward(obs_history [B,T_o,D], obs_valid [B,T_o]) → [B,T_p,A]`
  normalized actions. The trainer and the rollout policy call the same function.
- **BC-MLP:**
  - MLP 2 × 1024, ReLU, linear output; T_o = 1, T_p = 1, T_a = 1;
  - 1,077,255 parameters. This is robomimic's BC architecture.
- **BC-RNN:**
  - LSTM, 2 layers × 296, with a linear head on the newest step; context L = 10, T_p = 1;
  - 1,080,703 parameters (+0.3 %);
  - it re-runs from a zero state over the valid steps of the last L observations at every
    decision, so training and inference are identical. Pre-start padding is skipped by
    left-aligning the valid suffix and packing.
  - **Deviation from robomimic:** robomimic carries the hidden state for up to 10 rollout
    steps and then resets it. Exact train/inference consistency was preferred.
  - LSTM rather than GRU matches robomimic's BC-RNN reference; GRU is implemented and
    tested.
- **BC-MLP-chunk** (H2-c control):
  - BC-MLP's hidden layers with diffusion's horizons T_o = 2, T_p = 16, T_a = 8;
  - 1,204,336 parameters (+11.8 %, from the larger input and output layers).
- **Capacity tiers:** widths matched to 0.27 M / 1.08 M / 4.3 M are tabulated in
  `reports/session3/param_counts.json` (all within ±0.4 %). They are ready for matching
  diffusion's count in Session 4.
- **Not run:** GMM-BC (optional, spec §5.4).

### D-034 Training stack defaults (DEFAULT; identical for every BC family)
`src/roboforge/train/{config,data,loop,run}.py`.
- **Config:** strict schema `roboforge.train_config/1`, where unknown or missing keys raise.
  `--set` overrides are recorded. `config_sha256` is part of the run ID
  `<prefix>_<name>_<task>_seed<s>_<sha8>`.
- **Data:** all windows of a split are materialized from `SequenceDataset.from_committed`.
  The `valid` split gives offline losses only.
- **Sampler:** one seeded permutation per epoch from `default_rng([seed, 1])`, served as
  consecutive batches that may straddle two epochs.
- **Optimization:**
  - AdamW, lr 1e-4, betas (0.9, 0.999), weight decay 0, constant LR, batch 100;
  - masked MSE; no gradient clipping (the global norm is logged); no EMA;
  - 1 torch thread, CPU, float32.
- **Failures:** a non-finite loss or gradient norm, or a parameter without a gradient, raises.
- **Pilot budget:** 30,000 steps, checkpoints every 5,000 (RESEARCH_PROTOCOL §6.3).

### D-035 Checkpoints, run manifests and evaluation/selection artifacts (FIXED)
- **Checkpoints** (`runs/<run_id>/checkpoints/step_<N>.pt`, schema `roboforge.checkpoint/1`):
  - only tensors and JSON-compatible values, loaded with `torch.load(weights_only=True)`
    and a strict `load_state_dict`;
  - model, optimizer, scheduler, sampler (with numpy generator state as JSON) and torch RNG
    state;
  - config and hash, dims, parameter count, the committed normalizers, every data/spec hash,
    git, torch version, device and threads;
  - written atomically and never overwritten.
  - Resuming is bitwise identical to uninterrupted training (unit test).
- **Training manifest** (`runs/<run_id>/manifest.json`, `roboforge.train_run/1`):
  - written once with exclusive create, also on failure (`status: failed` + traceback);
  - it includes a reload check: the final checkpoint, reloaded into a fresh network, must
    reproduce parameters and outputs bitwise.
- **Evaluation run** (`runs/<train_run_id>__val_step<N>__<platform_id>/`): every episode
  record plus an immutable manifest. The checkpoint SHA-256 is verified against the training
  manifest before loading and appears in every record's `policy.params`.
- **Selection** (`reports/session<N>/selection/<train_run_id>.json`): the protocol §7 rule.
  It refuses missing, partial, duplicate, foreign-hash or mixed-platform evaluations.

### D-036 Learned-policy inference contract (FIXED)
`src/roboforge/eval/learned.py`.
- **Inputs:**
  - Observations are cast to float32 before normalization, as the processed training data
    were stored.
  - Windows come from `window_indices`, the training index arithmetic. They are tested equal
    to `SequenceDataset` windows at every step of a demo, for T_o = 1, 2 and 10.
  - The env's t = 0 features equal the processed training observation bitwise.
- **Outputs:** the first T_a actions of the chunk, decoded by the checkpoint's `ActionCodec`
  (unnormalize, then clip to [−1, 1]).
- **Contract check:** a policy refuses an episode or a committed contract (spec and
  normalization hashes) that differs from its checkpoint's.
- **Isolation:** the history is cleared at every reset, so no state carries between episodes
  (tested).
- **Determinism:** CPU, `no_grad`, 1 torch thread. The same checkpoint reproduces
  `physics_sha256` across loads and processes on one platform.

### D-037 Validation-only evaluation entry points (FIXED until the freeze)
- `scripts/eval_policy.py` and `make eval` have no suite option; `make eval` refuses any
  `EVAL_SET` other than `validation`.
- Results from a non-canonical platform go to `reports/session3/dev/`, labeled DEVELOPMENT
  ONLY, and are never eligible for selection.
- Partial evaluations (`--n-states`) are never eligible either.

### D-038 Session 3 evidence location (FIXED)
- Command outputs go under `logs/session3/` and `reports/session3/`. Container runs use
  `ROBOFORGE_EVIDENCE_SESSION=3`.
- Training and evaluation records go under `runs/` (gitignored). Their manifests' summaries
  and hashes are committed in the reports.

## Session 4 (2026-11-06)
Evidence: `reports/session4/` and `logs/session4/` (D-044). Commits, in order:
- `a01548d`: diffusion code, tests and tooling;
- `22adb8b`: `benchmark_policies.py --max-decisions`;
- `4c68ca1`: the container runner forwards `ROBOFORGE_EVIDENCE_SESSION`;
- `3aed462`: the protocol amendment (RESEARCH_PROTOCOL §6.4, §13), committed **before** the
  first reported diffusion run;
- `183b56c`: diffusion-aware `scripts/diagnose_actions.py`.

The pilot training ran from the clean `3aed462`. Every validation evaluation, the selection, the equivalence checks, latency, denoising
exports and diagnostics ran from the clean `183b56c`. `git diff --stat 3aed462 183b56c`
lists only `scripts/diagnose_actions.py`.

### Research protocol amendment 2026-11-06 (FIXED for model development)
Every diffusion item marked TO FINALIZE is filled in RESEARCH_PROTOCOL §6.4, with a dated
§13 row, at `3aed462`. Before that commit, only development runs existed: smoke runs,
overfit checks and a 200-step latency checkpoint. None was evaluated on the validation suite.
The amendment fixes:
- the architecture and its exact parameter count, **1,092,551**. The §6.1 capacity rule is
  not triggered (≤ 2 × 1,077,255);
- the schedule (`squaredcos_cap_v2`, K = 100), ε-prediction, the DDIM-10 sampler with η = 0
  and `clip_sample`;
- EMA as the evaluated network, and the horizons T_o 2 / T_p 16 / T_a 8;
- the noise-seed rule, with policy_seed 0 and R = 1;
- the pilot as diffusion's tuning trial 1 on Lift.

### D-039 Evidence session from `ROBOFORGE_EVIDENCE_SESSION` (FIXED)
- `roboforge.paths.evidence_session()` reads `$ROBOFORGE_EVIDENCE_SESSION`. It has **no
  default** and raises if the variable is unset. `evidence_reports_dir(*parts)` returns
  `reports/session<N>/…`.
- Every evidence script derives its default output directory from it:
  - `eval_policy`, `select_checkpoint`, `check_checkpoint_equivalence`, `diagnose_actions`;
  - `benchmark_policies`, `compare_diffusion_reference`, `export_denoising`.
- In the Makefile, `EVIDENCE_SESSION` defaults to `$ROBOFORGE_EVIDENCE_SESSION`, else 4. The
  container targets use `RUN_S`/`IN_SIM_S`.
- `infra/containers/run_sim_cpu.sh` forwards the variable into the container (`4c68ca1`).
  Only `reports/session<N>`, `logs/session<N>` and `runs/` are mounted writable.
- `check_checkpoint_equivalence.py` takes `--out-dir`. It reads the Session 2 sanity
  reference from `<out-dir>/session2_sanity_reference.json`, which is copied unchanged from
  Session 3's evidence.
- **Why:** the Session 3 tooling hardcoded `reports/session3`. A Session 4 command could have
  overwritten committed Session 3 evidence, or lost its report outside the mounted directory
  (HANDOFF §9 of Session 3).

### D-040 Diffusion policy family `diffusion_unet` (FIXED; RESEARCH_PROTOCOL §6.4)
- **Code:**
  - `src/roboforge/models/{diffusion,unet1d,noise_schedule}.py`;
  - `src/roboforge/train/ema.py`;
  - `configs/models/diffusion_unet.yaml`.
- **Encoder:** no parameters. The normalized `T_o × 19` window is flattened into a 38-D
  global condition, robomimic's low-dim path and BC-MLP-chunk's input.
- **Noise-prediction network:** `ConditionalUnet1D`, adapted from robomimic `d309eae` (MIT,
  `THIRD_PARTY_NOTICES.md`) with identical parameter names.
  - Down widths (64, 128): stages 16 → 8 → mid → 16.
  - Kernel 5, GroupNorm with 8 groups, Mish.
  - FiLM on `[64-D timestep embedding ‖ condition]`.
  - Parameters: step encoder 33,088, down 452,992, mid 381,952, up 203,392, final 21,127.
    Total **1,092,551**.
- **Schedule and samplers** (`noise_schedule.py`):
  - our own implementation of the diffusers 0.11.1 equations; the betas are computed in
    Python floats (`betas_for_alpha_bar`), then stored as float32;
  - DDPM refuses a strided timestep set: diffusers 0.11.1 always steps to `t − 1`, so DDPM
    is only correct when the inference steps equal the training steps;
  - DDIM uses `steps_offset` 0 and `set_alpha_to_one`;
  - every step returns a `StepRecord` with its coefficients.
- **Loss:** masked MSE between ε̂ and ε over valid action steps. Per sample, the noise is
  drawn first, then `t ~ U{0..K−1}`, from the run's own `torch.Generator`
  (`SeedSequence([seed, 2])`). That order is robomimic's, so the same seed gives the same
  draws (reference test).
- **EMA:** diffusers 0.11.1 `EMAModel` semantics (power 0.75, inv_gamma 1, update_after_step 0).
  - Update number s (0-based, one per optimizer step) uses decay 0 for s ≤ 1, which copies
    the weights. After that, the decay is `min(1 − s^−0.75, 0.9999)`.
  - Buffers are copied.
  - The EMA network is the `inference_model`: offline metrics, checkpoint selection and
    rollouts all use it.
- **Equivalence with the pinned references:** bitwise on macOS and on the canonical
  container (`reports/session4/reference/`). The checks:
  - betas, ᾱ, `add_noise`;
  - DDPM-100 and DDIM-10 steps;
  - U-Net forward with robomimic weights at three widths;
  - full DDIM-10 and DDPM-100 sampling against robomimic `_get_action_trajectory`;
  - EMA against diffusers `EMAModel`.

  The training loss agrees to |Δ| = 0.0 with `train_on_batch`.
- **Deviation:** robomimic predicts `a[t−1 .. t+14]` and executes from index `T_o − 1`.
  RoboForge keeps BC-MLP-chunk's window, `a[t .. t+15]`, executed from index 0. Both
  execute `a[t]` first.
- **Backend:** `unet_backend()` disables NNPACK around U-Net calls only (KI-026). The
  results are bitwise unchanged (reference tests). The context is scoped, so BC code paths
  are untouched.

### D-041 Stochastic inference contract (FIXED)
- `LearnedPolicy` is `stochastic` for the diffusion family.
  - `reset_episode` creates a fresh `torch.Generator`, seeded by
    `noise_seed(policy_seed, state_id, repeat)`: `SeedSequence([policy_seed,
    uint64_le(sha256(state_id)[:8]), repeat]).generate_state(1, uint64)[0]`
    (RESEARCH_PROTOCOL §6.4).
  - Each decision calls `model.sample(x, v, generator)`. The global RNG is never read.
- `run_episode` stores `policy.episode_info()` (seed triple, noise seed and rule) in each
  record's meta as `policy_episode`. It is not part of the arrays or of `physics_sha256`.
  The Session 2 sanity episodes still reproduce their recorded hashes (equivalence check E).
- **Evaluation run ID:** a stochastic policy's ID ends in `__ps<policy_seed>r<repeat>`, so
  repeats never share a directory. Each episode summary carries `noise_seed`, and the summary
  carries `stochastic_policy`.
- **Checkpoint equivalence** adds check F: two loads give bitwise-identical seeded samples.
  The rollout seed equals the rule's value.
- **BC policies are unchanged:** IDs, records and `physics_sha256` stay the same (the
  Session 3 selected checkpoints pass checks A–E under the new code).

### D-042 Model registry and optional checkpoint keys (FIXED)
- `roboforge.models.registry` maps each family to its builder and loss kind:
  - `bc_mlp`/`bc_rnn`/`bc_mlp_chunk`: `mse`;
  - `diffusion_unet`: `denoising_mse`.

  The config validator requires that loss kind. Adding a family does not change any Session 3
  config hash or run ID (unit test).
- Checkpoint `REQUIRED` keys are unchanged. Diffusion checkpoints add `noise_rng_state` and
  `ema_state`, which resume restores. Resume is bitwise, including the noise generator and
  the EMA (unit test).
- `reload_check` compares the inference (EMA) network and the trained weights separately.
  For a stochastic model it compares seeded samples (`RELOAD_SAMPLE_SEED = 0`) instead of a
  deterministic forward pass.

### D-043 Denoising export contract (FIXED)
- `scripts/export_denoising.py` / `make export-denoising EVAL_RUN=runs/<eval_run_id>`
  (`roboforge.viz.denoising`, schema `roboforge.denoising_trace/1`).
- **Source:** it replays recorded validation episodes with their own noise seed, so every
  decision consumes the generator as in the rollout. Every replayed chunk must equal the
  recorded `raw_policy_action` **bitwise**; otherwise the export raises. So it runs on the
  recording platform.
- **Content per selected decision:** for every reverse step, `t_k`, `t_prev`, ᾱ values and
  coefficients, plus these arrays in normalized and denormalized space:
  - `A_{t_k}`, ε̂, x̂₀ before and after the clip, the injected noise and the step output.

  It also stores the conditioning window (raw, float32 cast, normalized, mask), the noise
  seed and the final decoded chunk.
- **Labels:** every array is a **proposal**. Only the first `n_executed` actions were
  executed.
- **Selection, declared before export:** the first success and the first failure by sorted
  `state_id`, and decisions first / middle / last. Plus 8 never-executed diversity samples
  per decision (seeds `1,000,000 + i`).

### D-044 Session 4 evidence location (FIXED)
- Command outputs go under `logs/session4/` and `reports/session4/`, with
  `ROBOFORGE_EVIDENCE_SESSION=4`.
- Training and evaluation records go under `runs/` (gitignored). Their summaries and hashes
  are committed in the reports.
- Development-only artifacts are labeled `dev` or `pre_amendment`.

## Session 5 (2026-11-06 to 2026-11-07)
Evidence: `reports/session5/`, `logs/session5/` and the cumulative cost ledger
`reports/compute/cost_ledger.jsonl` (D-052).

Code commits, in order:
- `c7f043f`: resumable runs and the Modal integration;
- `2477312`: Volumes created before `app.run`;
- `af9c93a`: BC-RNN device fix;
- `a2fe706`: probe sections and launch results;
- `1d04672`: preemption drill, attempt cap, `repo_relative`;
- `6951c7a`: render probe and measured Modal rates;
- `db10a1b`: attempt-aware upper bound and 5k-cadence projections.

The last verified code commit is `db10a1b`. Each evidence record names the commit it ran
from, and all were clean.

### D-045 The Modal SDK stays outside the project venv; one Modal entrypoint (FIXED)
- `uv.lock` and `pyproject.toml` are unchanged. The Modal client (1.6.3) lives in a separate
  Python (here `/opt/anaconda3`, set with `MODAL_PY`).
- `infra/modal_app.py` is the only file that imports `modal`. It never imports RoboForge on
  the local side. It calls the project venv (`uv run --frozen python -m
  roboforge.compute.plan`) for:
  - plans and run IDs;
  - git state, data hashes and prices;
  - the approval gate and the cost ledger.
- A unit test asserts that no module under `src/` imports `modal`.
- **Why:** adding the SDK to the lock would change every platform's environment, the
  canonical image included (D-017). The local side needs only the SDK.

### D-046 Resumable, interrupt-safe training runs (FIXED)
`src/roboforge/train/{journal,stop}.py`, `train_run(..., launch_id, resume, stop)`,
`scripts/train.py --resume --launch-id --max-wall-s`.
- **Launch lock.** `launch.json` (exclusive create) holds the creating launch's nonce.
  - The same launch ID resumes (Modal restarts a preempted function on the same input).
  - A different launch ID raises `RunExistsError`, unless it is an explicit `resume` launch.
  - A run with a manifest is immutable; a restart of its own launch returns the manifest.
- **Stops act between optimizer steps only.** SIGINT/SIGTERM handlers (main thread only),
  a deadline and `stop_at_step` set a flag. The loop then writes a **resume checkpoint**
  (`resume/step_<N>.pt`, indexed in `resume.jsonl`) and returns `status: interrupted`. No
  manifest is written then.
  - Without handlers, a `KeyboardInterrupt` records an `aborted` segment, and the run
    resumes from its last indexed checkpoint.
- **Only cadence checkpoints enter selection.** `checkpoints.jsonl` indexes each
  `checkpoints/step_<N>.pt` after its atomic write. The manifest's `checkpoints` lists only
  these; resume checkpoints are listed separately (`resume_checkpoints`).
- **What a hard kill can leave:**
  - an open segment, recorded as `lost` with its last heartbeat;
  - a torn `.pt.tmp`, or a checkpoint renamed into place without its index line: both are
    moved to `orphans/`, never used;
  - metric rows past the resume point, which are truncated.

  An indexed checkpoint whose SHA-256 mismatches raises; there is no fallback to an older
  file.
- **Metrics rows** of a step are now written before its checkpoint, and the unaveraged
  train-row window (`fit_state.log_window`) travels in every checkpoint. A resumed run
  therefore writes the same rows. Wall-time fields are cumulative over segments.
- **Equivalence:** an interrupted and resumed run equals an uninterrupted one:
  - every cadence checkpoint is content-identical (every tensor bitwise, every value
    equal);
  - the metric rows match apart from wall-clock fields, and so do the final offline
    metrics.

  This holds for off-cadence and on-cadence stops, for BC-MLP and diffusion (EMA and noise
  generator), for real SIGINT/SIGTERM, and after simulated hard-kill debris
  (`tests/integration/test_resumable_run.py`, 9 tests). File SHA-256 can differ for
  content-identical checkpoints (KI-029), so equivalence compares content
  (`checkpoint_differences`). File hashes stay the integrity check.
- **Compatibility:** uninterrupted runs produce the same manifest fields as before, plus
  `launch`, `segments`, `resume_checkpoints` and `timing.n_segments`. Checkpoints add
  `fit_state`. Loading pre-Session-5 checkpoints is unchanged. The Session 3–4 selected
  checkpoints still pass the equivalence checks (E5-14).

### D-047 Modal launch policy and remote layout (FIXED for Session 5; DEFAULT for later sessions)
`configs/compute/modal.yaml`, `infra/modal_app.py`, `src/roboforge/compute/`.
- **Image:**
  - `debian_slim(python 3.11)` plus the apt packages of `Dockerfile.sim-cpu`;
  - `uv_sync(frozen=True, --no-dev, uv 0.11.11)` from `uv.lock`. Nothing calls `uv` at run
    time (KI-020).
  - The code directories are mounted at `/app`: `src`, `scripts`, `configs` and
    `data/{manifests,splits,normalization}`. No evaluation suite and no test state leave
    the machine (KI-033).
- **Volumes:** `roboforge-data` at `/data` (`ROBOFORGE_DATA_ROOT`; the processed HDF5,
  hash-checked against the plan in every attempt) and `roboforge-runs` at `/app/runs`.
  Modal mounts `/app/runs` through a symlink. Recorded paths stay `runs/<run_id>/...`
  (D-050) and resolve unchanged after download.
- **Policy for every function:**
  - `max_containers=1`, `retries=0`, `single_use_containers=True`;
  - `scaledown_window=2` s;
  - preemptible;
  - an explicit per-launch timeout via `with_options`, with the job's own deadline
    `deadline_margin_s` (180 s) earlier;
  - one small GPU at most (`gpu_probe`: L4); never multi-GPU.
- **Attempts:** Modal reschedules an input whose container exits on an interrupt, even with
  `retries=0`, and each attempt gets the full timeout (KI-031). The job therefore counts its
  launch's attempts in the run journal. At `max_attempts_per_launch` (4) it returns
  `status: error` without training. The launch upper bound prices all 4 attempts.
- **Functions:**
  - `train_cpu` (2 cores, 4 GiB): CPU training with 1 torch thread, the protocol's training
    setting (RESEARCH_PROTOCOL §6.2);
  - `gpu_probe` (L4, 4 cores, 16 GiB, 15 min): Gate A only.
    - Each section (CUDA info, EGL and OSMesa render checks, each family's CPU and CUDA
      training profile) is recorded separately, with tracebacks, and printed as one JSON
      line.
    - The render check is `scripts/render_probe.py`, which needs no dataset or suite.
  - GPU training is not a protocol path. It would need a dated amendment.
- **Engineering smoke only:**
  - `simulate_preemption_s` arms `remote.arm_preemption_drill`: a real SIGINT, then SIGKILL
    30 s later, as in a Modal preemption, on a launch's first attempt only.
    `modal.exception.simulate_preemption` is not used: it raises `KeyboardInterrupt` and
    bypasses the job's handler.
  - `stop_at_step` ends a launch cleanly, so a separate `resume` launch has work.
- **Default is a dry run.** `python infra/modal_app.py train|probe ...` prints the plan and
  the upper-bound estimate, logs a `dry_run` event and contacts no server. It was verified
  with an empty `HOME` and no Modal token.
- **A paid launch** needs `--budget-approved A-xx`. The gate (`roboforge.compute.plan
  authorize`) checks:
  - a clean tree;
  - an unused run ID (`new`) or a previously launched one (`resume`);
  - the approval row in `docs/EXPERIMENT_LEDGER.md` (status `granted`, function in scope);
  - committed spend + upper bound ≤ cap.

  Only then does it log the `launch` event. The launcher records the result even when the
  remote call raises `KeyboardInterrupt`.
- **Kill path:** Ctrl-C in the launching terminal (it stops the ephemeral app), or
  `python infra/modal_app.py cancel --app-id ap-...` (`modal app stop --yes`).
- **Platform:** a Modal image is a different platform from the canonical container (D-017).
  It never produces canonical evaluation evidence. Modal-trained checkpoints are evaluated
  only on the canonical container after a hash-checked download. Session 5 did exactly that
  (E5-08).

### D-048 Makefile defaults bumped to Session 5 (FIXED)
`EVIDENCE_SESSION` now falls back to 5 and `PREFIX` to `s5`, so that a Session 5 command
cannot write into `reports/session4` or reuse the `s4` prefix by omission. Every Session 5
command also passed `ROBOFORGE_EVIDENCE_SESSION=5` explicitly. `make modal-sim` takes
`SIM_ROOT`, `SIM_PREFIX` and `SIM_REPORT`. **Session 6 must pass `EVIDENCE_SESSION=6
PREFIX=s6` (or bump the defaults as a recorded change).**

### D-049 Cost ledger and owner approvals (FIXED)
- `reports/compute/cost_ledger.jsonl` (`roboforge.cost_event/1`) is append-only, committed
  and cumulative across sessions. Its events:
  - `dry_run`;
  - `launch` (approval, resources, timeout, upper bound);
  - `result` (status, measured container seconds, list-price cost of that time);
  - `billing` (amounts from `modal billing report`, never computed).
- The spend counted against a cap is per launch: billed if reconciled, else the
  measured-time estimate, else the upper bound. The upper bound covers
  `max_attempts_per_launch` attempts of the full timeout, plus the build/start allowance
  (D-047).
- Approvals are rows of the table "Owner approvals for paid compute" in
  `docs/EXPERIMENT_LEDGER.md`: ID, date, gate, functions, cap, status, evidence (the owner's
  words). `roboforge.compute.approval` parses them.
- **Not a hard stop:** the gate is client-side, and Modal can act outside it (KI-031).
  Only a Modal workspace budget caps spending.
- A closed approval is set to `exhausted`. A-01 was closed at the end of Gate A (E5-17).

### D-050 Repository-relative paths are lexical first (FIXED)
`roboforge.paths.repo_relative` now tries the unresolved absolute path first, and resolves
symlinks only if that path is not inside the repository.
- **Why:** a directory mounted through a symlink inside the repository keeps its
  repository path. Modal mounts `/app/runs` that way, and the old resolve-first rule wrote
  `/__modal/volumes/<id>/...` into the run journal (KI-032).
- **Effect:**
  - Paths without symlinks are unchanged. Every Session 1–4 manifest still verifies.
  - Paths outside the repository are recorded resolved, as before (`/out` in a container).
  - `tests/integration/test_remote_job.py` and the local simulation mount runs through a
    symlink and assert `runs/<run_id>/...`.

### D-051 Main-suite compute path: local CPU only (FIXED by owner Gate C, 2026-11-07)
- The owner chose **local CPU only, $0** for the main suite (Gate C; EXPERIMENT_LEDGER
  Gate C entry).
  - Training: the development host, 1 torch thread, at most 8 concurrent runs (measured
    slowdown 1.24 at 8).
  - Evaluation: the canonical container, at most 4 in parallel (measured 1.78× throughput;
    `physics_sha256` unchanged).
- **No Modal resource is used in Sessions 6–7** without a new recorded approval. A-01 is
  closed.
- **Training budget** (RESEARCH_PROTOCOL §13, 2026-11-07): 100,000 steps, checkpoints every
  5,000, identical across the 4 families and both tasks.
- **Why local:**
  - It is the protocol's training platform, the same as tuning trial 1.
  - It is free.
  - It is faster in wall time than Modal CPU at `max_containers=1`: 97–132 sequential
    container-hours for the full suite, $12.35–16.72.
  - Modal GPU ($12.36–16.02) would be a protocol deviation, and its CUDA kernels are not
    bitwise deterministic.
  - Figures: `reports/session5/cost/suite_projection.json` (estimates).

### D-052 Session 5 evidence location (FIXED)
`reports/session5/` and `logs/session5/`, with `ROBOFORGE_EVIDENCE_SESSION=5` set
explicitly, plus the cumulative `reports/compute/cost_ledger.jsonl`. The fetched
Modal-trained run (`runs/s5modal2_*`) and the simulated runs (`runs/s5sim*`) are gitignored
like every run. Their manifests, hashes and retrieval reports are in `logs/session5/modal/`
and `reports/session5/modal_sim/`.

## Session 6 (2026-11-07)
Evidence: `reports/session6/` and `logs/session6/` (D-056). The suite code, the plan and the
predeclared search space (EXPERIMENT_LEDGER, Session 6) were committed before the first
main-budget run. Every reported run was launched by `scripts/run_suite.py` from that clean
commit; no tracked file was edited while runs were in flight.

### D-053 Makefile defaults bumped to Session 6; recorded overrides for `make train` (FIXED)
- `EVIDENCE_SESSION` now falls back to 6 and `PREFIX` to `s6` (as D-048 did for Session 5),
  so no Session 6 command can write into `reports/session5` or reuse the `s5` prefix by
  omission.
- `make train ... SET="section.key=value ..."` passes recorded `--set` overrides.
- `make train-main` adds the main budget (`MAIN_BUDGET`: `train.max_steps=100000`,
  `train.ckpt_every=5000`) and refuses any `SET` that touches `train.*`, so a single run can
  never change the budget.
- `make suite-plan` / `make suite` resolve and run a suite plan (D-054).
- The model configs keep the 30k pilot budget: they are the record of the pilot
  configuration. The main budget is applied as overrides, which every run stores in its
  resolved config and manifest.

### D-054 Main-suite plan, orchestrator and the §6.3 choice rule (FIXED)
- **Plan** (`configs/experiments/level1_core_lift.yaml`, `roboforge.train.suite`,
  schema `roboforge.suite_plan/1`): task, prefix, the budget (exactly `train.max_steps` and
  `train.ckpt_every`, applied to every run), the tuning seed 0, the replicate seeds 1 and 2,
  and per family the base config and its trials.
  - Trial 1 must be the base config unchanged. Trials 2–4 may change only `name`,
    `optim.lr`, `optim.schedule` and `optim.weight_decay` (`TRIAL_KEYS`). Architecture,
    horizons, parameter count, batch, loss and budget therefore cannot differ between
    trials, and the loader also checks that the resolved configs agree on them.
  - At most 4 trials per family; no two trials may resolve to the same configuration.
  - `load_plan` resolves every run's config and run ID with the same functions as
    `scripts/train.py` (unit test), so the trial → run-ID table is fixed in advance
    (`logs/session6/suite/level1_core_lift/plan_resolved.json`).
- **Choice rule** (`roboforge.eval.tuning.choose_configuration`): among a family's seed-0
  trials, the one whose §7-selected checkpoint has the highest validation success count,
  then the lower mean first-success step, then the lower trial number. A trial whose run
  failed with a manifest (`status: failed`) counts against the budget, is never retried and
  ranks last; if every trial failed the rule raises. Missing trials, other seeds, other
  tasks or differing budgets raise. Output: `reports/session6/tuning/<family>_lift.json`.
- **Orchestrator** (`scripts/run_suite.py`): a dependency graph, not manual steps:
  train every trial (seed 0) → evaluate each checkpoint in its own canonical container
  (`eval_policy.py --steps N`, all 50 validation starts) → `select_checkpoint.py` (§7) →
  choice rule (§6.3) → train the chosen trial with seeds 1 and 2 → evaluate → select.
  - Limits (Gate C, D-051): ≤ 8 training processes, ≤ 4 containers, ≤ 10 processes in all
    on the 10-core host. Priority: the first family's training (diffusion, the critical
    path), then evaluations, then other training.
  - Every launch checks that code, configs and docs are committed (`git_revision`), except
    with `--allow-dirty`, which only the rehearsal used.
  - Failures are recorded, never retried. An evaluation directory without an eligible
    summary stops the suite for a manual decision. A restart never retrains a finished run
    and recomputes each tuning choice, refusing one that differs from the recorded file.
  - SIGINT/SIGTERM stop new launches and ask training runs for a resume checkpoint;
    `--resume-interrupted` continues them (D-046).
  - Events: `logs/session6/suite/<plan>/events.jsonl`; per-job logs beside it; summary
    `reports/session6/suite/<plan>.json`.

### D-055 Session 6 tuning search space: the same three optimizer changes for every family (FIXED)
- Trial 2: cosine learning-rate schedule (1e-4 annealed to 0 over the 100k steps); trial 3:
  constant learning rate 3e-4; trial 4: AdamW weight decay 0.1. Each changes one setting of
  trial 1, identically for BC-MLP, BC-MLP-chunk, BC-RNN and diffusion. Values and reasons:
  EXPERIMENT_LEDGER, Session 6 search space.
- **Why identical optimizer axes:** the protocol requires an equalized, bounded search
  (MASTER_SPEC §5.7, RESEARCH_PROTOCOL §6.3). Optimizer settings are DEFAULT in §6.2 and
  apply to every family, whereas the alternatives would break a fixed part of the protocol:
  widths (capacity rule §6.1), horizons (H2-c matching and the H3 ablation), DDIM steps and
  EMA (Session 7 ablations, §6.4), dropout (absent from the diffusion U-Net, so the search
  would differ between families) and observation noise (not implemented).

### D-056 Session 6 evidence location (FIXED)
- `reports/session6/{validation,selection,tuning,suite,figures,diagnostics,cost_latency}/`
  and `logs/session6/{prereq,suite,rehearsal}/`, with `ROBOFORGE_EVIDENCE_SESSION=6`.
- Training and evaluation records stay under `runs/` (gitignored); their summaries and hashes
  are committed in the reports.
- The orchestrator rehearsal (`s6dev_*`, 200 steps) wrote to `logs/session6/rehearsal/`. It
  is development only: never a trial, never in `reports/session6/validation/`.

### D-057 Validation figures and episode montages: inputs, labels and a fixed example rule (FIXED)
- **`scripts/plot_validation.py`** (`make figures`) reads only the suite summary
  (`reports/session6/suite/level1_core_lift.json`), the validation summaries it cites, the
  training metrics of those runs, the committed training split (action histograms, via
  `SequenceDataset.from_committed`) and the canonical latency report. It writes:
  - `training_curves.png`: training loss, and offline validation-demo MSE marked "never
    selects" (KI-024);
  - `validation_success.png`: successes per checkpoint, with the §7 selection ringed;
  - `seed_spread.png`;
  - `action_histograms.png`, with motion and gripper dimensions apart (KI-022);
  - `success_vs_latency.png`;
  - `validation_table.md` and `index.json`, with input and output hashes.
- **Labels:** every figure and the table carry the same footer: "VALIDATION ONLY …
  Development numbers, not the held-out test", the source file and the git revision.
  Colours: one fixed hue per trial (validated categorical palette, CVD-checked). Replicate
  seeds are dashed (seed 1) or dotted (seed 2), and a legend is always present.
- **`scripts/render_episodes.py`** draws the success and failure examples by a rule fixed in
  the script before any rendering. For each family, among the chosen configuration's
  selected checkpoints ordered by seed, it takes the first success (sorted `state_id`) of
  the lowest seed, and the first failure of the lowest seed that has one.
  - Tiles are drawn from the **recorded** physics states with `ShadowRenderer` on scratch
    data after `restore()`. Nothing is re-simulated, and no live data is touched (KI-019).
  - The recorded `physics_sha256` is checked against the summary.
  - Frames are labelled with the render platform: illustration only, the records are the
    evidence.
  - Output: `episodes_<family>.png` and `episodes_index.json` (rule, chosen episodes, record
    hashes).
- Neither script loads an evaluation suite other than validation, and neither selects
  anything (contamination check, LEDGER E6-11).
- **Evidence locations** extend D-056: `logs/session6/{analysis,diagnostics,final}/` and
  `reports/session6/{suite,selection,tuning,cost_latency,diagnostics,figures}/`, plus
  `reports/session6/lift_test_configuration.json`.

### D-058 Lift test configuration recorded, not frozen (FIXED)
- **Machine-readable record:** `reports/session6/lift_test_configuration.json`, written by
  `logs/session6/analysis/test_configuration_record.py` from the tuning choices and the
  selection records. It lists, per family:
  - the chosen trial, its overrides and its config SHA-256;
  - for seeds 0, 1 and 2, the training run ID, the §7-selected step, the checkpoint SHA-256
    and the validation result it was selected on.

  That is 12 checkpoints. HANDOFF repeats them.
- **Held-out initial-state schedule:** referenced **only** by its manifest content hash
  `e02498651b9d5bb88b6aec0870c5e3277cadaaeaf0e325abf86c79e89b1f22da` (50 states). The hash
  is the one field read from `reports/session2/eval_states_lift.json`. No test state, ID or
  statistic was loaded, printed or named. It differs from the validation hash
  (`0b1368fa…`) used by every Session 6 summary.
- **This is not the freeze.** Rollout counts, the paired CI method, aggregation, the failure
  taxonomy and the example-selection rule for the test belong to `docs/FINAL_TEST_FREEZE.md`
  in Session 7, together with Can. The Lift entries may be frozen as recorded here or not at
  all; they may not be re-selected.

### D-059 Session 6 scheduling under host contention: renice BC training, never relaunch (FIXED)
- While a foreign CPU-bound process shared the host (KI-035),
  `logs/session6/suite/renice_loop.sh` set BC training processes to nice 15 every 60 s.
  Diffusion stayed at the inherited nice 5, so the diffusion runs on the critical path won
  contention.
- Scheduling only: CPU training with 1 thread is deterministic, and evaluation physics does
  not depend on wall time. The trial-1 checkpoints and evaluations reproduce the pilots
  exactly (E6-07). No run was stopped, retried or relaunched.

## Session 7 (2026-11-07 onward)
Evidence: `reports/session7/` and `logs/session7/` (D-062). The Can plan and its predeclared
search space (EXPERIMENT_LEDGER, Session 7) were committed before the first Can training run.
Lift selections are final (D-058): no Lift run is retrained, retuned or reselected.

### D-060 Makefile defaults bumped to Session 7 (FIXED)
- `EVIDENCE_SESSION` now falls back to 7 and `PREFIX` to `s7` (as D-048 and D-053 did), so
  no Session 7 command can write into `reports/session6` or reuse the `s6` prefix by
  omission.
- `SUITE_PLAN` defaults to `configs/experiments/level1_core_can.yaml`, and `SUITE_SUMMARY`
  follows the plan's stem (`reports/session$(EVIDENCE_SESSION)/suite/<plan stem>.json`), so
  `make figures` reads the summary of the plan it is given. The Session 6 figures are
  reproduced with `make figures SUITE_PLAN=configs/experiments/level1_core_lift.yaml
  EVIDENCE_SESSION=6`.

### D-061 Can tuning search space: the Lift trials of D-055, unchanged (FIXED)
- Can uses exactly the Session 6 trials for every family: trial 1 the pilot configuration
  at the main budget, trial 2 cosine schedule, trial 3 lr 3e-4, trial 4 weight decay 0.1
  (`configs/experiments/level1_core_can.yaml`, SHA-256 `881b4fb7…`).
- **Why the default:** the reasons of D-055 hold on Can (an equalized, bounded search on
  optimizer settings that apply to every family; widths, horizons, DDIM steps, EMA, dropout
  and observation noise stay out for the same reasons). Reusing the same axes keeps the two
  tasks comparable as a protocol, and choosing Can-specific trials before any Can policy
  exists would have no evidence behind it. No Lift result was used to change the Can space.
- Budget, seeds, the §7 rule and the §6.3 choice are those of Session 6; prefix `s7`.

### D-062 Session 7 evidence location (FIXED)
- `reports/session7/` and `logs/session7/` with `ROBOFORGE_EVIDENCE_SESSION=7`. Training and
  evaluation records stay under `runs/` (gitignored), with summaries and hashes in the
  reports, as in D-056.

### D-063 Session 7 robustness starts: a fresh base set and OOD sets derived from it (FIXED)
- `roboforge.sim.robustness`, config `configs/eval/robustness.yaml`, CLI
  `scripts/generate_robustness_states.py` (`make robustness-states` on the canonical
  container, `make check-robustness-states`). Definitions: RESEARCH_PROTOCOL §14.1–14.2.
- **Separate from the evaluation suites:** the robustness suites are not in
  `configs/eval/suites.yaml`, so `partitions.json`, the validation and held-out manifests and
  their checks are unchanged. They use the same file format and load with
  `eval_states.load_suite`. New partition `eval_robustness` (`snapshot.PARTITIONS`) and a
  reserved seed block [300000, 400000) that `check_seed_allowed` maps to it.
- **Why a fresh base set:** the validation starts selected every checkpoint and
  configuration, so a robustness drop measured against them would include the selection
  bias. Base state `i` and OOD state `i` are paired.
- **Validity** is checked on scratch environments only (KI-019). The development dry run
  corrected two checks before any canonical generation (second amendment of 2026-11-07).

### D-064 Evaluator extensions for robustness suites and inference variants (FIXED)
- `evaluate_checkpoint(..., suite=, variant=)` and `scripts/eval_policy.py --suite --variant`.
  `check_suite_allowed` accepts only `validation` and the task's robustness suites; the
  held-out suites have their own entry point after the freeze.
- Variants (`roboforge.eval.learned.VARIANTS`, §14.4): `trained-weights` (A1: the non-EMA
  weights stored in the same checkpoint) and `ddpm100` (A2: DDPM with all 100 timesteps;
  the U-Net weights are unchanged and load strictly). Diffusion checkpoints only.
- Validation evaluation IDs, summaries and selection eligibility are unchanged. A robustness or
  variant evaluation has ID `<train>__<suite>_step<N>__<platform>[__ps0r0][__var-<variant>]`,
  schema `roboforge.robustness_eval/1`, `eligible_for_selection: false`, and its summary goes
  to `reports/session7/robustness/evals/`.

### D-065 Preregistered failure taxonomy (FIXED; replaces the post-hoc categories)
- `roboforge.eval.taxonomy`, config `configs/eval/failure_taxonomy.yaml`, CLI
  `scripts/failure_taxonomy.py`. Rules: RESEARCH_PROTOCOL §14.5. Computed only from the
  recorded arrays; contacts are not recorded, so "grasp-lifted" means a rise ≥ 1 cm above the
  rest height (minimum z over frames 0–10) while the executed gripper command was closing.
- **Why the rest height:** Lift drops the cube 1 cm onto the table at t = 0, so the post-hoc
  `diagnose_actions` rule (rise from frame 0) missed lifts of 1–2 cm.
- **Dry run** (LEDGER E7-02): on all 12,000 Session 6 Lift episodes of the 12 chosen runs,
  every episode has exactly one label and the failure totals equal the post-hoc ones
  (BC-MLP 439, BC-MLP-chunk 257, BC-RNN 146, diffusion 18); some post-hoc "missed grasp then
  drift" episodes become `dropped`. The Can dry run follows on the Session 7 validation
  records before the freeze.
- `scripts/diagnose_actions.py` keeps its post-hoc categories for the Session 3–6 reports;
  every Session 7 failure analysis uses this taxonomy.

### D-066 The §9 hierarchical paired bootstrap as implemented (FIXED)
- `roboforge.eval.statistics.hierarchical_paired_bootstrap`: per family an outcome matrix
  (seeds × states, one state order for all families); B = 10,000 replicates, seed
  `20261106`. Each replicate draws the state indices once for every family (pairing) and the
  seed indices independently per family; statistic = mean over resampled seeds × states;
  95 % percentile intervals (numpy linear interpolation); differences on the same replicates.
- **Draw order** (fixed, part of the freeze): one `default_rng(20261106)`; first the
  `(B, 50)` state indices, then each family's `(B, n_seeds)` seed indices in the declared
  family order.
- `separation()`: "separated (+/−)" when the interval excludes 0, else "not separated"
  (§14.6). Wilson intervals per checkpoint as before.
- Unit tests on synthetic controls: identical families not separated, certain differences
  exact, one shared state draw (paired difference exactly 0 under identical outcomes), seed
  resampling per family, reproducibility, a binomial sanity interval, input validation.

### D-067 The held-out freeze, the one-shot evaluator and the container attestation (FIXED)
- **Freeze format.** `docs/FINAL_TEST_FREEZE.md` holds prose plus exactly one machine-readable
  YAML block (schema `roboforge.heldout_freeze/1`) between `<!-- roboforge:freeze:begin -->`
  and `<!-- roboforge:freeze:end -->`, written by `scripts/write_freeze_block.py` from the
  committed test-configuration records (`scripts/test_configuration_record.py`). Per task: the
  held-out manifest content hash, and per family (order bc_mlp, bc_rnn, bc_mlp_chunk,
  diffusion_unet) the chosen trial and seeds 0–2 with run ID, selected step, checkpoint and
  config SHA-256; the random baseline (policy seed 0); rollouts (R = 1, policy seed 0, 400
  steps, first success); the analysis (bootstrap B, seed, draw order, comparisons, secondary
  outcome, separation rule, aggregation per task); the taxonomy hash; the example rule; and
  the SHA-256 of every file that runs or analyses the test (`code_sha256`).
- **One-shot evaluator.** `roboforge.eval.heldout.evaluate_entry` (via
  `scripts/eval_heldout.py`, orchestrated by `scripts/run_heldout.py`, ≤ 4 containers)
  refuses unless the freeze is committed unmodified, the frozen code is unchanged, the entry is
  in the freeze, the platform is canonical, the manifest hash equals the frozen one and the
  checkpoint SHA-256 equals the frozen and training-manifest values. States load only through
  `eval_states.load_suite` (lock check). An existing evaluation is never rerun; one
  engineering rerun (§14.6) needs an explicit reason, keeps the incomplete attempt and records
  the incident. Summaries are opened exclusively (`x`), never overwritten.
- **Container attestation.** The container has no `.git`, so the lock check could not ask git
  there. `provenance.tracked_and_unmodified` asks git on the host; inside the container it
  accepts only `docs/FINAL_TEST_FREEZE.md`, only when the wrapper attests a clean host tree
  (`ROBOFORGE_GIT_DIRTY=0`) and passes the host's committed blob ID
  (`ROBOFORGE_FREEZE_BLOB = git rev-parse HEAD:docs/FINAL_TEST_FREEZE.md`) equal to the git
  blob hash of the mounted file. `eval_states.lock_status` uses it.
- **Rehearsal.** The identical path runs on a non-locked suite with an uncommitted draft
  freeze (`--rehearsal-suite`, `--draft-freeze`), never on the locked suite; output under
  `logs/session<N>/rehearsal/`, labelled REHEARSAL, never reported as a test result.
- **Analysis.** `scripts/analyze_heldout.py` reads only the frozen summaries and their
  records and writes `reports/session<N>/results/` (results.json, generated tables.md,
  taxonomy.json) once: per checkpoint k/n with Wilson intervals; the §9 bootstrap per task for
  success (H1, H2, H2-c, H3) and the censored episode length (H2, H2-c, H3); engineering
  status (missing summaries, short runs, failed audits kept apart from in-episode errors,
  which count as outcomes); taxonomy counts; the frozen examples; cost and latency.
- Tests: `tests/unit/test_freeze.py`, `tests/unit/test_analyze_heldout.py`.

### D-068 Running and analysing the §14 robustness and ablation evaluations (FIXED)
- `scripts/run_evals.py` runs every preregistered robustness/ablation evaluation from the
  committed test-configuration records (12 selected checkpoints per task) in its own
  canonical container, ≤ 4 at a time: batch `robustness` (each checkpoint on every set of
  `robustness.sets_for(task)`; Lift 36, Can 24), `a1` (diffusion, `robustness_base`,
  `--variant trained-weights`), `a2` (`--variant ddpm100`), `a3` (the A3 suite's selected
  checkpoints on `robustness_base`). Existing summaries are skipped, failures not relaunched,
  clean tree at every launch; events in `logs/session<N>/robustness/events.jsonl`.
- `scripts/analyze_robustness.py` writes `reports/session<N>/robustness/` (analysis.json,
  generated tables.md, taxonomy.json). H4: one §9 bootstrap per task over the keys
  family:set; the base index is drawn once for every family and set, and each family's seed
  draw is shared by its sets (**seed groups**, `hierarchical_paired_bootstrap(groups=...)`)
  so the drop base − OOD is paired by seed and state; between-family differences in OOD
  success and (as **contrasts**) in drop for the H2, H2-c and H3 pairs. A1/A2: primary −
  variant, one seed group. A3: T_o = 2 − T_o = 1, independent seed draws (different runs).
  Without `groups` the bootstrap's draws are unchanged (tested), so the held-out analysis
  is unaffected.
- **A3 training.** The arm is the Can-chosen diffusion configuration with `horizons.obs = 1`,
  written as its own base config (horizons cannot change in a tuning trial, TRIAL_KEYS) and
  run as a one-trial plan. With one trial there is nothing to choose, so the plan key
  `replicates_with_tuning: true` (allowed only when every family has exactly one trial)
  trains seeds 0–2 together; §7 selection per run is unchanged. A 200-step native smoke run
  on Can trained and reloaded (1,063,367 parameters at T_o = 1; 1,092,551 at T_o = 2).

### D-069 Pre-freeze fixes to the held-out path; one host-wide container pool (FIXED)
Found by review before any freeze, rehearsal or held-out run:
- **The container could not see `infra/`.** `code_problems` runs inside the container for
  real held-out runs and the frozen files include `infra/containers/run_sim_cpu.sh`, which
  the wrapper did not mount: every held-out evaluation would have refused itself. The
  wrapper now mounts `infra/` read-only.
- **Frozen code is the whole package.** The held-out path imports models, training-manifest,
  data and simulator code, so `code_sha256` covers every `src/roboforge/**/*.py`, the three
  held-out scripts, the wrapper and `configs/eval/*.yaml` (all mounted in the container).
- **The analysis writes last.** `analyze_heldout.py` computes everything (and checks the
  latency reports exist) before it creates `reports/session<N>/results/`, so a failure
  leaves nothing that would block the single run.
- **An evaluation still incomplete after its one rerun** makes its family
  ENGINEERING-INCOMPLETE on that task: listed per checkpoint, left out of that task's
  bootstrap (the draw order skips it), its declared comparisons reported as not computed,
  the others computed as frozen. Declared in the freeze.
- **One host-wide container pool.** Each orchestrator's `--max-eval` alone would let two
  orchestrators exceed Gate C's 4 containers together. `roboforge.slots.ContainerSlots`
  is a pool of 4 slot files (`logs/container_slots/`, created with `O_EXCL`, reclaimed when
  their PIDs are dead) that `run_suite.py`, `run_evals.py` and `run_heldout.py` all draw
  from. Containers started directly by Make targets (state generation, latency) are run
  only when no orchestrator runs.
- **Free-disk check** (≥ 4 GiB) before every launch in the three orchestrators: below it
  they stop launching and report what was not launched.
- `scripts/test_configuration_record.py` reproduces the committed Lift record
  (`reports/session6/lift_test_configuration.json`) exactly, apart from its `status`
  wording and `note`, `git` and `generated_utc`.

### D-070 The suite orchestrator tolerates files that running jobs are still writing (FIXED)
- **Why:** the Can main suite (started 2026-11-07 23:26:38 UTC at 96aa972) died at
  2026-11-08 02:14:19 UTC on a `JSONDecodeError`, reading an empty validation summary that a
  running evaluation container was writing (KNOWN_ISSUES KI-036).
- **Change (`scripts/run_suite.py` only):**
  - `tick` does not read the summary of a step whose evaluation is running;
  - a summary or manifest that does not parse is "not there yet" while its job runs, and a
    `SuiteError` (manual decision) otherwise;
  - any unexpected exception now writes `suite_error` (with the traceback), waits for the
    running jobs, and still writes `end`.
- **Not changed:** the training and evaluation code. `git diff 96aa972` for the restart
  commit lists only `scripts/run_suite.py`, its unit test (`tests/unit/test_run_suite_reads.py`)
  and these docs, so the evaluations and replicate runs after the restart use the same
  `src/`, `scripts/train.py` and `scripts/eval_policy.py` as the first 44 evaluations and the 16
  tuning runs. The writers stay non-atomic until the suite is done.
- **Restart:** same plan (sha256 881b4fb7…), same limits, no `--resume-interrupted` (no run
  was interrupted). The first run's `plan_resolved.json`, `status.json` and stdout are kept as
  `*.run1.*` under `logs/session7/suite/`.
- **Merge with the s7-dev tooling (D-069):** the drain fallback after a `reap_error` also
  releases the host-wide container slot of every job it drops.

### D-071 A3 configuration and plan (FIXED; written after the Can diffusion choice)
- **Arm T_o = 1:** `configs/models/diffusion_unet_to1_can.yaml` is the chosen Can diffusion
  configuration (`reports/session7/tuning/diffusion_unet_can.json`, written 2026-11-08 07:37
  UTC: trial 2, cosine schedule, run `s7_diffusion_unet_cos_can_seed0_99251da3`, selected step
  65,000, 48/50; trials 1 and 4 also scored 48/50 and lost on the mean first-success step)
  with `horizons.obs = 1`. `tests/unit/test_suite_plan.py` checks that the resolved configs
  of seeds 0-2 differ from the chosen trial's only in `horizons.obs` and `name`.
  1,063,367 parameters (19-D FiLM condition) vs 1,092,551.
- **Arm T_o = 2:** the main suite's three chosen diffusion runs, unchanged.
- **Plan:** `configs/experiments/ablation_a3_can.yaml`, one trial, `replicates_with_tuning:
  true` (D-068), the main budget, the §7 selection on all 20 checkpoints. Its family label
  `diffusion_unet_to1` gives it its own tuning record, so it can never overwrite or contradict
  the main suite's `diffusion_unet_can.json`.
- **Why written now:** §14.4 fixes A3 as "the chosen Can diffusion configuration retrained
  with `horizons.obs = 1`", which is known only after the choice. Nothing else about A3 was
  decided after any Can result: the arms, seeds, budget, evaluation suites and analysis are
  those of §14.4.

### D-072 Executing the freeze and the one held-out run (FIXED, Session 7)
- **Order kept (D-067):**
  1. The robustness and ablation evaluations ended and their evidence was committed
     (`a92c74d`).
  2. The container attestation showed both tasks **locked**
     (`logs/session7/analysis/lock_status_before_freeze.log`).
  3. `scripts/write_freeze_block.py --document 2026-11-08` wrote `docs/FINAL_TEST_FREEZE.md`
     from the committed records at `a92c74d`.
  4. The freeze was committed alone with its tests (`bf0a0fa`).
  5. `make freeze-check`, `make lint` and `make test` (372 passed) ran next.
  6. A second container attestation showed both tasks **unlocked**, with the wrapper's freeze
     blob `bfdb4b77…` equal to `git rev-parse HEAD:docs/FINAL_TEST_FREEZE.md`
     (`logs/session7/analysis/lock_status_after_freeze.log`).
  7. `make eval-final` ran once, then `make analyze-final` ran once.
- **The rehearsal still holds.** The committed freeze differs from the rehearsed draft
  (`logs/session7/freeze_prep/draft_freeze.md`, generated at `005bc94`) in one line only, the
  date. Every frozen-code SHA-256 is identical. The commits between the rehearsal and the
  freeze touched no frozen file: `565c6e4` changed `scripts/analyze_robustness.py`, a unit test
  and docs (KI-038); `a92c74d` added evidence and `.gitignore` lines.
- **Tests changed with the freeze.**
  - `tests/research_integrity/test_final_test_lock.py` is replaced by its post-freeze form
    (drafted and reviewed before the freeze). It checks that the committed freeze names both
    held-out manifests and opens the lock, and that the lock stays shut for a missing,
    uncommitted or non-naming freeze.
  - `tests/simulator/test_env_adapter.py::test_seed_resets_and_the_final_test_seed_lock`
    asserted that `env.reset(900003)` is refused. With the lock open, that call would have
    built a held-out start on macOS during `make test`. The test now points the lock at a
    missing freeze, asserts the lock is shut, and then expects the refusal. No test ever builds
    a held-out start; only `scripts/eval_heldout.py` reads them.
- **Not committed:** `logs/session*/freeze_prep/`, which named the held-out suites before the
  freeze, and the rehearsal episode records under `logs/session*/rehearsal/heldout/runs/`
  (218 MB). Both are `.gitignore`d. The rehearsal summaries, check and analysis are committed.
- **Boundary (MASTER_SPEC Session 7 STOP):** from here on, nothing may change the Level 1 test
  configuration, the labels or the findings. That includes later RGB or point-cloud results.

## Session 8 (2026-11-08 onward)
Level 1 is frozen (`docs/LEVEL1_COMPLETE.md`). Nothing below changes a Level 1 configuration,
label or finding; RGB work is a separately labelled observation-modality study (MASTER_SPEC
§3.2 item 5).

### D-073 Makefile defaults bumped to Session 8 (FIXED)
- `EVIDENCE_SESSION` falls back to 8 and `PREFIX` to `s8` (as D-048, D-053 and D-060 did), so
  no Session 8 command can write into `reports/session6`–`7` or reuse the `s7` prefix by
  omission. `SUITE_PLAN`, `FREEZE_CONFIGS`, `TEST_TASK` and `LATENCY_ARGS` keep their Session 7
  values; the held-out targets (`eval-final`, `analyze-final`) are never run again (Session 7
  STOP boundary).
- New targets: `make rgb-feasibility` (canonical container) and `make rgb-feasibility-native`
  (development only), both writing `reports/session<N>/rgb_feasibility/<platform_id>/`.

### D-074 RGB frames are rendered from recorded states through the RoboForge render path (FIXED for the feasibility pilot; binding for any RGB dataset unless a later decision changes it)
`src/roboforge/data/rgb.py`, `src/roboforge/data/rgb_checks.py`, `scripts/rgb_feasibility.py`.
- **Source:** the raw files `data/raw/robomimic_v1.5/<task>/ph/demo_v15.hdf5` (SHA-256 checked
  against the manifest) hold per demo the states `[time, qpos, qvel]` and `model_file`, no
  images. Frame `t` is rendered from `states[t]`; it shares the index with `obs[t]` and
  `actions[t]` of the committed processed dataset. The raw file's XML, states and actions are
  checked bitwise against the low-dim file the processed dataset was built from, and the
  features of the raw states against the processed observations, for every demo.
- **Path:** the demo's `model_file` is compiled by `reset_to_state` (robosuite's reset toggles
  every visualization site off) and rendered by `ShadowRenderer` on scratch `MjData` (geom
  groups `(0, 1, 1, 0, 0, 0)`, the robosuite offscreen convention), never on live data
  (KI-019). Closed-loop evaluation renders through the same class (`RoboForgeEnv.render`).
  No robosuite transform helper is used (KI-018): projections use MuJoCo's `cam_xpos`,
  `cam_xmat` and `cam_fovy` in float64.
- **Warm-up:** the first render of a fresh GL context can differ from every later render of the
  same state by a few pixels (measured natively with CGL: up to 6 px on Can). Every kept frame
  is a steady-state frame: each new context first renders once and discards it
  (`WARM_UP_RENDERS = 1`). Any RGB policy's closed-loop observation path must do the same.
- **Cameras:** `agentview` and `robot0_eye_in_hand`, square, the robomimic image setup
  (`generate_paper_configs.py` at `d309eae`: these two cameras at 84 × 84 with
  `robot0_eef_pos`, `robot0_eef_quat`, `robot0_gripper_qpos`). Sizes 84 and 128 are both
  measured; the choice belongs to the RGB experiment freeze.
- **Alignment record** per frame (schema `roboforge.rgb_frame/1`): task, demo, step index,
  state SHA-256 (as `sha256_state`), model XML SHA-256, camera name with pinhole intrinsics and
  world extrinsics, `MUJOCO_GL` backend and frame SHA-256.
- **Platform:** frames that train or evaluate a reported RGB policy are rendered on the
  canonical container (OSMesa), like every reported rollout (D-017). Native CGL frames are
  development-only.
- **Nothing here reads a held-out state:** inputs are demonstration states and, for the
  evaluation-model parity check and the step cost, validation start 0.

### D-075 Session 8 evidence location (FIXED)
- `reports/session8/` and `logs/session8/` with `ROBOFORGE_EVIDENCE_SESSION=8`, as D-062.
  RGB feasibility evidence: `reports/session8/rgb_feasibility/<platform_id>/`.

