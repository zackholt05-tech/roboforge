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


### D-076 Gate D decided: a $1 L4 timing probe first, no main RGB training yet (OWNER, 2026-11-08 21:21 EST)
- The owner chose option D of `docs/COMPUTE_PLAN.md` §6.3, "Modal GPU probe first (≤ $1)".
  It is recorded as approval **A-02**: function `rgb_probe` only, cap $1.00
  (`docs/EXPERIMENT_LEDGER.md`).
- The approval does not cover any main RGB training, local or remote. That needs a
  separately capped plan and a new owner decision.
- GPU training would also need a dated RESEARCH_PROTOCOL amendment. The protocol trains on
  CPU in float32, and default CUDA kernels are not bitwise deterministic.

### D-077 The RGB probe's design (FIXED for A-02)
`src/roboforge/compute/{rgb_bench,rgb_probe}.py`, the `rgb_probe` function of
`infra/modal_app.py`, and `configs/compute/modal.yaml`.
- **No data.** Inputs are random, so nothing touches a dataset, an evaluation state or a
  checkpoint. The function mounts only the runs Volume, and only for attempt markers.
- **Same step everywhere.** `rgb_bench.Bench.step` mirrors `Trainer.step`: host-to-device
  copy, crops, loss, backward, gradient norm, AdamW, and EMA where the head config enables
  it. One generator lives on the data's device, and the EMA is built after the move (KI-030).
  Inference is not timed on CUDA (KI-042).
- **Bound that holds:**
  - the job enforces `max_attempts_per_launch` itself through `_probes/<launch_id>/attempt_<n>.json` on the runs Volume, committed before any GPU work;
  - it starts no section after `timeout − 60 s`;
  - the timeout is 360 s, so the upper bound is $0.53 at list price and $0.92 even at the 1.75× region maximum, which the estimate does not model.
- **Amendment evidence in the same launch.** For ResNet-18 at 84² with both heads, the probe
  ran:
  - deterministic kernels (`use_deterministic_algorithms`, cuDNN deterministic, benchmark
    off, `CUBLAS_WORKSPACE_CONFIG=:4096:8`), timed;
  - two same-seed 20-step repeats in each mode.
- `_launch` maps functions by name explicitly and refuses unknown names. Only `train_cpu`
  and `gpu_probe` upload data.

### D-078 Gate D2 decided: L60, GPU training of the Lift RGB comparison (OWNER, 2026-11-08 21:52 EST)
- **Choice:** "L60: Lift, GPU, 60k (Recommended)" (COMPUTE_PLAN §7.4).
- **Approval A-03:** function `train_gpu`, cap $15. Expected cost is about $9.8 at list
  price.
- **Runs:** 6 on Lift, visual BC-MLP-chunk and visual diffusion, seeds 0–2.
- **Model and budget:** ResNet-18 trunks (GroupNorm, from scratch) at 84² with random crop 76;
  batch 64, 60k steps, a checkpoint every 5k (K = 12).
- **Device:** deterministic CUDA under the RESEARCH_PROTOCOL §13 amendment of 2026-11-08.
- **Launches:** at most 1 h each, resumed, strictly sequential.
- **Before the first launch:**
  - `docs/RGB_EXPERIMENT_FREEZE.md` is committed;
  - the RGB dataset is built in the canonical container and recorded by a decision;
  - the trainer, `train_gpu` and the closed-loop RGB policy path are implemented and tested
    on the CPU.
- **Not covered:** Can, tuning trials, and any run outside the freeze.


### D-079 The Lift RGB dataset: an image-only file paired with the low-dim dataset (FIXED)
`src/roboforge/data/rgb_dataset.py`, `scripts/build_rgb_dataset.py`, `make rgb-dataset` and
`make rgb-dataset-install`.
- **New data, added by decision.** Nothing existing in `data/` changes. The new pieces are:
  - `data/processed/robomimic_v1.5_ph/lift/rgb84_v1.hdf5`, gitignored like every processed
    file;
  - its committed manifest, `data/manifests/rgb/lift_rgb84_v1.json`.
- **Content:**
  - both cameras at 84², `uint8`, of every frame of all 200 demos, rendered under D-074 on
    the canonical container;
  - the per-step state SHA-256;
  - schema `roboforge.rgb_dataset/1`.
- **What it does not hold:** proprioception, actions and the split. These come from the
  committed low-dim dataset. The file pins that dataset's content hash
  (`processed_content_sha256`), and the builder checks every demo's raw-state features
  against it bitwise before rendering.
- **Build and install:**
  - the container mounts `data/` read-only, so the build writes `runs/rgb_data/lift/`;
  - the install step re-verifies the file and content hashes on the host before copying;
  - it never overwrites an installed file;
  - it refuses a partial (`--limit`) build.

### D-080 The RGB experiment is frozen in `docs/RGB_EXPERIMENT_FREEZE.md` (FIXED, before any RGB training run)
The freeze fixes:
- arms, observation, encoder, initialization rule and augmentation;
- budget (60k steps, batch 64, K = 12, seeds 0–2), device and launch order;
- data hashes (RGB content `41c0ea02…`, paired with low-dim `74e70119…`);
- the §7 selection rule on the 50 Lift validation starts;
- the closed-loop policy path, the held-out test and its analysis, the §9 bootstrap;
- the hashes of the 12 files that define the experiment.

Held-out access stays forbidden until three things are committed:
- all 6 selections;
- a decision re-gating the lock on the freeze;
- a test (freeze §9).


### D-081 The held-out lock is re-gated on the RGB freeze (FIXED, before any RGB held-out rollout)
**Context.** The Level 1 lock (`eval_states.lock_status`) has been open since the Level 1 freeze
(Session 7). RGB freeze §9 forbids RGB held-out access until three things are committed:
- all 6 selections;
- this decision;
- a test.

**Decision.** A second gate, `roboforge.eval.rgb_heldout.gate_status`, stands in front of every
RGB held-out evaluation. It opens only when all of these hold:
1. `docs/RGB_EXPERIMENT_FREEZE.md` is committed, with SHA-256 `47e7cb8e…`, the file of `d9c6217`;
2. every file in the freeze's §10 table has its frozen SHA-256;
3. the 6 selection artifacts in `reports/session8/rgb_selection/` exist with the SHA-256 values
   pinned in the module, and nothing else is in that directory. On the host they must also be
   tracked and unmodified;
4. D-081 is in HEAD's `docs/DECISIONS.md`;
5. the Level 1 lock is open for Lift, and Lift's held-out manifest is `e0249865…` (freeze §9).

**How "committed" is checked.**
- On the host, git answers.
- The container has no `.git`, and its frozen wrapper attests only the Level 1 freeze file, so
  inside it the gate uses the wrapper's clean-tree flag (`ROBOFORGE_GIT_DIRTY=0`). That flag
  covers `src`, `docs` and `configs` but not `reports`, `logs` or `runs`. The selection
  artifacts are therefore pinned by hash in the committed module.
- No frozen Level 1 file changes: `eval_states.py`, `heldout.py`, `provenance.py` and
  `run_sim_cpu.sh` are untouched, so `make freeze-check` stays valid.
- The new module reads the held-out states only through `eval_states.load_suite`, by the
  constant `LOCKED_SUITE`, so the research-integrity name scan needs no exception.
- No RGB freeze §10 file changes either: the runner and the analysis are new files
  (`scripts/run_rgb_heldout.py`, `scripts/analyze_rgb_heldout.py`).

**The one-shot runner** (`evaluate_selected`, `make rgb-heldout RGB_RUN=<run_id>`):
- checks the gate, the canonical platform and "already ran" before it reads any held-out state;
- uses each run's committed selection: the step, and a checkpoint SHA-256 checked against the
  training manifest and the file;
- follows freeze §8 (`policy_seed = 0`, `repeat = 0`, CPU, 1 thread);
- writes `runs/rgb_heldout/<eval_run_id>/` and `reports/session8/rgb_heldout/<run_id>.json`,
  and never overwrites either.

An **engineering rerun** follows RESEARCH_PROTOCOL §14.6, as in Level 1. It is allowed only
when a fault outside the policy stopped the attempt before its summary. It happens at most once,
needs a written reason, and keeps the incomplete attempt. A rehearsal mode runs the identical
path on the first validation starts, is labelled, is never reported, and does not open the gate.

**Analysis choices the freeze leaves open, fixed here before any rollout**
(`scripts/analyze_rgb_heldout.py`, run once):
- the family keys are `rgb_diffusion` and `rgb_bc_chunk`;
- the bootstrap draw order is states, then `rgb_diffusion` seeds, then `rgb_bc_chunk` seeds;
- the one comparison is `rgb_diffusion − rgb_bc_chunk`;
- states are in held-out manifest order, and seed rows in order 0, 1, 2;
- success is the only inferential statistic. First-success step, latency, the failure taxonomy,
  training wall time and billed cost are descriptive;
- a family with an engineering problem (an incomplete run, a failed `verify_record` or an
  episode error) is not bootstrapped, and is reported as engineering-incomplete;
- the Level 1 low-dim Lift numbers appear only as labelled context, never as a comparison.

**Test.** `tests/research_integrity/test_rgb_heldout_gate.py`:
- the gate is open now, and shut when any precondition is removed: decision, selection,
  freeze, §10 code, the Level 1 lock, or the container's clean tree;
- a refused RGB run never reads a held-out state (an autouse guard fails any read of the
  held-out suite or reset to its seed block);
- `heldout.evaluate_entry` refuses all 26 Level 1 entries: off the canonical platform, and,
  on it, because each already ran. It also refuses an RGB entry.

## Session 9 (2026-11-09 onward)
Level 1 (`docs/LEVEL1_COMPLETE.md`) and the RGB study (`docs/RGB_EXPERIMENT_FREEZE.md`, D-081)
are frozen. Nothing below changes a Level 1 or RGB configuration, label or finding. Level 2 is
a separately labelled, DP3-inspired explicit-3D-perception study (MASTER_SPEC §3.2 item 6,
§7.3).

### D-082 Makefile defaults bumped to Session 9 (FIXED)
- `EVIDENCE_SESSION` falls back to 9 and `PREFIX` to `s9` (as D-048, D-053, D-060 and D-073
  did). No Session 9 command can then write into `reports/session6`–`8` or reuse the `s8` prefix
  by omission. Evidence goes to `reports/session9/` and `logs/session9/` with
  `ROBOFORGE_EVIDENCE_SESSION=9`.
- New targets: `make pc-feasibility` (canonical container), `make pc-feasibility-native`
  (development only), `make pc-dataset` (container, staged under `runs/pc_data/<task>/`) and
  `make pc-dataset-install` (host; installing needs its own decision).
- The held-out targets of Level 1 and of the RGB study are never run again.

### D-083 Level 2 depth and point clouds come from recorded states through the D-074 path, with sites hidden (FIXED for the feasibility; binding for any Level 2 dataset unless a later decision changes it)
`src/roboforge/data/pointcloud.py`, diagnostics in `src/roboforge/data/pc_checks.py`,
`scripts/pc_feasibility.py`, `scripts/build_pc_dataset.py`.
- **Source and alignment** are those of D-074. Cloud `t` of a demo is built from raw
  `states[t]` and pairs with `obs[t]` and `actions[t]` of the committed processed dataset.
  The builder checks the raw-state features against the processed observations bitwise for
  every demo.
- **Depth.** `DepthRenderer` renders on scratch `MjData` that shares the compiled model, never
  the live data (KI-019).
  - It uses geom groups `(0, 1, 1, 0, 0, 0)`, and one discarded warm-up render per GL context.
  - It also switches **every site group off**. robosuite hides its visualization sites in
    colour only, with a non-positive alpha, and a hidden site still writes the depth buffer.
    Lift's grip-site sphere and its 10 m grip-site cylinder would otherwise put a phantom
    sphere and a line along the gripper axis into every depth image. Measured on
    `demo_0`, t = 0: 45 agentview pixels and 92 wrist pixels change, by up to 0.43 m.
  - E8-03 showed that switching the site groups off changes no RGB pixel. This is D-074's
    "every visualization site hidden", carried over to depth.
- **Conversion.** MuJoCo 3.2.7's `Renderer` converts the reversed-Z buffer (`mjDEPTH_ZEROFAR`)
  to metric z-depth along the viewing axis, with `near = znear · extent` and
  `far = zfar · extent`. Lift gives 0.0106 m and 530.5 m. The convention is recorded per
  model and checked against exact `mj_ray` casts on every pixel.
  - Native CGL lacks `ARB_clip_control`, and MuJoCo warns that its depth accuracy is limited.
    So native depth is **development-only**, like native RGB (D-074). Reported data come from
    the canonical container (OSMesa).
- **Back-projection.**
  - Pixel centres `(j + 0.5, i + 0.5)`.
  - The pinhole intrinsics of `KinematicsExtractor.camera`.
  - World pose `cam_xpos`, `cam_xmat` in float64.
  - The exact inverse of `project_world_point` (unit test).
- **Cloud** (`PointCloudSpec`, hashed into data and checkpoints):
  - cameras `agentview` and `robot0_eye_in_hand` at 84²;
  - pooled in camera order, row-major;
  - valid pixels: finite depth, `0 < d < 0.99 · far`;
  - a fixed world box x, y ∈ [−0.25, 0.25] m, z ∈ [0.805, 1.20] m (the table top is at
    0.800 m);
  - deterministic farthest-point sampling to N = 512: start at index 0, float64 squared
    distances, lowest-index ties, a cyclic repeat if fewer than N points;
  - `float32` world metres.
- **Privilege: sensor-like, not "privileged geometry".** A cloud uses only:
  - depth;
  - the cameras' intrinsics and world poses. The wrist pose comes from robot kinematics,
    which a real robot knows from its encoders and a hand–eye calibration;
  - the static box.

  No segmentation, object pose or simulator label enters a cloud. `pc_checks` uses them only
  to measure clouds. Limitations to state: the depth is noise-free and the calibration exact.

### D-084 The Lift point-cloud dataset is installed as new data (FIXED, Session 9, after the owner's GO)
`src/roboforge/data/pc_dataset.py`, `scripts/build_pc_dataset.py`, `make pc-dataset` and
`make pc-dataset-install`. The D-079 pattern applies.
- **New data, added by decision.** Nothing existing in `data/` changes. The new pieces are:
  - `data/processed/robomimic_v1.5_ph/lift/pc512_v1.hdf5`, gitignored like every processed
    file;
  - its committed manifest, `data/manifests/pointcloud/lift_pc512_v1.json`.
- **Content:**
  - one fused world-frame cloud of 512 points per frame of all 200 demos, under the D-083
    specification (sha256 `973e27fe…`);
  - per demo: `points` float32 [T, 512, 3], `n_crop` and the per-step state SHA-256;
  - schema `roboforge.pc_dataset/1`.
- **Build:**
  - on the canonical container at `a20bfe3` (clean), in 594 s;
  - content `39fa3e52…`, file `6ab4b31d…`, 56.3 MB.
- **What it does not hold:** proprioception, actions and the split. These come from the
  committed low-dim dataset, whose content hash the file pins. The install re-verifies hashes
  on the host, never overwrites an installed file, and refuses a partial build.
- **Why now:** the owner chose GO on 2026-11-09 (approval A-04; `docs/LEVEL2_GO_NO_GO.md`).
  The data is derived from the MIT-licensed robomimic demonstrations and is never
  redistributed.

### D-085 Paid Modal launches run detached, single-flight, and survive client network loss (FIXED, Session 9)
`infra/modal_app.py` (`_launch`, `attach`), `src/roboforge/compute/launch_wait.py`,
`tests/unit/test_launch_wait.py`.
- **Why:** this host lost DNS twice in 90 minutes (KI-047). A launch runs in an ephemeral app,
  which Modal stops when its client disconnects:
  - the first loss stopped seed 0's first launch at step 25,000, which it resumed from;
  - the second killed a resume launch while it was still queued for an L4.
- **Change, client side only:**
  - each launch spawns its job into `app.run(detach=True)`, so the job keeps running if the
    client dies;
  - the function call ID is written to `<plan>.call.json` before any waiting;
  - the client polls `FunctionCall.get` and retries only transport failures (Modal
    `ConnectionError`, grpclib `StreamTerminatedError`, socket and connection errors), for a
    continuous outage of up to 1 h;
  - anything the remote job raises propagates unchanged, so `error`, `interrupted` and
    `complete` keep their meanings;
  - `attach --plan <plan>` re-attaches to a recorded call whose client died, and records its
    result; the ledger keeps a launch's last result;
  - after a result, the app is stopped.
- **Single flight:** `_launch` refuses (exit 2) when any app of this project is not stopped,
  or when Modal cannot be asked. This keeps A-04's `max_containers = 1` and "strictly one
  launch after another" even when a dead client leaves a job running.
- **What does not change:** training code, configs, budgets, the queue
  (`scripts/pc_train_queue.py`, frozen in Level 2 freeze §9), launch order, timeouts, the
  approval gate and the cost ledger. `infra/modal_app.py` is not in any freeze table.
- **Operational consequence:** Ctrl-C in the launching terminal stops only the client. To
  stop a job, run `python infra/modal_app.py cancel --app-id <ap-…>`.

### D-086 The held-out lock is re-gated on the Level 2 freeze (FIXED, Session 9, after all 3 point-cloud selections)
`src/roboforge/eval/pc_heldout.py`, `scripts/run_pc_heldout.py`,
`tests/research_integrity/test_pc_heldout_gate.py`, `make pc-heldout[-rehearse|-gate]`. The D-081
pattern: a new module, no frozen file changed.
- **The gate opens only when all of these hold:**
  - `docs/LEVEL2_EXPERIMENT_FREEZE.md` is committed and still the file of `b93c50c`
    (SHA-256 pinned);
  - its §9 table's 12 files still have their frozen hashes;
  - the 3 point-cloud selection artifacts in `reports/session9/pc_selection/` are present,
    tracked, unmodified, and equal to the hashes pinned in the module;
  - this decision is in HEAD's DECISIONS;
  - the RGB study's own gate (D-081) is still open, and the 3 reused RGB diffusion held-out
    reports still have their frozen hashes;
  - the Level 1 lock is open, and Lift's held-out manifest has the frozen hash.
- **One-shot:** `evaluate_selected` checks the gate, the canonical platform and "already ran"
  before it reads any held-out state. A second run is refused. An engineering rerun follows
  RESEARCH_PROTOCOL §14.6: a reason, at most once, and the incomplete attempt is kept.
- **Rehearsal:** the identical path runs on the first validation starts. It is labelled, not
  reported and not gated.
- **Tests:** an autouse guard fails any test that reads a held-out state. The tests prove four
  things:
  - the gate shuts when each precondition is removed;
  - a run is refused before any state is read;
  - the point-cloud evaluation path admits only validation and the two placement sets;
  - the RGB gate is unaffected.
- **Analysis:** `scripts/analyze_level2.py` (`make level2-analyze`) implements Level 2 freeze §8
  as written. It runs once, refuses to overwrite, and refuses while the gate is shut.
- **Amendment (2026-11-09, ≈ 17:55 EST, before any held-out Level 2 rollout):** *(Corrected in Session 11, audit C-20: the amendment commit `08c3aa8` is at 17:49:10 EST; "≈ 17:55" was wrong. The held-out run started after it and ended at 17:52:26 EST.)*
  - The first `make pc-heldout` attempts (3, from `4ebebc1` clean) were refused by this
    gate inside the container, before any held-out state was read; nothing was written.
  - Cause: the frozen wrapper `infra/containers/run_sim_cpu.sh` mounts only
    `reports/session<N>` (here 9), so the Session 8 checks (the RGB gate, D-081, and the 3
    reused RGB held-out reports) can never pass in a Session 9 container.
  - Change:
    - those Session 8 checks now run on the host only, where git also verifies the files;
    - `make pc-heldout` runs the full gate on the host first and stops if it is shut;
    - the container gate checks everything else, as before;
    - each held-out report records where its gate ran.
  - `scripts/analyze_level2.py` re-checks the RGB reports' hashes and the full gate on the
    host before analysing.
  - No frozen file changed. A new test checks the split and the order in the Makefile.

## Session 10 (2026-11-09 onward)
Level 1, the RGB study and Level 2 are frozen and tested once. Session 10 builds the RoboForge
Observatory (MASTER_SPEC §8) from that frozen evidence. Nothing below changes a configuration,
selection, label, result or finding of any study, and no held-out start is rolled out again.

### D-087 Makefile defaults bumped to Session 10 (FIXED)
- `EVIDENCE_SESSION` falls back to 10 and `PREFIX` to `s10` (as D-048, D-053, D-060, D-073 and
  D-082 did). No Session 10 command can then write into `reports/session6`–`9` or reuse the `s9`
  prefix by omission. Evidence goes to `reports/session10/` and `logs/session10/` with
  `ROBOFORGE_EVIDENCE_SESSION=10`.
- New targets:
  - `make observatory-select` (host): applies the D-089 rules to the frozen manifests and
    results and writes `reports/session10/observatory/selection.json`;
  - `make observatory-export` (canonical container, clean tree): clips, reference frames,
    media, denoising traces, candidate paths and point clouds from that selection;
  - `make observatory-findings` (host): `findings.json` from the three frozen `results.json`;
  - `make web-data` (host): copies the committed Session 10 outputs into
    `website/public/data/` and checks every file against the export index's SHA-256;
  - `make web-assets` (host): extracts the visual meshes (D-090);
  - `make web-build`, `make web-test`, `make web-e2e` and `make web-verify`.
- The held-out targets of all three studies are never run again.

### D-088 The web stack: a separate `website/` package (FIXED, before any frontend dependency)
- **Location:** `website/`, the MASTER_SPEC §9 layout, which the committed `.gitignore`
  already anticipates (`website/node_modules/`, `website/dist/`). The Session 9 handoff called
  it "web/"; the meaning is the same: a separate npm package with its own lockfile
  (`website/package-lock.json`), installed with `npm ci`.
- **Runtime:** React 19 + TypeScript + Vite, React Three Fiber 9 with `three@0.187.0` (the
  version `tests/web_contract` already pins), and `zod` for runtime schemas of every data file.
  `@react-three/drei` is not used: its dependency tree is large, and the few helpers needed
  (orbit controls, fat lines, a room environment) come from `three/examples/jsm`. Versions were
  read with `npm view` on 2026-11-09 (R3F 9 needs React `>=19 <19.4`). Exact versions are in
  `website/package.json` and the lockfile.
- **Dev tooling:** `vitest` (schema and unit tests), `@playwright/test` driving the
  **installed Google Chrome** (`channel: "chrome"`), so no browser is downloaded (disk,
  KI-008).
- **Python unchanged:** `uv.lock` and `pyproject.toml` do not change. The Python export uses
  only packages already locked (mujoco, numpy, imageio, Pillow).
- **No backend:** the site is static. It reads verified, compressed JSON and media, and runs no
  policy and no simulator in the browser (MASTER_SPEC §1.5).

### D-089 Observatory artifacts come from frozen evidence only, by rules fixed here (FIXED, before any clip is exported)
`src/roboforge/viz/observatory.py`, `scripts/export_observatory.py`,
`tests/unit/test_observatory.py`, `tests/simulator/test_observatory_export.py`.
- **Sources:**
  - recorded episodes of the frozen evaluations under `runs/` (`record.json` +
    `arrays.npz`, hash-checked on load);
  - the frozen `results.json` and `taxonomy.json` of `reports/session7/results/`,
    `reports/session8/rgb_results/` and `reports/session9/level2_results/` (+ `ERRATA.md`),
    each cited by path and SHA-256.

  No number on the site is typed by hand.
- **Held-out records are rendered, never rolled out.** The held-out episodes are the measured
  result, and the Level 1 freeze's example rule targets them ("rendered only from recorded
  states, labelled illustration"). The Session 8 videos set the precedent (E8-16):
  - a held-out start loads only through `eval_states.load_suite`, only while all three
    gates are open, and only for state IDs that already have a frozen record;
  - its model XML compiles the scene;
  - every transform is recomputed by `mj_kinematics` on scratch data from the **recorded**
    states, and must equal the recorded body poses;
  - no policy is built for a held-out record, and `env.step` is never called on one;
  - a test enforces this.
- **New policy calls** (denoising) replay **validation** records only. Each replay is checked
  bitwise against the recorded actions, which is the Session 4 `viz/denoising.py` method. The
  output is labelled an export, never a result.
- **Candidate paths** (MASTER_SPEC §8.4) come from **clone simulation**, on validation records
  only:
  1. restore the episode's start;
  2. re-execute the recorded executed actions up to the decision frame, checking every
     state bitwise against the record (the hidden controller and gripper state is then
     exactly the recorded one);
  3. branch, executing the candidate chunk decoded and clipped by the policy's own codec;
  4. record the `gripper0_right_eef` body position.

  The executed chunk's branch must reproduce the recorded frames bitwise. Every other branch
  is labelled "clone simulation of a never-executed proposal". Raw action vectors are never
  drawn as positions.
- **Selection rules (declared before any export, never by appearance):**
  1. **Lab and failure gallery, Level 1 held-out:** the Level 1 freeze rule verbatim (§4):
     - per task and family, among the frozen checkpoints ordered by seed, the first success
       and the first failure (sorted `state_id`) of the lowest seed that has one;
     - plus, per failure category, the first episode in (family order, seed, `state_id`);
     - families in freeze order: `bc_mlp`, `bc_rnn`, `bc_mlp_chunk`, `diffusion_unet`, and
       `random_uniform` (the H1 baseline).
  2. **Hero:** the rule-1 first success of Lift `diffusion_unet`.
  3. **Head-to-head (Can and Lift held-out, seed 0 of `diffusion_unet` vs seed 0 of
     `bc_mlp`, the H2 pair):**
     - (a) the first `state_id` on which both succeed (representative);
     - (b) case study: the first on which `bc_mlp` fails and `diffusion_unet` succeeds;
     - (c) case study: the first on which `diffusion_unet` fails and `bc_mlp` succeeds.

     A missing case is recorded as "none". Every pair is shown with the family denominators,
     and the full 50-start outcome strips of both families come from the manifests.
  4. **Denoising:** the validation evaluations of the frozen selected checkpoints
     `s6_diffusion_unet_lr3e4_lift_seed0_2f635398` (step 100000) and
     `s7_diffusion_unet_cos_can_seed0_99251da3` (step 65000):
     - episodes: the first success and the first failure in sorted `state_id` (the Session 4
       rule);
     - every decision of each episode;
     - 16 diversity samples per decision, with seeds `1_000_000 + i`, never executed;
     - clone-simulated paths for each sampler state `A_t`, each predicted clean chunk `x̂₀`,
       the final chunk and each diversity sample.
  5. **Beyond Pixels, Level 2 placement records (validation regime):**
     - on `ood_cube_shift`, `pc_diffusion` seed 0 (step 30000) against `rgb_diffusion`
       seed 0 (step 55000):
       - (a) case study: the first `state_id` on which pc fails and RGB succeeds;
       - (b) the first on which both succeed;
       - (c) the first on which pc succeeds and RGB fails, or "none";
     - on `robustness_base`, both arms' first `state_id`;
     - for every pc clip, the 512-point cloud at each decision frame, recomputed from the
       recorded state by the frozen `CloudMaker` (D-083);
     - for every RGB clip, the two 84² policy camera frames at each decision frame (D-074).
  6. **Reference frames for viewer verification:** at least 3 frames (0, ⌊T/2⌋, T, plus
     the first-success frame) of the hero, of the Can case study (b) for `diffusion_unet`,
     and of the Level 2 case study (a) for pc. Each frame gets the MuJoCo `frontview` and
     `agentview` RGB and segmentation at 480², plus each camera's pose and field of view.
- **Outputs:** `reports/session10/observatory/` (committed), with an `index.json` that holds
  every file's SHA-256 and the source records' `arrays_sha256` and `physics_sha256`.
  `website/public/data/` is generated from it by `make web-data` and is gitignored. Clip MP4s
  are frontview renders of the recorded states (D-074 path, 256², 20 fps); the inset video and
  the reduced-motion or no-WebGL fallback use them.
- **Labels:**
  - every clip is labelled "recorded closed-loop rollout (illustration of a frozen
    evaluation)", with its partition, checkpoint, seed, state and outcome;
  - every proposal is labelled "proposal" and every schematic "illustrative";
  - the export fails if checkpoint hash, run provenance or simulator build metadata is
    missing (MASTER_SPEC §8.5).
- **Amendment (Session 10 STOP): Level 1's robustness and ablation analysis is a source.**
  - **Why:** MASTER_SPEC §8.2 F asks for ablations. Before STOP, section F showed only the
    held-out results, and the Level 1 robustness (H4) and ablation (A1–A3) results were
    missing.
  - **Change:** `reports/session7/robustness/analysis.json` is now an eighth cited source,
    copied verbatim by `roboforge.viz.obs_findings.robustness`. Its blocks are the H4
    bootstrap and taxonomy and each ablation's arms, bias disclosure and bootstrap; each
    evaluation is kept as a reference with its summary path and SHA-256.
  - **Check:** the robustness configuration's content hash must equal the analysed one.
  - **Display:** the site shows these results in their own panel, labelled "validation regime,
    not held-out". The callouts only pick values from the data.
  - **Scope:** no rollout, recomputation or selection is involved. `findings.json` was
    regenerated from clean `8679aa9` (E10-06).
  - **Modules:** the module named above was never committed under that name. The code
    landed in `d12729f` as `roboforge.viz.obs_select`, `obs_scene`, `obs_media`,
    `obs_branch`, `obs_denoise` and `obs_findings`; the rules are unchanged.

### D-090 Owner decisions: no hosting this session; meshes are extracted at build time and never committed (OWNER, 2026-11-09)
- **Hosting:** the owner chose "No hosting". The site is built and verified locally
  (`npm run build`, preview, E2E). No deployment, no URL, no approval row. Hosting needs a
  later owner approval.
- **Meshes:** the owner chose "Extract at build time, gitignored". The visual meshes of the
  Panda, gripper and task objects ship inside robosuite 1.5.1 under its MIT license
  (THIRD_PARTY_NOTICES). `make web-assets` reads them from the **compiled** MuJoCo model of the
  pinned local install (`mesh_vert`, `mesh_face`; MuJoCo recentres and reorients meshes at
  compile time, so the raw asset files would be placed wrongly). It writes quantized binaries
  to `website/public/assets/` (gitignored, never committed) and checks each mesh's SHA-256
  against the container export's scene manifest.

  The site's credits name robosuite and its MIT license. If a later hosting approval ships the
  build, the MIT notice ships with it. THIRD_PARTY_NOTICES' "RoboForge does not redistribute
  meshes" stays true for the repository.

### D-091 Viewer verification: fixed thresholds; reference masks drawn without sites; the export restarts from one commit (FIXED, before the committed export and verification)
- **Thresholds**, fixed in `website/src/verify/VerifyPage.tsx` before any committed
  verification run, per reference frame at 480²:
  - robot IoU ≥ 0.95;
  - task-object IoU ≥ 0.85 (when the object is visible);
  - robot centroid offset ≤ 1.5 px.

  Every frame must pass. The browser renders the clip's flat-colour silhouette through the
  exported MuJoCo camera (pose and vertical field of view) and compares it with the class mask
  of MuJoCo's segmentation render of the same recorded state. `npm run verify` writes
  `reports/session10/observatory/viewer_verification.json` and a review montage.
- **Reference masks hide sites (KI-050).** A development comparison (uncommitted export of the
  Can case (b) clip, 6 frames) measured:
  - robot IoU 0.971–0.998;
  - object IoU 0.994–1.000;
  - two frames over the centroid limit (1.86 px and 1.61 px).

  Coloring each viewer geom uniquely located the viewer-only robot pixels under thin lines
  where the mask said "background". MuJoCo's segmentation draws sites whatever their alpha,
  including robosuite's transparent `grip_site_cylinder` (radius 5 mm, half-length 10 m). This
  is the same cause as KI-046 for depth. `ShadowRenderer.segmentation(..., sites=False)` now
  hides every site group, and `reference_frames` uses it. RGB references are unchanged. The
  thresholds were not changed.
- **One export, one commit.** The first evidence export (from `5f398d7`) stopped at clip 19 of
  37: the random baseline's manifest has `checkpoint: null` (KI-050). It wrote 18 clips and no
  `index.json`. Its partial output is discarded. After the fix, the whole export reruns from a
  single clean commit, so every committed file has one provenance.
- **Amendment (Session 10 STOP): the frozen renderer is restored.** The D-091 change was made
  in `roboforge.sim.shadow_render`. That file is pinned by `docs/FINAL_TEST_FREEZE.md`, and
  `make freeze-check` failed at STOP. The evidence export ran from `b5fc2ee`, which held the
  edited file. The edit added an optional `sites` argument whose default left the frozen code
  path unchanged, and no frozen evaluation was rerun.
  - The file is restored to its frozen bytes.
  - The geoms-only segmentation now lives in Session 10 code,
    `roboforge.viz.obs_media.segmentation_geoms_only`.
  - A development re-render of the 3 verification clips from the restored tree must reproduce
    the committed reference PNGs byte for byte. It is recorded in the ledger (E10-08), so the
    export is not rerun.
  - Rule: run `make freeze-check` before any commit that touches `src/roboforge/sim` or
    `src/roboforge/eval`.

## Session 11 (2026-11-09 onward)
Level 1, the RGB study and Level 2 are frozen and tested once. Session 11 writes the paper
(MASTER_SPEC §11) from that frozen evidence and audits it. Nothing below changes a
configuration, selection, label, result or finding of any study, and no held-out start is
rolled out again.

### D-092 Makefile defaults bumped to Session 11 (FIXED)
- `EVIDENCE_SESSION` falls back to 11 and `PREFIX` to `s11` (the D-048, D-053, D-060, D-073,
  D-082 and D-087 pattern). No Session 11 command can then write into `reports/session6`–`10`
  or reuse the `s10` prefix by omission. Evidence goes to `reports/session11/` and
  `logs/session11/` with `ROBOFORGE_EVIDENCE_SESSION=11`.
- **`OBS_OUT` stays pinned to `reports/session10/observatory`.** It used to follow
  `EVIDENCE_SESSION`, so the bump would have pointed `make web-data`, `make web-assets`,
  `make web-verify` and `make visual-provenance` at a directory that does not exist. The
  committed export is Session 10 evidence and does not move.
- New targets (D-094): `make paper-assets`, `make paper-check`, `make paper-bibcheck`,
  `make paper`.
- The held-out, placement and analysis targets of all three studies are never run again.

### D-093 The paper toolchain: Tectonic, BibTeX through Tectonic, matplotlib from the lock (FIXED, before any paper file)
- **Checked on this host, 2026-11-09:** `/opt/homebrew/bin/tectonic`, Tectonic 0.17.0 (XeTeX
  engine). `pdflatex`, `latexmk`, `xelatex`, `lualatex`, `bibtex` and `biber` are not
  installed. Tectonic runs its own BibTeX pass when the `.aux` asks for one, so the
  bibliography uses BibTeX with `natbib` (no biber or biblatex).
- **Bundle:** Tectonic fetches its TeX bundle over the network on first use and caches it, here
  in `~/Library/Caches/TectonicProject.Tectonic` (46 MB). `make paper` passes
  `--only-cached` so that a build never reaches the network silently; a fresh machine first
  runs `tectonic -X compile paper/main.tex` once with network access.
- **Reproducible PDF:** a two-build test (same input, one second apart) gave different PDFs
  without `SOURCE_DATE_EPOCH` and byte-identical PDFs with it. `make paper` sets
  `SOURCE_DATE_EPOCH` to a fixed value (`PAPER_EPOCH`), so `paper/main.pdf` is reproducible on
  this host with this cache. Byte equality is not claimed across Tectonic versions, bundles or
  platforms.
- **Figures:** matplotlib 3.11.2, already in `uv.lock` through robomimic, with the precedent of
  `scripts/plot_validation.py` and `roboforge.data.explore`. Agg backend, embedded TrueType
  (`pdf.fonttype 42`), `metadata={"CreationDate": None}`, so figure PDFs are byte-identical
  across rebuilds.
- **No new dependency:** `pyproject.toml`, `uv.lock` and `website/package-lock.json` do not
  change. Bibliography checks (D-094) use the standard library only.
- `paper/main.pdf` is committed with its SHA-256 recorded in the ledger; it is a local,
  unreviewed preprint-style report, never called a published paper (MASTER_SPEC §11.3).

### D-094 Paper tables, figures and numbers come from frozen files by a generator, with provenance (FIXED, before any paper file)
`src/roboforge/paper/` (new package), `scripts/paper_assets.py`,
`scripts/verify_bibliography.py`, `tests/unit/test_paper_*.py`.
- **Sources** (`roboforge.paper.sources.SOURCES`): the original frozen files, each pinned by
  SHA-256 in code; a mismatch stops the build. They are:
  - Level 1 held-out: `reports/session7/results/{results.json,taxonomy.json,tables.md}`;
  - Level 1 robustness and ablations: `reports/session7/robustness/{analysis.json,tables.md}`;
  - RGB study: `reports/session8/rgb_results/{results.json,tables.md}`;
  - Level 2: `reports/session9/level2_results/{results.json,tables.md,ERRATA.md}` and the
    committed seed-0 launch record and queue log of `logs/session9/pc_train/` (for the KI-049
    correction);
  - spend: `reports/compute/cost_ledger.jsonl` (the billing events);
  - setup facts: the freezes (`docs/FINAL_TEST_FREEZE.md`, `docs/RGB_EXPERIMENT_FREEZE.md`,
    `docs/LEVEL2_EXPERIMENT_FREEZE.md`), `data/manifests/robomimic_v1.5_ph.json`,
    `data/splits/{lift,can}_ph.json`, `reports/session4/param_counts.json`,
    `configs/eval/robustness.yaml`, `uv.lock` and the suite summaries that hold the validation
    numbers.

  `reports/session10/observatory/findings.json` is a convenience copy and is never a source.
  Nothing under the gitignored `runs/` is read, so a fresh clone regenerates the same files.
- **Outputs** under `paper/generated/` (committed):
  - `tables/*.tex`;
  - `figures/*.pdf`, each with a JSON sidecar of the plotted values;
  - `numbers.tex`, the only place a number used in the prose is defined. It holds key–value
    macros read with `\rf{key}`; an undefined key is a LaTeX error;
  - `index.json`, which lists every output with its SHA-256 and its source paths and SHA-256.

  Tables, figures and `numbers.tex` contain no timestamp. `index.json` carries the git SHA and
  dirty flag of the generating tree.
- **No new statistic.** The paper reports only values stored in the frozen files and plain
  arithmetic on them, each named in `index.json`: sums of taxonomy counts (k/150), unit
  conversions (fractions to percent, seconds to hours), and the KI-049 correction
  `train_wall_s / (end_step − resumed_from_step)` from the committed seed-0 launch record. The
  preregistered per-checkpoint Wilson intervals are the reported uncertainty where a bootstrap
  interval collapses at 0 % or 100 %. Any further analysis needs its own decision and a
  "not preregistered" label.
- **Checks (tests):**
  - every source matches its pinned hash;
  - a rebuild in memory equals the committed `paper/generated/` byte for byte
    (`make paper-check`);
  - every value in the generated tables equals the value in the frozen `tables.md` at the same
    rounding and sign;
  - every `\rf{key}` used in `paper/` is defined;
  - no bare number appears in `paper/sections/*.tex` outside `\rf{}` and a short list of
    identifiers (H1–H5, A1–A3, decision and issue IDs, "3D", names such as ResNet-18 and the
    sampler names DDIM-10 and DDPM-100);
  - every bibliography entry has a DOI or an arXiv identifier, or for software a URL, and
    `paper/bibliography_check.json` records that it resolved, with the title it resolved to.
- **Bibliography:** entries are taken from the publishers' or arXiv's metadata (DOI content
  negotiation, the Crossref API and the arXiv export API), never written from memory.
  `make paper-bibcheck` needs the network and is never part of `make test`.

### D-095 A seed-level sensitivity analysis, labelled not preregistered, beside the frozen intervals (FIXED, Session 11, after audit finding B-1 and before any paper result uses it)
`roboforge.paper.sensitivity`, tables `paper/generated/tables/sensitivity_main.tex` and
`sensitivity_robustness.tex`, macros `sens.*` (including `sens.all.boundary`, the count of
frozen intervals with a bound at exactly zero or zero width; audit B-4).
- **Why:** the independent audit (docs/AUDIT.md, B-1, B-2) found that, with 3 training seeds, the
  preregistered hierarchical bootstrap's percentile interval rests on 2 degrees of freedom per
  family for the dominant seed component. For example, Can diffusion − BC-MLP is separated under
  it, but a small-sample interval on the per-seed means includes 0. The frozen verdicts cannot
  change (freeze §4) and are reported as the primary result. Readers also need to see how much
  each verdict depends on the method.
- **What is computed**, from the frozen `per_seed` values only (no episode record, nothing
  under `runs/`, no new rollout):
  - a difference between two families or arms trained separately: a Welch t interval on the
    two sets of per-seed success rates (Welch–Satterthwaite df), 95 %;
  - a difference within one family on the same checkpoints (an OOD drop, A1, A2): a paired t
    interval on the per-seed differences, df = n − 1;
  - a difference of two drops: a Welch interval on the per-seed drops of the two families;
  - when both sets have zero seed variance, the interval is reported as undefined ("no seed
    variance"), never as a zero-width interval.

  scipy 1.17.1 supplies the t quantile. It is already in `uv.lock` (a transitive dependency),
  so `pyproject.toml` and `uv.lock` do not change, the same reasoning as matplotlib in D-093.
- **Scope:** every declared difference of the Level 1 held-out test (success), of H4 and A1–A3,
  of the RGB study and of Level 2. Each row shows the frozen verdict beside the sensitivity
  verdict; rows where they disagree are marked.
- **Labels:** "sensitivity analysis, not preregistered; the preregistered verdict is the
  result". It never replaces, reranks or relabels a frozen result. It ignores pairing by
  start (it treats per-seed means as the units), so it is a check on the seed dimension only,
  not a better test.
- **Also derived (descriptive, D-094 arithmetic):** the split of each censored-length
  difference into a success part and a speed part,
  ΔL = −Δp · (H − T̄_B) + p_A · ΔT̄, where H is the horizon, p is the success rate and T̄ is the
  mean first-success step over successful episodes. All of these come from `results.json`
  per-checkpoint values, pooled over seeds as k-weighted means.

### D-096 Dispositions of the independent audit; declared secondary metrics reported post hoc (FIXED, Session 11, after audits A–C and before the paper uses them)
`docs/AUDIT.md`, `roboforge.paper.secondary`, `Sources.chained_json`, table
`paper/generated/tables/secondary.tex`, macros `sec.*`.
- **Rule:** every audit finding gets an engineering correction or a disclosed limitation
  (`docs/AUDIT.md` lists each with its disposition). No frozen file, freeze or executed outcome
  changes. No new evaluation, training or paid compute is run in Session 11.
- **Declared secondary metrics (audit C-9):** RESEARCH_PROTOCOL §8's gripper sign changes and
  action smoothness were computed per episode in every Level 1 held-out summary but never
  aggregated. The generator reads the 26 summaries that `reports/session7/results/results.json`
  cites, each only after its SHA-256 equals the one recorded there (a chained hash, as strong as
  a direct pin, since the parent is pinned), and reports plain means per family and task
  (gripper sign changes per episode with the range of per-seed means; mean absolute
  step-to-step change of the executed motion command; the frozen motion clip fraction). No
  interval, no test, no comparison. Label: "declared secondary, aggregated after the frozen
  analysis". Plan diversity (measured only for the Session 4 pilot) and point-cloud throughput
  (Level 2 freeze, "also reported") are listed as declared, not reported.
- **Not run, by decision:** the evaluation-only clamp test of audit A-6 (BC checkpoints with
  outputs clamped to the training range on the robustness sets). It would be a new
  post-freeze evaluation; in this session the finding is a disclosed limitation. If run later
  it needs its own decision before any rollout, a clean tree, `ROBOFORGE_EVIDENCE_SESSION`, and
  the label "exploratory, after the freeze"; it never replaces a frozen result.
- **Optional items never run (audit C-11; MASTER_SPEC asks for a reason):** DiT, GMM-BC and the
  Square task were not run, to stay within the project's time and compute budget. No declared
  hypothesis depends on them.
- **Engineering corrections:** `scripts/analyze_robustness.py` refuses to overwrite an existing
  `analysis.json` (B-12, with a test); `scripts/analyze_level2.py` checks that a report exists
  before hashing it (B-13). Neither script is hash-frozen; `make freeze-check` passed before
  and after. The RGB cost row whose hours omit a preempted attempt gets a footnote in the
  paper's cost table (C-6). The D-086 amendment time and the E9-11 order are annotated in place
  (C-20).


## Session 12 (2026-11-09 onward)
Level 1, the RGB study and Level 2 are frozen and tested once; the paper is built. Session 12
reproduces the project from a fresh clone, runs the release checks and publishes the
repository as the owner authorized. Nothing below trains, evaluates, or changes a
configuration, selection, label, result or finding of any study.

### D-097 Makefile defaults bumped to Session 12; release targets (FIXED)
- `EVIDENCE_SESSION` falls back to 12 and `PREFIX` to `s12` (the D-092 pattern). Evidence goes
  to `reports/session12/` and `logs/session12/` with `ROBOFORGE_EVIDENCE_SESSION=12`.
- `OBS_OUT` stays pinned to `reports/session10/observatory` (D-092).
- New targets: `make readme-findings` (regenerates the README findings block, D-103),
  `make readme-preview` (the README preview GIF, D-103), `make release-check` (the offline
  release scans, D-104), `make pages-bundle` (the GitHub Pages staging directory, D-102).
- The held-out, placement and analysis targets of all three studies are never run again.

### D-098 Fresh-clone reproduction environment (FIXED, before the clone)
- **Where:** `git clone --no-local` of the committed tree into the session scratch directory
  (outside the repository, so the clone shares no object store, `.venv`, `data/`, `runs/` or
  `node_modules` with the working repository). The clone is never committed to; its logs are
  copied into `logs/session12/fresh_clone/`.
- **Host environment:** `uv sync --frozen` (`make setup`) on the host only, never inside the
  image (KI-020). `make download` fetches the datasets from the pinned Hugging Face revision and
  checks SHA-256 (this also checks the download links). `make check-data` and `make build-data`
  rebuild the processed data, which must equal the committed manifests.
- **Container:** a small simulator smoke on the canonical image through
  `infra/containers/run_sim_cpu.sh` from the clone, with `ROBOFORGE_EVIDENCE_SESSION=12`, writing
  only `reports/session12/` of the clone. It runs before any host command that rewrites committed
  files, and never at the same time as a full test suite (KI-043).
- **Checks in the clone:** lint, the full test suite (alone), `make smoke-sim` with its output
  sent to scratch, `make freeze-check`, both held-out gates, `make paper-check`, `make paper`
  (the PDF's SHA-256 is compared with the committed one and any difference is reported as found),
  `make web-install web-data web-assets web-build web-test web-e2e`.
- **Checkpoints:** not in git (gitignored `runs/`, not redistributed). One frozen checkpoint
  from the working repository's `runs/` is hash-checked against the freeze and loaded with the
  clone's code. No rollout, no policy evaluation.
- **Limits stated, not hidden:** the Tectonic bundle cache already exists on this host (E11-07),
  so the clone does not test a first networked Tectonic build; the clone runs on the same host
  and OS as development, so platform independence is tested only by CI (D-101) and the
  canonical container.

### D-099 What "release" means (OWNER, 2026-11-09, Session 12)
The owner chose, when asked once with the facts (gh logged in as `zackholt05-tech`, repo scope;
`github.com/zackholt05-tech/roboforge` absent; ~65 MB, 140+ commits including evidence logs):
- **"Public repo + tag + release":** create the public repository
  `github.com/zackholt05-tech/roboforge`, push `main`, create the annotated tag `v1.0.0` and a
  GitHub release with `paper/main.pdf` attached. Each step is done only after every check passes
  and is verified (`git ls-remote`, `gh repo view`, `gh release view`, a download of the
  attached PDF compared by SHA-256). Nothing is reported as published unless verified.
- **"GitHub Pages":** host the Observatory on GitHub Pages (D-102). This supersedes the hosting
  part of D-090 ("no hosting"); D-090's mesh rule (extracted at build time, never committed to
  `main`) still holds.
- **"Add CI, run on push":** GitHub Actions on push (D-101).
- **History:** never rewritten (D-100). Commits, the tag, the release text, the `gh-pages`
  branch and the repository description carry no AI co-author trailer or AI byline
  (MASTER_SPEC §0.2), whatever a tool's default footer says.
- **Versions:** the release is `v1.0.0` (tag, `CITATION.cff`). `pyproject.toml` keeps
  `version = "0.0.1"` because the root package version is recorded in `uv.lock`, a hash-pinned
  paper source (D-094) and the frozen environment record; `website/package.json` keeps
  `0.10.0` for the same reason with its lockfile. The Python and npm package metadata versions
  are not release identifiers.
- **Push identity:** before the first push, the credential git uses is confirmed to be the
  `zackholt05-tech` account. If GitHub rejects the push (for example an email-privacy rule),
  stop and ask; author emails are never rewritten.

### D-100 Local absolute paths in committed evidence are disclosed, not scrubbed (OWNER, 2026-11-09)
135 committed files (logs of Sessions 0–9, `reports/feasibility/` and `reports/session5/`
evidence, `docs/KNOWN_ISSUES.md`) contain absolute local paths under the developer's home directory (`/Users/<user>/…`), and every
commit in history has them (KI-009). No secret was found. The owner chose **"Disclose, no
rewrite"**: the evidence bytes stay unchanged (some logs are hash-pinned paper sources), and the
paths, which reveal only the local username, are listed as an accepted item in
`docs/FINAL_RELEASE_AUDIT.md`. `make release-check` fails if such a path appears in any other
file.

### D-101 Continuous integration (OWNER "Add CI, run on push", 2026-11-09)
- Workflows in `.github/workflows/`: `python-tests.yml` (lint, unit tests that need no
  simulator, dataset, renderer, node or long run), `research-integrity.yml`
  (`tests/research_integrity/`, `make freeze-check`, both held-out gates, `make paper-check`,
  `make release-check`), `web-build.yml` (`make web-data web-assets`, `npm ci`, build with its
  budgets, vitest).
- Runner `ubuntu-24.04` (Linux x86_64, unverified for this project before CI, KI-001). The
  workflows install `cmake`, `build-essential` and `libegl-dev` for `egl_probe`, then
  `uv sync --frozen`. No dataset download, no paid compute, no secrets, no publishing of
  numbers, no pushes (`permissions: contents: read`). Actions are pinned by commit SHA. The
  workflows are linted with `actionlint` before commit.
- Not in CI, with the reason: simulator, dataset, render and slow tests (no dataset in CI, and
  physics is not bitwise across platforms, KI-013); Playwright E2E (needs a GPU-backed Chrome,
  D-088); `make paper` (Tectonic is not pinned in CI); every evaluation target.
- A failure on the first GitHub run is recorded as found. Tests are never weakened to pass; a
  fix is a code change and is rechecked.
- **Amendment (Session 12, after the first GitHub run on `b4bd070`):** on `ubuntu-24.04`
  (x86_64), research-integrity passed, but python-tests and web-build failed for platform
  reasons, recorded in KI-061: loading triton 3.9.0's native library segfaults when
  `torch.optim.AdamW` imports `torch._dynamo` (the same test passes in the canonical aarch64
  container), and the meshes compiled on x86_64 differ from the canonical export by 7.45e-9 m,
  so the exact-equality mesh check refuses, as designed. The workflows now run on
  `ubuntu-24.04-arm`, Linux aarch64, the architecture of the canonical evaluation platform
  (D-017). No test, tolerance or check was changed; Linux x86_64 stays unverified (KI-001,
  KI-061).
- **Amendment 2 (Session 12, after the run on `e11ad04`):** python-tests crashed the same way
  on `ubuntu-24.04-arm`, so the triton crash was not architectural. Its root cause, isolated in
  the canonical container (KI-061), is Mesa's `libLLVM-15` in the global symbol scope (loaded by
  PyOpenGL's OSMesa platform on `import mujoco`) before triton loads its bundled LLVM. Fix: the
  test harness (`tests/conftest.py`) imports `torch._dynamo` before any test imports mujoco. No
  test skipped or changed; `src/` unchanged. The aarch64 runners stay: the web-build mesh check
  needs them (KI-061).

### D-102 GitHub Pages deployment of the Observatory (FIXED, after the owner's choice in D-099)
- **Method:** a `gh-pages` branch holding the static build, published by GitHub Pages from that
  branch. The build is made locally from a clean, verified commit with `make web-data web-assets
  web-build` and staged by `make pages-bundle` into a scratch directory, which adds:
  - `.nojekyll`;
  - `THIRD_PARTY_LICENSES.txt`: robosuite's MIT license text (from the pinned install; it covers
    the Panda, cube and can meshes in `assets/`), the SIL Open Font License 1.1 texts of the
    bundled fontsource fonts, and the license texts of every npm package bundled into the
    JavaScript;
  - `deploy.json`: the source commit, its dirty flag, and the SHA-256 of every staged file.
- **Consequence for the mesh rule:** `main` never contains meshes (D-090); the `gh-pages`
  branch does, under robosuite's MIT license with its notice. `THIRD_PARTY_NOTICES.md` and the
  site's credits say so.
- **Base path:** the site already uses a relative base (`base: "./"`, D-088) and fetches through
  `import.meta.env.BASE_URL`, so it works under `/roboforge/` without a code change.
- **Verification:** after GitHub reports the build as `built`, every staged file is fetched from
  the live URL and its SHA-256 compared with `deploy.json`; the site E2E is also run against the
  live URL when its config allows. The Pages URL is linked from the README only after this
  passes.
- The `gh-pages` commit is authored by Zackary Holt, with no AI trailer.

### D-103 README content generated from frozen files (FIXED, before the README is rewritten)
- **Findings block:** the README's key-findings block (between `<!-- BEGIN GENERATED: findings
  -->` and `<!-- END GENERATED: findings -->`) is written by `scripts/readme_findings.py` from
  `paper/generated/numbers.tex`, which `make paper-check` proves equal to a fresh build from the
  hash-pinned frozen files (D-094). No number in that block is typed by hand.
  `make release-check` fails if the block differs from a fresh generation. Numbers in the rest
  of the README are limited to counts and identifiers checked by the same rule.
- **Caveats carried with the numbers:** the Can H2 separation is fragile (a seed-level interval,
  not preregistered, includes 0; KI-053); H2-c is not separated on either task; the diffusion
  sampler bounds its actions to the training range and the BC heads are not bounded (KI-052);
  Lift saturates (KI-034); H5 is negative; the paper is a local build, not peer reviewed.
- **Preview:** `scripts/readme_preview.py` builds `docs/media/readme_preview.gif` from the
  committed Session 10 MP4s of the D-089 rule 3(a) head-to-head pair on Can (seed 0 diffusion
  U-Net and seed 0 BC-MLP on the same held-out start, the first start where both succeed,
  labelled "representative"). The state ID is read from `selection.json`, never written in
  code. Transformations: decode, resize, place side by side, add labels (task, policy, seed,
  start, outcome, source), hold the shorter clip's last frame. A JSON sidecar records the
  source paths and SHA-256, the selection rule and the transformation.
- **Amendment (Session 12, before the README was committed):** a second generated block,
  `setup` (simulator, data, policies and evaluation facts), is written by the same script from
  the same macros, so no configuration number in the README is typed by hand either. Outside
  the two blocks the README states no result; sizes and times measured in Session 12 are in
  `docs/FINAL_RELEASE_AUDIT.md`.

### D-104 The release checks (FIXED, before they are run)
`make release-check` (offline, also in CI) and `scripts/release_check.py --network` (manual):
- **History:** every commit on every ref has author and committer `Zackary Holt
  <zackholt05@gmail.com>`; no message contains a co-author trailer or an AI byline; every
  annotated tag's tagger is the same identity.
- **Secrets:** regular expressions for common key and token formats (AWS, GitHub, Slack, Google,
  OpenAI- and Anthropic-style keys, private-key blocks, Modal token variables with values) over
  `git log -p --all` and the working tree. `gitleaks` is not installed; this is a pattern scan,
  stated as such.
- **Private paths:** no absolute home path outside the D-100 disclosed set.
- **Tracked content:** no dataset (`*.hdf5`, `*.h5`), checkpoint (`*.pt`, `*.pth`, `*.ckpt`,
  `*.safetensors`) or mesh file; no tracked file over 50 MB; `.env` and key files ignored.
- **Docs:** no placeholder (`TODO`, `TBD`, `FIXME`, `<repo>`, lorem ipsum) in the public-facing
  documents; every relative link in tracked Markdown resolves; `CITATION.cff` names Zackary Holt
  as the only author, with version, date and license; the README findings block equals a fresh
  generation (D-103).
- **Network (`--network`, not in `make test` or CI):** every external URL in the README,
  `CITATION.cff`, `THIRD_PARTY_NOTICES.md` and the docs answers with HTTP < 400 (some hosts
  refuse automated requests; those are listed, not hidden). Dependency vulnerability and license
  scans (`pip-audit` on the exported lock, `npm audit`, a license inventory) are run separately
  and recorded in `reports/session12/`.

## Post-release extensions (2026-11-10 onward)

### D-105 Three follow-up studies after v1.0.0 (OWNER, 2026-11-10)
- **Owner request (2026-11-10):** continue building with three follow-up studies, each prepared
  as handoff material before any work starts: X1, the evaluation-only action-clamp test
  (KI-052, deferred by D-096); X2, a harder or multimodal task; X3, more training seeds.
  Specifications: `docs/extensions/` (README and one file per study). Session 13 runs X1;
  later session numbers are assigned when each study starts.
- **Boundaries:** Level 1, the RGB study and Level 2 stay frozen and tested once; no frozen
  file, freeze, result, label or executed outcome changes; `make freeze-check` passes before and
  after every code change, and new behavior goes into new files. Each study records its own
  design decision and commits a preregistration before its first reported rollout. Labels: X1
  "exploratory, after the freeze"; X2 a separate study with its own freeze and held-out suite;
  X3 "separate replication". None replaces a frozen number.
- **Publication:** the published v1.0.0 (tag, paper, README, Observatory) stays as released.
  Whether extension results are published (for example a v1.1 with an addendum) is the owner's
  decision, asked when results exist. Nothing is pushed, tagged, released or deployed without
  asking.
- **Compute:** local CPU and the local container ($0) by default; any Modal job needs a new
  approval row with a cost cap first. Disk is the binding local limit (KI-008).
- **Planning facts recorded in `docs/extensions/` (2026-11-10):** the Level 1 freeze pins
  nearly every file under `src/roboforge/` and three held-out scripts; the 45 BC robustness
  evaluations X1 repeats recorded 3.12 container-hours in Session 7; Square PH is in the data
  manifest with SHA-256 but has no simulator contract; Can, Lift and Square MH files exist at
  the pinned Hugging Face revision but are not in the manifest; one extra seed of the four
  families used 2.3 GB of disk on Can and 1.5 GB on Lift.

## Session 13 (2026-11-10): post-release extension X1
Level 1, the RGB study and Level 2 stay frozen and tested once. Session 13 runs X1, the
evaluation-only action-clamp test of KI-052, under D-105. Every X1 result is labelled
"exploratory, after the freeze" and never replaces a frozen number. No training, no held-out
rollout, no paid or remote compute.

### D-106 Makefile defaults bumped to Session 13; X1 targets (FIXED, before any code change)
- `EVIDENCE_SESSION` falls back to 13 and `PREFIX` to `s13` (the D-092/D-097 pattern). Evidence
  goes to `reports/session13/` and `logs/session13/` with `ROBOFORGE_EVIDENCE_SESSION=13`.
- `OBS_OUT` stays pinned to `reports/session10/observatory` (D-092).
- New targets: `make x1-run` (`scripts/run_x1_clamp.py`, the 45 clamped evaluations, once) and
  `make x1-analyze` (`scripts/analyze_x1_clamp.py`, once). The engineering checks are run by
  their logged commands (E13 rows), not by a target.
- The held-out, placement and analysis targets of the three frozen studies are never run again.

### D-107 X1 design: BC with the diffusion sampler's action bound on the robustness sets (FIXED, Session 13, before any X1 rollout)
**Question (KI-052, audit A-6, D-096).** The diffusion sampler clamps its predicted clean chunk
to [−1, 1] in min–max normalized space (`_Sampler._x0`, `clip_sample=True`). The last DDIM step
returns that clamped chunk unchanged (`abar_prev = 1`, direction coefficient 0), so every
diffusion action lies in each dimension's training range. BC heads are clipped only to the
controller limits. Do the frozen H4 and H2-c contrasts survive when BC gets the same bound?

**The clamp.**
- Each normalized action of the BC policy's executed chunk is clipped to [−1, 1] *before* the
  frozen `ActionCodec.decode`, which then runs unchanged. This is the sampler's bound bit for
  bit.
- Statistics: the committed action normalizer of each task,
  `data/normalization/<task>_ph_lowdim_v1.json` (content SHA-256 `253bfc7e…` Lift,
  `e0437cd7…` Can), mode `minmax`. It was fit on the training split. It is the only
  normalization any Level 1 checkpoint can load (`check_committed_contract` compares
  `normalization_sha256`), so BC and diffusion share it exactly.
- In environment units the bound is [min_i, max_i] of the training split, up to float64
  rounding: max |decode(±1) − stats| = 1.1e-16 (measured).
- The controller limits are ±1 on every dimension. The clamp can bind only where the training
  range is narrower:
  - Lift dims 1, 3, 4, 5, both sides: dy [−0.560, 0.652], drot_x [−0.151, 0.119],
    drot_y [−0.140, 0.305], drot_z [−0.518, 0.465];
  - Can dim 3 on both sides ([−0.556, 0.730]), and dims 4 and 5 on the upper side only
    (≤ 0.450, ≤ 0.745).
- On every other dimension the training range equals the controller bounds, so clamping and the
  controller clip give bitwise-identical executed actions. That includes the gripper on both
  tasks: **the gripper clamp is a no-op** on executed actions. The record's `clip_mask` still
  marks it, as the controller clip did before.
- Can's clamp is weaker by construction.

**Implementation: new files only.** No file pinned by `docs/FINAL_TEST_FREEZE.md` changes, and
`make freeze-check` runs before and after.
- `src/roboforge/exploratory/clamp.py`: `ClampedCodec` and `ClampedPolicy` wrap the frozen
  `LearnedPolicy`, and `evaluate_x1` reuses the frozen `validation.run_states`.
- Scripts:
  - `scripts/eval_x1_clamp.py`: one checkpoint × one set, in the container;
  - `scripts/run_x1_clamp.py`: host orchestrator; ≤ 4 containers, host-wide slots, clean tree
    at every launch;
  - `scripts/check_x1_engineering.py`: the engineering checks;
  - `scripts/analyze_x1_clamp.py`: the analysis, run once.
- Tests: `tests/unit/test_x1_clamp.py` and `tests/research_integrity/test_x1_scope.py`.
- **What the records keep:**
  - `raw_policy_action` keeps the network's own output, with the meaning it has in every
    earlier record;
  - `executed_action` is the clamped and decoded action;
  - `clip_mask` marks the controller clip or the clamp;
  - per-episode clamp counts go into `policy_episode`, outside `physics_sha256`.
- The container does not mount `reports/session6`/`7` (KI-048 class). The host therefore
  resolves each checkpoint from the frozen test-configuration records and passes the train
  run, step, family and checkpoint SHA-256 in. The container checks the hash against the
  training manifest and again when it loads the file.

**Evaluations (45, each once).**
- **Checkpoints:** the 9 frozen BC checkpoints per task (BC-MLP, BC-RNN, BC-MLP-chunk × seeds
  0–2) of `reports/session6/lift_test_configuration.json` (SHA-256 `38ac3c29…`) and
  `reports/session7/can_test_configuration.json` (`116f10a4…`). These are the SHA-256 values
  recorded by the Session 7 robustness analysis.
- **Sets, 50 starts each:**
  - Lift `robustness_base` (content SHA-256 `b1ffb682…`), `ood_cube_shift` (`c6083afc…`) and
    `ood_arm_offset` (`43c7c3d2…`);
  - Can `robustness_base` (`b91082ea…`) and `ood_arm_offset` (`32388f79…`).
  Lift 9 × 3 + Can 9 × 2 = 45 evaluations.
- **Run settings:** canonical container, R = 1, policy seed 0, first success or 400 steps, the
  first 3 episodes re-simulated (`verify_record`), from the preregistration commit on a clean
  tree.
- **Out of reach:** the code admits only `rb.sets_for(task)`, so the validation suite and the
  locked held-out suite are refused (research-integrity test). Diffusion checkpoints are refused
  on these sets. Their unclamped Session 7 evaluations already carry the bound.

**Outputs.**
- Records: `runs/x1_clamp/<eval_id>/`. The ID is the Session 7 ID of the same checkpoint and set
  plus `__x1-clamp`. The precedent for a subdirectory is `runs/level2_eval/`.
- Summaries: `reports/session13/x1_clamp/evals/<eval_id>.json`, schema
  `roboforge.x1_clamp_eval/1`.
- Logs: `logs/session13/x1_clamp/`.
- No existing directory is written into. The writer refuses an existing record directory and
  creates summaries exclusively.
- Engineering checks: records in `runs/x1_checks/`, results in `logs/session13/x1_checks/`.
  The analysis never reads either.

**Engineering checks, before the preregistration (not results).**
1. **Reproduction, clamp disabled:** the wrapper with the clamp disabled must reproduce a
   recorded Session 7 summary exactly, with equal per-episode `physics_sha256`, success,
   first-success step and step count on all 50 starts. The targets are heavily clipped OOD
   evaluations:
   - Lift BC-MLP-chunk seed 0 on `ood_arm_offset` (26/50);
   - Can BC-MLP seed 0 on `ood_arm_offset` (13/50).
   Each Session 7 summary is read only after its SHA-256 equals the value in the Session 7
   analysis, whose own hash is pinned.
2. **No-op on diffusion:** the clamp applied to the seed-0 diffusion checkpoint of each task, on
   5 development starts (seeds 13,000,000 + i, partition `eval_adhoc`, outside every reserved
   block), must leave `physics_sha256`, raw and executed actions identical, with zero clamped
   entries. The same run on the seed-0 BC-MLP-chunk checkpoints records how often the clamp
   acts and checks that executed actions lie inside the bounds.
3. **Isolation:** before the 45 evaluations, none of their record directories exists. A content
   digest of every file in the 60 Session 7 H4 record directories, and the `runs/` top-level
   listing, are taken before and after. The digests must be equal, and the only new top-level
   entries may be `x1_clamp` and `x1_checks`.

**Analysis (fixed now; the preregistration repeats it).**
- One §9 hierarchical paired bootstrap per task (B = 10,000, seed 20261106).
  - Keys: the frozen `family:set` matrices, in the frozen order, then `<bc>+clamp:set`.
  - Every key's seed group is its family. The draws are therefore those of the frozen analysis,
    and the script asserts that every frozen estimate, difference and contrast equals
    `analysis.json` exactly before it reports anything new.
  - Clamped − unclamped is paired by seed and start; diffusion − clamped BC by start.
- **Primary:** Lift `ood_arm_offset`, drop(diffusion) − drop(clamped BC-MLP-chunk), against the
  frozen −40.0 pp [−54.0, −26.0], separated (−).
- **Secondary:** every other X1 difference and contrast. See the preregistration for the count.
- The readings are fixed verbatim in the preregistration, in separated / not-separated terms
  only. "Not separated" is never read as equality.

**Not done, by decision.**
- No held-out revisit. The owner may choose one later, as an exploratory revisit with its own
  decision.
- No `clip_sample=False` diffusion arm. The audit named it as optional; it would be a second
  question, and the diffusion no-op check already shows the bound is all the sampler adds.
- No change to the paper, README or Observatory (D-105).

**Estimates, from measured Session 7 times.**
- The same 45 evaluations recorded 3.12 container-hours (Lift 1.41, Can 1.71). With ≤ 4
  containers that is about 0.8 h of wall-clock time (estimate). The clamp may lengthen
  episodes; a success ends an episode early.
- The engineering checks take about 15 min (estimate).
- Disk about 0.5 GB (estimate); 17 GB are free.
- $0.
- **Amendment (Session 13, after the engineering checks, before the preregistration commit
  `ab02469`).** Check 3 showed that most normalized BC entries outside [−1, 1] lie on dimensions
  where the clamp cannot bind (the gripper, and dimensions whose training range is ±1). There
  the clamp changes nothing. Counting them alone would overstate what the clamp does. The
  analysis therefore also:
  - recomputes from the records the fraction of executed actions the clamp **changed** (the
    frozen decode vs the clamped decode of `raw_policy_action`), and reports it beside the
    out-of-bound fraction;
  - stops unless every recorded `executed_action` equals the clamped decode bitwise. Its record
    checks were verified on the 100 clamp-disabled reproduction records first.
  Reading R1 was worded "does not account for all of the frozen H4 gap", so that it cannot
  contradict the supporting leg. The preregistration states these readings verbatim, and a
  unit test keeps them equal to the script's. Renaming a partial record directory to
  `.incomplete-1` (§14.6 rerun) is a manual, logged step; the orchestrator only refuses it.
  Nothing here was changed after any X1 rollout.
- **Outcome (exploratory, after the freeze):** see KI-052 and LEDGER E13-04, E13-05.
  `reports/session13/x1_clamp/` is analysed once and is not rerun.
- **Correction (Session 13, at STOP, after the analysis).** The "Not done" item above says
  that "the diffusion no-op check already shows the bound is all the sampler adds". That
  reasoning is wrong.
  - `clip_sample` clamps x̂₀ at every DDIM step, and the clamped value feeds the next sample,
    so the bound shapes diffusion's whole sampling path. The no-op check shows only that the
    final actions already lie in [−1, 1].
  - X1 therefore tests the BC side of KI-052, and finds that BC's missing bound does not
    explain the H4 gaps. It does not test whether in-sampler clipping contributes to
    diffusion's own robustness.
  - That needs a `clip_sample=False` arm of the 6 diffusion checkpoints on the same sets: 15
    evaluations, about 1.6 container-hours (an estimate from the Session 7 times). It was not
    run. KI-052 stays half-open.
  - The preregistration, the evaluations and the analysis are unchanged by this correction;
    only the interpretation in the records is rescoped.

### D-108 Session 14 runs X3, more training seeds (OWNER, 2026-11-10, at the Session 13 STOP)
The owner was asked once, with the facts: X1's outcome; X3 at 1–2 sessions, about 5 CPU-hours
and 2.3 GB per extra Can seed, so about 6 seeds fit in the 15 GB free (estimates); X2 at 3 or
more sessions. The owner chose **"X3: more seeds"**. Session 14 runs X3
(`docs/extensions/X3_more_seeds.md`), a separate replication. Its first step is to measure disk
and per-seed cost and to ask the owner to confirm the task(s) and the number of extra seeds.
X2 stays specified and unscheduled.

## Session 14 (2026-11-10): post-release extension X3
Level 1, the RGB study, Level 2 and X1 stay frozen. Session 14 runs X3, more training seeds on
Can, under D-105 and D-108. Every X3 result is labelled "separate replication" and is reported
beside, never instead of, the frozen results. No paid or remote compute; no rollout on any
frozen held-out suite.

### D-109 X3 scope: Can only, seeds 3–5, npm cache cleared first (OWNER, 2026-11-10, Session 14)
The owner was asked once, with facts measured at the start of Session 14:
- free disk 12.1–13.8 GiB, moving with swap (16.8 of 17.4 GB in use); the orchestrators stop
  launching below 4 GiB (D-069);
- per extra Can seed, from the recorded seed 1–2 runs: 5.18 CPU-h of training (diffusion
  3.98, BC-RNN 0.85, BC-MLP-chunk 0.18, BC-MLP 0.16; diffusion 4.1 h of wall time), 6.39
  container-hours of validation, 2.33 GiB of disk (training run plus 20 validation records);
- the seed-level interval half-width for Can diffusion − BC-MLP, estimated from the frozen
  per-seed spread (SD 10.3 pp BC-MLP, 3.1 pp diffusion): ±23.1 pp with 3 seeds (the frozen
  value), ±12.5 with 5, ±10.8 with 6, ±8.7 with 8 (estimates);
- elapsed time, estimated from the Session 7 throughput: about 4.5 h for 2 extra seeds,
  5.5 h for 3, 9 h for 5.

Answers: **Can only** (Lift saturates; KI-034); **3 extra seeds (3, 4, 5), 6 in all**;
**clear the npm cache first**. Done before anything else: `npm cache clean --force` (a
regenerable cache outside the repository): free disk 16.24 → 34.07 GiB (measured). Nothing
under `runs/` was touched.

### D-110 Makefile defaults bumped to Session 14; X3 targets (FIXED, before any code change)
- `EVIDENCE_SESSION` falls back to 14 and `PREFIX` to `s14` (the D-092/D-097/D-106 pattern).
  Evidence goes to `reports/session14/` and `logs/session14/` with
  `ROBOFORGE_EVIDENCE_SESSION=14`. `OBS_OUT` stays pinned to Session 10 (D-092).
- New targets: `make x3-states` (the X3 held-out starts, on the canonical container, once),
  `make check-x3-states` (host check), `make x3-plan` (resolve the X3 plan, launches nothing),
  `make x3-train` (the X3 training suite). The evaluation and analysis targets (`make x3-run`,
  `make x3-analyze`) are added with their code, before the first rollout on the X3 suite.
- The held-out, placement and analysis targets of the frozen studies and of X1 are never run.

### D-111 X3 design: the frozen Can configurations with 3 new seeds, on fresh held-out starts (FIXED, Session 14, before any X3 training run or rollout)
**Question (KI-053, audit B-1/B-2, D-095).** With 3 training seeds per family, two Can
separations of the preregistered bootstrap (H2 diffusion − BC-MLP +18.7 pp; H3 BC-RNN −
BC-MLP +19.3 pp) are not separated by a seed-level t interval ([−4.5, +41.8], [−2.8, +41.5]).
Do the declared Can contrasts H2, H2-c and H3 hold with 6 seeds, on fresh held-out starts?

**Label.** "Separate replication". The frozen Level 1 result (3 seeds × 50 frozen starts,
`reports/session7/results/results.json`, SHA-256 `b8b097c3…`) stays the reported result. X3
is shown beside it, never pooled with it, and never replaces, reranks or relabels it.

**Training (new seeds only, no retuning).**
- Each family trains its frozen chosen Can configuration (`docs/FINAL_TEST_FREEZE.md` §3):
  diffusion trial 2 (cosine), BC-MLP trial 2 (cosine), BC-MLP-chunk trial 4 (weight decay
  0.1), BC-RNN trial 2 (cosine), with seeds 3, 4 and 5.
- The suite plan admits only an unchanged trial 1 per family, so each chosen configuration is
  written out as a base config in `configs/extensions/x3/`: the frozen base config with that
  override applied, nothing else (the A3 precedent). With seeds 0–2 these configs resolve to
  the frozen config SHA-256 and run ID of all 12 frozen Can runs. With seeds 3–5 they differ
  from them in the seed alone (`tests/unit/test_x3_configs.py`). So each new run's
  configuration hash equals the frozen one except for the seed.
- Plan `configs/experiments/x3_can.yaml`: prefix `s14`, tuning seed 3, replicate seeds 4 and
  5, `replicates_with_tuning: true` (D-068). The one-trial "tuning" record chooses trivially;
  it is not a tuning step. The 12 run IDs are fixed by the plan and listed in the
  preregistration.
- Budget as in Session 7: 100,000 steps, batch 100, checkpoints every 5,000 (K = 20), 1 torch
  thread. Every checkpoint is evaluated on the 50 Can validation starts on the canonical
  container and selected by the RESEARCH_PROTOCOL §7 rule (`scripts/select_checkpoint.py`).
  Selection uses validation starts only.
- Training runs on the macOS host (KI-061), through the frozen `scripts/run_suite.py`, from a
  clean tree. At most 6 training runs at a time: the 3 diffusion runs (the critical path) and 3
  BC runs. That is enough, because every BC run finishes well inside a diffusion run, and it
  keeps memory below the Session 7 load (KI-035). At most 4 containers through the host-wide
  slots, and no launch below 4 GiB free (D-069).
- The training path is unchanged since the frozen runs. Between `f0715c4` (the seed 1–2 runs)
  and now, `src/roboforge/train`, `models` and `data` gained new files only, and the base
  model configs are unchanged; `scripts/run_suite.py` and `train/suite.py` changed in
  orchestration only (slots, disk guard). The `.venv` package versions equal the frozen
  manifests'.
- Nothing is committed while the suite runs: any new or modified tracked file makes the tree
  dirty, and the orchestrator then refuses its next launch.

**The X3 held-out suite.**
- Name `x3_heldout`, task Can, **100 starts**, seeds 14,000,000 + i (i = 0…99), reserved block
  [14,000,000, 15,000,000). It is drawn exactly as the frozen suites are
  (`roboforge.sim.snapshot.generate`: seed `np.random` and `random`, then `env.reset()`), and
  each start is checked to restore exactly and not to succeed at t = 0. Config
  `configs/extensions/x3.yaml`; module `src/roboforge/replication/x3_states.py`; script
  `scripts/generate_x3_states.py`.
- **Why 100 starts.** The frozen suite has 50. Doubling it halves the state-sampling variance
  of every per-seed rate (about 6 pp binomial SD at 50 starts for BC-MLP's 77 %). The extra
  cost is about 1.7 container-hours and 0.5 GiB (estimates).
- **Disjoint by construction and by test.** The block lies outside the validation
  [100,000, 200,000), held-out [900,000, 1,000,000) and robustness [300,000, 400,000) blocks,
  and outside X1's development seeds 13,000,000 + i. The starts are checked against every
  committed Can suite's open manifest by state hash (`tests/research_integrity/test_x3_scope.py`).
- **Format and files.** The suite uses the frozen format (`roboforge.eval_suite/1`), so the
  frozen `eval_states.load_suite` reads it unchanged:
  `data/eval_states/can/x3_heldout.{json,data.jsonl.xz}`. `partitions.json` (existing data)
  is not edited.
- **Partition label `eval_adhoc`.** The frozen vocabulary (`snapshot.PARTITIONS`, pinned) has
  no X3 label. The frozen code itself files these seeds as ad hoc:
  `eval_states.suite_of_seed(14_000_000)` returns `None`, and
  `check_seed_allowed('can', 14_000_000)` returns `eval_adhoc` (test). `eval_final_test` would
  make X3 records look like a revisit of the frozen held-out test, and `eval_robustness` means
  the H4 validation regime. The manifest's `role`, `decision` (D-111) and `label` fields and
  every X3 summary carry the X3 meaning. X3 analyses select by suite name and manifest content
  hash, never by partition.
- **Generated on the canonical container** (`ROBOFORGE_WRITE_EVAL_STATES=1`, `make
  x3-states`), then checked by regeneration: bitwise in the container, and on the host within
  `CROSS_PLATFORM_STATE_ATOL` with identical XML. Committed with its content hash **before any
  rollout on it**.
- The frozen lock does not cover these seeds, so discipline must: no X3 start is stepped
  before the preregistration and the evaluator are committed, not even by a smoke test. The
  evaluator's engineering check runs on validation starts.

**Evaluations (24, each once, on all 100 X3 starts).**
- Checkpoints:
  - the 12 frozen Can checkpoints, 4 families × seeds 0–2, read from the committed freeze
    (`docs/FINAL_TEST_FREEZE.md`, blob `bfdb4b77…`) with their SHA-256. This is a new suite,
    not a revisit of the frozen held-out suite;
  - the 12 new checkpoints, the §7 selections of the 12 X3 runs.
- Settings as frozen: canonical container, R = 1, policy seed 0, diffusion noise from
  `SeedSequence([0, sha256(state_id)[:8], 0])`, first success or 400 steps, resets only from
  the stored snapshots, the first 3 episodes re-simulated (`verify_record`).
- **No random baseline.** H1 is not replicated: every frozen H1 difference is separated by
  ≥ 76.7 pp, and nothing in KI-053 concerns it.
- **Evaluator, in new files only** (`src/roboforge/replication/x3.py`, `scripts/eval_x3.py`,
  `scripts/run_x3.py`, `scripts/analyze_x3.py`). These are written after the preregistration,
  during training, outside the repository. They are committed, tested and given a dated ledger
  row with their SHA-256 before the first rollout on the X3 suite, and they implement the
  preregistration exactly.
  - The evaluator reuses the frozen `LearnedPolicy` and `validation.run_states`. It admits only
    the X3 suite, except in an engineering-check mode that admits only the validation suite and
    writes to `runs/x3_checks/` and `logs/session14/x3_checks/`.
  - The host orchestrator launches at most 4 containers through the host-wide slots, on a clean
    tree. It refuses unless the preregistration is committed unmodified and names the suite's
    content hash, and every new run's selection record is committed unmodified.
  - No evaluation is ever rerun, except one engineering rerun under the frozen §14.6 rule.
- **Engineering check (not a result), before the first X3 rollout:** in check mode, the
  evaluator must reproduce two committed Session 7 Can validation summaries exactly, with equal
  per-episode `physics_sha256`, success and first-success step on all 50 starts. They are
  BC-MLP seed 0 at step 55,000 and diffusion seed 0 at step 65,000; the diffusion one checks
  the noise seeding.
- **Outputs.** Records go to `runs/x3/<eval_id>/`, where `<eval_id>` follows the frozen pattern
  with suite tag `x3_heldout`. Summaries go to `reports/session14/x3/evals/<eval_id>.json`
  (schema `roboforge.x3_eval/1`), logs to `logs/session14/x3/`, and the analysis to
  `reports/session14/x3/`. Nothing is overwritten.

**Analysis (fixed now; the preregistration states it in full and verbatim).**
- Outcome matrices per family: 6 seeds (0–5) × 100 starts, in the manifest's state order.
- **Primary interval:** the frozen §9 hierarchical paired bootstrap,
  `roboforge.eval.statistics.hierarchical_paired_bootstrap`, unchanged: B = 10,000, seed
  20261106, draw order states, bc_mlp, bc_rnn, bc_mlp_chunk, diffusion_unet; 95 % percentile
  intervals.
- **Seed-level check, co-primary for robustness:** the D-095 Welch t 95 % interval on the 6 + 6
  per-seed success rates (`roboforge.paper.sensitivity.welch`).
- **Declared contrasts (Can):** H2 diffusion − BC-MLP, H2 diffusion − BC-RNN, H2-c diffusion −
  BC-MLP-chunk, H3 BC-RNN − BC-MLP.
- **Verdict per contrast:**
  - **separated, robust (±):** both intervals exclude 0 on the same side;
  - **separated by the bootstrap only:** method-dependent, as in KI-053;
  - **separated by the seed-level interval only:** method-dependent;
  - **not separated:** neither interval excludes 0. This is not evidence of equality.
- **Secondary:** the censored episode length (first-success step, else 400) for the 4
  contrasts, with the same bootstrap. The contrasts are also computed on the frozen seeds 0–2
  alone (same checkpoints, new starts) and on the new seeds 3–5 alone (new checkpoints, new
  starts), to tell the start draw from the seed count. Descriptive only: per-seed rates, family
  means and the frozen failure taxonomy counts.
- **Side by side:** each contrast is shown with the frozen value and verdict beside the
  replication. The frozen side (3 × 50) is read through the pinned SHA-256 of `results.json`,
  and its seed-level interval is recomputed from that file's `per_seed` values with the same
  Welch function, as D-095 did. The two are never pooled.
- **Engineering incompleteness:** the frozen rule of `docs/FINAL_TEST_FREEZE.md` §4.
- The readings are fixed verbatim in the preregistration, in separated / not-separated terms
  only.

**Estimates, from measured Session 7 times.**
- Training: 15.5 CPU-h (3 × 5.18), about 4.2 h of wall time.
- Validation: 19.2 container-hours.
- X3 evaluations: about 3.4 container-hours (24 × about 0.14 h at 100 starts).
- In all, about 6.5 h elapsed and about 8 GiB of disk (34 GiB free).
- $0.

**Not done, by decision:** Lift; more than 3 new seeds; H1 and the robustness sets (H4); the
diffusion side of KI-052; any change to the paper, README or Observatory (D-105).

**Outcome (separate replication; Session 14, 2026-11-11; LEDGER E14-08 to E14-12).** All 12
new runs trained and were selected on validation starts only (D-112, D-113); the 24 X3
evaluations ran once and the analysis once (`reports/session14/x3/results.json`, SHA-256
`4299a149…`). H2 diffusion − BC-MLP +22.7 pp and H3 BC-RNN − BC-MLP +23.8 pp are separated
under both the bootstrap and the seed-level interval (*holds, robust*); H2 diffusion − BC-RNN
and H2-c are not separated (*unchanged*). The frozen result stands as reported. Measured
against the estimates above: training 33.0 CPU-hours summed (12 runs; 8.1 h of wall time, with
validation and swap contention) against 15.5; validation 35.5 container-hours against 19.2 (the
211 evaluations of driver run 2 alone took 17.0); X3 evaluations 3.1 container-hours (47 min)
against 3.4; 13.2 h elapsed against 6.5; 7.1 GiB of disk against 8; $0.

### D-112 X3 validation and selection finished by an evaluation-only driver after the suite stopped (FIXED, Session 14, 2026-11-11, before any X3 rollout)
**What happened (KI-062).** `make x3-train` (the frozen `scripts/run_suite.py`, from clean
`fb9f05e`, launched 21:37:25 UTC) trained the 12 X3 runs as preregistered. At 23:57:37 UTC,
right after it selected BC-MLP seed 3 (step 75,000, 37/50), it stopped at its first
configuration choice: `ValueError: bc_mlp trial 1: task/seed can/3 != can/0`. The frozen
`roboforge.eval.tuning.choose_configuration` (pinned) admits only seed-0 tuning trials. The X3
plan's one-trial "tuning" seed is 3, and `make x3-plan --dry-run` never reaches the chooser. The
suite then launched nothing and waited for its running jobs: the 6 training runs went on (all 9
BC runs complete; the 3 diffusion runs still training at this record), and the 3 BC-MLP seed 4
evaluations in flight finished with exit 0. Nothing failed and nothing is lost: each family has
one configuration, fixed before training (D-111), so the choice decides nothing.

**Decision.** The remaining validation evaluations and §7 selections are run by a new
evaluation-only driver, `scripts/x3_validate.py` with `src/roboforge/replication/x3_validation.py`
(`make x3-validate`), committed and tested before its first launch:
- It launches exactly what the suite would have launched: the frozen
  `scripts/eval_policy.py` per checkpoint (50 Can validation starts, canonical container), then
  the frozen `scripts/select_checkpoint.py` per run (RESEARCH_PROTOCOL §7), with the suite's
  argv and environment. A unit test pins the evaluation argv to every evaluation the suite
  logged, and the training-process match to every training launch it logged.
- It mirrors the suite's rules: one eligible 50-start summary per checkpoint, more than one is
  an error, an evaluation directory without an eligible summary stops it, clean tree and
  ≥ 4 GiB free at every launch, ≤ 4 containers through the host-wide slots (D-069), diffusion
  first, no automatic retry.
- It never trains, resumes or reuses the suite orchestrator (research-integrity test). A run
  without a final manifest is waited for while its `train.py` process lives; otherwise it stops
  for a manual decision.
- The BC-MLP seed 3 selection the suite wrote is checked against its 20 sources and kept, not
  rewritten. The suite's 23 evaluation summaries are used as they are.
- When all 12 runs are selected (or failed in training) it writes
  `reports/session14/suite/x3_validation.json` (schema `roboforge.x3_validation/1`). The X3 gate
  (`x3_jobs`) now requires that file committed and unmodified, and checks every new checkpoint
  against it, in place of the suite summary `reports/session14/suite/x3_can.json`. The suite
  still writes its own summary when its last training run ends; that partial file is committed
  as the record of the stop, not used by the gate.

**Unchanged:** the configurations, seeds, budget, run IDs, validation starts, the §7 rule, the
X3 suite, the 24 evaluations and the analysis. The preregistration is not edited.

**What this supersedes in `docs/extensions/X3_PREREGISTRATION.md` (§2, §4) and D-111:**
- "The one-trial 'tuning' record chooses trivially; it is not a tuning step." No tuning record
  exists (`reports/session14/tuning/` is empty): the frozen chooser refused the seed-3 trial.
- "through `scripts/run_suite.py` (`make x3-train`)": true of training. Validation and selection
  went through the suite for BC-MLP seed 3 (20 evaluations and its selection) and BC-MLP seed 4
  steps 5,000–15,000 (3 evaluations), and through `make x3-validate` for the rest.
- §4 "Code": the code list gains `src/roboforge/replication/x3_validation.py`,
  `scripts/x3_validate.py` and their tests (`tests/unit/test_x3_validation.py`,
  `tests/research_integrity/test_x3_validation_scope.py`). The orchestrator also refuses
  unless the validation summary is committed unmodified.
- D-111 "Estimates": the 4.2 h of training wall time was low. Diffusion trained at about
  4 steps/s against Session 7's 6.7 while 6 runs and 4 containers shared 10 cores
  (measured from `progress.json`). Validation evaluations took a median of 1,087 s per
  BC-MLP checkpoint under that load, against 404 s in Session 7.

**Recorded** as a dated deviation in the ledger (E14-05) before the first X3 rollout, as the
preregistration's §4 requires. The analysis's engineering section points here.

### D-113 X3 validation paused while the diffusion runs train, after the host thrashed swap (OWNER, 2026-11-11, Session 14)
- **Facts given to the owner (03:29 UTC):** with the Docker VM (7.3 GB footprint), 4
  validation containers, the 3 diffusion trainings and the desktop apps on a 16 GB host, the
  machine swapped about 3,200 pages in and 6,800 out per second (load average about 100).
  Diffusion trained at 0.9 steps/s (35,900 → 45,500 in 3 h; 4.0 steps/s before validation
  resumed), and a BC-MLP validation evaluation took 1.6–2.6 h instead of about 7 minutes
  (Session 7 median 404 s).
- **Decision (owner: "Pause validation"):** no new validation launches until the diffusion runs
  finish training; running evaluations finish (nothing killed); then `make x3-validate` resumes
  with 4 containers. The driver was stopped with SIGINT at 03:32:46 UTC and exited 3 at
  03:40:11 after its 2 running evaluations finished (exit 0).
- **Results-neutral:** scheduling only, as D-059. The driver is idempotent: evaluated
  checkpoints are skipped and existing selections are kept. The evaluation code, starts,
  settings and the §7 rule are unchanged.

### D-114 Three Session 14 evidence files with local paths are disclosed, not scrubbed (OWNER, 2026-11-11, Session 14 STOP)
- **Facts given to the owner:** at the STOP, `make release-check` failed `private_paths` on 3
  committed Session 14 evidence files outside the D-100 set:
  `logs/session14/suite/stdout_x3.log` and `logs/session14/suite/x3_can/events.jsonl` (the
  frozen `run_suite.py` traceback and argv when the suite stopped, KI-062) and
  `logs/session14/x3/check_host.log` (a robosuite warning naming its install path). Each path
  reveals only the local username, as in D-100.
- **Decision (owner: "Disclose 3 files"):** these 3 files are added by name to
  `DISCLOSED_PATH_FILES` in `src/roboforge/release/scan.py` (not pinned by `make
  freeze-check`), with a unit test. The evidence bytes stay unchanged. Any other file with such
  a path still fails the check; `logs/session14/` is not disclosed as a prefix.
- `docs/FINAL_RELEASE_AUDIT.md` describes v1.0.0 and is not edited; a later release audit
  would list these files. Nothing is pushed.

### D-115 Session 15 runs X2 (OWNER, 2026-11-11, at the Session 14 STOP)
The owner was asked once, with the facts: X3's outcome (both frozen Can separations hold under
both interval methods with 6 seeds on 100 fresh starts; the two parity contrasts are unchanged);
nothing pushed; options stop, a v1.1 addendum, or X2. The owner chose **"X2"**. Session 15 runs
X2 (`docs/extensions/X2_harder_task.md`), under D-105 and its own decisions; the exact prompt is
in `docs/HANDOFF.md` (# Session 14 Handoff). Whether X1 or X3 is published stays the owner's
decision (D-105).

## Session 15 (2026-11-12): post-release extension X2, phase X2-a (data and contract)
Level 1, the RGB study, Level 2, X1 and X3 stay frozen. Session 15 runs phase X2-a of X2
(`docs/extensions/X2_harder_task.md`) under D-105 and D-115. X2 is a separate study with its
own decisions, freeze and held-out suite; none of its results replaces or revises a frozen
number. No held-out rollout of any kind in X2-a: the headroom pilot steps X2 validation starts
only. No paid or remote compute.

### D-116 X2 option A: Can, mixed-human (MH) (OWNER, 2026-11-12, Session 15)
The owner was asked once, with facts measured at the start of Session 15:
- free disk 30 GiB (`df`), `runs/` 45 GB; swap 11.9 of 13.3 GB in use before any job;
- file sizes and SHA-256 from the Hugging Face tree API at the pinned revision `74fa0184…`
  (read 2026-11-12): Can MH `low_dim_v15.hdf5` 112,792,088 B (`c4a34c83…`) and `demo_v15.hdf5`
  79,780,388 B (`6f5a0e96…`), not in any manifest; Square PH 51,145,256 / 68,982,198 B, already
  in `data/manifests/robomimic_v1.5_ph.json` with SHA-256 equal to the LFS ids re-read now;
  Square MH 123,300,120 / 73,913,499 B;
- new infrastructure per option (X2_harder_task.md §2): A needs manifest entries, an
  operator-aware split if the file carries operator identity, a demonstration that the
  validated Can contract applies, new X2 suites and the pilot; B needs a full Session 0–2
  contract for Square; C needs both;
- downloads are < 0.25 GB in every case; the binding disk cost is later training (2.3 GB per
  seed of the four families on Can PH).

Answer: **"A: Can MH (Recommended)"**. X2 tests whether the generative head helps when the
demonstrations come from several operators of different quality, on the task, simulator,
controller and evaluation machinery already validated for Can. Square and Lift MH are not
part of X2.

### D-117 Makefile defaults bumped to Session 15; X2 targets (FIXED, before any code change)
- `EVIDENCE_SESSION` falls back to 15 and `PREFIX` to `s15` (the D-092/D-097/D-106/D-110
  pattern). Evidence goes to `reports/session15/` and `logs/session15/` with
  `ROBOFORGE_EVIDENCE_SESSION=15`. `OBS_OUT` stays pinned to Session 10 (D-092).
- X2 targets are added with their code, each before its first use: `x2-download` (fetch and
  verify the MH files), `x2-inspect` (the inspection report), `x2-data` / `check-x2-data`
  (build the X2 data artifacts; rebuild and compare), `x2-contract` (the Can-contract checks
  and the open-loop replay), `x2-states` / `check-x2-states` (the X2 suites on the canonical
  container; regenerate and compare), and the pilot targets (D-121).
- The held-out, placement and analysis targets of the frozen studies, X1 and X3 are never run.

### D-118 X2 data: the Can MH files enter a new manifest, SHA-256 from the pinned revision, verified on download (FIXED, before any download)
- **New manifest file** `data/manifests/robomimic_v1.5_mh.json` (schema and source block of the
  PH manifest: repository `robomimic/robomimic_datasets`, revision
  `74fa018461f479cd9fd15b924a16103012096203`, license MIT, the robomimic citation). The PH
  manifest is existing data and is not edited.
- **Entries** (sizes and SHA-256 = the LFS object ids of the Hugging Face tree API at that
  revision, read 2026-11-12):

  | Task | Type | path_in_repo | Size (bytes) | SHA-256 |
  |---|---|---|---|---|
  | can | low_dim | `v1.5/can/mh/low_dim_v15.hdf5` | 112,792,088 | `c4a34c837913446d295c6ca38f2b31888d69812198478af470ecb84a99e9f7d7` |
  | can | raw | `v1.5/can/mh/demo_v15.hdf5` | 79,780,388 | `6f5a0e96fc375468ae19ccac86a6dee3b4d3862ef0fe37ca2d02e93d9046fe6b` |

  Local paths `raw/robomimic_v1.5/can/mh/{low_dim_v15,demo_v15}.hdf5` under
  `$ROBOFORGE_DATA_ROOT` (gitignored, never redistributed).
- **Download and verification** through the frozen `roboforge.data.fetch` (`load_manifest`
  with the new manifest name, `fetch_file`, `verify_file`): a file is accepted only if its size
  and SHA-256 equal the manifest; a mismatch deletes the partial file and stops. A new thin
  script (`scripts/x2_download.py`, `make x2-download`) does this; it prints repository-relative
  paths only. The frozen `scripts/download_data.py` and the PH manifest are unchanged.
- **Why both files.** `low_dim` is the training source (as for PH, D-018). `raw` is downloaded
  to look for operator identity in the teleoperation logs (`user_info`, `controller_info` and
  related keys; X2_harder_task.md §3) and to check that its states, actions and model XML equal
  the low-dim file's, as Session 1 did for PH.
- Nothing about the data is assumed: episode count and lengths, masks and filter keys,
  operator information, `env_args` and the registered evaluation horizon are read from the
  files and recorded before the split (D-119) and before any training.

### D-119 X2 split: robomimic's operator-stratified upstream masks; operator identity recorded per episode (FIXED, Session 15, after the inspection and before any build or training)
**What the files carry** (`reports/session15/x2/inspection.json`, from the hash-verified files):
300 demos, 62,756 samples, lengths 98–1,050 control steps (median 178; Can PH 82–151). Six
operator masks, `{better,okay,worse}_operator_{1,2}`, 50 demos each, contiguous index blocks;
they partition all 300 demos, and each quality mask (`better`, `okay`, `worse`) is the union of
its two operators. Each operator also has `_train`/`_valid` masks of 45/5, and the upstream
`train` (270) and `valid` (30) masks are exactly their unions. The raw file's teleoperation
arrays carry no identity: `interventions` and `policy_acting` are false at every step of every
demo, `user_acting` true at every step; there is no `user_info`. So operator identity exists
only as the six masks (an anonymous operator index within a quality tier).
- **Split:** whole episodes, robomimic's upstream masks `train` (270) / `valid` (30), the D-019
  precedent. They are operator-stratified: 45 train / 5 valid per operator, every operator in
  both partitions. MASTER_SPEC §3.3's hash split would serve the same purpose; the upstream
  masks are kept because they are already deterministic, operator-aware and published.
- **Operator identity is recorded** per episode in the X2 task manifest (`operator`, `quality`),
  from the masks only. Nothing else is inferred about the operators.
- **Claims:** every family trains on all six operators, and closed-loop evaluation has no
  operator. Every X2 result is therefore a within-operator result: no held-out-operator claim
  is made. The `valid` demos give offline losses only, as in Level 1.
- **Normalization:** per-dimension min-max on the 270 training episodes' per-step arrays only
  (the frozen `Normalizer.fit`, `configs/data/can.yaml` settings).
- **Features:** the frozen Can observation config `configs/data/can.yaml`, unchanged (19-D,
  KI-003 correction included), so the observation and action specs must hash equal to the
  committed Can PH specs.
- **Flagged episodes:** the frozen sanity checks (`episode_issues`, the saved-final-state
  success check) run on every demo. A flagged demo is listed with its reason in the split
  manifest and is never silently dropped; X2 training refuses a split that contains one, as
  `SequenceDataset.from_committed` does. Any flag stops the pipeline for a dated decision
  before training.
- **Training data for the pilot:** all 270 training episodes, all quality tiers (the natural MH
  setting, and robomimic's MH benchmark setting). Training on a quality subset (for example
  `better` only) is an X2-b design question, not decided here.

### D-120 X2 episode horizon: 500 control steps, robomimic's registered Can MH horizon (FIXED, Session 15, before any X2 rollout)
- **Facts:** robomimic@`d309eae` registers an evaluation horizon of 500 for every Can MH file
  and 400 for Can PH (the value Level 1 adopted, D-026). 14 of the 300 MH demos are longer than
  400 steps and 10 longer than 500, all from the two "worse" operators (p95 397, max 1,050).
  The longest Can PH demo is 151 steps.
- **Decision:** X2 episodes end at the first success or at **500** steps. This is the
  principle that set Level 1's 400 (the dataset's registered horizon), applied to the MH files.
  A 400-step limit would cut off slow, demonstrated behaviour of the slower operators and could
  create failures the data do not imply.
- **Where it lives:** the X2 task spec (`eval_horizon_steps: 500`), which the X2 evaluator's
  environment reads (D-121). Everything else in the episode rule is the frozen one (first
  success ends the scored episode; resets only from stored snapshots; R = 1, policy seed 0).
- **Consequence:** X2 success rates are not on Level 1's 400-step scale and are never compared
  with or pooled with Level 1 numbers. A failed X2 episode costs 500 steps of simulation.

### D-121 X2 data artifacts and how frozen code is bound to them (FIXED, Session 15, before any X2 build, training or rollout)
**Artifacts** (new paths; nothing under the `*_ph` names changes), built by a new module
`src/roboforge/x2/data.py` from the hash-verified MH low-dim file with the frozen building
blocks (`build_observation_spec`, `build_action_spec`, `build_task_spec`, `build_episodes`,
`episode_issues`, `upstream_split`, `Normalizer`) and the frozen Can simulator config:
- processed float32 file (gitignored): `processed/robomimic_v1.5_mh/can/lowdim_v1.hdf5`;
- `data/splits/can_mh.json`, `data/normalization/can_mh_lowdim_v1.json`,
  `data/manifests/tasks/can_mh.json` (schema `roboforge.task_manifest/1`, `dataset_id`
  `robomimic_v1.5/can/mh`, per-episode operator and quality, task spec with the D-120 horizon).
`make x2-data` writes them; `make check-x2-data` rebuilds in memory and requires equality, as
`make check-data` does for PH.

**Binding frozen code to X2 data.** The frozen training and evaluation code finds a task's
data through `roboforge.data.pipeline.load_task_artifacts(task)`, which reads the `*_ph`
paths; `roboforge.train.run.load_windows` builds the windows from the PH processed file; and
`LearnedPolicy` refuses a checkpoint whose spec or normalization hashes differ from the
committed Can PH ones. The frozen files stay byte-identical. A context manager in a new module
(`roboforge.x2.binding.x2_bindings`) rebinds, for task `can` only and only inside the context,
exactly these module attributes:
- `load_task_artifacts` in `roboforge.data.pipeline` (so `load_specs` and
  `SequenceDataset.from_committed` follow), `roboforge.train.run`, `roboforge.eval.learned` and
  `roboforge.sim.env_adapter`, to the integrity-checked X2 loader;
- `roboforge.train.run.load_windows`, to the X2 windows (the frozen `SequenceDataset` and
  `WindowTensors` over the X2 processed file, refusing flagged episodes).
Everything else runs unchanged: `train_run`, `Trainer`, checkpoints, `LearnedPolicy`,
`run_states`, `run_episode`, `verify_record`. Inside the context the run manifest records the
X2 data hashes and `dataset_id`, the environment records the X2 task spec (horizon 500), and
`LearnedPolicy` checks a checkpoint against the X2 normalization and specs. Only new X2 entry
scripts enter the context; nothing launched through the frozen scripts sees it.

**Guards (tests, before first use):**
- the X2 loader refuses unless the X2 observation and action spec hashes equal the committed
  Can PH ones (read with the frozen loader), so "the validated Can contract applies" is an
  enforced equality;
- inside the context the rebound names resolve to the X2 functions; on exit (also on error) the
  frozen ones are restored;
- the frozen `LearnedPolicy` refuses an X2 checkpoint outside the context, and the X2 context
  refuses a Can PH checkpoint (different normalization);
- a short training run inside the context, with the X2 loader pointed at the Can PH artifacts,
  is bitwise equal to the same run without the context: only the data source changes.

### D-122 X2 data: one flagged demo, `demo_81`, is kept as an accepted flag (FIXED, Session 15, before any X2 data are written or any training)
- **Found by** the first `scripts/x2_data.py` build: the frozen sanity check flags `demo_81`
  (operator `worse_operator_2`, upstream `train` mask, 246 steps) with "final step has no
  success reward/done". The stored reward is 1.0 at steps 241–244 and 0.0 at the last step
  245 (`done` is 1 at all five). The simulator's own success predicate holds on the demo's
  saved final state (`build_episodes`' `final_state_success` is true), and every recomputed
  feature equals the stored observations. No other demo is flagged.
- **Decision:** keep `demo_81` in the training split. It is a successful demonstration by the
  simulator's predicate; the flag concerns only the stored reward label of its last step, and
  no X2 model reads rewards or dones (inputs are the 19-D observations, targets the actions).
  Dropping it would remove a valid demonstration of a slow operator.
- **Mechanism:** the split manifest keeps the flag and its reason under `flagged` and adds
  `accepted_flags: {demo_81: {reason, decision: D-122}}`. The X2 builder refuses to write, and
  the X2 window loader refuses to train, unless every flagged episode is accepted with exactly
  that reason; any new or different flag still stops for a dated decision (D-119).

### D-123 X2 evaluation suites: `x2_validation` (50) and `x2_heldout` (100) from fresh seed blocks (FIXED, Session 15, before generation and before any X2 rollout)
- **Suites** (config `configs/extensions/x2_suites.yaml`, module `src/roboforge/x2/states.py`,
  script `scripts/generate_x2_states.py`, targets `make x2-states` / `make check-x2-states`):

  | Suite | Task | n | Seeds | Reserved block | Role |
  |---|---|---|---|---|---|
  | `x2_validation` | Can | 50 | 15,000,000 + i | [15,000,000, 16,000,000) | X2 development and §7 checkpoint selection; the X2-a headroom pilot |
  | `x2_heldout` | Can | 100 | 16,000,000 + i | [16,000,000, 17,000,000) | one rollout per selected checkpoint after `docs/extensions/X2_FREEZE.md` (X2-c); never stepped before |

- **Why these sizes.** 50 validation starts keep the §7 selection cost and rule identical to
  Level 1 and X3. 100 held-out starts halve the state-sampling variance of every per-seed rate
  relative to 50 (the X3 precedent, D-111).
- **Draw.** Exactly as the frozen suites (`roboforge.sim.snapshot.generate`: seed `np.random`
  and `random` with `seed_base + i`, then `env.reset()` of the Can environment built from
  `configs/simulator/can.yaml`), each start checked to restore exactly and not to succeed at
  t = 0. The MH demonstrations come from the same environment and reset distribution, so the
  starts are in distribution by construction.
- **Disjoint by construction and by test.** Both blocks lie outside the validation
  [100,000, 200,000), held-out [900,000, 1,000,000) and robustness [300,000, 400,000) blocks, X1's
  development seeds 13,000,000 + i and X3's block [14,000,000, 15,000,000), and outside each
  other. After generation each suite is checked disjoint by state hash from every committed Can
  suite's open manifest and from the other X2 suite
  (`tests/research_integrity/test_x2_scope.py`).
- **Format and label.** The frozen format (`roboforge.eval_suite/1`) under
  `data/eval_states/can/{x2_validation,x2_heldout}.{json,data.jsonl.xz}`; partition `eval_adhoc`
  (the frozen vocabulary has no X2 label; the D-111 precedent); `partitions.json` is not edited.
  Names are X2's own; no X2 file contains a locked suite's name.
- **Generated on the canonical container** (`ROBOFORGE_WRITE_EVAL_STATES=1`, `make x2-states`),
  checked by regeneration bitwise in the container (twice) and within
  `CROSS_PLATFORM_STATE_ATOL` with identical XML on the host, and committed with their content
  hashes before any rollout on them.
- **Held-out protection.** The frozen lock does not cover these seeds, so discipline and tests
  must: only the X2 suite module and its generator may name `x2_heldout` in `src/` or
  `scripts/` (research-integrity test); the X2-a evaluator admits only `x2_validation`. No
  X2-a step rolls out on `x2_heldout`, not even a smoke test.

### D-124 X2-a headroom pilot: design, measurements and the reading rule (FIXED, Session 15, before the first pilot launch)
**Question** (`docs/extensions/X2_harder_task.md` §3): with the frozen Level 1 configurations,
is Can MH neither saturated nor at zero, so that X2 can rank the families? Development numbers
only: one seed, the best of five checkpoints on the same 50 starts (biased upward, RESEARCH_PROTOCOL
§7), never a result, never pooled with or compared to any frozen number.

**Runs** (`configs/extensions/x2_pilot.yaml`, `src/roboforge/x2/pilot.py`): each family's frozen
chosen Can configuration (`docs/FINAL_TEST_FREEZE.md` §3: diffusion trial 2 cosine, BC-RNN trial
2 cosine, BC-MLP-chunk trial 4 weight decay 0.1, BC-MLP trial 2 cosine), as written out for X3
(`configs/extensions/x3/`), with seed 0 and the frozen budget (100,000 steps, batch 100,
checkpoints every 5,000, 1 torch thread). Each resolves to the frozen seed-0 Can configuration
hash (test). Run IDs `s15x2_<name>_can_seed0_<hash>` under `runs/x2/`; they differ from the frozen
`s7_*` IDs in the prefix only, and their manifests record the X2 dataset. Training on the macOS
host (KI-061) through the frozen `train_run` inside the X2 binding (D-121), by
`scripts/x2_train.py`.

**Evaluations:** checkpoints 20k, 40k, 60k, 80k and 100k of each run (stride 4 of the 20; 20
evaluations), each on the 50 `x2_validation` starts once, canonical container, policy seed 0,
R = 1, 500-step episodes (D-120), by `scripts/x2_eval.py` (admits only `x2_validation`).
Records `runs/x2/evals/<eval_id>/`; summaries `reports/session15/x2/pilot/evals/`. The other 15
checkpoints per run stay unevaluated. X2-b may count these four runs as tuning trial 1 of each
family only if its preregistration says so (they are exactly what a seed-0 trial 1 of the
frozen configuration trains, KI-062); then their remaining checkpoints are evaluated like any
trial's. No X2 held-out start is stepped.

**Orchestration and limits** (`scripts/x2_pilot.py`, `make x2-pilot`): from a clean tree; the
four trainings at once; evaluations as their checkpoints are indexed, at most 2 containers
while the diffusion run trains and 4 after it ends (KI-035, D-113); no launch below 4 GiB free
disk, on a dirty tree, or while the host swaps more than 3,000 pages/s (idle rate at launch
about 390/s; swap 13.5 of 14.3 GB in use before the pilot); no automatic rerun. Nothing is
committed while it runs.

**Measured:** training steps/s, wall and CPU time (run manifests), peak RSS (`/usr/bin/time
-l`), disk per run and per evaluation; evaluation wall time, control steps/s and latency;
host swap rate, swap use, load, free disk and container memory every 5 minutes. Report:
`reports/session15/x2/pilot/{pilot.json,tables.md}`.

**Reading rule** (fixed now). Let b_f be the highest success count among family f's five
evaluated checkpoints (of 50).
- **saturated:** b_f ≥ 48 for at least 3 of the 4 families (the Can PH pattern: three families
  at 95–100 % on validation cannot be ranked);
- **at zero:** b_f ≤ 2 for all 4 families;
- **headroom:** otherwise.
Only "headroom" supports the X2-b design proposal as planned. "Saturated" or "at zero" goes
back to the owner with options (for example Square, D-116's alternatives) before X2-b.

**Estimates** (labelled estimates, from measured Session 7 and Session 14 times): diffusion
about 4.1 h of training at Session 7's 6.7 steps/s alone, up to about 7 h at Session 14's
loaded 4 steps/s; BC-RNN 0.9–2.5 h; BC-MLP and BC-MLP-chunk about 0.2 h each; the 20
evaluations about 2–3.5 container-hours (Session 7 Can median 353–404 s per 50 starts; MH
failures run 500 steps); disk about 1.5–2 GB; $0.

**Outcome (development numbers, not a result; Session 15, 2026-11-12; LEDGER E15-06).** The
pilot ran once from clean `fe9df74`: 4 trainings and 20 evaluations, every exit 0, all 20
summaries eligible and verified, no launch hold, `runs/` gained only `x2` (isolation hash
unchanged). Best of five checkpoints out of 50: diffusion 48, BC-MLP-chunk 48, BC-RNN 34,
BC-MLP 6. **Reading: headroom** (two families at 48, three needed for "saturated"). Headroom
rests on one family margin: the top two are at the 50-start ceiling on best-of-5 validation,
so X2-b may separate the single-step families from the rest but may not separate diffusion
from BC-MLP-chunk. Measured against the estimates above: diffusion 4.62 h wall (6.0 steps/s,
estimate 4.1–7), BC-RNN 1.11, BC-MLP-chunk 0.19, BC-MLP 0.17; the 20 evaluations 2.8
container-hours (mean 503 s, range 205–819, estimate 2–3.5); disk 1.63 GiB (estimate
1.5–2); $0. The 3,000 pages/s swap guard was exceeded once (4,874 pages/s sampled at 06:38 UTC,
3 minutes after launch, with 4 trainings and 2 evaluation containers running and 516 MB of
swap free); no hold fired because the guard is checked only at launch (KI-035).
**Discrepancy, recorded and not resolved here:** the text above says X2-b may count these
runs as "tuning trial 1" of each family. In the Level 1 search space (D-055, D-061) the frozen
chosen configurations are trial 2 (cosine; diffusion, BC-RNN, BC-MLP) and trial 4 (weight
decay 0.1; BC-MLP-chunk), not trial 1 (the base configuration). Whether X2-b counts each
pilot run as the seed-0 run of the trial whose configuration it has, or retrains, is an
owner decision for the X2-b preregistration (HANDOFF # Session 15 Handoff §5).

## Session 16 (2026-11-14): post-release extension X2, phase X2-b (tuning, training, selection)
Session 16 runs X2-b on Can MH (D-116): the preregistered search, training on local CPU and
selection on X2 validation starts only. No X2 held-out start is stepped. Dates are the owner's
(EST); 2026-11-15 UTC.

### D-125 X2-b owner decisions: proceed on Can MH; pilot runs count; concurrency; disk (OWNER, 2026-11-14, Session 16)
**Owner's words** (answering # Session 15 Handoff §5): "Follow what you would recommend for
each of these decisions and continue". The recommendations taken:
1. **Proceed** with X2-b on Can MH (the D-124 reading is headroom, by one family).
2. **The pilot runs count** as the seed-0 runs of the trials whose configuration they have
   (diffusion, BC-RNN, BC-MLP trial 2; BC-MLP-chunk trial 4), mapped by config SHA-256; their
   other 15 checkpoints are evaluated like any trial's. Not retrained. This resolves the
   "trial 1" discrepancy in D-124's outcome note.
3. **Concurrency** (`configs/extensions/x2b.yaml`): at most 4 trainings; trainings plus
   evaluation containers at most 5 (1 container beside 4 trainings); a baseline host sample
   before the first launch; at most one launch per 60-s sample, each after a fresh sample;
   launches held while the last swap rate exceeds 3,000 pages/s; below 2 GiB free disk the
   trainings stop with a resume checkpoint. **Correction of # Session 15 Handoff §5 item 3:**
   the X2-a pilot already sampled swap every 60 s and gated launches on it. What it lacked was
   a sample before its first launches (its rate was undefined at startup), spacing between
   launches, and a container cap beside trainings. Other projects' containers are not touched;
   none was running at the start.
4. **Disk** (KI-008). `uv cache prune` was run at about 01:20 UTC and failed after its 300-s
   lock timeout: long-running `uvx` MCP-server processes hold the uv cache lock. It was **not
   forced**, because `--force` may delete the cached environments those running tools execute
   from. **Substituted:** `docker builder prune -f` (2.38 GB of unused build cache, listed as
   reclaimable in # Session 15 Handoff §5; the canonical image is untouched), +2.1 GiB on the
   host. Free disk at the dry run: 24.5 GiB, against a conservative need of 23.1 GiB plus the
   4 GiB floor. So the tuning phase launches, and replicate trainings launch only when free
   disk covers the remaining estimate plus the floor. The owner was notified with the options:
   quit the apps running those servers so the prune can run; approve `--force`; or approve
   removing the non-frozen development scratch runs (about 2.2 GB, citations checked first).
   Swap files share the volume: swap grew from 10.2 to 13.3 GB in 15 minutes with no X2 job
   running, which cost about 2 GiB of free disk.

### D-126 Makefile defaults bumped to Session 16; X2-b targets (FIXED, before any code change)
`PREFIX ?= s16` and `EVIDENCE_SESSION` default 16, so that no Session 16 command writes into
Session 6–15 evidence or reuses the `s15` prefix by omission. New targets: `x2b-plan` (dry run),
`x2b-suite`, `x2b-report`, `x2b-benchmark`. X2-b runs use the prefix `s16x2`.

### D-127 X2-b design: the preregistration, plan, drivers and limits (FIXED, Session 16, before the first X2-b training)
`docs/extensions/X2_PREREGISTRATION.md` governs X2-b and fixes the X2-c analysis.
- **Plan** `configs/experiments/x2_can_mh.yaml`: Level 1's four trials per family (D-055,
  D-061), the frozen budget, tuning seed 0, replicate seeds 1 and 2; resolved by the frozen
  `load_plan`. The config hashes equal Level 1's Can trials (test).
- **Selection:** every checkpoint once on the 50 `x2_validation` starts, the frozen §7
  `select_checkpoint` per run and the frozen §6.3 `choose_configuration` per family. The X2-a
  pilot summaries (`reports/session15/x2/pilot/evals/`) and X2-b's
  (`reports/session16/x2/suite/evals/`) feed the same selections; a duplicate step raises.
- **Code** (new files only; `make freeze-check` before and after):
  - `src/roboforge/x2/suite.py` (plan, pilot mapping, selection and choice from files);
  - `src/roboforge/x2/host.py` (host guards, D-125);
  - `scripts/x2b_train.py` (refuses a pilot run);
  - `scripts/x2b_eval.py` (the X2-a evaluator, X2 validation only);
  - `scripts/x2b_suite.py` (resumable orchestrator);
  - `scripts/x2b_report.py`;
  - `scripts/x2b_benchmark.py` (latency, X2_harder_task §4);
  - `configs/extensions/x2b.yaml`;
  - tests `tests/unit/test_x2_suite.py` (a synthetic pass through the frozen §7 and §6.3 rules
    to the test configuration, KI-062) and `tests/research_integrity/test_x2b_scope.py`.
- **Not reused:** `scripts/x2_eval.py` (its summaries go to the pilot's directory) and
  `scripts/x2_pilot.py` (unchanged X2-a evidence code).
- **Estimates** (labelled; from the X2-a measured times): 30.5 training-hours summed;
  evaluations about 64 container-hours (biased low); about 23 h of wall time; disk 18–23 GiB;
  $0.

**Outcome note (D-127, Session 16 STOP).** `make x2b-suite` ran once from clean `9df6628`,
01:44 UTC 2026-11-15 to 01:50 UTC 2026-11-16 (24.1 h). All 20 new runs completed and all 460
evaluations ended with exit 0. Every one of the 480 summaries (20 pilot, 460 X2-b) is eligible,
verified and on the canonical platform (`logs/session16/x2/suite/verification.txt`). The frozen
§6.3 rule chose diffusion trial 2, BC-RNN trial 4, BC-MLP-chunk trial 2 and BC-MLP trial 1.
Two choices were ties broken by the lower mean first-success step: diffusion trials 1 and 2
both 48/50 (197 vs 206), and BC-MLP-chunk trials 1 and 2 both 49/50 (156 vs 172). The X2 test
configuration is `reports/session16/x2/suite/x2_test_configuration.json`; the 12 checkpoint
hashes were re-hashed from the files. Measured cost against the labelled estimates:

| Item | Measured | Estimate |
|---|---|---|
| Training, summed | 33.86 training-hours | 30.5 |
| Evaluations | 68.98 container-hours (mean 540 s) | about 64 |
| Wall time | 24.1 h | about 23 |
| Disk added to `runs/x2` | 17.96 GiB | 18–23 GiB |
| Spend | $0 | $0 |

The latency benchmark took 8.7 min (`reports/session16/x2/suite/latency/`). No X2 held-out
start was stepped.

### D-128 Disk reclaim beyond D-125: two scratch runs and download caches (OWNER, 2026-11-14, Session 16)
Free disk at the dry run (24.5 GiB) was tight against the conservative X2-b need (D-125
item 4), so the owner approved two further reclaims.
- **Scratch runs.** Owner approval at 21:17 EST: delete non-frozen development scratch runs
  after a citation check. The check ran `git grep` for each `runs/` entry name over the
  tracked files, excluding isolation name listings. Only two entries had no other citation:
  `runs/s8smoke_rgb_bc_chunk_resnet18_lift_seed0_1a7d0da4` (465,964 KiB) and
  `runs/s8smoke_rgb_diffusion_resnet18_lift_seed0_0ce63766` (746,004 KiB). They are
  engineering smokes, never selected or reported (LEDGER E8-09), and were deleted at 02:19 UTC
  2026-11-15. Their manifest SHA-256 values are in `logs/session16/x2/suite/scratch_deleted.txt`.
  `runs/rgb_eval/s8smoke_*` was kept.
- **Download caches** (owner's answer "Download caches (~5.3 GB)"). Cleared outside the repo:
  the pip cache, Poetry's cache and artifacts (virtualenvs kept), the npm download cache
  (`~/.npm/_npx` kept) and Homebrew downloads.
- **Not approved, untouched:** untagged Docker images and `uv cache prune --force`.

No frozen run, evidence file or data file was touched. The post-suite isolation check found
the top-level `runs/` listing equal to the baseline minus exactly these two entries, by hash
(`logs/session16/x2/suite/isolation_{before,after}.txt`).

## Session 17 (2026-11-15 EST, 2026-11-16 UTC): post-release extension X2, phase X2-c (freeze, held-out once, analysis once)
Session 17 runs X2-c on Can MH (D-116): it commits `docs/extensions/X2_FREEZE.md`, runs the 12
checkpoints of the X2 test configuration once on the 100 `can/x2_heldout` starts and runs the
preregistered analysis (`docs/extensions/X2_PREREGISTRATION.md` §7) once. X2 (Can MH) results
are reported beside the frozen Level 1 Can PH result, never instead of it, never pooled with
it, and are within-operator only. Every prerequisite of the Session 17 prompt passed by
executable check (LEDGER E17-01), so no owner decision was needed.

### D-129 Makefile defaults bumped to Session 17; X2-c targets (FIXED, before any code change)
`PREFIX ?= s17` and `EVIDENCE_SESSION` default 17 (the D-092/D-097/D-106/D-110/D-117/D-126
pattern), so that no Session 17 command writes into Session 6–16 evidence or reuses the `s16`
prefix by omission. Evidence goes to `reports/session17/` and `logs/session17/` with
`ROBOFORGE_EVIDENCE_SESSION=17`; `OBS_OUT` stays pinned to Session 10 (D-092). The X2-c
targets are added with their code, before first use: `x2c-check` (the engineering
reproduction on X2 validation starts), `x2c-freeze` (write `docs/extensions/X2_FREEZE.md`),
`x2c-plan` (the gate as a dry run), `x2c-run` (the 12 evaluations, once) and `x2c-analyze`
(the analysis, once). The held-out, placement and analysis targets of the frozen studies, X1,
X3, X2-a and X2-b are never run again.

### D-130 X2-c design: the freeze file, the held-out evaluator, the analysis and their limits (FIXED, Session 17, before any X2-c code and before any held-out rollout)
**What runs.** The 12 checkpoints of `reports/session16/x2/suite/x2_test_configuration.json`
(SHA-256 `387ecbe5…`; 4 families × seeds 0–2, each the frozen §7 selection of the family's
§6.3-chosen trial), each once on all 100 `can/x2_heldout` starts (content `40b8228c…`), on the
canonical container, policy seed 0, R = 1, 500-step episodes (D-120), first success ends an
episode, resets only from the stored snapshots, the first 3 episodes re-simulated
(`verify_record`). No random baseline (X2 declares no H1). Preregistration §7 governs; this
decision adds only the mechanics and the clarifications marked below.

**Code, in new files only (KI-064; `make freeze-check` before and after every change).**
- `src/roboforge/x2/heldout.py`: the X2 freeze parser and loader, the 12 jobs, and the
  evaluator core `evaluate_x2c`. It reuses the frozen `LearnedPolicy`, `run_states`,
  `_summarize` and `eval_run_id` and the X2 binding (`x2_bindings`). It does not reuse
  `roboforge.x2.evaluate.evaluate_x2`, which refuses the held-out suite by design and writes
  under `runs/x2/`. That file is unchanged.
- `src/roboforge/x2/heldout_gate.py`: the host gate, the scan for held-out outputs and the
  isolation snapshot.
- `src/roboforge/x2/analysis.py` and `src/roboforge/x2/freeze_doc.py`: the analysis rules and
  tables; the generator of the freeze document.
- Scripts: `scripts/x2c_eval.py` (container), `scripts/x2c_run.py` (host orchestrator),
  `scripts/x2c_check.py` (engineering check), `scripts/write_x2_freeze.py` and
  `scripts/analyze_x2c.py`.
- Tests, synthetic outcomes first (the KI-062 lesson), and
  `tests/research_integrity/test_x2c_scope.py`.

**The X2 freeze (`docs/extensions/X2_FREEZE.md`).**
- Generated by `make x2c-freeze` from committed files, after the code is committed. The
  template is `docs/FINAL_TEST_FREEZE.md`.
- It holds:
  - a verbatim copy of the test configuration (its SHA-256 is checked on parse);
  - the held-out suite's content hash;
  - the rollout rule;
  - the analysis design;
  - the host limits;
  - the output locations;
  - the SHA-256 of every file that runs or analyses the test (the new X2-c files, the reused
    X2 modules, `roboforge.paper.sensitivity`, the X2 suite config and data artifacts, the
    container wrapper and the failure taxonomy);
  - the commit order.
- Its YAML block is authoritative.
- Committed alone, from a clean tree, before any held-out rollout; never edited after.

**Only through the X2 freeze.**
- The evaluator reads the held-out suite's name and content hash, and the 12 (family, seed,
  run, step, SHA-256), from the freeze's block, and refuses any checkpoint it does not list.
- The X2-a/X2-b evaluator still refuses the held-out suite.
- No `src/` or `scripts/` file other than the X2 suite module, its generator and the X2-c
  files names the held-out suite (research-integrity test).

**Freeze attestation inside the container.**
- The container has no `.git`.
- The pinned wrapper attests only `docs/FINAL_TEST_FREEZE.md`, and passes only its own `-e`
  variables.
- So the host passes the X2 freeze's committed git blob ID as a command-line argument. The
  container requires:
  - `ROBOFORGE_GIT_DIRTY=0`;
  - the mounted file's blob ID equal to that argument;
  - every pinned file and the Level 1 frozen code unchanged.
- The checkpoint file is hashed before it is loaded.

**"Refuses if any held-out summary already exists", at two levels.**
- **Host (batch):**
  - The gate searches `reports/`, `logs/` and `runs/` for any file or directory named with the
    held-out evaluation-ID tag or a held-out state ID.
  - It also searches every JSON file under `reports/` and `logs/` for an evaluation summary of
    the held-out suite.
  - At the first launch nothing may be found.
- **Container (job):** that job's own summary and record directory must not exist. The
  container sees only `reports/session17`, and the 12 jobs run in parallel, so the global
  search can only run on the host.

**Outputs.**
- Records: `runs/x2c/<eval_id>/`. A new tree, so `runs/x2` stays byte-identical (isolation).
- Summaries: `reports/session17/x2/heldout/evals/<eval_id>.json`, schema
  `roboforge.x2c_eval/1`, written with exclusive create, `eligible_for_selection: false`.
- Analysis: `reports/session17/x2/` (`results.json`, `tables.md`, `taxonomy.json`,
  `analysis_record.json`), each written once.
- Logs: `logs/session17/x2/heldout/`.
- Isolation before and after the batch:
  - the `runs/` listing;
  - the digests (names and sizes) of `runs/x2`, `runs/x3` and the 26 frozen Level 1 held-out
    record directories;
  - after the batch, only `runs/x2c` may be new, and every held-out output must belong to a
    frozen job.

**Engineering check (not a result), before the freeze.** The X2-c evaluator in check mode
(X2 validation starts only, `runs/x2c_checks/`, `logs/session17/x2c_checks/`) must reproduce
two committed validation summaries exactly. For every one of the 50 starts it must give equal
`physics_sha256`, success, first-success step, steps and stop reason. The two are diffusion
seed 0 at 100k (pilot; noise seeding) and BC-MLP-chunk seed 1 at 5k (X2-b). No held-out start
is stepped by any check or test.

**Launch and limits.**
- From the clean commit that contains the freeze.
- At most 4 containers through the host-wide slots (D-069).
- The D-125 limits:
  - one launch per fresh 60-s host sample;
  - no launch above 3,000 swap pages/s, below 4 GiB free disk or on a dirty tree (a dirty
    tree stops all further launches).
- Diffusion first, then BC-RNN, BC-MLP, BC-MLP-chunk (longest first).

**No automatic rerun.**
- A failed evaluation is recorded and not relaunched.
- The frozen engineering rule allows one rerun, only for a fault outside the policy and
  before the summary is written. That rerun, or resuming an interrupted batch, needs a new
  decision recorded in `docs/DECISIONS.md`, passed as `--decision D-NNN`.
- A failed first attempt reruns once as attempt 2, in `runs/x2c/attempt2/`.
- A job that fails twice is never run again; its family is reported ENGINEERING-INCOMPLETE
  (frozen `docs/FINAL_TEST_FREEZE.md` §4).
- The analysis accepts a summary only if it was written from a clean commit whose X2 freeze
  blob and pinned-file hashes equal the freeze's. A rerun's decision commit only adds records,
  so this rule admits it.

**Analysis: preregistration §7, plus these clarifications** (fixed now, before any held-out
data exist; they go beyond the preregistration's text and are disclosed as such in
`results.json` and the freeze):
1. **Relation to Level 1 Can PH** compares like with like: the frozen bootstrap separation
   against the X2 bootstrap separation.
   - Both separated, same sign: "same direction".
   - Both separated, opposite signs: "reversed".
   - Frozen separated, X2 not: "not reproduced".
   - Frozen not, X2 separated: "new separation".
   - **Neither separated: "neither separated".** The preregistration has no label for this
     case, which is likely for H2 diffusion − BC-RNN and H2-c (both not separated on Can PH).
   - **X2 verdict "conflicting": "no relation"** (the contrast has no verdict; the X3
     precedent).
2. **The Level 1 side** shows the frozen difference, its bootstrap interval and verdict
   (`reports/session7/results/results.json`, read through its pinned SHA-256 `b8b097c3…`).
   Beside them, the D-095 seed-level interval is recomputed from that file's `per_seed`
   values with the same Welch function, as X3 did. These are descriptive, never pooled, on
   the 400-step scale.
3. **No preregistered readings exist for X2.** The tables carry one generated sentence per
   contrast in the preregistered terms only: difference, verdict, relation.
4. **Draw order** is passed explicitly: states, then bc_mlp, bc_rnn, bc_mlp_chunk,
   diffusion_unet, minus any ENGINEERING-INCOMPLETE family. The censored length uses 500 for
   failures.

**Estimates (labelled estimates, in hours from measured times).**
- The 12 evaluations take about **3.6 container-hours**: 13,008 s, from the X2-b per-family
  means per 50 starts (diffusion 642 s, BC-RNN 667 s, BC-MLP-chunk 228 s, BC-MLP 631 s) × 2
  × 3 seeds. That is above the preregistration §9's 2–3 container-hours.
- About **0.9–1.0 h of wall time** at 4 containers, plus launch spacing and contention.
- The engineering check takes about 0.2 h (642 s and 228 s at once).
- Disk about 0.6–0.9 GiB (X2-b records were 10–37 MiB per 50 episodes by family).
- $0.

**Outcome note (D-130, Session 17 STOP; LEDGER E17-02 to E17-07).** The design ran as fixed,
in the fixed commit order: code `a1c3db7`; engineering check PASS from it (both reproductions
exact, 7.0 min); `docs/extensions/X2_FREEZE.md` written once from `a1c3db7`; the check
evidence and ledger rows committed in `45edda2` (the freeze file left out); the freeze
committed alone in `3a64dbb` (SHA-256 `952d232c…`, git blob `9929dc7c…`); `make x2c-run` once from clean
`3a64dbb`, 05:34 to 06:20 UTC 2026-11-16; the summaries and logs committed in `29ce47a` (the
episode records stay in `runs/x2c/`, outside git);
`make x2c-analyze` once from clean `29ce47a`, committed in `3df25ff`
(`reports/session17/x2/results.json`, SHA-256 `30c01f55…`). All 12 evaluations completed
attempt 1 with exit 0: 100 of 100 starts each, 0 engineering errors, `verify_record` audit PASS,
canonical platform, clean tree, the freeze blob attested. No rerun and no rerun decision.
Isolation PASS: the listing digests (path and size of every file) of `runs/x2`, `runs/x3` and
the frozen Level 1 held-out records are equal before and after; `runs/x2c` is the only new top-level entry. Result, X2 (Can MH), beside the
frozen Level 1 Can PH result, never pooled, within-operator only:

| Contrast | X2 difference, bootstrap 95 %, seed-level 95 % | X2 verdict | Relation to Level 1 Can PH |
|---|---|---|---|
| H2 diffusion − BC-MLP | +63.3 pp [+53.0, +73.0], [+51.9, +74.8] | robust (+) | same direction |
| H2 diffusion − BC-RNN | +27.3 pp [+16.7, +37.7], [+14.9, +39.8] | robust (+) | new separation |
| H2-c diffusion − BC-MLP-chunk | +2.0 pp [−6.3, +11.0], [−11.6, +15.6] | not separated | neither separated |
| H3 BC-RNN − BC-MLP | +36.0 pp [+23.7, +48.0], [+22.9, +49.1] | robust (+) | same direction |

Measured against the labelled estimates: the 12 evaluations 2.66 container-hours, launch to
exit (2.61 inside the evaluator; per-family means per 100 starts: diffusion 826 s, BC-RNN
939 s, BC-MLP-chunk 398 s, BC-MLP 972 s), against 3.6; wall time 47.0 min against 0.9–1.0 h;
the engineering check 7.0 min against 0.2 h; `runs/x2c` 529 MB against 0.6–0.9 GiB; the
analysis 1.5 min; $0.

## Session 18 (2026-11-16): v1.1, the post-release extensions in the repository and the Observatory (build only; publication in Session 19)
Level 1, the RGB study, Level 2, X1, X2 and X3 stay frozen with all their evidence. No rollout,
no training, no evaluation, no paid or remote compute. Session 18 pushes, tags, releases and
deploys nothing (D-131).

### D-131 Publish X1, X2 and X3 as v1.1; the Observatory first; build now, publish in Session 19 (OWNER, 2026-11-16, Session 18)
**First ask.** The owner was asked once, with facts measured at the start of Session 18:
- every prerequisite passed:
  - the three extension `results.json` files, `docs/extensions/X2_FREEZE.md` and
    `paper/main.pdf` hash to their recorded values;
  - lint, `make freeze-check`, the full suite alone (705 passed), `tests/research_integrity`
    (81) and `make release-check` all pass;
  - a fresh `git clone --no-local` ran the D-101 CI commands and passed (E18-01);
- a push is 37 commits, a 19 MiB pack and about 1,880 files, mostly Session 13–17 logs and
  reports;
- README, CITATION, `paper/`, `website/` and `.github/` are unchanged since v1.0.0;
- `origin/main` is `abbb9c3`, the owner's README edit after the v1.0.0 record;
- the oldest unpushed commit, `536fa24`, is the D-105 handoff (documents only), made before
  Session 13.

Answers:
- **"publish"**, the answer to the Session 17 STOP question. This is the session's first
  decision.
- Scope **"(b) + README"**: push `main`, an annotated tag `v1.1.0`, a GitHub release with an
  addendum PDF, and a generated README section on the extensions.
- The Observatory: **"Update it too"**.

**Owner instruction during the session (2026-11-16 02:05 EST), which supersedes the "later
session" wording of that option:** "before putting anything on github, we should fully update
the observatory with the new results. I want all the changes we've done past version 1.0 to be
visually and writtenly updated in both the repo and the observatory."

**Second ask**, once, with the facts:
- clips can only be re-rendered from recorded simulator states;
- new denoising traces or candidate paths would call the policy and the simulator, so they are
  excluded;
- at most about 15 s of container time per plain clip (estimate). Session 10's 37-clip export
  took 3,548 s, of which 3,036 s were two denoising clips (E10-05).

Answers:
- **"X2 + X1 clips"**: X2 (Can MH) held-out head-to-heads, and X1 under the Lift arm offset;
- **"Build now, publish in S19"**.

**Consequences.**
- **Session 18 builds and verifies locally, then stops with nothing pushed:**
  - the addendum PDF (D-133);
  - the README section and its chart (D-133);
  - the Observatory extension section and its clips (D-134);
  - the records.
- **Session 19 publishes, in this order:**
  1. push `main`;
  2. wait for green CI on that exact SHA (D-101);
  3. create the annotated tag `v1.1.0` on it;
  4. create the GitHub release with the addendum PDF, then download the PDF and compare its
     SHA-256;
  5. redeploy Pages as a **new commit on top of `c14e041`** on `gh-pages`, never an orphan or
     a force push, so the v1.0.0 site stays in history;
  6. append the v1.1 publication record to `docs/FINAL_RELEASE_AUDIT.md`.
- **Pages redeployment is now inside the owner's chosen scope**, in Session 19 only.
- `CITATION.cff` gets `version: 1.1.0` and its `date-released` in the Session 19 commit that is
  tagged, because the date is the release date. `pyproject.toml` and `website/package.json`
  keep their versions (D-099).
- v1.0.0 stays as published: its tag, its release, `paper/main.pdf` (`c648e753…`) and the
  frozen findings on the site.
- No AI co-author trailer or AI byline anywhere (MASTER_SPEC §0.2, D-099).

**Outcome note (Session 19, 2026-11-16).** Applied in the D-131 order, each step verified
before the next (`docs/FINAL_RELEASE_AUDIT.md` §10, E19-02 to E19-05):
1. `main` pushed as a fast-forward `abbb9c3..907c901`; the release commit `907c901` sets
   `CITATION.cff` to 1.1.0, released 2026-11-16.
2. CI 3/3 pass on `907c901` (python-tests 43832425639, research-integrity 43832425616 with the
   addendum-check step, web-build 43832429173).
3. Annotated tag `v1.1.0` (object `d805cb5`) on `907c901`, tagger Zackary Holt, no trailer.
4. Release v1.1.0 with `addendum.pdf`; the downloaded asset's SHA-256 equals `bedfb277…`. The
   v1.0.0 release is unchanged.
5. Pages redeployed as `gh-pages` commit `d3f47dd` on top of `c14e041` (no orphan, no force
   push), built from the clean tagged commit; `built`; all 263 `deploy.json` files match live;
   live E2E 8/8.
6. `docs/FINAL_RELEASE_AUDIT.md` §10 appended; the README Observatory paragraph updated after the
   live checks.
Nothing outside D-131's scope was published; no AI trailer or byline anywhere.

### D-132 Makefile defaults bumped to Session 18; v1.1 targets (FIXED, before any code change)
- `PREFIX ?= s18` and `EVIDENCE_SESSION` default 18 (the D-092/D-097/D-106/D-110/D-117/D-126/
  D-129 pattern). Evidence goes to `reports/session18/` and `logs/session18/` with
  `ROBOFORGE_EVIDENCE_SESSION=18`.
- `OBS_OUT` stays pinned to `reports/session10/observatory` (D-092). The extension export gets
  its own `OBS_EXT_OUT ?= reports/session18/observatory`, so nothing writes into Session 10
  evidence.
- New targets, each added with its code before first use:
  - `addendum-assets`, `addendum-check` and `addendum` (D-133);
  - `readme-extensions` (D-133);
  - `obs-ext-select`, `obs-ext-export` (canonical container, clean tree) and
    `obs-ext-findings` (D-134);
  - `web-verify-ext` (D-134).
- `web-data` also copies the Session 18 tree, hash-checked (D-134).
- The research-integrity workflow gains `make addendum-check`, beside `make paper-check` (a
  D-101 amendment; no test or check is weakened).
- No eval, analysis, check or freeze target of Level 1, the RGB study, Level 2, X1, X2 or X3 is
  run.

### D-133 v1.1 content: sources, labels and caveats; the addendum and the README section (FIXED, before any code)
**Sources.** Exactly the three hash-pinned extension results. A mismatch stops every build, and
nothing under `runs/` is read:

| Study | File | SHA-256 |
|---|---|---|
| X1 | `reports/session13/x1_clamp/results.json` | `ad50368645b96fb87489310243535f97694b1e47a2ff993915b039b2199a5ee7` |
| X3 | `reports/session14/x3/results.json` | `4299a14930790bfdb2f91d7a3216c9bd7f6db1fa2171469a29bfed3c5c5b71d1` |
| X2 | `reports/session17/x2/results.json` | `30c01f551333feafba70bfa4c9c6a414eda691f179800022d3871776be2df238` |

- Values are copied at their tables' rounding (one decimal, in pp or %).
- Derived values are plain arithmetic and are listed in the generated index, as D-094 did.
- The frozen Level 1 values shown beside X2 and X3 are the ones those `results.json` files
  carry (`frozen_families`, `relation`). The X1 frozen reference values are the ones X1's
  file carries.
- No number in the addendum, the README section or the site's extension section is typed by
  hand (the D-094/D-103 rule).

**Labels and caveats, carried wherever the numbers appear.**
- **X1:** "exploratory, after the freeze".
  - Validation regime: the Session 7 robustness sets, not the held-out test.
  - KI-052: X1 tested the BC side. The diffusion side, in-sampler clipping
    (`clip_sample=False`), stays open (D-107 correction).
- **X3:** "separate replication".
  - The frozen Level 1 result stands as reported; X3 is shown beside it and never pooled.
  - KI-053.
- **X2:** "X2 (Can MH)".
  - Beside the frozen Level 1 Can PH result, never instead of it, never pooled.
  - Within-operator only (D-119).
  - X2 episodes end at 500 steps, Level 1 Can PH at 400, so the success scales differ (D-120).
  - KI-053: X2 does not change the fragility of the frozen Can PH separations.
- **Every document:** simulation only; the addendum is a local build, not peer reviewed;
  "not separated" is never read as equality.

**Addendum PDF.**
- New files only:
  - `paper/addendum/addendum.tex`, whose prose carries every number through generated macros;
  - `paper/addendum/generated/`: `numbers.tex`, tables, figure PDFs with their data sidecars,
    and an `index.json` holding each output's SHA-256 and its sources;
  - `paper/addendum/addendum.pdf`.
- Generator `src/roboforge/addendum/`, script `scripts/addendum_assets.py` (build, check,
  lint). `make addendum-check` proves the committed outputs equal a fresh build and that every
  number in the `.tex` comes from a macro.
- **Toolchain as v1.0.0:** Tectonic 0.17.0, `--only-cached`, and only the packages already in
  `paper/main.tex`'s preamble. The fixed `SOURCE_DATE_EPOCH` is `1794787200` (2026-11-16
  00:00 UTC). The PDF is built twice and the two copies must be byte-identical.
- Title: "RoboForge v1.1 addendum: three follow-up studies (X1, X2, X3)". Author: Zackary Holt
  only. Bibliography: entries reused from `paper/references.bib` only.
- **Unchanged and byte-identical to v1.0.0:** `paper/main.pdf`, `paper/main.tex`,
  `paper/sections/`, `paper/generated/` and `src/roboforge/paper/`. `make paper-check` and the
  PDF hash are rechecked afterwards.

**README.**
- A new generated block, `extensions`, between its own markers. It is written by
  `scripts/readme_extensions.py` from `paper/addendum/generated/numbers.tex` (which
  `make addendum-check` proves equal to a fresh build) and states each study's label and
  caveats.
- A chart, `docs/media/readme_extensions.svg`, is generated deterministically (matplotlib SVG,
  fixed hash salt, no date, text as paths) with a JSON sidecar of the plotted values.
- `make release-check` gains a `readme_extensions` check: the block equals a fresh generation,
  and the chart equals a fresh build.
- The v1.0 `findings` and `setup` blocks stay as generated from `paper/generated/numbers.tex`.
- **Hand-written edits, number-free:**
  - the Limitations line on KI-052, which still says the evaluation-only test was not run,
    now says X1 tested the BC side and the diffusion side stays open;
  - a link to the addendum;
  - the "Extensions" heading covers X1–X3.
- The Observatory line ("built from commit `b4bd070`") is updated in Session 19, when the new
  site is deployed and checked (D-102).

### D-134 The Observatory's extension section and its clips (FIXED, before any code and before any export)
**Section.** "After v1.0: three follow-up studies", placed after the frozen findings (F).
- It holds generated charts and text for X1, X3 and X2, with the D-133 labels and caveats.
  Every number comes from `findings_ext.json`, which copies blocks of the three pinned
  `results.json` files, cites their SHA-256 and computes nothing.
- The v1.0 sections, their data (`reports/session10/observatory/`) and their wording about the
  frozen results are unchanged.
- The site's tests and schemas grow with it: zod schemas, vitest and E2E assertions for the new
  section, inside the D-088 budgets (first view ≤ 6.5 MB, all data ≤ 80 MB).

**Clips: re-rendered from recorded states only.**
- Each clip's body poses are recomputed from the recorded MuJoCo states, as Session 10's
  held-out clips were.
- No policy call, no clone simulation, no `env.step`. There are no denoising traces, branches or
  candidate paths for the extensions.
- The export runs on the canonical container from a clean tree, as one process, never alongside
  the full suite (KI-043).
- It only reads `runs/`; the listing digests (path and size) of every record directory it reads
  are equal before and after.

**Selection rules, fixed now and applied by `make obs-ext-select` (host).** It reads committed
summaries only, and suite names come from those summaries.
- **X2 (Can MH, the 12 X2-c held-out summaries listed in X2's `results.json`):**
  - seed 0 diffusion against seed 0 of each BC family (BC-MLP, BC-RNN, BC-MLP-chunk), on the
    same start, in the summary's state order;
  - (a) the first start where only diffusion succeeds;
  - (b) the first start where only the BC family succeeds;
  - (c) the first start where both succeed.
  Up to 9 pairs; a diffusion clip is shared when two pairs use the same start.
- **X1 (Lift `ood_arm_offset`, seed 0):**
  - three arms on the same start:
    - diffusion (its Session 7 robustness record; the sampler already bounds its actions);
    - BC-MLP-chunk unclamped (Session 7 robustness record);
    - BC-MLP-chunk clamped (X1 record);
  - (a) the first start where diffusion succeeds and both BC-MLP-chunk arms fail;
  - (b) the first start where all three succeed;
  - (c) the first start where the clamped and unclamped BC-MLP-chunk outcomes differ.
- Session 7 summaries are read only after their SHA-256 equals the value pinned in X1's
  `results.json` inputs or in the Session 7 analysis it pins.
- **Missing cases** are recorded as `null` with a reason, never dropped. **Representativeness:**
  the clips are rule-selected cases, labelled as such, not representative samples; the outcome
  strips show all starts for the seed-0 checkpoints.

**Labels on clips.**
- The X2 records' recorded policy label reads "trained on robomimic PH demonstrations". That
  string comes from the frozen evaluator's label template, which the X2 binding does not change
  (KI-065).
- The export keeps the recorded string in the clip's provenance. The site shows the study label:
  "X2 (Can MH): trained on the robomimic Can MH demonstrations".

**Outputs.** `reports/session18/observatory/`:
- `selection.json`, `clips/`, `media/`, `verification/`, `meshes.json`, `index.json` (every
  file with its SHA-256), `findings_ext.json` and `viewer_verification.json`.
- `make web-data` copies it, hash-checked, into `website/public/data/ext/` and writes
  `ext_registry.json`.

**Verification (MASTER_SPEC §8.3).**
- The first X2 (a) diffusion clip and the X1 (a) diffusion clip carry MuJoCo reference frames.
- `make web-verify-ext` checks them against the D-091 thresholds: robot IoU ≥ 0.95, object IoU
  ≥ 0.85, centroid ≤ 1.5 px. It writes to the Session 18 tree, never to Session 10's.

**Integrity.**
- No new file in `src/` or `scripts/` names the X2 held-out suite (D-130 test), the X3 suite or
  the locked Level 1 suite.
- The X2-c output gate (D-130) would now find these files, because they are named by held-out
  state ID. This is consistent: `make x2c-run` is never run again.

**Outcome note (Session 18 STOP, 2026-11-16).** Applied as fixed; no rule changed after the
selection.
- **Selection** (`make obs-ext-select`, host, from clean `2fdf21d`; committed `7eb5817` before any
  export): 23 clips, 12 cases, none missing. X2: 9 pairs (diffusion shared across pairs on
  `0000` and `0008`); X1: 3 starts × 3 arms. Verification clips: the X2 (a) and X1 (a)
  diffusion clips.
- **Export** (`make obs-ext-export`, canonical container, clean `7eb5817`; committed `f26e4e6`,
  videos `a8b5a74`): 23/23 clips, recomputed body poses equal the records (0.0 m), 350.5 s of
  container time (the D-131 estimate was about 15 s per clip; measured 15.2 s on average,
  29.7 s at most for 501-frame episodes). Isolation PASS: the 7 record directories read are
  unchanged. Every mesh equals Session 10's. The `.mp4` files are force-added, as Session 10's
  were, because `*.mp4` is ignored.
- **Findings** (`findings_ext.json`): copied from the three pinned results at `7eb5817`.
- **Site** (`fb1f6be`): the section is titled "After v1.0.0: three follow-up studies" (nav
  "After v1.0"). Its prose is gated on the recorded verdicts: a sentence that a changed verdict
  would falsify is not shown. Counting verdicts or adding taxonomy counts is the only arithmetic.
  `make web-verify-ext`: 12 reference frames, robot IoU 0.9998–1.0000, object IoU 1.0000,
  centroid ≤ 0.03 px, pass. Budgets: all data 29.3 MB of 80 MB, first view 3.95 MB of 6.5 MB.
- **Labels:** KI-065 recorded; the X2 display label is checked by a site test.
- `docs/VISUAL_PROVENANCE.md` gained the extension section (`f675490`); the Session 10 rows are
  unchanged.
- Nothing was pushed, tagged, released or deployed.

## Session 19 (2026-11-16): v1.1 publication
Level 1, the RGB study, Level 2, X1, X2 and X3 stay frozen with all their evidence, and so does
the Session 18 build (the addendum, the README extensions block and chart, the Observatory
extension export and the site code). No rollout, no training, no evaluation, no paid or remote
compute. Session 19 publishes only what D-131 authorizes, in its order.

### D-135 Makefile defaults bumped to Session 19 (FIXED, before any other change)
- `PREFIX ?= s19` and `EVIDENCE_SESSION` default 19 (the D-092/D-097/D-106/D-110/D-117/D-126/
  D-129/D-132 pattern). Evidence goes to `reports/session19/` and `logs/session19/` with
  `ROBOFORGE_EVIDENCE_SESSION=19`.
- `OBS_OUT` stays `reports/session10/observatory` and `OBS_EXT_OUT` stays
  `reports/session18/observatory`: `make web-data` only reads them, and nothing writes into
  Session 10 or Session 18 evidence. `PAGES_DIR` stays the gitignored `logs/tmp/pages`.
- Nothing else changes in code: no new target, no change to `src/`, `scripts/`, `tests/`,
  `website/`, `paper/`, `.github/`, `uv.lock` or any data.
- Targets Session 19 runs: the checks (`lint`, `test`, `freeze-check`, `addendum-check`,
  `paper-check`, `release-check`, `scripts/release_check.py --network`), the site build
  (`web-data`, `web-assets`, `web-build`, `web-test`, `web-e2e`) and `pages-bundle`. Never run:
  any eval, analysis, check or freeze target of Level 1, the RGB study, Level 2, X1, X2 or X3;
  `obs-ext-select`, `obs-ext-export`, `obs-ext-findings`, `web-verify-ext`; `addendum`,
  `addendum-assets`, `readme-extensions`, `paper`, `paper-assets` (the committed outputs stand).
- No test pins the Makefile defaults (checked by grep before the change).

## Session 20 (2026-11-16): Observatory visual quality pass
Level 1, the RGB study, Level 2, X1, X2 and X3 stay frozen with all their evidence, and so do
the Observatory exports (`reports/session10/observatory/`, `reports/session18/observatory/`),
the generated `website/public/data`, `paper/`, the README's generated blocks, preview and chart,
and `CITATION.cff`. No rollout, no policy call, no `env.step`, no training, no evaluation, no
paid or remote compute. Session 20 changes how the site presents the recorded evidence, and
nothing else.

### D-136 A full visual quality pass on the Observatory (OWNER, 2026-11-16)
**The owner's request, verbatim:** "a full visual quality improvement pass on the observatory,
making sure things are best visually represented, rendered, that there isnt too much text strewn
across the observatory, that everything makes sense".

- **Authorizes:** changes to `website/` source, styles and site tests, for presentation only, in
  this session.
- **Supersedes**, for `website/` code only, Session 19's "the site code is frozen".
- **Authorizes nothing else.** No new result, no new clip, no re-export, no change to any number
  or verdict. In particular it does **not** authorize publication: the push of `main`, the Pages
  redeploy and any README or audit record of a new build need a separate owner answer at the end
  of the session (Session 20 prompt, step 6).
- **Rules the pass keeps** (Session 20 prompt, step 4):
  - recorded physical state, the policy's proposals and illustrative material stay visually
    distinct and labelled (MASTER_SPEC §0.4, §8.1, §8.3 to §8.5);
  - every clip keeps a reachable source (run id, checkpoint hash, start, outcome), which may move
    into a collapsible "Provenance" element but is never deleted; selection labels stay with
    their clips;
  - every number still comes from the generated data; no number is typed into new TSX or CSS
    text; no arithmetic beyond what exists (D-134: counting verdicts only); the verdict gating
    stays;
  - the caveats listed in the Session 20 prompt stay visible by default wherever their numbers
    are shown; `docs/OBSERVATORY_VISUAL_AUDIT.md` lists each caveat's location before and after;
  - kinematics, transforms, meshes, frame conversion and the MuJoCo camera intrinsics used for
    verification do not change what pose is shown; if anything under `website/src/three/` or
    `website/src/verify/` changes, the viewer is re-verified with new spec files that write to
    `reports/session20/observatory/`, with the D-091 thresholds unchanged;
  - no new npm dependency, font, texture, environment map or other third-party asset without the
    owner's approval; `website/package-lock.json` does not change without it; the D-088 budgets
    hold and are never loosened;
  - tests are never weakened (D-101); a test that pins moved text or a moved selector is changed
    to check the same fact on the new element, and each such change is recorded with its reason.
- **Asked once, before implementing, only for:** removing a section, reordering the narrative,
  changing the visual identity of MASTER_SPEC §8.1, or adding any third-party asset.

### D-137 Makefile defaults bumped to Session 20; site audit and verification targets (FIXED, before any code change)
- `PREFIX ?= s20` and `EVIDENCE_SESSION` default 20 (the D-092/D-097/D-106/D-110/D-117/D-126/
  D-129/D-132/D-135 pattern). Evidence goes to `reports/session20/` and `logs/session20/` with
  `ROBOFORGE_EVIDENCE_SESSION=20`.
- `OBS_OUT` stays `reports/session10/observatory` and `OBS_EXT_OUT` stays
  `reports/session18/observatory`: `make web-data` only reads them. Nothing writes into Session
  10 or Session 18 evidence.
- New site targets, each added with its spec file before first use and recorded here:
  - `web-audit` (`website/e2e/audit-s20.spec.ts`): screenshots of every section of the built
    site and DOM word counts, into `logs/session20/visual_audit/$(AUDIT_TAG)/`
    (`AUDIT_TAG` = `before` or `after`). It reads the built site only;
  - `web-verify-s20` and `web-verify-ext-s20` (`website/e2e/verify-s20.spec.ts`,
    `website/e2e/verify-ext-s20.spec.ts`): the checks of `web-verify` and `web-verify-ext`
    with the D-091 thresholds, writing to `reports/session20/observatory/` and
    `reports/session20/observatory/ext/`. Added only if step 4 changes `website/src/three/` or
    `website/src/verify/`.
- Never run in Session 20: `make web-verify`, `make web-verify-ext` (they would overwrite the
  Session 10 and Session 18 `viewer_verification.json` and montages); any eval, analysis, check
  or freeze target of Level 1, the RGB study, Level 2, X1, X2 or X3; `obs-ext-select`,
  `obs-ext-export`, `obs-ext-findings`, `observatory-export`, `observatory-findings`;
  `addendum`, `addendum-assets`, `readme-extensions`, `paper`, `paper-assets`.
- Nothing else changes in the Makefile. No test pins the Makefile defaults (checked by grep
  before the change, as in D-135).


**Outcome note (D-136, Session 20 STOP, 2026-11-16; LEDGER E20-02 to E20-05).** Applied within
the rules above; no rule changed after it was fixed.
- **Not asked before implementing:** no section was removed, the narrative order and the §8.1
  identity are unchanged, and no dependency, font, texture, environment map or other
  third-party asset was added (`website/package-lock.json` unchanged).
- **Done:** `docs/OBSERVATORY_VISUAL_AUDIT.md` Parts 1 to 4 (baseline committed at `586ef34`
  before any site change; plan; changes; after measurements). Reading words visible by default
  4,291 → 3,471 and page height 21,477 → 18,637 px at 1440×900; the hero now states three
  results read from `findings.json`. Every caveat of the Session 20 prompt is visible by default
  where its numbers are shown (audit §3.6, pinned by a new E2E test); KI-052 is now also shown in
  Findings and the hero, and the X2 (Can MH) label is visible by default.
- **Tests:** no existing assertion changed or removed; 2 E2E tests, 4 vitest tests and 1 pytest
  test added (audit §3.5). D-088 budgets unchanged and met.
- **3D:** `Viewer.tsx`, `robotScene.ts` (colours and material parameters) and `cameras.ts`
  (viewer presets) changed; kinematics, transforms, meshes, `mujocoCameraPose` and `src/verify/`
  did not. Both verifications were run again into `reports/session20/observatory/`: 18/18 and
  12/12 frames pass, each frame record equal to the Session 10 or Session 18 record.
- **One label was clarified:** X2 captions read "X2 held-out start" in place of "ad hoc
  evaluation start"; Provenance keeps the recorded partition `eval_adhoc` beside it (D-123).
- **Publication:** not done at STOP. The owner was asked once (Session 20 prompt, step 6) and
  answered yes: D-138.

**Outcome note (D-137, Session 20 STOP).** The three targets were added as recorded
(`web-audit`, `web-verify-s20`, `web-verify-ext-s20`) and `make web-verify`,
`make web-verify-ext` were not run. One Makefile change beyond the list above:
`visual-provenance` passes `--reverify-dir $(OBS_REVERIFY)` (`reports/session20/observatory`) so
that the generated `docs/VISUAL_PROVENANCE.md` gains a section for the Session 20
re-verification; `scripts/visual_provenance.py` appends it after the existing sections, which
stay byte-identical (`tests/unit/test_visual_provenance.py`). `scripts/visual_audit_montage.py`
(before/after montages from the audit screenshots) is new.

### D-138 Publish the Session 20 Observatory build (OWNER, 2026-11-16)
**The owner's answer, verbatim**, to the one question asked at the Session 20 STOP ("Publish
this site build?", with the four steps and the facts of that report): "yes please publish
everything and and make all of the updates you mentioned to main".

- **Authorizes**, in this order, each step verified before the next (HANDOFF # Session 20
  Handoff §5, as Session 19 E19-04):
  1. push `main` as a fast-forward and wait for CI 3/3 on the pushed head;
  2. build the Pages bundle from the pushed commit and add one new `gh-pages` commit on top of
     `d3f47dd`, pushed as a fast-forward, never an orphan commit or a force-push;
  3. verify the live site by `scripts/pages_bundle.py verify` and by the E2E suite against the
     live URL;
  4. then update the build commit in the README Observatory paragraph, append an "Observatory
     redeploy" section to `docs/FINAL_RELEASE_AUDIT.md`, and push those records to `main`.
- **Authorizes nothing else.** No new tag, no new release, no edit of either existing release,
  no change to any frozen file, result, export, number or verdict. `CITATION.cff` stays at 1.1.0.
- **Known when the owner answered** (stated in the STOP report): three Session 20 logs in the
  local commits `88707cd` and `586ef34` contain the absolute home directory; pushing `main`
  publishes those blobs, since history is never rewritten (D-100).

**Outcome note (D-138, 2026-11-16; LEDGER E20-06 to E20-08).** Applied in the order above.
`main` `92c8cc4..57d8b7b`; CI 3/3 on `57d8b7b`; bundle of 263 files from `57d8b7b`
(`deploy.json` `38a02eac…`); `gh-pages` `d3f47dd..b4467f7`; Pages `built` in 31 s; all 263 live
files match on the second attempt (the first met one HTTP 503 from the CDN); live E2E 10/10;
then the records. Measured: 5.7 minutes from the push of `main` to the end of the live E2E
(estimate given: about 0.1 h). Tags and releases unchanged. Cost: $0.
