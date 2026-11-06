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

### D-010 Observation features for Can (DEFAULT for Session 1)
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

### D-013 No LICENSE chosen (OWNER DECISION PENDING)
The spec makes the RoboForge code license an owner choice. Third-party licenses are
recorded in `THIRD_PARTY_NOTICES.md`.

### D-014 Session 0 model check is forward-only (FIXED for Session 0)
A plugin hook (ml-pipeline) blocked code containing a backward pass. Session 0 only
requires a one-batch forward pass (§2.3 item 6). The owner has since set
`ML_PIPELINE_ENFORCE=0` globally and disabled the cto-toolkit plugin; both take effect
after a Claude Code restart.

### D-015 Compute split: GPU for training only; rollouts on CPU (DEFAULT)
State-based Level 1 evaluation needs no rendering, and MuJoCo stepping is CPU-bound
(~160–200 control steps/s/core measured). Training goes on a single Modal GPU only if local
benchmarks show it is needed. See `docs/COMPUTE_PLAN.md`.

### D-016 Repository layout (FIXED)
The spec's functional boundaries are kept. Packages (`models`, `train`, `eval`, `viz`,
`compute`) exist only as documented boundaries until their session. CLIs live in `scripts/`,
and Make targets exist only where a working command exists (spec §6.5).
