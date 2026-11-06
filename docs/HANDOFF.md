<!-- parsable handoff header (values below are real, filled at the end of Session 1) -->
```yaml
project: roboforge-3d
current_session: 1
session_status: PASS
next_session: 2
last_verified_commit: c69551fc02f00841a05cbe2352539f1c6aca400c   # evidence regenerated from this clean tree
python: 3.11.15
sim_engine_version: mujoco 3.2.7
robosuite_revision: 51cc01785bab80ffeed20da15e67d7dd4140e76a   # v1.5.1
robomimic_revision: d309eaecc18acf4152a830a895a6984b8ac71b05   # v0.5.0+15
dataset_source: hf://robomimic/robomimic_datasets@74fa018461f479cd9fd15b924a16103012096203
dataset_sha256:
  lift_low_dim: 2067777cb8b532e9263dd09fd6448c41cc31224bb27be4a3b734010ae13eb540
  lift_raw: 68f2a8cfe2502de1224f0708fb82e46da07abce108c63e721d72fc1618272f37
  can_low_dim: 3f2eb92e0a5025d0095e866ac16cc8092d6a762abe27dec90dbaff9027282962
  can_raw: 86961df85af1b9c6b9d4182a3755a8a4db6d0660cd550e45cca1f7accdb6d73d
converter_version: lowdim_v1
processed_content_sha256:
  lift: 74e70119993830798c8b4ed87f202f64d8d2c9d573c0f376d54c4130e52d63c8
  can: e6055a6f066f53b987ed27a7b5894f20ec7053fc9cdf3222b4ad79ed3834ea7b
split_sha256:
  lift: 73097265526c03dfb7535aa9bff50aadd285cbfb554e563636d66395909c331e
  can: 743779b0ad9cacd5bab85a74421f6fba4ace5e980f5979a4a3534c82f042b46f
normalization_sha256:
  lift: 253bfc7e63570bf3e8f8646a0094d1da4c5b729ba8f463b4955f01410b4812d6
  can: e0437cd7b8e48403366d19f7b0e6116e2224723b8a5240b2d7ba008403b3228d
observation_spec_sha256:
  lift: dfbd868e0ec11da76f66a71a9c3468e58590728ca95e93bbfb0a50ce182a7721
  can: 71e89629bc3447b02b4bb7ec2f470e4a72f4d44ec04a74f90b6da146f425b71c
action_spec_sha256: bcae0b7110c56e1bfdf8e2754c03ff95c8c9e77a34c56ef9ebae11e60d7fa77f   # identical for both tasks
task_manifest_sha256:
  lift: 2ac6070d18cc53f565ec11e67e01f6b5a91de7e268df6a491590b0e45f0528ac
  can: d8f85ecea43c0e616162079757598af6b06c0edd8f03a3c563c08b06dea47162
obs_dim: 19              # both tasks; privileged (exact object state)
action_dim: 7            # OSC_POSE delta (3 pos + 3 axis-angle) + gripper, bounds [-1, 1], 20 Hz
training_spend_usd: 0
next_required_owner_approval: none for Session 2
license: Apache-2.0
```

# Session 1 Handoff

Do not trust this file blindly: re-run the checks in "Exact starting instructions".

## 1. Session completed
Session 1 (canonical datasets, episode alignment, train/val splits): **PASS**. Session 2
can start.

## 2. Completed work
- **Specs** (`src/roboforge/data/specs.py`):
  - `TaskSpec`, `ActionSpec` and `ObservationSpec` are frozen dataclasses with canonical
    SHA-256 hashes.
  - They are built from `configs/data/<task>.yaml` (the feature selection) and validated
    against the HDF5 file: obs block widths, MuJoCo names present in the model XML, and
    actions/controller read from `env_args`.
  - They are also checked against the pinned simulator (`env.action_spec`, `control_freq`,
    `model_timestep`).
  - Later sessions load them, hash-checked, with `roboforge.data.pipeline.load_specs(task)`.
- **The one state→feature function** (`src/roboforge/data/features.py`, D-010):
  - It is a pure function of the flattened MuJoCo state: scratch `MjData` + `mj_kinematics`.
  - It equals the robomimic obs exactly on all 400 demos for every Lift block and Can's
    robot and world blocks.
  - It recomputes Can's relative position from same-step poses, in float64 (KI-003, KI-018),
    and drops Can's discontinuous relative quaternion.
  - Closed-loop tests show that it equals the restored-state value bitwise after every
    `env.step`, and that computing it does not perturb the rollout.
- **Processed data** (`processed.py`, `pipeline.py`, D-021):
  - float32 per-demo arrays in `$ROBOFORGE_DATA_ROOT/processed/…` (gitignored), with a
    content hash.
  - Committed, byte-deterministic, self-hashed artifacts: the task manifest, the split
    manifest and the normalization stats.
  - `make check-data` rebuilds everything and fails on any difference. It passes on macOS
    arm64 and in the Linux aarch64 image.
- **Split** (D-019): the upstream masks, 180 train / 20 valid per task, with identical IDs
  for Lift and Can.
- **Normalization** (D-020): train-only min-max to [−1, 1] (Diffusion Policy `limits`
  semantics) for observations and actions. `ActionCodec` maps model output to clipped env
  actions.
- **Window sampler** (`sequence.py`, D-022): `SequenceDataset.from_committed(task, split)`
  verifies the processed hash, refuses flagged episodes and normalizes. The Session 0 raw-obs
  loader was removed.
- **Sanity, audit and explorer:**
  - `sanity.py` flags (never drops) bad episodes; none were flagged.
  - `audit.py` and `scripts/audit_data.py` write `reports/data_audit.json`, 3 figures per
    task and the generated `docs/DATA_DICTIONARY.md`.
- **Fetcher:** HTTP Range resume of `.part` files; a hash failure deletes the file and raises.
- **Make targets:**
  - new: `build-data`, `check-data`, `audit-data`;
  - `verify-data` now also runs `check-data`;
  - `VERIFY_OUT` and `SMOKE_OUT` keep Session 0 evidence intact (D-023).

## 3. Files created or changed
- **New:**
  - `configs/data/{lift,can}.yaml`;
  - `data/manifests/tasks/{lift,can}_ph.json`, `data/splits/{lift,can}_ph.json`,
    `data/normalization/{lift,can}_ph_lowdim_v1.json`;
  - `src/roboforge/data/{specs,features,processed,pipeline,split,normalization,sequence,sanity,audit,explore,dictionary}.py`;
  - `scripts/{build_data,audit_data}.py`;
  - tests: `tests/unit/{test_normalization,test_split,test_sequence,test_fetch_resume}.py`,
    `tests/integration/test_data_pipeline.py`,
    `tests/simulator/test_features_closed_loop.py`;
  - `docs/DATA_DICTIONARY.md` (generated), `reports/data_audit.json`,
    `reports/data_audit/*.png`, `logs/session1/`.
- **Changed:**
  - `src/roboforge/data/{fetch,windows}.py`, `src/roboforge/provenance.py` (canonical JSON
    helpers);
  - tests `tests/unit/{test_windows,test_forward_pass}.py` and
    `tests/integration/test_datasets.py`;
  - `Makefile`, `README.md`;
  - `data/manifests/robomimic_v1.5_ph.json` (`acquired_date`);
  - docs: DECISIONS, KNOWN_ISSUES, EXPERIMENT_LEDGER, RESEARCH_PROTOCOL, DATA_PROVENANCE,
    SIMULATOR_CONTRACT.

## 4. Data contract (Session 1 STOP list)
- **Observation (19-D float32, normalized, slices in spec order):**

  | Task | 0–2 | 3–6 | 7–8 | 9–11 | 12–15 | 16–18 |
  |---|---|---|---|---|---|---|
  | Lift | `robot0_eef_pos` | `robot0_eef_quat` (xyzw) | `robot0_gripper_qpos` | `cube_pos` | `cube_quat` (xyzw) | `gripper_to_cube_pos` (world) |
  | Can | `robot0_eef_pos` | `robot0_eef_quat` (xyzw) | `robot0_gripper_qpos` | `Can_to_robot0_eef_pos` (eef frame, corrected) | `Can_pos` (12–14) | `Can_quat` (15–18, xyzw) |

  Can's blocks are 9–11, 12–14 and 15–18. Exact slices are in the task manifests.
- **Action (7-D float32):** `dpos_x, dpos_y, dpos_z` (±0.05 m at ±1), `drot_x, drot_y,
  drot_z` (±0.5 rad axis-angle at ±1), `gripper` (−1 open, +1 close); bounds [−1, 1].
- **Window tensors** (`collate`), with defaults T_o = 2 and T_p = 16 (per-run parameters):

  | Tensor | Shape | Type |
  |---|---|---|
  | `obs_history` | [B, T_o, 19] | float32 |
  | `obs_valid` | [B, T_o] | bool |
  | `future_action` | [B, T_p, 7] | float32 |
  | `valid_mask` | [B, T_p] | bool |
  | `start_t` | [B] | int |
  | `obs_index` | [B, T_o] | int |
  | `action_index` | [B, T_p] | int |
  | `episode_id` | list[str] | — |

- **Training IDs (both tasks):** `demo_0` … `demo_199` except the 20 validation IDs
  `demo_0, 5, 14, 21, 27, 32, 36, 44, 56, 93, 103, 105, 111, 125, 131, 155, 168, 176, 180,
  191`. The full lists are in `data/splits/`.
  - Lift: 8,640 train / 1,026 valid samples.
  - Can: 20,883 train / 2,324 valid samples.
- **Hashes:** see the header. Hash sources: `data/manifests/tasks/*.json` and
  `docs/DATA_DICTIONARY.md`.

## 5. Executed test commands (final run from clean `c69551f`, `logs/session1/`)
| Command | Result |
|---|---|
| `make test` | **123 passed** (66 s): unit 68, simulator 18, integration 37, including the slow Lift rebuild check |
| `make lint` | clean |
| `make smoke-sim SMOKE_OUT=logs/session1/smoke_sim` | Lift 11/11, Can 10/10 PASS |
| `make verify-data VERIFY_OUT=logs/session1/verify_data` | 12/12 integrity checks PASS; replay Lift 199/200, Can 190/200 (unchanged from Session 0); `check-data` PASS for both tasks |
| `make audit-data` | written; 0 flagged episodes; 400/400 final states succeed; `DATA_DICTIONARY.md` regenerated unchanged |
| Linux aarch64 container `build_data.py --check` (read-only mounts) | **PASS** for Lift and Can: bit-identical processed content, split, normalization, manifests |

All JSON evidence is stamped `c69551f`, `dirty: false`. Prerequisite runs at `833277d`:
setup OK, 4/4 hashes, 56/56 tests, smoke 21/21 (`logs/session1/prereq/`).

## 6. Key findings
- **KI-003 resolved.** The stored Can relative block equals robosuite's one-update-lag
  formula to 2.2e-16. It is off by up to 0.861 m at t = 0 and 0.091 m at t > 0. The Can
  input is therefore not robomimic's default 23-D vector.
- **KI-015.** robosuite `env.step` observations are one physics substep stale: up to
  5.8e-4 m on eef position and 7.3e-3 on `Can_quat`. Policies must only see
  `FeatureExtractor.from_sim` outputs.
- **KI-018.** robosuite's `quat2mat` is float32 and platform-dependent. It broke Can
  cross-platform reproducibility until the feature moved to float64.
- **KI-014, KI-016.** Default robosuite models carry two non-colliding mocap markers that
  the dataset XMLs lack, and a `get_xml()` round trip changes the model at about 1e-7.
  Both matter for Session 2's serialized evaluation states.
- **Data facts:**
  - every quaternion input is sign-continuous within episodes;
  - no operator identity exists (single teleop channel);
  - success persists to the end of every demo (4–5 steps after first success);
  - Lift demos are short, so 31 % of Lift training windows have padded actions at T_p = 16.

## 7. Decisions
D-010 (now FIXED), D-016 (layout extended), D-018 (observation spec), D-019 (split),
D-020 (normalization, DEFAULT), D-021 (processed/committed artifacts), D-022 (windows),
D-023 (evidence location).

## 8. Outstanding issues (none blocks Session 2)
- **KI-014, KI-016:** Session 2 must choose the evaluation model and store exact XML
  strings.
- **KI-013 / D-017:** choose the evaluation platform in Session 2.
- **KI-007:** partially verified. A file with a training step was not blocked outside the
  repo; re-check inside it in Session 3.
- **KI-001:** linux/x86_64 and EGL unverified (Session 5/8). Until x86_64 reproduces
  `check-data`, copy the hash-verified processed files to remote storage instead of
  rebuilding them there (KI-018).
- **KI-008:** 33 GB free.

## 9. Budget
Spent $0.00, no remote compute. No approval is needed for Session 2.

## 10. Exact starting instructions for Session 2
Paste into a fresh Claude Code session:

```text
Implement RoboForge 3D SESSION 2 ONLY from MASTER_SPEC.md. Begin by reading all of
MASTER_SPEC.md, docs/STATUS.md, docs/HANDOFF.md, docs/DECISIONS.md, docs/KNOWN_ISSUES.md,
docs/EXPERIMENT_LEDGER.md, docs/RESEARCH_PROTOCOL.md, docs/SIMULATOR_CONTRACT.md,
docs/DATA_DICTIONARY.md, docs/DATA_PROVENANCE.md and reports/feasibility/STACK_REPORT.md,
and inspect the actual repository/test state. Do not blindly trust prior handoff claims:
first run `make setup`, `make download`, `make build-data`, `make check-data`, `make test`
and `make smoke-sim SMOKE_OUT=logs/session2/prereq` and confirm they pass; run
git log --format=fuller -n 5 to confirm the verified commits and identity
Zackary Holt <zackholt05@gmail.com>.

Session 2 goal: the research measurement apparatus (MASTER_SPEC Session 2): a RoboForgeEnv
adapter per spec §4.1 (reset from an exact serialized initial snapshot, action stepping with
clipping to the ActionSpec bounds, reward/success/truncation, camera rendering, robot and
object world transforms); demonstration replay comparison on >= 10 appropriately sampled
demos per task with a divergence summary and frame inspection; an evaluation initial-state
generator with separate validation and final-test suites (RESEARCH_PROTOCOL §5 seeds), the
final-test states saved and hashed but blocked from every script until
docs/FINAL_TEST_FREEZE.md exists; random and no-op policies plus demo-replay instrumentation
with clear non-autonomous labels; a closed-loop rollout recorder with frame-indexed
kinematics; and one exported real simulation clip with a verified MuJoCo -> three.js
coordinate conversion. Decide and record the evaluation platform (D-017) and the
success/termination rule (first success vs persistence). Acceptance: two fresh processes
replaying the same controlled seed match within declared tolerances; the environment accepts
a valid dynamic-dimension action; a random policy produces correctly recorded closed-loop
outcomes; test metadata identifies train/validation/test partitions; `make smoke-sim` and
simulator tests pass.

Keep the Session 1 data contracts (DECISIONS D-010, D-018..D-022): policy observations come
only from roboforge.data.features.FeatureExtractor.from_sim after every step, never from
robosuite env.step observations (KI-015); load specs with roboforge.data.pipeline.load_specs
and decode policy outputs with roboforge.data.normalization.ActionCodec; never use robosuite
transform helpers for precision-critical inputs or metrics (KI-018); serialized initial
states must store the exact compiled XML string (KI-016); resolve KI-014 (robosuite default
models add two non-colliding mocap markers absent from the dataset XMLs: choose which model
the evaluation states use, verify dynamics against the dataset model, and check renders).
Do not train policies, do not change committed data artifacts without a recorded decision,
no paid compute, no remote publication.

Use the current verified simulator/data contracts. Run relevant tests and record commands,
real results and provenance. Never leak final-test states into model selection; never invent
results. Use Zackary Holt as the sole named RoboForge project author with the verified local
Git identity; no AI co-author trailers, while preserving external scientific and license
credit. At the SESSION 2 STOP boundary, update docs/STATUS.md, docs/HANDOFF.md,
docs/DECISIONS.md, docs/KNOWN_ISSUES.md and docs/EXPERIMENT_LEDGER.md, commit verified code
only, inspect commit metadata (git log --format=fuller -n 5), write the exact prompt for
SESSION 3, and STOP. Do NOT proceed automatically.
```
