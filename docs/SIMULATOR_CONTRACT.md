# Simulator Contract (Sessions 0–2)

Facts below were read from the pinned simulator and dataset metadata, not assumed.
Configs: `configs/simulator/{lift,can}.yaml`. Evidence:
`reports/feasibility/smoke_sim_{lift,can}.json` and `data_report_{lift,can}.json`.

## Stack
Python 3.11.15 · mujoco 3.2.7 · robosuite 1.5.1 (`51cc017`) · robomimic `d309eae`.
Backend: `MUJOCO_GL=cgl` on macOS (verified), `osmesa` on Linux aarch64 (verified in Docker),
`egl` unverified (KI-001). Physics is platform-dependent at floating-point level (KI-013).

## Shared (Lift, PickPlaceCan)
| Property | Value |
|---|---|
| Robot | Panda, 7 arm joints `robot0_joint1..7` + 2 finger joints; `nu = 9` |
| Controller | BASIC composite, right arm `OSC_POSE`, `control_delta=true`, `input_ref_frame=world`, kp=150, damping=1, `uncouple_pos_ori=true`, gripper `GRIP` |
| Action | 7-D, bounds [−1, 1]; maps to Δpos ±0.05 m, Δaxis-angle ±0.5 rad, gripper |
| Timing | control 20 Hz (0.05 s), physics dt 0.002 s, 25 substeps/control step |
| State vector | flattened `[time, qpos(nq), qvel(nv)]`; free-joint qpos `[x,y,z,qw,qx,qy,qz]` |
| Observation quaternions | robosuite obs use **xyzw**; MuJoCo qpos uses **wxyz** |
| Reward | sparse (`reward_shaping=false`): 1.0 on success, else 0 |
| Termination | `done` only at `horizon` when `ignore_done=False` (verified at horizon 5); RoboForge evaluation will stop on success or at 400 steps |
| Reset randomness | global `np.random` (KI-002); constructor `seed` kwarg unsupported |
| Determinism | same seeded reset → identical state; same restored state + actions in two processes → identical trajectory (max diff 0.0) |
| Throughput (macOS M-series, 1 core) | Lift ≈ 194–198 control steps/s, Can ≈ 163–164; offscreen render 256² ≈ 8–25 ms/frame |
| Cameras | Lift: `frontview, birdview, agentview, sideview, robot0_robotview, robot0_eye_in_hand`; Can: same without `sideview` |

## Lift
- nq=16, nv=15, state dim 32; free joint `cube_joint0`.
- Success: cube body z > table top + 0.04 m. Verified false at +0.039, true at +0.041.
- Initial distribution: cube x/y ±3 cm, random yaw; Panda joint noise σ=0.02 rad (KI-011).
- `object` obs (10): `cube_pos`, `cube_quat` (xyzw), `gripper_to_cube_pos` (cube − eef,
  world frame). All exact.

## PickPlaceCan
- nq=37, nv=33, state dim 71. Free joints `Milk/Bread/Cereal/Can_joint0`. The three
  inactive objects are parked at (10, 10, 10) and free-fall: exclude them from all metrics.
- Success: the can lies inside the goal bin (robosuite `PickPlace._check_success`,
  single-object mode).
- `object` obs (14): `Can_to_robot0_eef_pos`, `Can_to_robot0_eef_quat` (**lagged, KI-003**),
  `Can_pos`, `Can_quat` (exact).

## Reset-to-demo procedure (`roboforge.sim.state_replay.reset_to_state`)
`env.reset()` → `env.reset_from_xml_string(env.edit_model_xml(model_file))` →
`env.sim.reset()` → `set_state_from_flattened(state)` → `sim.forward()`. Read-back equals
the target exactly. This mirrors robomimic `EnvRobosuite.reset_to` for robosuite ≥ 1.4.

## Observation contract (Session 1, D-010)
- Policy observations come from `roboforge.data.features.FeatureExtractor.from_sim`, a
  pure function of the flattened state (scratch `MjData` + `mj_kinematics`). They never
  come from `env.step`'s observations, which are one physics substep stale (KI-015) and,
  for Can, one update lagged (KI-003).
- `robot0_eef_pos` is the `gripper0_right_grip_site` position; `robot0_eef_quat` is the
  `robot0_right_hand` body orientation (xyzw).
- robosuite's default models add two non-colliding mocap markers that the dataset XMLs
  lack (KI-014). A model recompiled from `get_xml()` differs at about 1e-7 (KI-016).

## Exact physics and control (Session 2; read from the evaluation models)
| Property | Value (Lift and Can) |
|---|---|
| Timestep | `opt.timestep = 0.002` s; 25 substeps per control step; control 20 Hz (0.05 s) |
| Integrator / solver | `mjINT_EULER`, `mjSOL_NEWTON`, `iterations = 100`, `tolerance = 1e-8`, `ls_iterations = 50`, `noslip_iterations = 0` |
| Contacts | `cone = elliptic`, `impratio = 20`, `jacobian = auto`; `<size njmax="5000" nconmax="5000"/>` |
| Medium / gravity | `density = 1.2`, `viscosity = 2e-5`, gravity (0, 0, −9.81); no disabled or enabled flags |
| Evaluation model size | Lift nq 16, nv 15, nu 9, nbody 26, nmocap 2; Can nq 37, nv 33, nu 9, nbody 34, nmocap 2 (the 2 mocap bodies are the KI-014 markers) |
| Horizon / termination | 400 control steps; first success ends the scored episode (D-026) |
| Evaluation platform | Linux aarch64 container `roboforge-sim-cpu:session0` @ `sha256:1f685638…` (D-017) |

`RoboForgeEnv.constants()` returns these values together with the action spec and the
observation slices. Every rollout record stores them under `simulator`.

## Reset mechanism (Session 2, D-024/D-025)
1. **Snapshot** (`roboforge.sim.snapshot.InitialState`): task, `state_id`, partition, the
   portable saved XML (`mj_saveLastXML` of the live model, asset paths
   `robosuite/models/assets/…`), the flattened state `[time, qpos, qvel]` (float64), the
   source (seed or demo) and SHA-256 hashes of the XML and the state.
2. **Restore** (`snapshot.restore` → `reset_to_state`): `env.reset()` →
   `reset_from_xml_string(edit_model_xml(xml))` → `sim.reset()` →
   `set_state_from_flattened(state)` → `sim.forward()`. The read-back must equal the
   snapshot exactly.
   - The fresh compile and deterministic reset clear all hidden state: `qacc_warmstart`,
     `ctrl`, mocap poses, the gripper integrator, controller goals and the step counter.
3. **Seeds** are only a way to *generate* snapshots: `random.seed(s)` and
   `np.random.seed(s)`, then `env.reset()`, then capture.
   - Seeds in [900000, 1000000) are refused until `docs/FINAL_TEST_FREEZE.md` exists
     (D-025).
   - Evaluation always restores the stored snapshot. It never re-seeds.

## Step contract (Session 2, D-024)
- **Action:** shape `(7,)` and finite, or `InvalidActionError` (the state is unchanged).
  Finite actions are clipped to [−1, 1], and `clip_mask` reports the clipped entries.
- **Observation:** `FeatureExtractor.from_sim(post-step state)`, 19-D float32 in spec order
  (KI-015).
- **Reward / success:** sparse 1.0 iff `_check_success()` holds after `mj_kinematics` on
  the live data. robosuite's stale value is kept as `success_robosuite_raw` and is never
  scored.
- **Never** call `mj_forward` / `sim.forward()` on live data during an episode (KI-019).

## File and transform schema (Session 2)
| Artifact | Schema | Location |
|---|---|---|
| Eval suite manifest (open: hashes only) | `roboforge.eval_suite/1` | `data/eval_states/<task>/<suite>.json` |
| Eval suite data (XML + states) | JSON lines, xz | `data/eval_states/<task>/<suite>.data.jsonl.xz` |
| Partition index | — | `data/eval_states/<task>/partitions.json` |
| Rollout record | `roboforge.rollout/1` (D-028) | `runs/<run_id>/manifest.json`; episodes in `runs/<run_id>/episodes/<state_id>/{record.json, arrays.npz}` |
| Web clip | `roboforge.web_clip/1` (D-030) | `reports/session2/web_diagnostic/` |

- **World frame:** MuJoCo world, z-up, metres.
  - Body poses are `xpos` / `xquat` (wxyz) from `mj_kinematics` on a scratch `MjData`.
  - Joint values are `qpos`/`qvel` per named non-free joint.
  - Free objects are read from their bodies.
- **Robot links:** the subtree of `robot0_base`. Fixtures are every other non-world,
  non-mocap body. Mocap markers are excluded.
- **Cameras:** `cam_xpos`, `cam_xmat` (columns: camera x right, y up, −z view), `cam_fovy`.
  The intrinsics are `f = (H/2) / tan(fovy/2)` with the principal point at the image
  centre.
- **three.js:** `(x, y, z)_mj → (x, z, −y)`, `q_three = q_C ⊗ q_mj` with
  `q_C = (cos(−π/4), sin(−π/4), 0, 0)` (D-030).
