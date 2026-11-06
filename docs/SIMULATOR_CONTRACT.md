# Simulator Contract (Session 0 baseline)

Facts below were read from the pinned simulator and dataset metadata, not assumed.
Configs: `configs/simulator/{lift,can}.yaml`. Evidence:
`reports/feasibility/smoke_sim_{lift,can}.json` and `data_report_{lift,can}.json`.

## Stack
Python 3.11.15 · mujoco 3.2.7 · robosuite 1.5.1 (`51cc017`) · robomimic `d309eae`.
Backend: `MUJOCO_GL=cgl` on macOS (verified), `egl`/`osmesa` on Linux (KI-001).

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

## Open items (Session 2)
- `RoboForgeEnv` adapter per spec §4.1.
- Serialized eval-state suites.
- The episode-termination rule.
- Kinematics export with documented frames, and the MuJoCo→three.js conversion test.
