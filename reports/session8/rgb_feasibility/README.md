# Session 8 RGB feasibility evidence (MASTER_SPEC §7.2; no training, no RGB dataset)

Frames are rendered from the **recorded demonstration states** of the raw robomimic v1.5 files
(`demo_v15.hdf5`: per-demo `states` and `model_file`, no images) through the RoboForge render path
(DECISIONS D-074). Nothing here reads a held-out state; the only evaluation states used are
validation start 0 of each task (evaluation-model parity and closed-loop step cost).

| Directory / file | Platform | Status |
|---|---|---|
| `linux-aarch64-container-1f685638f28c/` | canonical container, `MUJOCO_GL=osmesa`, `make rgb-feasibility` at `f3c9170` (clean) | **canonical** |
| `darwin-arm64-native/` | macOS arm64, `MUJOCO_GL=cgl`, `make rgb-feasibility-native` at `f3c9170` | development only |
| `compare_container_two_processes.json` | container run 1 vs a second container process (`logs/session8/rgb_feasibility/container_run2/`) | bitwise check |
| `compare_container_vs_native.json` | OSMesa vs CGL on identical states | cross-platform difference |

## Results (canonical container)
| Check | Lift | Can |
|---|---|---|
| Raw file = low-dim file (model XML, states, actions, bitwise), every demo | PASS: 200 demos, 9,666 frames | PASS: 200 demos, 23,207 frames |
| Features of the raw states = committed processed `obs` (float32, bitwise); actions equal | PASS, all 200 | PASS, all 200 |
| Projection test, t = 0 of the 20 validation-split demos × 2 cameras × 84/128 (80 frames): projected `*_pos` from the processed obs lands on the object's segmentation pixels (≤ 1 px) | 80/80 visible, 80/80 on object, max 0.70 px from the mask | 68/80 visible (the wrist camera does not see the can at t = 0 in 6 demos), 68/68 on object, max 0.65 px |
| Negative controls (where informative: the control moves the projection by more than the object's pixel radius + 1 px) | vertically flipped mask 0/54 on object; other camera's mask 0/75 | 0/50; 0/60 |
| Fixed inspection sample (12 frames × 2 cameras × 2 sizes): agentview projection on object | 24/24 | 24/24 |
| Re-render bitwise; no visualization site drawn (all site groups off = identical); state reached by stepping the live sim renders like the same state in a freshly compiled model | PASS, both cameras, 84 and 128 | PASS |
| First render of a fresh GL context vs later renders | 0 px differ (OSMesa) | 0 px |
| Evaluation-start model (robosuite default model with the KI-014 markers) renders the demo state like the demo model | differences only on the cube and within 2 px of it (cube size is sampled per reset) | bitwise identical |
| Two container processes, every sample and clip frame | 166/166 frame hashes equal | 284/284 |

Native CGL (development): all checks pass too, but the first render of a fresh context differs
from later renders by up to 1 px (Lift) and 6 px (Can); every kept frame is a steady-state
frame (`WARM_UP_RENDERS = 1`). CGL vs OSMesa on identical states: no frame is bitwise equal;
mean absolute difference 0.64–2.06 grey levels (Lift) and 1.79–3.12 (Can); at most 0.34 % of
pixels differ by more than 16 levels. So frames that train or evaluate a reported RGB policy
come from the container only.

## Inspection by eye (fixed sample, rule declared in `scripts/rgb_feasibility.py` before looking)
`<task>_inspection.png` (overlays exist only on this montage; `frames/` holds the raw PNGs).
Inspected 2026-11-09 on the canonical montages:
- **Lift** (demos 5, 15, 24, 27, 44, 67, 82, 85, 94, 162, 164, 185): every agentview crosshair
  is on the cube; wrist-camera crosshairs are on the cube, or at its occluded centre between the
  fingers when it is grasped. Arm poses match the step (approach early, grasp/lift later).
- **Can** (demos 8, 26, 28, 61, 64 ×2, 93, 155, 161, 162, 183, 189): every agentview crosshair is
  on the can; when the can is held, the wrist-camera crosshair is at the can's centre in the
  gripper; where the can is outside the wrist view, no crosshair is drawn (projection off image).
- Images are upright; no visualization site (coloured sphere) appears in any frame.

## Aligned clips with ground-truth actions
`<task>_clip.mp4`: the lowest-index validation demo (`demo_0` for both tasks), both cameras at
128², 20 fps (the control rate), each frame labelled with task, demo, step, the state's SHA-256
prefix and the ground-truth action `a[t]`. `<task>_clip.jsonl` holds the full alignment record of
every frame (schema `roboforge.rgb_frame/1`).

## Cost (canonical container, this host; `<task>_cost.json`)
| Quantity | Lift | Can |
|---|---|---|
| Render, ms per frame, median (84² / 128²) | agentview 53.6 / 37.1; wrist 65.4 / 49.5 | agentview 65.6 / 64.5; wrist 71.3 / 58.9 |
| Control step, physics only (ms, median) | 5.8 | 11.6 |
| Control step + both cameras rendered (84² / 128²) | 122.6 / 102.7 ms | 149.3 / 127.3 ms |
| Storage, both cameras, HDF5 gzip-4 per frame pair (84² / 128²) | 14.2 / 33.0 kB | 23.4 / 49.4 kB |
| Projected dataset size, all demos, gzip-4 (84² / 128²) | 0.13 / 0.30 GiB | 0.51 / 1.07 GiB |

OSMesa cost is dominated by per-frame overhead, not by pixels: 128² is no slower than 84² here.
Native CGL renders a frame in 3.6–4.1 ms (development only).
