# RGB study: held-out results (Lift, generated)

Freeze `d9c6217`, decision D-081, engineering complete. 3 training seeds per family: the seed-level resampling is coarse; every per-seed value is listed (RESEARCH_PROTOCOL §9).

| Family | Seed | Step | k/n | Wilson 95 % | Mean first-success step | Latency ms (median) | Errors |
|---|---|---|---|---|---|---|---|
| rgb_bc_chunk | 0 | 5000 | 49/50 | 89.5–99.6 % | 42.816326530612244 | 66.4 | 0 |
| rgb_bc_chunk | 1 | 5000 | 50/50 | 92.9–100.0 % | 42.16 | 65.0 | 0 |
| rgb_bc_chunk | 2 | 55000 | 50/50 | 92.9–100.0 % | 44.7 | 63.0 | 0 |
| rgb_diffusion | 0 | 55000 | 50/50 | 92.9–100.0 % | 44.54 | 183.1 | 0 |
| rgb_diffusion | 1 | 55000 | 50/50 | 92.9–100.0 % | 44.26 | 187.2 | 0 |
| rgb_diffusion | 2 | 60000 | 50/50 | 92.9–100.0 % | 44.06 | 166.4 | 0 |

| Family | Success | 95 % interval | Per seed |
|---|---|---|---|
| rgb_diffusion | 100.0 % | 100.0–100.0 % | 100.0, 100.0, 100.0 |
| rgb_bc_chunk | 99.3 % | 96.0–100.0 % | 98.0, 100.0, 100.0 |

**rgb_diffusion − rgb_bc_chunk:** +0.7 points, 95 % interval [+0.0, +4.0]: **not separated**.

| Family | success | engineering_error | gripper_never_closed | no_grasp_drift | no_grasp_near | timeout_holding | in_goal_bin_not_success | dropped |
|---|---|---|---|---|---|---|---|---|
| rgb_diffusion | 150 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| rgb_bc_chunk | 149 | 0 | 0 | 0 | 1 | 0 | 0 | 0 |

| Run | Segments | Train wall h | Billed USD |
|---|---|---|---|
| s8_rgb_bc_chunk_resnet18_lift_seed0_73a80bad | 2 | 1.01 | 1.14 |
| s8_rgb_bc_chunk_resnet18_lift_seed1_c49230ce | 2 | 1.01 | 1.14 |
| s8_rgb_bc_chunk_resnet18_lift_seed2_2e677ad6 | 2 | 1.00 | 1.13 |
| s8_rgb_diffusion_resnet18_lift_seed0_dddc7f1c | 2 | 1.35 | 1.52 |
| s8_rgb_diffusion_resnet18_lift_seed1_51deeb3e | 2 | 1.36 | 1.53 |
| s8_rgb_diffusion_resnet18_lift_seed2_ec213b6a | 3 | 1.29 | 1.60 |

*CONTEXT ONLY: Level 1 low-dim policies, same 50 Lift held-out starts; different observations, not a declared comparison:* bc_mlp_chunk 98.0 % [94.0, 100.0]; diffusion_unet 97.3 % [93.3, 100.0].
