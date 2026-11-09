# Level 2 study: results (Lift, DP3-inspired, generated)

Freeze `b93c50c`, decision D-086, approval A-04. sensor-like: depth + calibration only (D-083); not privileged geometry. 3 training seeds per arm: the seed-level resampling is coarse; every per-seed value is listed (RESEARCH_PROTOCOL §9). Power: with 3 seeds x 50 starts, binomial noise alone gives about +/-8 points near 85 % success (an estimate); 'not separated' is not equivalence.

## Placement (primary H5 on ood_cube_shift)

| Arm | Seed | Step | Set | k/n | Wilson 95 % | Mean first-success step | Errors |
|---|---|---|---|---|---|---|---|
| pc_diffusion | 0 | 30000 | robustness_base | 50/50 | 92.9–100.0 % | 43.8 | 0 |
| pc_diffusion | 1 | 40000 | robustness_base | 50/50 | 92.9–100.0 % | 46.02 | 0 |
| pc_diffusion | 2 | 60000 | robustness_base | 50/50 | 92.9–100.0 % | 44.1 | 0 |
| pc_diffusion | 0 | 30000 | ood_cube_shift | 32/50 | 50.1–75.9 % | 135.5625 | 0 |
| pc_diffusion | 1 | 40000 | ood_cube_shift | 28/50 | 42.3–68.8 % | 117.67857142857143 | 0 |
| pc_diffusion | 2 | 60000 | ood_cube_shift | 25/50 | 36.6–63.4 % | 153.72 | 0 |
| rgb_diffusion | 0 | 55000 | robustness_base | 50/50 | 92.9–100.0 % | 43.38 | 0 |
| rgb_diffusion | 1 | 55000 | robustness_base | 50/50 | 92.9–100.0 % | 44.36 | 0 |
| rgb_diffusion | 2 | 60000 | robustness_base | 50/50 | 92.9–100.0 % | 43.02 | 0 |
| rgb_diffusion | 0 | 55000 | ood_cube_shift | 50/50 | 92.9–100.0 % | 54.84 | 0 |
| rgb_diffusion | 1 | 55000 | ood_cube_shift | 49/50 | 89.5–99.6 % | 57.69387755102041 | 0 |
| rgb_diffusion | 2 | 60000 | ood_cube_shift | 48/50 | 86.5–98.9 % | 59.666666666666664 | 0 |

| Key | Estimate (%) | 95 % interval | Per seed |
|---|---|---|---|
| pc_diffusion:robustness_base | 100.0 | 100.0–100.0 | 100.0, 100.0, 100.0 |
| pc_diffusion:ood_cube_shift | 56.7 | 43.3–70.0 | 64.0, 56.0, 50.0 |
| rgb_diffusion:robustness_base | 100.0 | 100.0–100.0 | 100.0, 100.0, 100.0 |
| rgb_diffusion:ood_cube_shift | 98.0 | 93.3–100.0 | 100.0, 98.0, 96.0 |

| Difference | Kind | Estimate (pp) | 95 % CI | Outcome |
|---|---|---|---|---|
| pc_diffusion:ood_cube_shift − rgb_diffusion:ood_cube_shift | PRIMARY H5: between-arm difference in OOD success | -41.3 | [-54.0, -28.0] | **separated (-)** |
| pc_diffusion:robustness_base − pc_diffusion:ood_cube_shift | drop (base - OOD), pc_diffusion | +43.3 | [+30.0, +56.7] | **separated (+)** |
| rgb_diffusion:robustness_base − rgb_diffusion:ood_cube_shift | drop (base - OOD), rgb_diffusion | +2.0 | [+0.0, +6.7] | **not separated** |
| pc_diffusion:robustness_base − rgb_diffusion:robustness_base | between-arm difference in base success (context) | +0.0 | [+0.0, +0.0] | **not separated** |
| drop pc_diffusion - drop rgb_diffusion | difference in drop | +41.3 | [+28.0, +54.0] | **separated (+)** |

## Held-out (secondary; RGB records reused, never rerun)

| Arm | Seed | Step | k/n | Wilson 95 % | Mean first-success step | Latency ms | Errors |
|---|---|---|---|---|---|---|---|
| pc_diffusion | 0 | 30000 | 50/50 | 92.9–100.0 % | 45.36 | 147.4 | 0 |
| pc_diffusion | 1 | 40000 | 50/50 | 92.9–100.0 % | 45.56 | 146.0 | 0 |
| pc_diffusion | 2 | 60000 | 50/50 | 92.9–100.0 % | 45.28 | 147.0 | 0 |
| rgb_diffusion | 0 | 55000 | 50/50 | 92.9–100.0 % | 44.54 | 183.1 | 0 |
| rgb_diffusion | 1 | 55000 | 50/50 | 92.9–100.0 % | 44.26 | 187.2 | 0 |
| rgb_diffusion | 2 | 60000 | 50/50 | 92.9–100.0 % | 44.06 | 166.4 | 0 |

**Success**

| Key | Estimate (%) | 95 % interval | Per seed |
|---|---|---|---|
| pc_diffusion | 100.0 | 100.0–100.0 | 100.0, 100.0, 100.0 |
| rgb_diffusion | 100.0 | 100.0–100.0 | 100.0, 100.0, 100.0 |

| Difference | Kind | Estimate (pp) | 95 % CI | Outcome |
|---|---|---|---|---|
| pc_diffusion − rgb_diffusion | difference | +0.0 | [+0.0, +0.0] | **not separated** |

**Censored episode length (steps; failures = 400)**

| Key | Estimate (steps) | 95 % interval | Per seed |
|---|---|---|---|
| pc_diffusion | 45.4 | 43.6–47.4 | 45.4, 45.6, 45.3 |
| rgb_diffusion | 44.3 | 42.5–46.5 | 44.5, 44.3, 44.1 |

| Difference | Kind | Estimate (steps) | 95 % CI | Outcome |
|---|---|---|---|---|
| pc_diffusion − rgb_diffusion | difference | +1.1 | [-1.4, +3.4] | **not separated** |

## Failure taxonomy

| Arm:set | success | engineering_error | gripper_never_closed | no_grasp_drift | no_grasp_near | timeout_holding | in_goal_bin_not_success | dropped |
|---|---|---|---|---|---|---|---|---|
| pc_diffusion:robustness_base | 150 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| pc_diffusion:ood_cube_shift | 85 | 0 | 0 | 4 | 45 | 0 | 0 | 16 |
| pc_diffusion:heldout | 150 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| rgb_diffusion:robustness_base | 150 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| rgb_diffusion:ood_cube_shift | 147 | 0 | 0 | 0 | 3 | 0 | 0 | 0 |

## Training (point-cloud arm, L4)

| Run | Segments | Train wall h | ms/step | Billed USD |
|---|---|---|---|---|
| s9_pc_diffusion_dp3_lift_seed0_8fd35613 | 3 | 0.41 | 24.62 | 0.80 |
| s9_pc_diffusion_dp3_lift_seed1_a66a2758 | 1 | 0.70 | 41.73 | 0.78 |
| s9_pc_diffusion_dp3_lift_seed2_402a1f24 | 1 | 0.71 | 42.55 | 0.80 |
