# X2-b on Can MH (development numbers on the 50 X2 validation starts; not a result)

Selection by the frozen §7 rule per run; configuration by the frozen §6.3 rule per family (tuning seed 0). Pilot runs (X2-a, D-124) count as their trials' seed-0 runs (D-125).

| Family | Trial | Seed | Run | Pilot | Status | Wall h | Peak RSS GiB | Evaluated | Selected step | Successes / 50 | Mean first-success step |
|---|---|---|---|---|---|---|---|---|---|---|---|
| diffusion_unet | 1 | 0 | `s16x2_diffusion_unet_can_seed0_b0cb87bb` |  | complete | 5.51 | 0.49 | 20 | 90000 | 48 | 206 |
| diffusion_unet | 2 | 0 | `s15x2_diffusion_unet_cos_can_seed0_99251da3` | yes | complete | 4.62 | 0.37 | 20 | 100000 | 48 | 197 |
| diffusion_unet | 3 | 0 | `s16x2_diffusion_unet_lr3e4_can_seed0_a5e9939f` |  | complete | 5.51 | 0.49 | 20 | 65000 | 47 | 186 |
| diffusion_unet | 4 | 0 | `s16x2_diffusion_unet_wd1e1_can_seed0_0cb7d4f8` |  | complete | 5.52 | 0.50 | 20 | 100000 | 45 | 207 |
| bc_rnn | 1 | 0 | `s16x2_bc_rnn_can_seed0_9755b24b` |  | complete | 1.33 | 1.22 | 20 | 80000 | 34 | 199 |
| bc_rnn | 2 | 0 | `s15x2_bc_rnn_cos_can_seed0_5ff5ce91` | yes | complete | 1.11 | 1.08 | 20 | 100000 | 34 | 207 |
| bc_rnn | 3 | 0 | `s16x2_bc_rnn_lr3e4_can_seed0_d59b2d0c` |  | complete | 1.19 | 1.03 | 20 | 85000 | 35 | 210 |
| bc_rnn | 4 | 0 | `s16x2_bc_rnn_wd1e1_can_seed0_14e39a6c` |  | complete | 1.22 | 1.05 | 20 | 65000 | 36 | 194 |
| bc_mlp_chunk | 1 | 0 | `s16x2_bc_mlp_chunk_can_seed0_e245ceb1` |  | complete | 0.15 | 0.40 | 20 | 15000 | 49 | 172 |
| bc_mlp_chunk | 2 | 0 | `s16x2_bc_mlp_chunk_cos_can_seed0_7d7745e8` |  | complete | 0.15 | 0.41 | 20 | 15000 | 49 | 156 |
| bc_mlp_chunk | 3 | 0 | `s16x2_bc_mlp_chunk_lr3e4_can_seed0_d4114f71` |  | complete | 0.15 | 0.40 | 20 | 10000 | 48 | 162 |
| bc_mlp_chunk | 4 | 0 | `s15x2_bc_mlp_chunk_wd1e1_can_seed0_0c2f9d4b` | yes | complete | 0.19 | 0.39 | 20 | 80000 | 48 | 158 |
| bc_mlp | 1 | 0 | `s16x2_bc_mlp_can_seed0_ad021ce7` |  | complete | 0.16 | 0.37 | 20 | 85000 | 16 | 230 |
| bc_mlp | 2 | 0 | `s15x2_bc_mlp_cos_can_seed0_ae8152e3` | yes | complete | 0.17 | 0.33 | 20 | 95000 | 11 | 273 |
| bc_mlp | 3 | 0 | `s16x2_bc_mlp_lr3e4_can_seed0_d8ea4251` |  | complete | 0.13 | 0.40 | 20 | 90000 | 14 | 242 |
| bc_mlp | 4 | 0 | `s16x2_bc_mlp_wd1e1_can_seed0_72e464be` |  | complete | 0.15 | 0.38 | 20 | 70000 | 9 | 246 |
| diffusion_unet | 2 | 1 | `s16x2_diffusion_unet_cos_can_seed1_51f8481d` |  | complete | 4.89 | 0.44 | 20 | 90000 | 46 | 208 |
| diffusion_unet | 2 | 2 | `s16x2_diffusion_unet_cos_can_seed2_9b0fde56` |  | complete | 4.92 | 0.43 | 20 | 65000 | 47 | 201 |
| bc_rnn | 4 | 1 | `s16x2_bc_rnn_wd1e1_can_seed1_7eddb00a` |  | complete | 1.14 | 1.16 | 20 | 65000 | 36 | 194 |
| bc_rnn | 4 | 2 | `s16x2_bc_rnn_wd1e1_can_seed2_14ae1f29` |  | complete | 1.19 | 1.04 | 20 | 90000 | 39 | 183 |
| bc_mlp_chunk | 2 | 1 | `s16x2_bc_mlp_chunk_cos_can_seed1_96b63a9d` |  | complete | 0.15 | 0.48 | 20 | 5000 | 50 | 175 |
| bc_mlp_chunk | 2 | 2 | `s16x2_bc_mlp_chunk_cos_can_seed2_a6007970` |  | complete | 0.15 | 0.41 | 20 | 30000 | 48 | 191 |
| bc_mlp | 1 | 1 | `s16x2_bc_mlp_can_seed1_8bf126a6` |  | complete | 0.13 | 0.37 | 20 | 45000 | 12 | 239 |
| bc_mlp | 1 | 2 | `s16x2_bc_mlp_can_seed2_8324158e` |  | complete | 0.13 | 0.37 | 20 | 75000 | 15 | 209 |

Chosen configurations: diffusion_unet trial 2 (48/50), bc_rnn trial 4 (36/50), bc_mlp_chunk trial 2 (49/50), bc_mlp trial 1 (16/50)

X2 test configuration (for X2-c): diffusion_unet: seed 0 step 100000 (48/50), seed 1 step 90000 (46/50), seed 2 step 65000 (47/50); bc_rnn: seed 0 step 65000 (36/50), seed 1 step 65000 (36/50), seed 2 step 90000 (39/50); bc_mlp_chunk: seed 0 step 15000 (49/50), seed 1 step 5000 (50/50), seed 2 step 30000 (48/50); bc_mlp: seed 0 step 85000 (16/50), seed 1 step 45000 (12/50), seed 2 step 75000 (15/50)

X2-b evaluations: 460, 68.98 container-hours (mean 539.8 s).
