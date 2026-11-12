# X2-a headroom pilot (development numbers on X2 validation starts; not a result)

Reading: **headroom** (rule: D-124, `configs/extensions/x2_pilot.yaml`).

| Family | Run | Steps/s | Wall h | CPU h | Peak RSS GiB | Disk GiB |
|---|---|---|---|---|---|---|
| diffusion_unet | `s15x2_diffusion_unet_cos_can_seed0_99251da3` | 6.01 | 4.62 | 4.58 | 0.37 | 0.34 |
| bc_rnn | `s15x2_bc_rnn_cos_can_seed0_5ff5ce91` | 24.95 | 1.11 | 1.14 | 1.08 | 0.25 |
| bc_mlp_chunk | `s15x2_bc_mlp_chunk_wd1e1_can_seed0_0c2f9d4b` | 145.70 | 0.19 | 0.21 | 0.39 | 0.28 |
| bc_mlp | `s15x2_bc_mlp_cos_can_seed0_ae8152e3` | 164.57 | 0.17 | 0.18 | 0.33 | 0.25 |

| Family | Step | Successes / 50 | Wilson 95 % | Mean first-success step | Wall min | Control steps/s |
|---|---|---|---|---|---|---|
| diffusion_unet | 20000 | 25 | [0.37, 0.63] | 284 | 11.1 | 29.4 |
| diffusion_unet | 40000 | 38 | [0.63, 0.86] | 234 | 7.5 | 33.1 |
| diffusion_unet | 60000 | 42 | [0.71, 0.92] | 204 | 6.4 | 32.8 |
| diffusion_unet | 80000 | 46 | [0.81, 0.97] | 199 | 5.6 | 33.5 |
| diffusion_unet | 100000 | 48 | [0.87, 0.99] | 197 | 4.1 | 42.5 |
| bc_rnn | 20000 | 0 | [0.00, 0.07] |  | 13.6 | 30.5 |
| bc_rnn | 40000 | 5 | [0.04, 0.21] | 280 | 13.7 | 29.2 |
| bc_rnn | 60000 | 29 | [0.44, 0.71] | 228 | 9.7 | 29.5 |
| bc_rnn | 80000 | 31 | [0.48, 0.74] | 222 | 9.4 | 29.0 |
| bc_rnn | 100000 | 34 | [0.54, 0.79] | 207 | 7.8 | 31.9 |
| bc_mlp_chunk | 20000 | 47 | [0.84, 0.98] | 160 | 4.7 | 31.9 |
| bc_mlp_chunk | 40000 | 37 | [0.60, 0.84] | 159 | 5.5 | 37.4 |
| bc_mlp_chunk | 60000 | 46 | [0.81, 0.97] | 169 | 3.8 | 42.4 |
| bc_mlp_chunk | 80000 | 48 | [0.87, 0.99] | 158 | 3.4 | 41.8 |
| bc_mlp_chunk | 100000 | 39 | [0.65, 0.87] | 190 | 4.8 | 45.3 |
| bc_mlp | 20000 | 0 | [0.00, 0.07] |  | 13.5 | 30.8 |
| bc_mlp | 40000 | 2 | [0.01, 0.13] | 336 | 11.1 | 37.0 |
| bc_mlp | 60000 | 5 | [0.04, 0.21] | 247 | 10.8 | 36.5 |
| bc_mlp | 80000 | 5 | [0.04, 0.21] | 237 | 10.6 | 37.1 |
| bc_mlp | 100000 | 6 | [0.06, 0.24] | 250 | 10.3 | 37.9 |

Best evaluated checkpoint per family (of 50): diffusion_unet 48, bc_rnn 34, bc_mlp_chunk 48, bc_mlp 6
