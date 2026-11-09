# Session 8 Gate D RGB probe: one Modal L4, approval A-02 (timing only)

**This is not training.** The policies have random weights, and every input is random
`uint8` images plus random normalized proprioception and actions. No dataset, evaluation state
or checkpoint was read or written, and nothing was kept. The only outputs are seconds, memory,
flags and parameter hashes.

| File | What |
|---|---|
| `rgb_probe_plan.json` | The launch plan (`roboforge.compute.rgb_probe plan`): git `5f2ca4a` (clean), 22 sections, timeout 360 s, deadline 300 s, upper bound $0.528 at list price |
| `rgb_probe_result.json` | The job's result: host, CUDA facts, and every section with its full record (`step_def`, flags, timings, peak memory) |
| `../../../logs/session8/modal/rgb_probe_{dry_run,launch}.log` | Client logs, including one JSON line per section from the container |

Launch facts:
- App `ap-v4h8hPxCAv9fSLgXDsFlBH`, launch `c087ef13659840d8aea71012da69fad2`, started
  2026-11-09 02:33 UTC.
- GCP `us-central`, x86_64, attempt 0 of at most 4.
- Status **complete**: 22 of 22 sections ran, 0 failed, 0 skipped.
- 92.5 s in the container; the client waited 160 s.

Platform:
- NVIDIA L4 (23,034 MiB), driver 610.57.04.
- torch 2.15.0+cu132 (CUDA 13.2), cuDNN 9.26.
- PyTorch defaults on this GPU: cuDNN TF32 **on**, matmul TF32 off.
- `CUBLAS_WORKSPACE_CONFIG=:4096:8` was set before the first CUDA call.

## Step definition (identical on every device; `roboforge.compute.rgb_bench`)
Each timed step does the following, in order:
1. copies the batch from pinned host memory to the GPU;
2. takes random crops (76 of 84 px, 116 of 128) of 2 cameras × T_o = 2 frames;
3. runs the encoder:
   - per camera: trunk, then SpatialSoftmax(32), then Linear(64);
   - then appends the 9-D proprioception;
4. applies the head's loss:
   - BC-MLP-chunk: masked MSE;
   - diffusion U-Net: ε-MSE;
5. runs backward and takes the total gradient norm;
6. takes an AdamW step (Level 1 optimizer settings, lr 1e-4);
7. updates the EMA (power 0.75), for diffusion only, as in its config.

The timing covers 30 steps after 5 warm-up steps; repeat checks run 20 steps twice.

Parameters (`params_*` in each record):

| Component | Parameters |
|---|---|
| Encoder, ResNet-18 trunks (two cameras) | 22.39 M |
| Encoder, compact trunks (two cameras) | 0.63 M |
| BC-MLP-chunk head (input 2 × 137) | 1.45 M |
| U-Net head (input 2 × 137) | 1.46 M |

## Results (median ms per optimizer step; peak GPU memory)
| Trunk, image | Head | batch 64 | batch 100 | peak MiB (b64 / b100) |
|---|---|---|---|---|
| ResNet-18, 84² | BC-MLP-chunk | **54.5** (18.3 steps/s) | 74.1 | 1,108 / 1,447 |
| ResNet-18, 84² | diffusion | **71.6** (14.0 steps/s) | 91.0 | 1,203 / 1,545 |
| ResNet-18, 128² | BC-MLP-chunk | 105.1 | 165.9 | 1,936 / 2,782 |
| ResNet-18, 128² | diffusion | 121.1 | 182.6 | 2,026 / 2,873 |
| compact, 84² | BC-MLP-chunk | 28.1 | 43.2 | 512 / 773 |
| compact, 84² | diffusion | 45.8 | 60.0 | 520 / 781 |
| compact, 128² | BC-MLP-chunk | 62.8 | 104.0 | 1,139 / 1,741 |
| compact, 128² | diffusion | 79.4 | 119.5 | 1,147 / 1,749 |

Every run is far below the L4's 23 GiB. The slowest step is at most 5.0 % above its median.

## CUDA determinism (ResNet-18, 84², batch 64)
"Deterministic" means `torch.use_deterministic_algorithms(True)`, cuDNN deterministic and
cuDNN benchmark off.

| Head | default ms | deterministic ms | slowdown | default: 2 same-seed runs of 20 steps | deterministic: same |
|---|---|---|---|---|---|
| BC-MLP-chunk | 54.5 | 58.4 | +7.2 % | **differ** (final loss 0.311046 vs 0.311085; parameter hashes differ) | **bitwise equal** (parameters, EMA and every loss) |
| diffusion | 71.6 | 79.8 | +11.5 % | **differ** (0.963112 vs 0.963125) | **bitwise equal** |

Every operation in both policies has a deterministic CUDA kernel: nothing raised. Repeats are
verified only within one container on one L4, at 20 steps. Whether runs are bitwise equal
across different L4 hosts or drivers is **not** verified.

## Cost
**$0.0317 billed.** Source: Modal's hourly report for 02:00 UTC
(`logs/session8/modal/billing_20261109_h.json`), recorded as the `billing` event in
`reports/compute/cost_ledger.jsonl`. Measured container time × list price was $0.0293; the
difference is start-up.

The A-02 cap was $1.00. The approval is now closed (`exhausted`).
