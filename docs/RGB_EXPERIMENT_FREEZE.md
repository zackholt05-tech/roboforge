# RGB experiment freeze: visual BC vs visual diffusion on Lift (Session 8)

**Status:** FROZEN by the Git commit that adds this file. It is written before any RGB training
run (MASTER_SPEC Session 8 step 3). Every item below is fixed: no configuration, budget,
selection rule, suite or analysis may change after a training run starts, except by a dated
amendment in §12 that states which runs it affects. Owner decisions: Gate D (D-076, A-02, the
probe) and **Gate D2 (D-078, A-03, "L60")**. Protocol: RESEARCH_PROTOCOL v1 with its §13
amendment of 2026-11-08 (RGB training on a deterministic L4).

Level 1 is frozen (`docs/LEVEL1_COMPLETE.md`). This is a separate, separately labelled
observation-modality study (MASTER_SPEC §3.2 item 5). Nothing here changes a Level 1
configuration, label or finding. Level 1 numbers are never pooled with these, and are never
reported as the same experiment.

## 1. Question and comparison
- **Question.** With RGB observations only (two cameras plus proprioception, no object
  state), does a conditional diffusion action-chunk policy match or beat a deterministic
  action-chunk regressor on Lift, when both share the encoder, data, augmentation, budget
  and selection rule?
- **Arms.** Two families that differ **only in the head**:
  - `rgb_bc_chunk` uses the BC-MLP-chunk head;
  - `rgb_diffusion` uses the U-Net head with DDIM-10.
  Both heads are copied verbatim from their Level 1 configs, and their horizons are
  T_o = 2, T_p = 16, T_a = 8.
- **Primary outcome.** Held-out closed-loop success on Lift, over 3 training seeds × 50 starts
  per family, and the paired difference diffusion − BC (§9).
- **Secondary outcomes:**
  - validation success at every checkpoint (development numbers);
  - first-success step;
  - failure categories;
  - inference latency per decision on the canonical container;
  - training time and cost.
- **Not studied:**
  - Can, other cameras or resolutions (128²), pretrained encoders and frame stacking beyond
    T_o;
  - tuning: one configuration per family, no trials;
  - comparisons with Level 1 low-dim policies.

## 2. Observation (RGB only; no object state)
| Item | Value |
|---|---|
| Cameras | `agentview`, `robot0_eye_in_hand`, square |
| Resolution | 84 × 84, `uint8` RGB, upright |
| Render | D-074: `ShadowRenderer` on scratch data, geom groups (0, 1, 1, 0, 0, 0), every visualization site hidden, one discarded warm-up render per GL context, canonical container (OSMesa) |
| Proprioception | `robot0_eef_pos` (3), `robot0_eef_quat` (4), `robot0_gripper_qpos` (2) = dims 0–9 of the committed 19-D low-dim spec, min-max normalized with its **train-split** statistics (`data/normalization/lift_ph_lowdim_v1.json`) |
| Excluded | `cube_pos`, `cube_quat`, `gripper_to_cube_pos` (privileged object state) |
| Window | frames t − 1 and t; frame indices clip at the episode start, with `obs_valid` false for the padded frame (Level 1 `window_indices`) |

## 3. Encoder (identical in both arms)
- **Per camera, separate weights:**
  - torchvision ResNet-18 `conv1` through `layer4`, with **GroupNorm** (groups = channels / 16)
    replacing every BatchNorm;
  - random initialization (`weights=None`; nothing is downloaded);
  - SpatialSoftmax with 32 keypoints, temperature 1;
  - Linear(64 → 64).
- **Output** per frame: 2 × 64 image features + 9 proprioception values = 137.
  Encoder parameters: 22,394,176.
- **Initialization:** `torch.manual_seed(seed)`, then the encoder is built, then the head.
  For a given seed both arms start from bitwise-identical encoder weights
  (`tests/unit/test_rgb_train.py`).
- **Heads:**
  - BC-MLP-chunk: MLP 2 × 1024, ReLU, 1,446,000 parameters;
  - U-Net: (64, 128), FiLM, 64-D step embedding, 100 training timesteps, squared-cosine
    betas, ε prediction, clipped samples, DDIM-10 with η = 0, 1,455,047 parameters.
- **EMA:** the U-Net head keeps its Level 1 EMA (power 0.75, maximum 0.9999) over the whole
  policy, and its EMA weights are the inference network. BC has no EMA, as in Level 1.

## 4. Augmentation (identical in both arms)
- **Training:** a random 76 × 76 crop per image, drawn independently per camera and frame,
  from a crop generator seeded `SeedSequence([seed, 2])` on the training device.
- **Shared streams:** both arms see the same batch stream (`EpochStreamSampler(seed)`), and
  diffusion noise comes from its own generator `[seed, 3]`. The crop sequence is therefore
  identical across arms at each seed.
- **Evaluation:** the centre 76 × 76 crop.
- No other augmentation.

## 5. Training (identical in both arms)
| Item | Value |
|---|---|
| Configs | `configs/models/rgb/bc_chunk_resnet18_lift.yaml`, `configs/models/rgb/diffusion_resnet18_lift.yaml` (differ only in name, family, head, model block and loss; test) |
| Data | all 180 train-split demos of Lift PH (8,640 windows); 20 valid-split demos for logged offline loss only |
| Optimizer | AdamW, lr 1e-4 constant, betas (0.9, 0.999), weight decay 0 (Level 1 defaults, untuned); no gradient clipping (norm logged) |
| Budget | **60,000 steps**, batch **64**, a checkpoint every **5,000** steps (**K = 12**), the last at 60,000 |
| Seeds | 0, 1, 2 per family: 6 runs, run ID `s8_<name>_lift_seed<s>_<config sha[:8]>` |
| Device | one NVIDIA L4 (Modal `train_gpu`), float32, `torch.use_deterministic_algorithms(True)`, cuDNN deterministic, cuDNN benchmark off, `CUBLAS_WORKSPACE_CONFIG=:4096:8`; TF32 at PyTorch defaults (on for convolutions, off for matmuls) |
| Launches | at most 1 h each, strictly sequential (`scripts/rgb_train_queue.py`): diffusion s0, BC s0, diffusion s1, BC s1, diffusion s2, BC s2; a run continues in resume launches from `resume/latest.pt` |
| Provenance | every segment records host, cloud, region, GPU, driver and the flags (`segments.jsonl`, `manifest.json`); bitwise repeatability is claimed only within an uninterrupted segment |
| Budget guard | approval A-03 ($15 cap) in `docs/EXPERIMENT_LEDGER.md` |

**Failure handling.**
- **Engineering failure** (crash, data error or preemption loop): a run may resume from its
  last resume file.
- **A run that cannot resume** is restarted from scratch under a new launch, with the incident
  recorded, and its configuration never changes. If a failed launch left its run ID marked
  launched but with nothing to resume, the restart uses prefix `s8r<k>`, k = 1, 2, …, and the
  restart is recorded in §12.
- **A run that trains but performs poorly** is kept. Nothing is retrained because of its
  results.

## 6. Data identity
| Item | Value |
|---|---|
| RGB frames | `data/processed/robomimic_v1.5_ph/lift/rgb84_v1.hdf5`, schema `roboforge.rgb_dataset/1` (D-079); content SHA-256 `41c0ea0253e196c55bdbbca972bf5ccad0136724896eae8e639aa6e1b901334f`; file SHA-256 `05c23eee534bafeeb6f627ceb12d525cfc20a5a54df2ff4ea11e510638c451a3`; manifest `data/manifests/rgb/lift_rgb84_v1.json` |
| Low-dim pairing | processed content SHA-256 `74e70119993830798c8b4ed87f202f64d8d2c9d573c0f376d54c4130e52d63c8` (proprioception, actions, windows, split) |
| Split | `data/splits/lift_ph.json`, content SHA-256 `73097265526c03dfb7535aa9bff50aadd285cbfb554e563636d66395909c331e` (180 train, 20 valid) |
| Normalization | `data/normalization/lift_ph_lowdim_v1.json`, content SHA-256 `253bfc7e63570bf3e8f8646a0094d1da4c5b729ba8f463b4955f01410b4812d6` |

## 7. Checkpoint selection (RESEARCH_PROTOCOL §7, unchanged)
- **Suite:** all 12 checkpoints of every run run all **50 Lift validation starts**
  (`validation` suite, the same as Level 1) on the canonical container.
- **Rule:**
  1. highest success count;
  2. ties to the lower mean first-success step;
  3. then to the earlier checkpoint.
- **Mechanics:** `scripts/eval_rgb.py validate` / `select`. The selection artifact is
  `reports/session8/rgb_selection/<run_id>.json`. Only canonical, complete, verified,
  error-free summaries whose checkpoint hash matches the training manifest can select.
- **Signals that never select:** offline losses, gradient norms and clip rates.
- **Bias:** validation numbers are development numbers, because the maximum over K biases
  them upward.

## 8. Closed-loop policy (`src/roboforge/eval/rgb.py`)
- **Images** come from the two cameras through `RoboForgeEnv.render`, the same D-074 path.
  The env renderer's first render is a discarded warm-up.
- **Frames rendered:** only those a decision reads. For decisions at frames 0, 8, 16, …, these
  are frame 0, then frames d + 7 and d + 8 after each decision d (simulator test).
- **Inference** runs on the CPU with 1 torch thread and the centre crop:
  - BC uses its trained weights;
  - diffusion uses its EMA weights, DDIM-10, one sample per decision, and Level 1's
    per-episode noise seed rule (`noise_seed(policy_seed = 0, state_id, repeat = 0)`).
- **Actions:** the first 8 actions of each chunk, decoded and clipped by Level 1's
  `ActionCodec`.
- **Episode rules:** Lift's 400-step horizon, the success predicate, the hold steps and the
  termination of the Level 1 suites.

## 9. Held-out test (run once)
- **Suite:** Lift's held-out suite, the same 50 starts as Level 1, schedule manifest SHA-256
  `e02498651b9d5bb88b6aec0870c5e3277cadaaeaf0e325abf86c79e89b1f22da`.
  - No RGB decision has used these states.
  - The Level 1 test does not inform any choice here: every RGB choice is fixed by this file
    and by the validation suite.
- **What runs:** each of the 6 selected checkpoints runs all 50 starts once on the canonical
  container (R = 1; DDIM noise per Level 1's rule). That is 6 evaluations in all.
- **Preconditions, in order. All must be committed before the first held-out RGB rollout:**
  1. all 6 selection artifacts;
  2. a recorded decision that **re-gates the held-out lock** on this file;
  3. a test proving two things:
     - the lock is shut for every RGB evaluation until (1) and (2) hold;
     - `heldout.evaluate_entry` still refuses every Level 1 configuration.
- **Never:** tuning, reselection or reruns after it. A later access is labelled
  exploratory or replication.
- **Analysis**, following RESEARCH_PROTOCOL §9 with its seed and B:
  - per checkpoint, k/50 with a Wilson 95 % interval;
  - per family, the mean over 3 seeds × 50 starts, with every per-seed value listed;
  - the hierarchical paired bootstrap: B = 10,000, seed `20261106`. Seeds are resampled per
    family, and states in one shared draw. It gives 95 % percentile intervals for each family
    and for **diffusion − BC**.
- **Wording:**
  - if the difference interval contains 0, the arms are reported as **"not separated"** on
    Lift RGB;
  - no p-value threshold;
  - with 3 seeds the seed resampling is coarse, and the report says so.
- **Also reported:**
  - failure categories (`configs/eval/failure_taxonomy.yaml`, Level 1's taxonomy, unchanged)
    on every failed test episode;
  - inference latency per decision;
  - training wall time and billed cost.
- **The three kinds of negative result stay apart:**
  - engineering failures;
  - a family that fails validation;
  - a held-out difference that is not separated.

## 10. Code identity at the freeze
SHA-256 of the files that define this experiment (a change after the freeze needs a §12
amendment):

| File | SHA-256 |
|---|---|
| `src/roboforge/train/rgb.py` | `6e02d4fbc775e3e4612c6c958d1810149e6215a02928be93c131ef9c11aaa886` |
| `src/roboforge/models/vision.py` | `164d72d23c0d616f705c9ac568d9878a4fc55c6a2d2e79c2ecc37eaab3e4db64` |
| `src/roboforge/compute/rgb_train.py` | `ad7867e4bb1b4afddb097e8da05a4ef5ff627e5e33b670fe94e0171ed5cc0b82` |
| `src/roboforge/data/rgb_dataset.py` | `2562e7a2344c87ae40724c7a47df3031c4bc3ad2956fc321258a4464201ea8f3` |
| `src/roboforge/data/rgb.py` | `fde8a3786c4dd2e9e4773f1e6f710e92493ab1c84b4bbc7eb72b5dd49a75996f` |
| `src/roboforge/eval/rgb.py` | `a6b56ca78c8e6948ef039652e04b520805d961a38c80f51ae8756ef3bcb83bb2` |
| `scripts/eval_rgb.py` | `e31dba0a8b5d2ceac83cda28d3fcdb65d7bd0f6a80071d9aa21e105381e2afe2` |
| `scripts/rgb_train_queue.py` | `ddfa138356d3f1d607df0cb4c302d0f91d3d8d6d4b49492ca727e1a9a38379ed` |
| `scripts/build_rgb_dataset.py` | `2b5e783077c22c4981de885dc518dd96d5281642b24da9f2457bb16bdcdf1457` |
| `configs/models/rgb/bc_chunk_resnet18_lift.yaml` | `0384eb0b5010eec3b4c56e1a20f69376da0389dee0886b7a1eed46a74e2af576` |
| `configs/models/rgb/diffusion_resnet18_lift.yaml` | `b66b8cb9948c41fe4102e25d944cd8db442ac1e1851b3b2fbaa5c465bfc319fb` |
| `data/manifests/rgb/lift_rgb84_v1.json` | `58f11710eab017550070bdd0437ffc63513c7ac2bf0811dac0f4b597f1034292` |

## 11. Boundaries
- **No held-out access** of any kind until §9's preconditions hold. Evaluation CLIs accept
  only the validation suite (`check_suite_allowed`).
- **Level 1:** its runs, checkpoints, reports and frozen files are never modified, pruned or
  moved.
- **Budget:** paid compute only under A-03, within its cap. Evaluation stays local at $0.

## 12. Amendments
| Date | Section | Change | Reason | Runs affected |
|---|---|---|---|---|
| — | — | none | — | — |
