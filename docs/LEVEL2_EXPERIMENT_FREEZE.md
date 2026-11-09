# Level 2 experiment freeze: DP3-inspired point clouds vs RGB on Lift (Session 9)

**Status:** FROZEN by the Git commit that adds this file. It is written before any Level 2
training run or Level 2 rollout (MASTER_SPEC Session 9 step 4). Every item below is fixed:
no configuration, budget, selection rule, suite, evaluation order or analysis may change
after a training run starts, except by a dated amendment in §11 that states which runs it
affects.
- Owner decision: **GO, L4, $9 cap** and **placement H5 primary** (2026-11-09; approval
  **A-04**; `docs/LEVEL2_GO_NO_GO.md`).
- Protocol: RESEARCH_PROTOCOL v1 with its §13 amendment of 2026-11-09 and **§15** (the
  preregistration this file repeats with hashes).

Level 1 (`docs/LEVEL1_COMPLETE.md`) and the RGB study (`docs/RGB_EXPERIMENT_FREEZE.md`) are
frozen. This is a separate, separately labelled advanced study (MASTER_SPEC §7.3). Nothing
here changes a Level 1 or RGB configuration, result or label. Its numbers are never pooled
with theirs. Every result is labelled **DP3-inspired** (Ze et al., RSS 2024,
arXiv:2403.03954), never a DP3 reproduction.

## 1. Question and comparison
- **H5 (primary):** on Lift starts whose cube lies outside the demonstrations' placement
  square, does a diffusion policy conditioned on a calibrated world-frame point cloud succeed
  more often than the same diffusion policy conditioned on RGB images?
- **Arms:** `pc_diffusion` (new, trained here) against `rgb_diffusion`, the RGB freeze's
  diffusion arm. The RGB arm is reused as committed and never retrained, reselected or rerun on
  the held-out starts.
- **Why placement is primary:** RGB diffusion scored 150/150 on the Lift held-out test
  (E8-14). In-distribution success cannot show a point-cloud advantage (KI-034).

## 2. Observation (point-cloud arm)
- Fused world-frame XYZ from the depth of `agentview` and `robot0_eye_in_hand` at 84², the
  RGB arm's two cameras, under D-083:
  - geom groups 1–2, sites hidden;
  - metric z-depth;
  - pixel-centre back-projection with logged intrinsics and extrinsics.
- **Crop:** the fixed box x, y ∈ [−0.25, 0.25] m, z ∈ [0.805, 1.20] m.
- **Sampling:** N = 512 by deterministic farthest-point sampling (start index 0).
- Specification `roboforge.point_cloud_spec/1`, sha256
  `973e27feb70c2bdf1befc92789962fc7b2e57ac401bda44e7cead477badb01a7`.
- Raw 9-D proprioception, as in the RGB arm. T_o = 2.
- **Privilege: sensor-like.** Only depth and calibration, with the wrist pose from
  kinematics. No segmentation, object pose or simulator label enters a cloud.
- Verified: `reports/session9/pc_feasibility/linux-aarch64-container-1f685638f28c/` at
  `b16b692`, where calibration, invariants and coverage pass.

## 3. Encoder (point-cloud arm only)
- `src/roboforge/models/point_encoder.py`, DP3-inspired:
  - points normalized by the fixed box (centre, half-extent);
  - per-point MLP 3 → 64 → 128 → 256, with LayerNorm and ReLU;
  - max pool over points;
  - Linear to 64, then LayerNorm;
  - concatenated with the raw proprioception (73 per frame).
- No augmentation. The RGB arm's random 76-px crop has no point-cloud counterpart.
- 1,317,511 parameters in the policy, 357,184 of them in the encoder.

## 4. Training (identical to the RGB diffusion arm except §2–§3)
- **Config:** `configs/models/pointcloud/diffusion_dp3_lift.yaml`. Its model, loss, optim,
  train and device blocks equal the RGB diffusion config verbatim (`tests/unit/test_pc_train.py`):
  - U-Net (64, 128); DDPM K = 100, ε-prediction, masked MSE; DDIM-10 inference;
  - EMA power 0.75 as the evaluated network; T_p/T_a = 16/8;
  - AdamW lr 1e-4, batch 64, **60,000 steps**, checkpoints every 5,000 (K = 12);
  - seeds 0, 1, 2.
- **Run IDs:**
  - `s9_pc_diffusion_dp3_lift_seed0_8fd35613`
  - `s9_pc_diffusion_dp3_lift_seed1_a66a2758`
  - `s9_pc_diffusion_dp3_lift_seed2_402a1f24`
- **Trainer:** `PCTrainer` subclasses the frozen `RGBTrainer` and inherits its step,
  checkpoint cadence, resume and run loop unchanged.
- **Device:** one NVIDIA L4 on Modal (`train_gpu_pc`), under the 2026-11-08 deterministic
  regime (§13). The run manifest records host, GPU and driver per segment.
- **Launches:**
  - `scripts/pc_train_queue.py --budget-approved A-04`, strictly one after another;
  - ≤ 1 h per launch, `max_containers=1`, `retries=0`, preemptible, ≤ 4 attempts per launch,
    resumes as needed.
- **Stop rule:** the queue stops and the owner is asked if a segment trains slower than
  66.7 ms/step, 1.3 × the 51.3 ms/step estimate.
- **No tuning:** one configuration, fixed by this file.

## 5. Data identity
| Item | Value |
|---|---|
| Demonstrations | Lift PH, 200 demos, the Level 1 train/valid split (low-dim processed dataset pinned by the point-cloud file) |
| Point-cloud dataset | `data/processed/robomimic_v1.5_ph/lift/pc512_v1.hdf5` (D-084), content `39fa3e523f6735cd95f0da879d171e589ff65305ec5f7db7c5b9be1464d05329`, file `6ab4b31d20b3cec694588534116cceb6045ea25bf16c3bfa22126e02272890b8` |
| Its manifest | `data/manifests/pointcloud/lift_pc512_v1.json` (hash in §9) |
| Built | canonical container, `a20bfe3` clean, 594 s |

## 6. Checkpoint selection (RESEARCH_PROTOCOL §7, unchanged)
- `make pc-validate PC_RUN=<run>` evaluates every checkpoint on the 50 validation starts
  (manifest `0b1368fae0d5e967955ad7b5c3a6191001bbad2758becad537412b45e71c8785`) on the
  canonical container.
- `scripts/eval_pc.py select` applies the rule:
  1. the highest success count;
  2. then the lower mean first-success step;
  3. then the earlier checkpoint.
- It uses only canonical, complete, verified, error-free summaries whose checkpoint hash
  matches the training manifest.
- The selection artifacts are `reports/session9/pc_selection/<run_id>.json`.

## 7. Evaluations, each run once, in this order
All of them run on the canonical CPU container with 1 torch thread, `policy_seed = 0` and
R = 1. There are at most 4 containers at once, and the full test suite never runs alongside
them (KI-043).

1. **Validation and selection** (§6): 36 evaluations.
2. **Placement:** `make pc-placement PC_ARM=pc|rgb PC_SEED=s PC_SUITE=…`.
   - Both arms' 3 selected checkpoints run on every start of `robustness_base`
     (`b1ffb682c2c94128c7c773f8899edba42f04d8e2631f3ab63f7f7085cb05855d`) and `ood_cube_shift`
     (`c6083afc0212bf06f6eb98e7fc5522fae4ed8fae49bb89d8faf1e74d65d1b7ac`): 12 evaluations.
   - **None starts before all 3 point-cloud selection artifacts are committed.**
   - The RGB arm's selections come from `configs/experiments/level2_pointcloud.yaml`, pinned
     by checkpoint, selection and held-out report hash, and verified against
     `reports/session8` by `tests/unit/test_level2_plan.py`.
   - Reports go to `reports/session9/level2_placement/<arm>/`. Each is written once and never
     overwritten.
3. **Held-out test:** the 3 point-cloud selections on Lift's 50 held-out starts (schedule
   manifest `e02498651b9d5bb88b6aec0870c5e3277cadaaeaf0e325abf86c79e89b1f22da`), 3
   evaluations.
   - **Preconditions, in order. All must be committed before the first held-out Level 2
     rollout:**
     1. the 3 selection artifacts;
     2. a recorded decision that **re-gates the held-out lock** on this file, in the D-081
        pattern: a new module, no frozen file changed;
     3. a test proving three things:
        - the lock is shut for every point-cloud evaluation until (1) and (2) hold;
        - the RGB gate and `heldout.evaluate_entry` still refuse what they refused;
        - the frozen RGB held-out reports are unchanged.
   - The RGB arm's 3 committed held-out reports are reused and **never rerun**.
4. **Analysis**, once (§8).

**Never:** tuning, reselection or reruns after any of these. A later access is labelled
exploratory or replication.

## 8. Analysis (RESEARCH_PROTOCOL §15.4)
- **Primary (H5):** `ood_cube_shift` success, `pc_diffusion` − `rgb_diffusion`.
- **Secondary:**
  - each arm's paired drop (`robustness_base` − `ood_cube_shift`), and pc − rgb in the drop;
  - held-out success, pc − rgb;
  - held-out censored episode length, pc − rgb (first-success step, else 400).
- **Bootstrap:** the §9 hierarchical paired bootstrap, B = 10,000, seed `20261106`.
  - State indices are resampled once per replicate and shared by both arms and both sets;
    OOD start *i* is paired with base start *i*.
  - Each arm's base and OOD outcomes share one seed draw.
  - Draw order: states, `pc_diffusion` seeds, `rgb_diffusion` seeds.
  - Held-out uses one separate bootstrap with the same settings, over the manifest order.
- **Wording:**
  - "separated (+/−)" when the 95 % interval excludes 0, otherwise **"not separated"**, which
    is never reported as equivalence;
  - no p-value threshold;
  - with 3 seeds the seed resampling is coarse, and the report says so;
  - power: about ±8 points from binomial noise alone near 85 % (an estimate).
- **Also reported:** per checkpoint k/50 with a Wilson 95 % interval; failure categories
  (`configs/eval/failure_taxonomy.yaml`, unchanged); inference latency; point-cloud
  throughput; training wall time and billed cost.
- **The three kinds of negative result stay apart:** engineering failures; a run that fails
  validation; a difference that is not separated.

## 9. Code identity at the freeze
SHA-256 of the files that define this experiment (a change after the freeze needs a §11
amendment). The inherited RGB files (`train/rgb.py`, `eval/rgb.py`, `compute/rgb_train.py`)
are pinned by RGB freeze §10.

| File | SHA-256 |
|---|---|
| `src/roboforge/data/pointcloud.py` | `05e7731db9ac213019f4ce0aa1b2d672a9869d7227b2b3ac0f970b2187f622f2` |
| `src/roboforge/data/pc_dataset.py` | `97046eea59051908336e8d18ea82c2fe0886c2599da6cdb5443362638c7090d9` |
| `src/roboforge/models/point_encoder.py` | `c9b285abde14bf6c682e55849dfc82eb1d54b7fcf655d5fe96dd75f3d815a4cd` |
| `src/roboforge/train/pointcloud.py` | `c957e11df0ae1e2559a5e6873ad0daea1a21406e5b5f0a0f9e2afde51cdcb953` |
| `src/roboforge/compute/pc_train.py` | `214c3ba303118fe865fc829912a6c34f509b085be8fc0330960902f1042d96d4` |
| `src/roboforge/eval/pointcloud.py` | `ce2a65d63f32852825e359b1f86741aa70cb57547c4ad72470e2c4b1bea241d4` |
| `scripts/eval_pc.py` | `55ed3024d34ef5abb947c6f7b7ae8c70a5b6e42d4008ac8105a7fa135a9ff2c8` |
| `scripts/pc_train_queue.py` | `53aac5352857d3b126b09fe15386d67c9902b258f0b9a259b70409151a49055b` |
| `scripts/build_pc_dataset.py` | `cc9ff778d8627b695e52047fee062b10dc378b846fffb4d6832f6721c77af483` |
| `configs/models/pointcloud/diffusion_dp3_lift.yaml` | `b56f515dfd861568c27d5c89ebaa9878bb87452a634daebf776f58174bfe802a` |
| `configs/experiments/level2_pointcloud.yaml` | `d91a546e1774779b27a16edbd5f33b0a2d4c11e6d68c71c13e2dff7e0a3e5c9d` |
| `data/manifests/pointcloud/lift_pc512_v1.json` | `a7694e8ae245bff8ab8f6fbb81f2fa749f8d77df0d48985bf29324ea1bb75013` |

## 10. Boundaries
- **No held-out access** of any kind until §7.3's preconditions hold. The point-cloud
  evaluation CLIs accept only validation and the two placement suites.
- **Level 1 and the RGB study:** their runs, checkpoints, reports, freezes and frozen files
  are never modified, pruned or moved. `make eval-final`, `make analyze-final`,
  `make rgb-heldout` and `scripts/analyze_rgb_heldout.py` are never rerun.
- **Budget:** paid compute only under A-04, within its $9 cap. Evaluation stays local at $0.

## 11. Amendments
| Date | Section | Change | Reason | Runs affected |
|---|---|---|---|---|
| — | — | none | — | — |
