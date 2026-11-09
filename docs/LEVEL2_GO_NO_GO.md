# Level 2 GO / NO-GO: DP3-inspired point clouds on Lift (Session 9)

**Status: GO (owner decision, 2026-11-09, ≈ 10:50 EST).** The owner chose "GO, L4, $9 cap
(Recommended)" (approval **A-04**, `docs/EXPERIMENT_LEDGER.md`) and "Placement H5 primary
(Recommended)". The proposal below was written before that decision and is kept as it was
(`85eeb2e`). The experiment is fixed by `docs/LEVEL2_EXPERIMENT_FREEZE.md` and RESEARCH_PROTOCOL
§15. Every number below is measured, unless it is marked *estimate*.

## Outcome (executed 2026-11-09; `reports/session9/level2_results/`, analysed once at `8c728f4`)
- **Executed:**
  - 3 DP3-inspired point-cloud runs on the L4 under A-04, **$2.37 billed** of the $9 cap;
  - 36 validation evaluations, then the §7 selections (steps 30k, 40k, 60k, all 50/50);
  - the 12 preregistered placement evaluations, both arms;
  - the held-out test, 3 evaluations through the D-086 gate;
  - the frozen analysis.
- **H5 (primary): separated (−).** On `ood_cube_shift`, success was point cloud **56.7 %**
  (64, 56, 50 per seed) against RGB **98.0 %** (100, 98, 96).
  - pc − rgb = **−41.3 points, 95 % CI [−54.0, −28.0]**.
  - Both arms scored 100 % on `robustness_base`.
  - The point-cloud arm's drop was +43.3 points [+30.0, +56.7]; the RGB arm's was +2.0
    [+0.0, +6.7], not separated.
  - Failures of the point-cloud arm under the shift: 45 `no_grasp_near`, 16 `dropped`,
    4 `no_grasp_drift`.
- **Held-out (secondary):**
  - success: 150/150 for both arms, not separated;
  - censored length: 45.4 against 44.3 steps, +1.1 [−1.4, +3.4], not separated.
- **Reading.** Under this matched protocol, on Lift, this DP3-inspired point-cloud policy
  generalised *worse* to cube placements outside the demonstrations' square than the RGB
  policy. That is a negative result for H5, and it is reported as one.
  - It is one task, one encoder configuration and 3 seeds.
  - It is not a test of the official DP3 method, and not evidence that depth is uninformative.
  - The RGB arm's random-crop augmentation, and the point arm's world-frame coordinates with no
    augmentation, are part of what was compared.

MASTER_SPEC §7.3 requires items (a)–(f) below. If any of them is missing, the answer is
`NOT EXECUTED`. Level 1 (`docs/LEVEL1_COMPLETE.md`) and the RGB study
(`docs/RGB_EXPERIMENT_FREEZE.md`) are frozen, and nothing here changes them.

## Recommendation in one table
| Item | Proposal |
|---|---|
| Arm | **DP3-inspired** point-cloud diffusion (Ze et al., RSS 2024, arXiv:2403.03954), reimplemented with no DP3 code. Point encoder plus the RGB arm's diffusion head, unchanged. |
| Reference | The frozen **RGB diffusion** arm. Its 3 selected checkpoints and held-out reports are reused by hash and never retrained or rerun. |
| Observations | Fused world-frame XYZ from the depth of the RGB arm's two cameras, plus calibration. **Sensor-like: no segmentation, object pose or simulator label.** |
| Primary test | **H5**: point cloud − RGB success on the committed Lift `ood_cube_shift` starts (object placement), paired with `robustness_base`. |
| Secondary | The base − OOD drop per arm; held-out success and censored episode length against the reused RGB held-out records. |
| Compute | **L4, approval A-04, cap $9.** Expected spend about $3.2 at list price (*estimate*). The alternative is local CPU at $0. |
| Fallback | Stop Level 2 and record what was learned. Level 1 and the RGB study stay as published. |

## (a) Hypothesis and baseline
- **H5 (primary).** On Lift starts whose cube sits outside the demonstrations' placement square,
  a diffusion policy conditioned on a calibrated world-frame point cloud succeeds more often
  than the same diffusion policy conditioned on RGB images. Inputs, demonstrations and
  training are otherwise identical.
  - Outcome: success on the 50 `ood_cube_shift` starts × 3 training seeds per arm.
  - Decision rule: the §9 interval of pc − rgb is "separated (+)", "not separated" or
    "separated (−)". "Not separated" is reported as such, never as equivalence.
- **Baseline.** The RGB diffusion arm of `docs/RGB_EXPERIMENT_FREEZE.md`:
  - ResNet-18 at 84², two cameras (`agentview`, `robot0_eye_in_hand`);
  - selected checkpoints s0 step 55,000, s1 step 55,000, s2 step 60,000;
  - pinned by checkpoint, selection and held-out report hashes in
    `configs/experiments/level2_pointcloud.yaml`, which `tests/unit/test_level2_plan.py`
    verifies against `reports/session8`.
- **Why placement, not held-out.** RGB diffusion scored **150/150 on the Lift held-out test**
  (E8-14). An in-distribution comparison can therefore end only as "not separated" or as a
  point-cloud deficit (KI-034, the Lift ceiling). Placement has headroom: Level 1's low-dim
  diffusion fell from 99.3 % on `robustness_base` to 85.3 % on `ood_cube_shift` (E7-09). An
  RGB policy that has seen the cube only inside the training square may fall further.
  - Point clouds in a world frame are the representation DP3 credits with spatial
    generalization. That is a claim to test, not to assume.

## (b) Compute and dataset budget
### Dataset (built, not installed)
| Item | Value |
|---|---|
| Source | the 200 Lift PH demonstrations' simulator states, rendered in the canonical container |
| Build | `make pc-dataset` at `a20bfe3` (clean): **594 s** |
| Size | 9,666 frames × 512 points × 3 float32: **56.3 MB** gzip |
| Content hash | `39fa3e523f6735cd95f0da879d171e589ff65305ec5f7db7c5b9be1464d05329` |
| File hash | `6ab4b31d20b3cec694588534116cceb6045ea25bf16c3bfa22126e02272890b8` |
| Staged at | `runs/pc_data/lift/pc512_v1.{hdf5,manifest.json,coverage.json}` |

Installing it under `data/processed/` needs a recorded decision, in the D-079 pattern: on GO,
D-084 and `make pc-dataset-install`. The data is a derived work of the MIT-licensed robomimic
data, so it stays local and is never redistributed, like the RGB dataset.

### Training throughput
| Path | Measured | Per run (60k steps) | 3 runs |
|---|---|---|---|
| CPU, this host (M1 Pro, native), 1 thread | 0.168 s/step alone; 0.191 s/step each with 3 concurrent (`reports/session9/pc_benchmark/darwin-arm64-native/cpu_train_step.json`) | 3.2 h (*estimate*) | 3.2 h wall in parallel (*estimate*), $0 |
| L4, deterministic kernels | not measured. The bound comes from the Session 8 probe: compact-CNN diffusion took 45.8 ms/step, and deterministic kernels cost ×1.12 (measured on ResNet-18 diffusion), so ≈ **51 ms/step** (*estimate*). That compact CNN at 84² is heavier than this point encoder, so the true step should be faster. | ≈ 51 min training, 1–2 launches (*estimate*) | ≈ 2.9 L4-hours with start-up, ≈ **$3.2** at $1.116/h list (L4 + 4 cores + 16 GiB) (*estimate*) |

- The network has 1,317,511 parameters, 357,184 of them in the point encoder. Checkpoints are
  small: about 0.4 GB for 36 checkpoints (*estimate*), against 17 GiB free on disk.
- The first L4 segment re-measures ms/step, as E8-10 did. If it exceeds the estimate by more
  than 30 %, the queue stops and the owner is asked before the next launch.

### Evaluation (local canonical container, $0)
- RGB diffusion validation measured a median of **259 s** per 50-start evaluation (36
  evaluations, Session 8).
- A point-cloud decision adds 2 clouds at **37 ms** each (two depth renders 22.6 ms plus FPS
  12.5 ms; container). That is about 0.4 s per episode, so about 4.7 min per evaluation
  (*estimate*).
- Total: 36 validation + 12 placement + 3 held-out = 51 evaluations, ≈ 4.0 container-hours.
  That is ≈ 1 h wall with 4 containers at once (*estimate*).

### Modal credit
- Month-to-date usage after Session 8: about $13.51 ($5.45 at the D2 decision plus $8.06 for
  A-03), so about **$16.49** of the $30 monthly credit remains.
- A-04 at $9 would leave at least $7.49 even if the cap were reached.
- The gate (`roboforge.compute.approval`, D-049) refuses any launch for which committed spend
  plus that launch's upper bound exceeds the cap. A 1-h `train_gpu_pc` launch has an upper
  bound of **$4.54** (4 attempts × $1.116 plus a $0.08 build allowance). Finished launches
  count at measured cost. The launches run strictly one after another, so the last launch
  needs committed spend ≤ $4.46.
  - At the estimate, committed spend before the last launch is ≤ $3.2.
  - The $9 cap therefore leaves room for about one more L4-hour if the step is slower than
    estimated.

## (c) Reproducible depth and point-cloud extraction
Code: `src/roboforge/data/pointcloud.py`, decision D-083.
- **Depth:**
  - rendered from each demonstration's recorded simulator state, through the D-074 shadow
    model;
  - same two cameras, 84²;
  - MuJoCo `Renderer` depth mode with geom groups 1 and 2, with **sites hidden**;
  - converted to metric z-depth.
- **Back-projection:**
  - pixel centres `(j + ½, i + ½)`;
  - camera frame `((u − cx)d/fx, −(v − cy)d/fy, −d)`;
  - world `R p + t` from `cam_xmat` / `cam_xpos`.
- **Valid pixels:** finite, `0 < d < 0.99·far`.
- **Crop:** a fixed world box, x, y ∈ [−0.25, 0.25] m and z ∈ [0.805, 1.20] m. The table top
  is at 0.800 m, so the table is removed and the cube and gripper kept.
- **Sampling:** deterministic farthest-point sampling to N = 512 over the fused cloud of both
  cameras:
  - start index 0, in float64;
  - ties go to the lowest index;
  - cyclic padding if fewer than 512 points;
  - an empty crop raises.
- **Specification:** `roboforge.point_cloud_spec/1`, sha256 `973e27fe…`. The dataset and the
  closed-loop policy both refuse a different specification.
- **Determinism:**
  - re-rendering is bitwise equal;
  - a live state and a restored state give bitwise-equal clouds;
  - **the closed-loop cloud equals the dataset cloud** for the same state
    (`tests/simulator/test_pc_closed_loop.py`).

## (d) Verified camera projection and frames
Evidence: `make pc-feasibility` at **`b16b692`** (clean), canonical container
`linux-aarch64-container-1f685638f28c`.
`reports/session9/pc_feasibility/linux-aarch64-container-1f685638f28c/`.

| Check | Result |
|---|---|
| Depth convention | reversed-Z (`mjDEPTH_ZEROFAR`). near = 0.01061 m, far = 530.49 m (extent 10.61 × znear 0.001 / zfar 50). Perspective; float32 metres. |
| Depth vs exact `mj_ray`, 30 frames × 2 cameras | **358,383 interior pixels: max error 3.6 µm, p99 0.26 µm, 0 over 1 mm.** 64,977 boundary pixels reported, not tested (37 over 1 mm). |
| Known point: cube top-face centre → pixel → back | 95 of 120 applicable (visible), all inside the pixel's footprint. Max distance 5.3 mm at 84² and 1.7 mm at 256². Reprojection 3e-14 px; plane residual 1.4e-7 m. |
| Surfaces | Cube points ≤ 1.9e-5 m from the cube's box; table points ≤ 2.1e-7 m from its plane. |
| Invariants | Re-render bitwise; live equals restored bitwise; evaluation model equals demo model (0 differing in-box pixels outside the cube's 2-px band); sites absent from depth. |
| Overlays | `lift_overlay_demo_0_t{000,029,058}.png` (points projected on the RGB frame) and `lift_scatter_*.png` (3-D scatter). |

- **Interior and boundary pixels.** An interior pixel satisfies three conditions:
  - the ray hits the geom that segmentation shows there;
  - its 8 neighbours show the same geom;
  - the exact ray depth is locally planar.

  At a boundary pixel the multisampled depth mixes two surfaces ("flying pixels", as on a
  real sensor). Boundary pixels are kept in the clouds and not filtered.
- **Disclosure.** The first container run, at `a20bfe3`, did not yet have the planarity
  condition. It failed on **1 of 368,570** interior pixels: 9.7 mm on the robot pedestal mesh,
  at a silhouette inside one geom. The condition uses exact ray depth only, with no labels. It
  moved 10,187 pixels (2.8 %) from interior to boundary. That run's summary is kept in
  `reports/session9/pc_feasibility/superseded_a20bfe3/`.

### Coverage (demonstrations and every evaluation start set, container)
| Set | Frames / starts | Points in crop (min) | Points on cube (min / p5 / median) |
|---|---|---|---|
| All demo frames (dataset build) | 9,666 | 1,693 (never below 512) | 13 / 26 / 46; 0 frames with none; lifted frames min 20 |
| Validation starts | 50 | 1,704 | 90 / — / 108 |
| `robustness_base` | 50 | 1,713 | 91 / — / 108 |
| `ood_cube_shift` | 50 | 1,711 | 95 / — / 109; max cube radius 0.070 m, all inside the box |

### Privilege statement
The clouds use only depth from two cameras plus their calibration. The wrist camera's pose
comes from robot kinematics, the same as a real calibrated arm. The workspace box is fixed.
- Segmentation, object poses and simulator labels are used **only** in the diagnostics
  (`src/roboforge/data/pc_checks.py`), never in a cloud, the dataset or a policy input.
- The run is **not** "privileged geometry".
- The world frame is a choice of coordinates, available to any calibrated robot. It is still
  part of the treatment, together with depth and the point encoder. H5 tests the package, not
  depth alone.

## (e) Comparison protocol
**Matched to the RGB arm** (`tests/unit/test_pc_train.py` checks the configs):
- the same 200 demonstrations and the same train/valid split;
- raw 9-D proprioception;
- action normalization, T_o = 2, K = 12 checkpoints, horizon and action chunk;
- U-Net (64, 128), DDIM-10 inference, EMA (power 0.75);
- AdamW lr 1e-4, batch 64, 60k steps, seeds 0–2;
- decision rate and rendering rule (`n_renders = 1 + 2 · decisions`);
- CPU, 1-thread evaluation with policy seed 0 and repeat 0;
- the canonical container;
- the §7 selection rule over all 12 checkpoints on the 50 validation starts.

**What differs:**
- the observation: a point cloud instead of RGB;
- the encoder: DP3-inspired, a per-point MLP 64-128-256 with LayerNorm, max pool and a linear
  layer to 64 with LayerNorm. The RGB arm used a ResNet-18 to 64.
- no augmentation: the RGB arm's random 76-px crop has no point-cloud counterpart.

**Order (fixed by the freeze before the first launch):**
1. Train 3 runs; validate every checkpoint; select one per seed by the §7 rule.
2. Then run the 12 placement evaluations, each once:
   - `robustness_base` and `ood_cube_shift`;
   - for each arm's 3 selected checkpoints.

   The RGB arm's placement results therefore do not exist while the point-cloud arm is
   trained or selected.
3. Re-gate the held-out lock in the D-081 pattern:
   - a new module and a recorded decision;
   - no frozen file changed;
   - a guarded test.
4. Run the held-out test once (3 evaluations).
5. Run the analysis once.

**Analysis** (RESEARCH_PROTOCOL §9; preregistered as a new §15 before any Level 2 rollout):
- A hierarchical paired bootstrap: B = 10,000, seed 20261106.
  - Starts are paired by index: OOD start *i* is derived from base start *i*.
  - Each arm's base and OOD outcomes share one seed draw, as in `analyze_robustness`.
  - Draw order: states, pc_diffusion seeds, rgb_diffusion seeds.
- **Primary:** pc − rgb on `ood_cube_shift`.
- **Secondary:**
  - each arm's base − OOD drop, and the difference of the drops;
  - held-out success pc − rgb, reusing the committed RGB records;
  - held-out censored episode length.
- **Power (*estimate*).** With 3 seeds × 50 starts, binomial noise alone gives a half-width of
  about ±8 points near 85 % success; seed variance widens it. Only a large effect will
  separate, and "not separated" is a likely honest outcome.
- Every result is labelled **DP3-inspired**. Failures are classified with the committed
  taxonomy.

Out of scope, and declared so: camera-pose perturbations, `ood_arm_offset`, Can, and any
tuning of the point-cloud arm. There is one configuration, fixed before training.

## (f) Fallback
| Failure | Action |
|---|---|
| Owner chooses NO-GO | Record the skip here and do not train. Session 10 proceeds. |
| L4 unavailable, or step time > 1.3 × the estimate | Stop the queue and ask the owner: continue under the cap, or switch to local CPU with the device mismatch declared. |
| Training error, divergence or non-finite loss | Stop. No tuning reruns. Record a negative engineering result with logs. |
| Validation success is low | Still select by §7 and run the preregistered evaluations once. Report a negative result, never relabelled. |
| Gate or determinism check fails | Fix the engineering only (RESEARCH_PROTOCOL engineering reruns), with a decision. Never touch frozen results. |

In every case: Level 1 and the RGB study remain the published results. Never gate RoboForge
on Level 2 (MASTER_SPEC Session 9).

## Blockers
**Technical**

| ID | Blocker | Status |
|---|---|---|
| T1 | Native macOS (CGL) lacks `ARB_clip_control`, so reversed-Z depth precision is poor. | Resolved: only the container (OSMesa) produces evidence. Native depth is development-only. New KI. |
| T2 | robosuite hides visualization sites by alpha, but they still write depth. A 10 m grip-site cylinder would enter the wrist cloud. | Resolved: `sitegroup` off in the depth renderer (D-083), with an invariant test. New KI. |
| T3 | Flying pixels at depth discontinuities. | Accepted as sensor-like: 37 of 64,977 boundary pixels exceed 1 mm. Not filtered. |
| T4 | The cube is a small part of the cloud: at least 13 of 512 points (p5 26) in demo frames, at least 90 at the starts. | A risk to performance, not a blocker. The encoder must find a sparse object. |
| T5 | L4 step time is an estimate; determinism across different L4 hosts is unverified (as in Session 8). | The first segment re-measures. Same-host bitwise determinism was verified in E8-06. |
| T6 | GPU memory | Not a blocker: 1.3 M parameters. The larger RGB model peaked at 1.2 GiB. |

**Licensing**
- robosuite is MIT and MuJoCo Apache-2.0; robomimic data is MIT (`THIRD_PARTY_NOTICES.md`,
  `docs/DATA_PROVENANCE.md`). They are used as in Level 1.
- No DP3 code, weights or data are used. The encoder is reimplemented from the paper, which is
  cited and added to `THIRD_PARTY_NOTICES.md` on GO.
- No blocker.

**Ethical**
- Simulation only, on public demonstration data, with no people and no personal data.
- The risk is overclaiming. Mitigations:
  - the "DP3-inspired" label;
  - the privilege statement;
  - preregistration before any rollout;
  - "not separated" is never reported as "equal" or as "better".

**Resource**
- Modal credit: about $16.49 left; A-04 caps the spend at $9.
- Disk: 17 GiB free (the disk-space constraint). Level 2 adds about 1 GB (*estimate*).
- Local CPU: at most 4 containers at once. The full test suite never runs alongside a
  container (KI-043).

## Decision requested from the owner
1. **GO or NO-GO, and the compute path:**
   - **L4 (recommended).** A new approval **A-04** for function `train_gpu_pc`:
     - 3 runs (seeds 0–2, 60k steps);
     - 1 L4, 4 cores, 16 GiB;
     - ≤ 1 h per launch, strictly one after another;
     - `max_containers=1`, `retries=0`, preemptible, ≤ 4 attempts per launch;
     - resume launches as needed; no tuning runs;
     - **cap $9**, expected about $3.2 at list price (*estimate*).
   - **Local CPU, $0.** About 3.2 h wall (*estimate* from the native host). The device
     mismatch to the RGB arm is declared, and it uses 3 cores that would otherwise run
     container validation.
   - **NO-GO.**
2. **Placement test.** Add the preregistered `ood_cube_shift` / `robustness_base` evaluation
   for both arms as the primary H5 (recommended), or compare on held-out only, where RGB is at
   the ceiling.

**On GO**, before any launch:
- D-084: install the point-cloud dataset;
- the RESEARCH_PROTOCOL §15 amendment (H5, and the device for the chosen path);
- `docs/LEVEL2_EXPERIMENT_FREEZE.md`, committed;
- the A-04 row in `docs/EXPERIMENT_LEDGER.md`.
