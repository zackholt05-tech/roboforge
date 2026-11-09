<!-- parsable status header -->
```yaml
CURRENT_SESSION: 9
SESSION_STATUS: PASS            # Session 9 GO path: DP3-inspired point-cloud policy trained (3 seeds, L4, A-04), matched closed-loop evaluation, calibrated overlays; H5 separated (-)
NEXT_SESSION: 10                # RoboForge Observatory; docs/HANDOFF.md "# Session 9 Handoff" §10, fresh context
BLOCKERS: none                  # open items: KI-001, KI-008, KI-011, KI-021, KI-023, KI-024, KI-028, KI-033, KI-034, KI-035, KI-042; KI-044, KI-045, KI-047 mitigated; KI-049 documented
LAST_VERIFIED_COMMIT: 620c541   # code; lint clean, 478 tests (run alone), freeze-check OK, RGB and Level 2 gates open on host; the STOP commit adds docs only
PROTOCOL_COMMITS: {v1: 8b08b95, diffusion_amendment: 3aed462, main_budget_amendment: ec9626b, lift_search_space: 6dce01b, can_search_space: 96aa972, section14: 2859d04, rgb_cuda_amendment: e433f11, section15_level2: b93c50c}
EVIDENCE_COMMITS: {level1_freeze: bf0a0fa, level1_heldout_results: 97cc48a, rgb_freeze: d9c6217, rgb_gate_d081: 378ea43, rgb_heldout: bc054bc, rgb_results: 3b23255, pc_feasibility: 85eeb2e, level2_freeze: b93c50c, pc_selections: b1f4da2, level2_gate_d086: 08c3aa8, level2_placement: 4ebebc1, level2_heldout: a031e88, level2_results: 620c541}
EVALUATION_PLATFORM: linux-aarch64-container-1f685638f28c   # D-017, configs/eval/platform.yaml
FINAL_TEST: "Level 1 TESTED_ONCE (never rerun). RGB study TESTED_ONCE under D-081 (never rerun). Level 2 TESTED_ONCE under D-086 (a031e88; analysis 620c541; never rerun). Any later held-out access is exploratory or replication"
RESOURCE_SPEND: {usd_billed_total: 10.61, session9_usd_billed: 2.37, notes: "A-01 $0.148; A-02 $0.0317; A-03 $8.06; A-04 Level 2 training $2.37 (5 launches); all four approvals closed; Modal month-to-date $15.89 of $30"}
APPROVED_LIMIT: {A-03: "closed, $8.06 of $15", A-04: "closed, $2.37 of $9", open: none, local: "$0"}
UPDATED: 2026-11-09
```

# RoboForge Status

## Session 9: Level 2, DP3-inspired point clouds on Lift (PASS, GO path)
| Step (Session 9 prompt) | Status | Evidence |
|---|---|---|
| 1. `docs/LEVEL2_GO_NO_GO.md`, (a)–(f), blockers, Lift ceiling, placement proposal | DONE (`85eeb2e`); outcome section added | the document |
| 2. Feasibility, no training: depth via D-074, metric conversion, back-projection, known-point test, overlays, crop and sampling, privilege, throughput, dataset size | DONE, canonical container at `b16b692`: calibration, invariants and coverage pass; superseded `a20bfe3` run disclosed | E9-01–E9-05, D-083, KI-045, KI-046 |
| 3. GO/NO-GO and budget, then STOP and ask; approval row | DONE: owner chose **GO, L4, $9 cap** and **placement H5 primary** (A-04) | E9-06 |
| 4. Freeze before the first training run; train, validate, select; re-gate; held-out once; analysis once | DONE: freeze `b93c50c`, §15 preregistration; 3 runs ($2.37); 36 validation evaluations; selections 30k / 40k / 60k; 12 placement evaluations; D-086 gate; 3 held-out evaluations; analysis | E9-07–E9-12, D-084–D-086 |
| 5. STOP: docs, lint, full suite alone, freeze-check, commit, Session 10 prompt | DONE | this file, `docs/HANDOFF.md` |

**Result (Lift; DP3-inspired; sensor-like depth and calibration only; `reports/session9/level2_results/tables.md`):**

| Arm | `robustness_base` | `ood_cube_shift` | Held-out |
|---|---|---|---|
| Point cloud (DP3-inspired, 512 pts) | 100.0 % [100, 100] | **56.7 % [43.3, 70.0]** (64, 56, 50) | 100.0 % (150/150) |
| RGB diffusion (frozen Session 8 arm) | 100.0 % [100, 100] | **98.0 % [93.3, 100]** (100, 98, 96) | 100.0 % (reused, never rerun) |

- **H5 (primary): pc − rgb on `ood_cube_shift` = −41.3 points, 95 % CI [−54.0, −28.0],
  separated (−).** The point-cloud policy generalised worse to cube placements outside the
  demonstrations' square.
- **Drops (base − OOD):**
  - point cloud: +43.3, separated;
  - RGB: +2.0, not separated.
- **Held-out:**
  - success: not separated;
  - censored length: +1.1 steps [−1.4, +3.4], not separated.
- **Point-cloud failures under the shift:** 45 `no_grasp_near`, 16 `dropped`,
  4 `no_grasp_drift`.
- The three kinds of negative result stay apart:
  - engineering: complete; three launch and analysis faults were fixed and recorded
    (KI-047–KI-049);
  - validation: every run reached 50/50;
  - test: H5 is separated in the negative direction.
- **Scope:** one task, one encoder configuration, 3 seeds. It is not a DP3 reproduction.

## Session 9 acceptance (MASTER_SPEC Session 9, GO path)
| Criterion | Status | Evidence |
|---|---|---|
| GO/NO-GO document with blockers; separate owner approval | DONE: `docs/LEVEL2_GO_NO_GO.md`; A-04 | E9-06 |
| Depth and point clouds aligned to training episodes and simulator state; calibration and reprojection checked numerically and visually | DONE: depth vs `mj_ray` max 3.6 µm on 358,383 interior pixels; known-point round trip; overlays `lift_overlay_demo_0_t{000,029,058}.png`; the closed-loop cloud equals the dataset cloud (test) | E9-04, D-083 |
| Compact point encoder + conditional diffusion vs a matched RGB diffusion reference on identical tasks and held-out starts | DONE: the same head, budget, demonstrations, proprioception, horizons and starts; RGB arm frozen and reused | freeze `b93c50c`, E9-07–E9-11 |
| Spatial-shift evaluation, physically meaningful and implemented; privilege logged | DONE: preregistered `ood_cube_shift` vs `robustness_base` for both arms; sensor-like (no segmentation, object pose or label) | E9-10, freeze §2 |
| Actual success metrics, point-cloud schema, compute costs, failures and negative result; labelled DP3-inspired | DONE: H5 separated (−); $2.37; failure taxonomy; schemas `roboforge.point_cloud_spec/1`, `roboforge.pc_dataset/1` | E9-12, COMPUTE_PLAN §8 |
| Level 1 and the RGB study unchanged | DONE: freeze-check OK; the RGB gate is still open; no frozen file edited | D-086 tests |

## Session 8: RGB observation-modality study (PASS)
| Step (Session 8 prompt) | Status | Evidence |
|---|---|---|
| Prerequisites; KI-040 (smoke-sim built a held-out start) fixed | DONE | LEDGER E8-01, E8-02 |
| 1. RGB feasibility, no training: alignment, projection test, clip, by-eye sample, cost at 84/128 | DONE, every check passes on the canonical container | E8-03–E8-05, `reports/session8/rgb_feasibility/README.md`, D-074 |
| 2. Gate D proposal; owner decision; approval row | DONE: Gate D "D: GPU probe ≤ $1" (A-02, D-076), run and closed at **$0.0317 billed**. Gate D2 "**L60**" (A-03, $15 cap, D-078) after the probe | E8-06, E8-07, COMPUTE_PLAN §6–§7, `reports/session8/modal/README.md` |
| Protocol amendment (deterministic CUDA training) | DONE: RESEARCH_PROTOCOL §13, 2026-11-08 | `e433f11` |
| RGB dataset (Lift, both cameras, 84²) | DONE: built on the container, installed with hash checks (D-079) | E8-08, `data/manifests/rgb/lift_rgb84_v1.json` |
| 3. `docs/RGB_EXPERIMENT_FREEZE.md` before the first RGB training run | DONE: `d9c6217`; §10 hashes checked against the commit | D-080 |
| 4a. Train the 6 frozen runs (visual BC-chunk vs visual diffusion, seeds 0–2) | DONE: 6/6 at 60k, 12 launches, **$8.06 billed**, 0 failures | E8-10 |
| 4b. Validate 12 checkpoints per run; select per §7 | DONE: 72/72 canonical; all runs reach 50/50 (ceiling); selections below | E8-11, `reports/session8/rgb_selection/` |
| 4c. Lock re-gate decision + test; held-out test once; analysis | DONE: D-081 gate and test (`378ea43`); rehearsal bitwise equal to validation; 6 held-out evaluations once (`bc054bc`); frozen analysis once (`3b23255`) | E8-12–E8-15 |
| Failure videos (MASTER_SPEC S8) | DONE: 5 rule-selected MP4s from recorded states | E8-16, `reports/session8/rgb_videos/` |

**RGB selections (development numbers; the held-out test is the unbiased estimate):**

| Family | Seed | Run ID | Selected step | Validation | Mean first-success step | Checkpoint SHA-256 |
|---|---|---|---|---|---|---|
| rgb_bc_chunk | 0 | `s8_rgb_bc_chunk_resnet18_lift_seed0_73a80bad` | 5000 | 50/50 | 42.38 | `5f8af84e8d0f…` |
| rgb_bc_chunk | 1 | `s8_rgb_bc_chunk_resnet18_lift_seed1_c49230ce` | 5000 | 50/50 | 43.08 | `7931609f5e07…` |
| rgb_bc_chunk | 2 | `s8_rgb_bc_chunk_resnet18_lift_seed2_2e677ad6` | 55000 | 50/50 | 44.04 | `b4f191054117…` |
| rgb_diffusion | 0 | `s8_rgb_diffusion_resnet18_lift_seed0_dddc7f1c` | 55000 | 50/50 | 44.06 | `6d919dfb989c…` |
| rgb_diffusion | 1 | `s8_rgb_diffusion_resnet18_lift_seed1_51deeb3e` | 55000 | 50/50 | 43.22 | `5a14e1510f4b…` |
| rgb_diffusion | 2 | `s8_rgb_diffusion_resnet18_lift_seed2_ec213b6a` | 60000 | 50/50 | 43.90 | `048031b33f05…` |

**RGB held-out results (Lift, 3 seeds × 50 held-out starts, run once; `reports/session8/rgb_results/tables.md`):**

| Family | Success | 95 % bootstrap CI | Per seed | Median `act` latency |
|---|---|---|---|---|
| Visual diffusion (ResNet-18, DDIM-10) | **100.0 %** | [100.0, 100.0] | 100, 100, 100 | 166–187 ms |
| Visual BC-chunk (ResNet-18) | **99.3 %** | [96.0, 100.0] | 98, 100, 100 | 63–66 ms |

- **Diffusion − BC = +0.7 points, 95 % interval [+0.0, +4.0]: not separated.**
- With 3 seeds the seed-level resampling is coarse.
- The one failure is BC s0 on start 49, `no_grasp_near`.
- The three kinds of negative result stay apart:
  - engineering: none, and engineering is complete;
  - validation: no family failed it;
  - held-out: the difference is not separated.
- Context only, not a declared comparison: Level 1's low-dim policies on the same starts scored
  BC-MLP-chunk 98.0 % and diffusion 97.3 %. Visual control from two 84² cameras plus 9-D
  proprioception sits at the same Lift ceiling. Lift saturates for every arm (KI-034), so it cannot
  rank visual BC against visual diffusion.

## Session 8 acceptance (MASTER_SPEC Session 8)
| Criterion | Status | Evidence |
|---|---|---|
| Raw snapshots, renderer, RGB extraction, time-step alignment, camera metadata, storage; a small aligned clip with ground-truth actions | DONE | E8-03–E8-05, E8-08, D-074, D-079 |
| Owner budget approval and an RGB experiment freeze (fair visual BC vs visual diffusion, Lift) | DONE: A-02, A-03; freeze `d9c6217` | D-076, D-078, D-080 |
| Image-encoder policies with a consistent backbone, frozen input keys, camera settings, identical evaluation states | DONE: shared ResNet-18 GN initialization per seed, identical augmentation and budget, deterministic CUDA | E8-10, freeze §3–§6 |
| Actual RGB closed-loop results and representative failure videos | DONE: 72 validation and 6 held-out evaluations with full traces; 5 videos | E8-11, E8-14–E8-16 |
| Level 1 metrics unchanged | DONE: freeze-check OK; no Level 1 file edited; `heldout.evaluate_entry` refuses all 26 Level 1 entries (test) | E8-12 |
| Costs recorded | DONE: $8.09 billed in Session 8; evaluation local, $0 | COMPUTE_PLAN §7.5 |

## Session 7 acceptance criteria (MASTER_SPEC Session 7)
| Criterion | Status | Evidence |
|---|---|---|
| Can trained and validated with the Lift protocol, or blocker recorded | DONE: 24 runs (4 families × 4 trials at seed 0, then seeds 1–2 of each choice), 100k steps, 480 canonical validation evaluations, 0 failed. One orchestrator crash (KI-036) was fixed and resumed without losing any run | LEDGER E7-02–E7-04, `reports/session7/suite/level1_core_can.json` |
| OOD perturbations, failure categories and ≥ 2 controlled ablations, preregistered, on validation | DONE: §14 committed before any such run (`2859d04`). 5 start sets (base, Lift cube shift, arm offset on both tasks), validated and frozen, bitwise in the container. Failure taxonomy (8 categories). **3** ablations: A1 EMA, A2 sampler, A3 T_o | E7-05, E7-08–E7-11, `reports/session7/robustness/` |
| Protocol frozen in `docs/FINAL_TEST_FREEZE.md` and locked by a Git commit | DONE: `bf0a0fa`. Configs, checkpoint hashes, held-out schedules by hash, R = 1, the §9 bootstrap, aggregation, taxonomy, example rule, "not separated" and the secondary outcome. Container attestation: locked before the freeze commit, unlocked after it | E7-13, D-072 |
| Final paired test executed once under the frozen protocol, all core models and tasks | DONE: 26 evaluations from `bf0a0fa`, failed none; engineering status complete for both tasks; no tuning, reselection or rerun | E7-14, `reports/session7/heldout/` |
| Immutable machine-readable results, CI and difference tables, failures, cost and latency; the three kinds of negative result kept apart | DONE: `reports/session7/results/{results.json,tables.md,taxonomy.json}` (`97cc48a`) | E7-14, `docs/LEVEL1_COMPLETE.md` §2, §4 |
| `docs/LEVEL1_COMPLETE.md` with executed results, omissions and reasons; follow-ups kept separate | DONE | `docs/LEVEL1_COMPLETE.md` §5–§6 |
| All-in cost record; every run recorded, failed ones included | DONE: **$0** in Session 7; 0 failed runs; 3 engineering failures recorded (KI-036, KI-037, KI-038) | LEDGER Session 7, COMPUTE_PLAN |
| Committed data, eval suites, `uv.lock` and dependencies unchanged without a decision | DONE: only the preregistered robustness start sets and their configs were added (D-063) | `git diff d956292 -- data/ uv.lock pyproject.toml configs/eval` |

## Level 1 held-out results (frozen protocol, run once; `reports/session7/results/tables.md`)
Family success, % of 3 seeds × 50 held-out starts, with the §9 bootstrap 95 % CI:

| Family | Lift | Can |
|---|---|---|
| BC-MLP | 90.7 [83.3, 96.7] | 76.7 [62.7, 90.0] |
| BC-RNN | 100.0 [100.0, 100.0] | 96.0 [89.3, 100.0] |
| BC-MLP-chunk | 98.0 [94.0, 100.0] | 96.0 [88.7, 100.0] |
| Diffusion U-Net | 97.3 [93.3, 100.0] | 95.3 [89.3, 100.0] |
| random_uniform | 0.0 | 0.0 |

- **H1** is separated on both tasks.
- **Diffusion − BC-MLP** is separated on Can, +18.7 pp. It is not separated on Lift in
  success (+6.7 pp) but is separated in the secondary episode length.
- **Diffusion − BC-RNN** and **diffusion − BC-MLP-chunk (H2-c)** are not separated on
  either task.
- **BC-RNN − BC-MLP (H3)** is separated on both tasks.

Interpretation and robustness: `docs/LEVEL1_COMPLETE.md`.

## Owner decisions pending
1. Optional: set a Modal **workspace budget** (the only hard spending cap). About $14.11 of the
   $30 monthly credit remains.
2. Optional: prune the Docker build cache (KI-008). About 17 GB is free (2026-11-09).
3. Optional: fix the host's intermittent DNS. Launches survive it now (D-085), but fetches
   and Modal CLI calls still need retries (KI-047).
4. Later, not approved: Can for the RGB study (about $15, COMPUTE_PLAN §7.4). Any further
   Level 2 work, such as point-cloud augmentation, ego-centric frames or Can, would be a new,
   separately preregistered study.
