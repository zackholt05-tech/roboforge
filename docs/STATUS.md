<!-- parsable status header -->
```yaml
CURRENT_SESSION: 12
SESSION_STATUS: PASS            # Session 12: fresh-clone reproduction, final QA, release v1.0.0 (publication record below); $0 paid compute
NEXT_SESSION: none              # project complete (MASTER_SPEC Session 12 STOP); no further phase is started
BLOCKERS: none                  # open items: KI-001, KI-008, KI-010, KI-011, KI-021, KI-023, KI-024, KI-028, KI-033, KI-034, KI-035, KI-042; documented: KI-052, KI-053, KI-056, KI-057; accepted: KI-009 (D-100), KI-060; fixed in Session 12: KI-058, KI-059
LAST_VERIFIED_COMMIT: release-candidate   # the Session 12 release-candidate commit; its SHA and STOP results are recorded in the follow-up commit
PROTOCOL_COMMITS: {v1: 8b08b95, diffusion_amendment: 3aed462, main_budget_amendment: ec9626b, lift_search_space: 6dce01b, can_search_space: 96aa972, section14: 2859d04, rgb_cuda_amendment: e433f11, section15_level2: b93c50c}
EVIDENCE_COMMITS: {level1_freeze: bf0a0fa, level1_heldout_results: 97cc48a, rgb_freeze: d9c6217, rgb_gate_d081: 378ea43, rgb_heldout: bc054bc, rgb_results: 3b23255, pc_feasibility: 85eeb2e, level2_freeze: b93c50c, pc_selections: b1f4da2, level2_gate_d086: 08c3aa8, level2_placement: 4ebebc1, level2_heldout: a031e88, level2_results: 620c541, observatory_selection: 5f398d7, observatory_export: ebfd0c3, observatory_findings_verification: 010867b, observatory_renderer_restored: d2c8add, observatory_findings_robustness: c73a080, paper: 8a08247, release_tooling: 21fc749, fresh_clone_fixes: [d9da1dd, bd3bf6b]}
PAPER: {pdf: paper/main.pdf, sha256: c648e7532e7112cd8bcb3b4e0f118342d6eb97dffd94d3bd1e5448c81b50939f, pages: 23, toolchain: "Tectonic 0.17.0, --only-cached, SOURCE_DATE_EPOCH=1794182400 (D-093)", fresh_clone_rebuild: "byte-identical (Session 12)", label: "local build, not peer reviewed"}
RELEASE: {version: v1.0.0, audit: docs/FINAL_RELEASE_AUDIT.md, decisions: "D-097-D-104", repository: "https://github.com/zackholt05-tech/roboforge (public, per D-099; verified state in the publication record)", observatory: "GitHub Pages per D-102 (verified state in the publication record)"}
EVALUATION_PLATFORM: linux-aarch64-container-1f685638f28c   # D-017, configs/eval/platform.yaml
FINAL_TEST: "Level 1 TESTED_ONCE (never rerun). RGB study TESTED_ONCE under D-081 (never rerun). Level 2 TESTED_ONCE under D-086 (never rerun). Sessions 10-12 read recorded evidence only; no start was stepped"
RESOURCE_SPEND: {usd_billed_total: 10.61, session12_usd_billed: 0, notes: "A-01 $0.148; A-02 $0.0317; A-03 $8.06; A-04 $2.37; all four approvals closed; Sessions 10-12 local only; CI on GitHub-hosted runners is free for public repositories (D-101)"}
APPROVED_LIMIT: {A-03: "closed, $8.06 of $15", A-04: "closed, $2.37 of $9", open: none, local: "$0"}
UPDATED: 2026-11-09
```

# RoboForge Status

## Session 12: fresh-clone reproduction, final QA, release (PASS)
| Step (Session 12 prompt) | Status | Evidence |
|---|---|---|
| 1. Decisions before acting: Makefile defaults s12; fresh-clone environment; what "release" means; owner asked before any public action | DONE: D-097 (s12), D-098 (clone), D-099 (owner: **public repo + tag v1.0.0 + GitHub release**), D-100 (owner: local paths **disclosed, no rewrite**), D-101 (owner: **CI on push**), D-102 (owner: **GitHub Pages**), D-103 (README generated), D-104 (release checks) | E12-01 |
| 2. Fresh-clone reproduction in a separate clean directory | DONE: full suite 519 passed, 7 skipped; container smoke 33/33; downloads SHA-256 equal; data rebuild equal; `make paper` byte-identical; site build, vitest 49/49, E2E 7/7; one frozen checkpoint hash-checked and loaded. Two test-infrastructure findings fixed (KI-058, KI-059) | E12-02, `docs/FINAL_RELEASE_AUDIT.md` §3 |
| 3. Download links and commands; scans for placeholders, private paths, secrets, dataset copies, broken citations | DONE: `make download` in the clone; `make release-check` PASS (136 disclosed local-path files, D-100) | E12-03, audit §5 |
| 4. Commit identity, AI trailers, CITATION.cff | DONE: every commit on every ref `Zackary Holt <zackholt05@gmail.com>`, no trailer or byline; `CITATION.cff` v1.0.0, Zackary Holt only | audit §5 |
| 5. Full test matrix, security and license scans | DONE: see the STOP row; pip-audit (KI-060, accepted), npm audit 0, license inventory; Python type checking not configured (stated) | audit §4 |
| 6. README per §11.2 from real artifacts | DONE: generated findings and setup blocks, preview from the D-089 rule 3(a) pair, paper link labelled not peer reviewed; Observatory link only after the hosted site is verified | README, D-103 |
| 7. `docs/FINAL_RELEASE_AUDIT.md` | DONE | the file |
| 8. STOP: docs, checks, commit, completion summary | DONE (results in the follow-up commit) | this file, `docs/HANDOFF.md` |

**Nothing here is a new experimental result.** The README's numbers are generated from the
same frozen macros as the paper. No held-out start was stepped; no training, no evaluation, no
paid compute.

## Session 11: research paper, reproducibility, independent audit (PASS, local only)
| Step (Session 11 prompt) | Status | Evidence |
|---|---|---|
| 1. Decisions first: Makefile defaults s11; paper toolchain; no new dependency | DONE: D-092 (PREFIX s11, evidence session 11, `OBS_OUT` pinned to Session 10), D-093 (Tectonic 0.17.0, matplotlib and scipy already in the lock; `uv.lock` and `pyproject.toml` unchanged), D-094 (generator contract) | E11-01 |
| 2. Every table and figure from frozen files by script and make target, with tests and provenance | DONE: `make paper-assets` / `make paper-check`; 35 files in `paper/generated/` (17 tables, 6 figures with data sidecars, 3 renders, `numbers.tex`, `index.json`); 30 pinned sources plus 26 hash-chained held-out summaries; KI-049 erratum beside its row; nothing read from `findings.json` or `runs/` | E11-02, `tests/unit/test_paper_assets.py`, `test_paper_tex.py` |
| 3. LaTeX paper, MASTER_SPEC §11.1 structure, no typed numbers, verified bibliography, single author, §11.3 claims, "DP3-inspired" | DONE: `paper/` (abstract, 10 sections, 2 appendices); a lint fails on any typed number or placeholder; 31 bibliography entries resolved (E11-03) | `paper/main.tex`, `paper/sections/` |
| 4. Independent critical audit, findings with severity and evidence, each corrected or disclosed | DONE: four fresh-context passes (A simulator/splits/privilege, B statistics, C spend/declared analyses/timeline, D claims of the paper). HIGH: A-6 (diffusion action bound, a confound), B-1/B-2 (method-dependent separations), D-1 ("not separated" written as equality, corrected). Post-hoc additions labelled: D-095 sensitivity analysis, D-096 declared secondary metrics. No new evaluation | `docs/AUDIT.md`, D-095, D-096, KI-052–KI-057, E11-04–E11-06 |
| 5. Build paper/main.pdf | DONE: `make paper` (cached bundle only, fixed SOURCE_DATE_EPOCH), no undefined reference, two builds byte-identical, 23 pages, SHA-256 `c648e753…` | E11-07 |
| 6. STOP: docs, lint, full suite alone, freeze-check, gates, web-build, web-test, commit, Session 12 prompt | DONE | this file, `docs/HANDOFF.md` |

**What the paper reports (every number generated from the frozen files):**
- Level 1 held-out: on Can, diffusion − BC-MLP +18.7 pp [+4.7, +32.7], separated by the
  preregistered bootstrap; a seed-level t interval (not preregistered) includes 0. H2-c
  (diffusion − BC-MLP-chunk) is not separated on either task, so there is no evidence that
  the generative head adds to history and chunking. Diffusion costs 19–26× the chunked MLP's
  CPU hours.
- H4: on Lift, diffusion degrades least under the arm offset; on Can its drop is not separated
  from the chunked MLP's. The diffusion sampler bounds actions to the training range (KI-052),
  a confound for these comparisons.
- Extensions: RGB visual diffusion vs visual BC-chunk +0.7 pp [0.0, +4.0], not separated
  (saturated). Level 2 H5: the DP3-inspired point-cloud arm minus the frozen RGB arm under cube
  shift −41.3 pp [−54.0, −28.0], separated (−), robust to the seed-level check.
- Limitations are quantitative and include every audit finding (§9 of the paper).

**Integrity:** no frozen file, freeze or executed outcome changed (`make freeze-check` before
and after the two analysis-script fixes of KI-055); both held-out gates open; no training, no
evaluation and no paid or remote compute. Network use: the bibliography check and one
download of Tectonic's TeX bundle files.

## Session 10: RoboForge Observatory (PASS, local only)
| Step (Session 10 prompt) | Status | Evidence |
|---|---|---|
| 1. Decisions before any frontend dependency: Makefile defaults s10; web stack | DONE (`086a346`): D-087 (PREFIX s10, evidence session 10), D-088 (`website/` with its own lockfile; uv.lock and pyproject unchanged); D-089 selection and provenance rules; owner: **no hosting**, meshes extracted at build time (D-090) | E10-01 |
| 2. Export from frozen evidence only; denoising traces only on validation starts; viewer checked against ≥ 3 reference frames; every approximation reported | DONE: 37 clips from clean `b5fc2ee` on the canonical container (E10-05); 18 reference frames pass (D-091 thresholds); approximations on the site (Methods) and in `docs/VISUAL_PROVENANCE.md` | E10-02–E10-05, E10-07 |
| 3. Sections A–H; findings from the frozen results files; panel G with the real Level 2 results | DONE: `findings.json` copies 8 frozen sources with hashes (E10-06). Section F shows the held-out results and, labelled "validation regime, not held-out", the Level 1 robustness (H4) and ablations A1–A3. Panel G shows H5 −41.3 points [−54.0, −28.0], the KI-049 erratum and the calibrated overlays | `website/src/sections/` |
| 4. Accessibility, keyboard, reduced motion, mobile, loading states, budgets; build and E2E; hosting only if authorized | DONE: build within budgets, vitest 49/49, E2E 7/7 (keyboard, reduced motion, phone, no WebGL, missing artifact); no hosting, no URL | E10-07 |
| 5. STOP: docs, lint, full suite alone, freeze-check, both gates, commit, Session 11 prompt | DONE | this file, `docs/HANDOFF.md` |

**What the Observatory shows (all from recorded episodes; nothing new is a result):**
- 28 Level 1 held-out episodes and 6 Level 2 placement episodes, re-rendered from their saved
  states and never stepped. Body poses recomputed with `mj_kinematics` equal the recorded
  poses to 0.0 m.
- 3 validation denoising traces (Lift `0000`; Can `0000` success, `0003` failure). The frozen
  checkpoint's sampler was re-run on each recorded episode, and every replayed chunk equals
  the recorded action bitwise. The candidate paths come from clone simulation.
- The viewer reproduces MuJoCo's silhouettes: robot IoU 0.9998–1.0000, object IoU
  0.9970–1.0000, centroid ≤ 0.04 px over 18 frames (`viewer_verification.json`).
- Approximations, listed on the site:
  - interpolation between 20 fps frames;
  - restyled materials and lighting;
  - meshes quantized to about 7 µm;
  - candidate paths drawn over the robot;
  - point clouds as fixed-size dots.
- Local only: `make web-build` then `npm --prefix website run preview`. No deployed or staging
  URL exists.

**Integrity:**
- At STOP, `make freeze-check` caught an edit to the frozen `roboforge.sim.shadow_render`. The
  file is restored, and the logic now lives in `viz/obs_media` (D-091 amendment, KI-050).
- A re-render from the restored tree is byte-identical to the committed export: 43 of 43
  files (E10-08).
- The full suite at STOP found the locked suite's partition label written in
  `viz/obs_select.py`, which breaks a research-integrity test (KI-051). It is fixed in
  `4348244`. The selection reproduces exactly, so no evidence changed (E10-09).

## Session 10 acceptance (MASTER_SPEC Session 10)
| Criterion | Status | Evidence |
|---|---|---|
| `npm run build` passes; E2E/UI sanity checks pass | DONE: `make web-build`, `make web-e2e` 7/7 | E10-07 |
| No frontend fixture data pretending to be measured results | DONE: every number is read from `findings.json` (copies of frozen files, hash-checked) or the export; the one hand-drawn figure is labelled "illustrative schematic, not data"; unit tests use only the exported data | `website/src/test/fixtures.ts`, `schema.test.ts` |
| Source provenance visible | DONE: each clip caption names its checkpoint, task, outcome and evaluation run. Its provenance panel adds the policy, seed, start, record and physics hash. Findings and Methods list every frozen source with its SHA-256, and section F lists the 75 robustness and ablation evaluation summaries (Lift 42, Can 33) (repository paths with SHA-256: the form a manifest link takes without hosting). Also `docs/VISUAL_PROVENANCE.md` | site, ledger |
| A reviewer can confirm the animated joint motion matches the saved MuJoCo transforms | DONE: forward-kinematics test vs recorded poses (vitest), and the `?verify=all` page with its overlap montage (`verification/viewer_montage.jpg`) | E10-07 |
| Screenshots, videos and GIFs are genuine | DONE: videos are MuJoCo renders of recorded states; screenshots in `site_screens/` are captures of the built site | `reports/session10/observatory/` |
| Optional RGB and point-cloud panels only for studies actually run | DONE: both studies ran; panel G shows their frozen results, labelled DP3-inspired, H5 negative | `BeyondPixels.tsx` |

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
2. Optional: prune the Docker build cache (KI-008). About 17 GB is free (2026-11-09, end of
   Session 10).
3. Optional: fix the host's intermittent DNS. Launches survive it now (D-085), but fetches
   and Modal CLI calls still need retries (KI-047).
4. Optional: hosting the Observatory (static build, `website/dist`) needs a new approval row
   and an owner choice of host (D-090).
5. Later, not approved: Can for the RGB study (about $15, COMPUTE_PLAN §7.4). Any further
   Level 2 work, such as point-cloud augmentation, ego-centric frames or Can, would be a new,
   separately preregistered study.
