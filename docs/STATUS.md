<!-- parsable status header -->
```yaml
CURRENT_SESSION: 18
SESSION_STATUS: PASS            # Session 18: v1.1 built and verified locally (addendum PDF, README extensions section and chart, Observatory section with 23 X1/X2 clips re-rendered from recorded states); nothing pushed, tagged, released or deployed; $0
NEXT_SESSION: 19                # v1.1 publication (owner decision D-131): push main, CI, tag v1.1.0, release with the addendum PDF, Pages redeploy on top of c14e041; exact prompt in docs/HANDOFF.md # Session 18 Handoff §6
BLOCKERS: none                  # open items: KI-001, KI-008, KI-010, KI-011, KI-021, KI-023, KI-024, KI-028, KI-033, KI-034, KI-035, KI-042; documented: KI-053 (Can H2/H3 replicated by X3; X2 Can MH shown beside), KI-056, KI-057; mitigated: KI-062, KI-064 (Session 15), KI-065 (Session 18); KI-052 tested by X1 (Session 13); accepted: KI-009 (D-100, D-114), KI-060, KI-063 (D-122); partial: KI-061
LAST_VERIFIED_COMMIT: 83b2f37   # Session 18 STOP checks ran on 83b2f37 (last code change f675490; documents only after); export from clean 7eb5817; nothing pushed
PROTOCOL_COMMITS: {v1: 8b08b95, diffusion_amendment: 3aed462, main_budget_amendment: ec9626b, lift_search_space: 6dce01b, can_search_space: 96aa972, section14: 2859d04, rgb_cuda_amendment: e433f11, section15_level2: b93c50c, x1_preregistration: ab02469, x3_preregistration: fb9f05e, x2_preregistration: 9df6628}
EVIDENCE_COMMITS: {level1_freeze: bf0a0fa, level1_heldout_results: 97cc48a, rgb_freeze: d9c6217, rgb_gate_d081: 378ea43, rgb_heldout: bc054bc, rgb_results: 3b23255, pc_feasibility: 85eeb2e, level2_freeze: b93c50c, pc_selections: b1f4da2, level2_gate_d086: 08c3aa8, level2_placement: 4ebebc1, level2_heldout: a031e88, level2_results: 620c541, observatory_selection: 5f398d7, observatory_export: ebfd0c3, observatory_findings_verification: 010867b, observatory_renderer_restored: d2c8add, observatory_findings_robustness: c73a080, paper: 8a08247, release_tooling: 21fc749, fresh_clone_fixes: [d9da1dd, bd3bf6b], x1_code: 09d1a90, x1_preregistration: ab02469, x1_results: c2a9ae6, x3_code: [e0b8a4e, e9289c6], x3_heldout_suite: 81afa3b, x3_preregistration: fb9f05e, x3_suite_end: d602d1b, x3_validation: 86e6c9b, x3_isolation_before: 13c5db2, x3_evaluations: 949a6d7, x3_results: 335c27c, x2_data: [5e4b015, 305f074], x2_suites: 4459c00, x2_pilot_code: [3d42786, fe9df74], x2b_preregistration_and_code: 9df6628, x2b_evidence: fdb1f37, x2c_decisions: ca1897a, x2c_code: a1c3db7, x2c_engineering_check: 45edda2, x2_freeze: 3a64dbb, x2_heldout: 29ce47a, x2_results: 3df25ff, v1_1_decisions: b96b44f, addendum_and_readme: 5e2cf46, obs_ext_code: 2fdf21d, obs_ext_selection: 7eb5817, obs_ext_export: [f26e4e6, a8b5a74], obs_ext_site: fb1f6be, visual_provenance_ext: f675490, ki_065: d00b195}
PAPER: {pdf: paper/main.pdf, sha256: c648e7532e7112cd8bcb3b4e0f118342d6eb97dffd94d3bd1e5448c81b50939f, pages: 23, toolchain: "Tectonic 0.17.0, --only-cached, SOURCE_DATE_EPOCH=1794182400 (D-093)", fresh_clone_rebuild: "byte-identical (Session 12)", label: "local build, not peer reviewed"}
ADDENDUM: {pdf: paper/addendum/addendum.pdf, sha256: bedfb27753722604e6f31b6fc171c646975ba6b7b635bdd9884dd6bc16917fe2, pages: 6, toolchain: "Tectonic 0.17.0, --only-cached, SOURCE_DATE_EPOCH=1794787200 (D-133)", sources: "the three pinned extension results.json", check: "make addendum-check (CI research-integrity)", label: "local build, not peer reviewed"}
RELEASE: {version: v1.0.0, audit: docs/FINAL_RELEASE_AUDIT.md, decisions: "D-097-D-104", repository: "https://github.com/zackholt05-tech/roboforge (public, per D-099; verified state in the publication record)", observatory: "GitHub Pages per D-102 (verified state in the publication record)", v1_1: "built and verified in Session 18, not published; Session 19 publishes (D-131)"}
EVALUATION_PLATFORM: linux-aarch64-container-1f685638f28c   # D-017, configs/eval/platform.yaml
FINAL_TEST: "Level 1 TESTED_ONCE (never rerun). RGB study TESTED_ONCE under D-081 (never rerun). Level 2 TESTED_ONCE under D-086 (never rerun). Sessions 10-12 read recorded evidence only. Session 13 (X1) ran on the Session 7 robustness sets only; no held-out or validation start was stepped. Session 14 (X3) stepped only the Can validation starts and the new can/x3_heldout suite (24 evaluations, once); no frozen held-out suite; isolation PASS. Session 15 (X2-a) stepped only the can/x2_validation starts (20 pilot evaluations); no held-out start of any suite; isolation PASS. Session 16 (X2-b) stepped only the can/x2_validation starts (460 evaluations); no held-out start of any suite; isolation PASS. Session 17 (X2-c) stepped can/x2_heldout once (12 evaluations from the X2 freeze 3a64dbb; never again) and, for the engineering check, 2 evaluations on the 50 can/x2_validation starts (100 episodes); no frozen held-out suite; isolation PASS. Session 18 read recorded evidence only: 23 X1/X2 records re-rendered from their recorded states on the canonical container (no policy call, no env.step); isolation PASS"
RESOURCE_SPEND: {usd_billed_total: 10.61, session13_usd_billed: 0, session14_usd_billed: 0, session15_usd_billed: 0, session16_usd_billed: 0, session17_usd_billed: 0, session18_usd_billed: 0, notes: "A-01 $0.148; A-02 $0.0317; A-03 $8.06; A-04 $2.37; all four approvals closed; Sessions 10-18 local only; CI on GitHub-hosted runners is free for public repositories (D-101)"}
APPROVED_LIMIT: {A-03: "closed, $8.06 of $15", A-04: "closed, $2.37 of $9", open: none, local: "$0"}
X1: {decision: D-107, preregistration: "docs/extensions/X1_PREREGISTRATION.md @ ab02469 (sha256 7405dc3d...)", evaluations: "45/45 once, reports/session13/x1_clamp/evals/, runs/x1_clamp/", analysis: "reports/session13/x1_clamp/results.json (sha256 ad503686...) @ c2a9ae6", primary: "Lift ood_arm_offset drop(diffusion) - drop(clamped BC-MLP-chunk) -40.0 pp [-54.0, -25.3], separated (-); reading R1", label: "exploratory, after the freeze"}
X3: {decisions: "D-109-D-115", preregistration: "docs/extensions/X3_PREREGISTRATION.md @ fb9f05e (sha256 b5e54125...)", suite: "can/x3_heldout, 100 starts, content 9c199e19... (81afa3b)", runs: "12 new (4 families x seeds 3-5), validation-selected (reports/session14/suite/x3_validation.json)", evaluations: "24/24 once, reports/session14/x3/evals/, runs/x3/", analysis: "reports/session14/x3/results.json (sha256 4299a149...) @ 335c27c", primary: "H2 diffusion - BC-MLP +22.7 pp and H3 BC-RNN - BC-MLP +23.8 pp separated under bootstrap and seed-level (holds, robust); H2 diffusion - BC-RNN -1.2 and H2-c -1.7 not separated (unchanged)", label: "separate replication"}
X2: {decisions: "D-116-D-130", preregistration: "docs/extensions/X2_PREREGISTRATION.md @ 9df6628", option: "A, Can MH (operator-stratified split; within-operator claims only)", suites: "can/x2_validation 50 starts 47b889cd..., can/x2_heldout 100 starts 40b8228c... (4459c00)", x2b: "reports/session16/x2/suite/ from clean 9df6628: chosen diffusion t2, BC-RNN t4, BC-MLP-chunk t2, BC-MLP t1; x2_test_configuration.json (12 checkpoints)", freeze: "docs/extensions/X2_FREEZE.md @ 3a64dbb (sha256 952d232c..., blob 9929dc7c...)", evaluations: "12/12 once from 3a64dbb (reports/session17/x2/heldout/evals/, runs/x2c/), isolation PASS", analysis: "reports/session17/x2/results.json (sha256 30c01f55...) @ 3df25ff", primary: "H2 diffusion - BC-MLP +63.3 pp robust (+), same direction; H2 diffusion - BC-RNN +27.3 pp robust (+), new separation; H2-c +2.0 pp not separated; H3 BC-RNN - BC-MLP +36.0 pp robust (+), same direction", phase: "X2 done (X2-a Session 15, X2-b Session 16, X2-c Session 17)", label: "X2 (Can MH): beside the frozen Level 1 Can PH result, never instead of it, never pooled; within-operator only"}
UPDATED: 2026-11-16
```

# RoboForge Status

## Next: Session 19, publish v1.1 (D-131)
The owner chose to publish X1, X2 and X3 as v1.1: push `main`, an annotated tag `v1.1.0`, a
GitHub release with the addendum PDF, the README section, and the Observatory updated first
(D-131). Session 18 built and verified all of it locally and stopped with nothing pushed,
tagged, released or deployed. Session 19 publishes it in the D-131 order; its exact prompt is in
`docs/HANDOFF.md` (# Session 18 Handoff §6). v1.0.0 (tag, release, `paper/main.pdf`
`c648e753…`) stays as published.

## Session 18: v1.1 built and verified locally (PASS; nothing published)
| Step (Session 18 prompt and owner instructions) | Status | Evidence |
|---|---|---|
| Prerequisites, by executable checks; fresh-clone CI commands | DONE: all PASS | E18-01 |
| Owner decisions, first | DONE: "publish"; scope "(b) + README"; Observatory "Update it too", then "update the Observatory before anything goes on GitHub"; "X2 + X1 clips"; "build now, publish in S19" (D-131) | DECISIONS D-131 |
| Makefile defaults and targets | DONE: D-132 (`PREFIX ?= s18`, evidence session 18, `OBS_EXT_OUT`, addendum and Observatory targets) | D-132 |
| Addendum PDF, generated from the three pinned results; `paper/main.pdf` unchanged | DONE: `paper/addendum/addendum.pdf`, 6 pages, SHA-256 `bedfb277…`, built twice byte-identical; `make addendum-check` in CI (D-133) | E18-02 |
| README section and chart, generated; `release-check` covers them | DONE: `extensions` block, `docs/media/readme_extensions.svg` | E18-02 |
| Observatory: selection, export, findings, section, verification | DONE: 23 clips by the D-134 rules (`7eb5817`); export from clean `7eb5817`, 0.0 m, isolation PASS (`f26e4e6`, `a8b5a74`); section "After v1.0.0: three follow-up studies" (`fb1f6be`); viewer check 12/12 pass | E18-03 to E18-05 |
| Provenance and known issues | DONE: `docs/VISUAL_PROVENANCE.md` v1.1 section (`f675490`); KI-065 (`d00b195`) | E18-06 |
| STOP | see below | E18-07 |

**What v1.1 adds (no new result; every number generated from the three pinned results).**
- The addendum: X1 "exploratory, after the freeze" (validation regime, KI-052), X3 "separate
  replication" (KI-053), X2 "X2 (Can MH)" (beside Level 1 Can PH, never pooled,
  within-operator only, 500- vs 400-step scales), each with its tables and caveats.
- The README: a generated `extensions` block and chart; the KI-052 limitation line now says X1
  tested the BC side.
- The Observatory, section "After v1.0.0: three follow-up studies":
  - X1: three arms (diffusion, BC-MLP-chunk unclamped and clamped) on rule-selected Lift
    arm-offset starts;
  - X2 (Can MH): diffusion against each BC family on rule-selected held-out starts;
  - X3: its contrasts beside Level 1;
  - all recorded episodes, re-rendered from their recorded states and never stepped. The X2
    clips show the Can MH label (KI-065).
- Cost: 350.5 s of container time for the export; $0.

**STOP checks.** On `83b2f37` plus the Session 18 STOP records (documents and logs only; the last code change is `f675490`), host, no container running during the suite: `make lint` clean; full suite alone **720 passed** (9:33; 705 of Session 17 plus 15 Session 18 tests; `logs/session18/stop_full_suite.log`); `make freeze-check` unchanged ("freeze committed; frozen code unchanged"); `tests/research_integrity` 81 passed; `make addendum-check` and `make paper-check` pass; `paper/main.pdf` `c648e753…` and its sources identical to `v1.0.0`; `paper/addendum/addendum.pdf` `bedfb277…`; `make release-check` PASS with the new records staged so that its scan covers them (139 disclosed local-path files, D-100 and D-114; no new local path); site: `make web-build` within budgets, `make web-test` 75 passed, `make web-e2e` 8/8, `make web-verify-ext` pass. On Linux aarch64 (the canonical image, read-only mount of the tree), the 15 addendum files and the README block and chart equal a fresh build, as CI will check (`logs/session18/addendum_check_linux.log`). A fresh `git clone --no-local` of `67f4419` ran the commands of all three CI workflows (lint; unit tests 455 passed, 12 skipped for data absent from a clone, none of them new; `tests/research_integrity` 77 passed, 4 skipped; freeze-check; both held-out gate checks (open, as since v1.0.0, which is not a permission to rerun); paper-check; addendum-check; release-check; web-data, web-assets, web-build within budgets, web-test 75 passed): all PASS (`logs/session18/fresh_clone_ci_stop/`). No frozen file, freeze, result or executed outcome changed; nothing was trained, evaluated or stepped; nothing was pushed, tagged, released or deployed.

## Session 17: post-release extension X2, phase X2-c, Can MH (PASS; X2 done)
| Step (Session 17 prompt) | Status | Evidence |
|---|---|---|
| Prerequisites, by executable checks; host facts | DONE: all PASS, no owner decision needed | E17-01 |
| 1. Decisions before acting | DONE: D-129 (Makefile s17, evidence session 17, X2-c targets), D-130 (the X2-c design), both in `ca1897a` before any code | DECISIONS, E17-01 |
| 2. New code in new files only; synthetic-outcome tests first; research-integrity test | DONE: `a1c3db7`; `make freeze-check` unchanged; full suite 705 passed; engineering check PASS (two X2-b validation summaries reproduced exactly) | E17-02, E17-03 |
| 3. `docs/extensions/X2_FREEZE.md` committed from a clean tree before any held-out rollout | DONE: alone in `3a64dbb` (SHA-256 `952d232c…`, git blob `9929dc7c…`) | E17-04 |
| 4. The 12 evaluations once on the canonical container from the freeze commit | DONE: 12/12 complete, attempt 1, 0 errors, audit PASS; isolation PASS; no rerun | E17-05, `reports/session17/x2/heldout/evals/` |
| 5. The analysis once; label "X2 (Can MH)" | DONE: `3df25ff` | E17-06, `reports/session17/x2/` |
| STOP | see # Session 17 Handoff §4 | E17-07 |

**Result: X2 (Can MH), beside the frozen Level 1 Can PH result, never instead of it, never
pooled; within-operator only** (`reports/session17/x2/tables.md`; 3 seeds × 100 held-out starts,
500-step episodes; Level 1 Can PH was 3 × 50 starts at 400 steps).
| Contrast | Level 1 Can PH (frozen) | X2: difference, bootstrap 95 % | X2 seed-level 95 % | X2 verdict | Relation |
|---|---|---|---|---|---|
| H2 diffusion − BC-MLP | +18.7 pp, separated (+) | +63.3 pp [+53.0, +73.0] | [+51.9, +74.8] | robust (+) | same direction |
| H2 diffusion − BC-RNN | −0.7 pp, not separated | +27.3 pp [+16.7, +37.7] | [+14.9, +39.8] | robust (+) | new separation |
| H2-c diffusion − BC-MLP-chunk | −0.7 pp, not separated | +2.0 pp [−6.3, +11.0] | [−11.6, +15.6] | not separated | neither separated |
| H3 BC-RNN − BC-MLP | +19.3 pp, separated (+) | +36.0 pp [+23.7, +48.0] | [+22.9, +49.1] | robust (+) | same direction |

Family success on the X2 held-out starts: diffusion 93.3 % (90, 96, 94 per seed), BC-MLP-chunk
91.3 % (91, 98, 85), BC-RNN 66.0 % (72, 60, 66), BC-MLP 30.0 % (24, 31, 35). Engineering
errors 0. The frozen Level 1 result stands as reported.

**Cost (measured; labelled estimates from D-130 in brackets).** Evaluations 2.66
container-hours, launch to exit [3.6]; wall 47.0 min [0.9–1.0 h]; engineering check 7.0 min
[0.2 h]; `runs/x2c` 529 MB [0.6–0.9 GiB]; analysis 1.5 min; $0. Host: swap rate peak 991
pages/s (no launch held), free disk ≥ 13.2 GiB.

## Session 16: post-release extension X2, phase X2-b, Can MH (PASS; development numbers only)
| Step (Session 16 prompt) | Status | Evidence |
|---|---|---|
| Prerequisites, by executable checks; facts measured; owner asked | DONE: owner followed every recommendation (D-125) | E16-01 |
| 1. Decisions before acting | DONE: D-125 (owner), D-126 (Makefile s16, evidence session 16, X2-b targets), D-128 (disk reclaim) | DECISIONS, E16-01, E16-03 |
| 2. Preregistration committed before the first X2-b training | DONE: `9df6628` (D-127) | `docs/extensions/X2_PREREGISTRATION.md`, E16-02 |
| 3. New code in new files only; research-integrity test on held-out isolation | DONE: suite driver, host guards, latency benchmark; `make freeze-check` unchanged | E16-02 |
| 4. Train and select on local CPU from a clean tree; X2 test configuration; cost against estimates | DONE: `make x2b-suite` once from clean `9df6628`, 24.1 h; 20/20 runs, 460/460 evaluations, 0 failed; benchmark on the canonical container | E16-04, E16-05, `reports/session16/x2/suite/` |
| STOP | see below | E16-06 |

**Choices and selections (development numbers on 50 validation starts; never a result).**
| Family | Chosen trial (frozen §6.3) | Seed 0 / 1 / 2: selected step (successes) |
|---|---|---|
| diffusion | 2, cosine (48; tie with trial 1, lower mean first-success step) | 100k (48) / 90k (46) / 65k (47) |
| BC-RNN | 4, weight decay 0.1 (36) | 65k (36) / 65k (36) / 90k (39) |
| BC-MLP-chunk | 2, cosine (49; tie with trial 1, lower mean first-success step) | 15k (49) / 5k (50) / 30k (48) |
| BC-MLP | 1, base (16) | 85k (16) / 45k (12) / 75k (15) |

The top two families sit at the 50-start ceiling on validation, and these are maxima over 20
checkpoints (selection optimism, preregistration §8). All X2 claims are within-operator.

**Cost (measured; estimates from the preregistration §9 in brackets).**
- Training: 33.86 training-hours summed [30.5]; diffusion 4.9–5.5 h per run at about 5.0
  steps/s.
- Evaluations: 68.98 container-hours, mean 540 s [about 64].
- Wall time: 24.1 h [about 23].
- `runs/x2`: +17.96 GiB [18–23].
- Spend: $0.
- Latency (observe + act median per decision): BC-MLP 1.4–1.6 ms, BC-MLP-chunk 1.7–1.8 ms,
  BC-RNN 9.4–9.8 ms, diffusion 48–53 ms.
- Host: swap peaked at 29,285 pages/s (KI-035); free disk was 11.0 GiB at its lowest
  (KI-008).

**STOP checks.** On `9df6628` plus the Session 16 records and evidence (documents, logs and reports only; no code changed), host, no container running: `make lint` clean; full suite alone **644 passed** (9:24; equal to the Session 16 baseline; `logs/session16/stop_full_suite.log`); `make freeze-check` unchanged ("freeze committed; frozen code unchanged"); `tests/research_integrity` 69 passed; `make release-check` PASS with the new evidence staged so that its scan covers it (139 disclosed local-path files, D-100 and D-114; no new local path). Evidence verification and isolation PASS (E16-05). No frozen file, freeze, result or executed outcome changed; no held-out start of any suite was stepped; nothing was pushed, tagged, released or deployed.

## Session 15: post-release extension X2, phase X2-a, Can MH (PASS; development numbers only)
| Step (Session 15 prompt) | Status | Evidence |
|---|---|---|
| Prerequisites, by executable checks; facts measured; owner asked | DONE: owner chose option A, Can MH (D-116) | E15-01 |
| 1. Decisions before acting | DONE: D-117 (Makefile s15, evidence session 15, X2 targets); D-118 (manifest with SHA-256 from the pinned revision, verified on download) | DECISIONS, E15-01 |
| 2. Download, verify, inspect, split, normalize | DONE: both files hash-verified; 300 demos, six operator masks (D-119, operator-stratified upstream split); horizon 500 (D-120); `demo_81` an accepted flag (D-122); normalization on the 270 training demos | E15-02, E15-03, `reports/session15/x2/inspection.json` |
| 3. The validated Can contract applies | DONE: identical `env_args`, observation and action specs; replay 285/300; X2 binding of frozen code (D-121, KI-064) | E15-03, `reports/session15/x2/contract.json` |
| 4. X2 suites, canonical container, fresh seed blocks, checked bitwise, committed with hashes | DONE: `can/x2_validation` (50, `47b889cd…`), `can/x2_heldout` (100, `40b8228c…`), `4459c00`, before any rollout (D-123) | E15-04 |
| 5. Headroom pilot on X2 validation starts only | DONE once from clean `fe9df74` (D-124): reading **headroom**; X2-b proposal with estimates in HANDOFF §5 | E15-05, E15-06, `reports/session15/x2/pilot/` |
| STOP | see below | E15-07 |

**Pilot (development numbers on 50 validation starts; never a result).** Best of five
checkpoints: diffusion 48, BC-MLP-chunk 48, BC-RNN 34, BC-MLP 6. The reading is headroom by one
family (three at ≥ 48 would be "saturated"); the top two are at the 50-start ceiling, so X2-b
may not separate diffusion from BC-MLP-chunk. All X2 claims are within-operator.

**Cost (measured).** Training 4.62 h wall for diffusion (6.0 steps/s), 1.11 BC-RNN, 0.19
BC-MLP-chunk, 0.17 BC-MLP, all four at once; evaluations 2.8 container-hours; disk 1.63 GiB;
$0. The swap rate peaked at 4,874 pages/s, above the launch guard (KI-035).

**STOP checks.** On `fe9df74` plus the Session 15 records (documents and evidence only), host, no container running: `make lint` clean; full suite alone **631 passed** (9:03; 28 more than Session 14's 603; `logs/session15/stop_full_suite.log`); `make freeze-check` unchanged ("freeze committed; frozen code unchanged", also before and after every code change); `tests/research_integrity` 64 passed; `make release-check` PASS (139 disclosed local-path files, D-100 and D-114; no new local path). No frozen file, freeze, result or executed outcome changed; no held-out start of any suite was stepped; nothing was pushed, tagged, released or deployed.

## Session 14: post-release extension X3, more training seeds (PASS; separate replication)
| Step (Session 14 prompt) | Status | Evidence |
|---|---|---|
| Prerequisites, by executable checks; free disk and per-seed cost measured; owner asked | DONE: owner chose Can only, seeds 3–5, npm cache cleared first (D-109) | E14-01 |
| 1. Decisions before acting | DONE: D-110 (Makefile s14, evidence session 14, X3 targets); D-111 (design) | DECISIONS, E14-01 |
| 2. X3 held-out suite, canonical container, fresh seed block, checked bitwise, committed with its hash before any rollout | DONE: `can/x3_heldout`, 100 starts, seeds 14,000,000 + i, content `9c199e19…` (`81afa3b`) | E14-02 |
| 3. Preregistration committed before the first new seed trains | DONE: `fb9f05e` (SHA-256 `b5e54125…`) | `docs/extensions/X3_PREREGISTRATION.md`, E14-03 |
| 4. New seeds trained on local CPU from a clean tree; selection by the frozen rule on validation starts only | DONE: 12/12 runs, 0 failed; 240 validation evaluations, 12 selections. The suite stopped at its configuration choice (KI-062); an evaluation-only driver finished validation and selection with the suite's commands (D-112); validation paused during diffusion training after swap thrashing (D-113) | E14-04 to E14-09 |
| 5. Every seed on the X3 suite once; analysis once; "separate replication" | DONE: 24/24 evaluations, 0 errors, isolation PASS; `make x3-analyze` once | E14-10 to E14-12, `reports/session14/x3/` |
| STOP | see below | E14-13 |

**Result (separate replication; `reports/session14/x3/tables.md`).** 6 seeds per family
(frozen 0–2, new 3–5) × 100 fresh Can held-out starts, beside the frozen 3 × 50 result:
- **H2 diffusion − BC-MLP** +22.7 pp, bootstrap [+10.5, +35.0], seed-level [+8.0, +37.3]:
  separated under both (frozen +18.7 pp, bootstrap only). *Holds, robust.*
- **H3 BC-RNN − BC-MLP** +23.8 pp, [+12.0, +36.0], [+9.2, +38.5]: separated under both
  (frozen +19.3 pp, bootstrap only). *Holds, robust.*
- **H2 diffusion − BC-RNN** −1.2 pp and **H2-c diffusion − BC-MLP-chunk** −1.7 pp: not
  separated under either. *Unchanged.*
- Family success: BC-MLP 72.3 %, BC-RNN 96.2, BC-MLP-chunk 96.7, diffusion 95.0.
- Declared secondary (no readings): on either 3-seed subset H2 and H3 separate under the
  bootstrap only; on seeds 3–5 alone H2-c separates (−) under the seed-level interval only.
- The frozen result stands as reported; X3 is never pooled with it. Scope: Can only, R = 1,
  not Lift and not H4.

**Cost (measured; estimates in D-111).** Training 33.0 CPU-hours summed (estimate 15.5);
validation 35.5 container-hours (estimate 19.2); X3 evaluations 3.1 container-hours
(estimate 3.4); 13.2 h elapsed (estimate 6.5); 7.1 GiB of disk; $0.

**STOP checks.** On `227cb26` (code as `bd7c899`; documents only after), host, no container running: `make lint` clean; full suite alone **603 passed** (8:58; the 550 of Session 13 plus 53 Session 14 tests; `logs/session14/stop_full_suite.log`); `make freeze-check` unchanged ("freeze committed; frozen code unchanged", also before and after every code change); `tests/research_integrity` 59 passed; `make release-check` PASS (139 disclosed local-path files, D-100 and D-114). No frozen file, freeze, result or executed outcome changed; no frozen held-out suite was rolled out; nothing was pushed, tagged, released or deployed.

## Session 13: post-release extension X1, the action-clamp test (PASS; exploratory, after the freeze)
| Step (Session 13 prompt) | Status | Evidence |
|---|---|---|
| Prerequisites, by executable checks | DONE: `make freeze-check` unchanged; the pinned image is present; the 60 Session 7 H4 summaries match their recorded SHA-256; baseline full suite alone 526 passed | `logs/session13/baseline_full_suite.log` |
| 1. Decisions before acting | DONE: D-106 (Makefile s13, evidence session 13, `x1-run`, `x1-analyze`); D-107 (the clamp, new files, 45 evaluations, outputs, checks, analysis) and its pre-preregistration amendment | DECISIONS, E13-01 |
| 2. Engineering checks before any reported rollout | DONE, all PASS: the clamp-disabled wrapper reproduces 2 Session 7 summaries exactly (50/50 episodes each); the clamp changes no diffusion action (5 development starts per task); the analysis's record checks hold on real records; isolation snapshot taken | E13-02, `logs/session13/x1_checks/` |
| 3. Preregistration committed before the first reported rollout | DONE: `ab02469`; the commit order shows it | `docs/extensions/X1_PREREGISTRATION.md` |
| 4. The 45 clamped BC evaluations, once, canonical container | DONE: 45/45, 0 failed; canonical, n = 50, 0 errors, audits PASS, from clean `ab02469`; isolation PASS | E13-04 |
| 5. Analysis, new script, Session 7 summaries through their SHA-256 | DONE: `make x1-analyze` once; the frozen bootstrap reproduced exactly; `reports/session13/x1_clamp/` | E13-05 |
| 6. KI-052 updated, labelled "exploratory, after the freeze" | DONE | KNOWN_ISSUES KI-052 |
| STOP | see below | |

**Result (exploratory, after the freeze; `reports/session13/x1_clamp/tables.md`).**
- **Setup.** The 18 frozen BC checkpoints were re-run on the Session 7 robustness sets with
  the diffusion sampler's exact action bound.
- **Primary.** Lift arm offset, drop(diffusion) − drop(clamped BC-MLP-chunk) =
  **−40.0 pp [−54.0, −25.3], separated (−)**. The frozen value is −40.0 [−54.0, −26.0].
  Reading R1: the bound does not account for all of the frozen H4 gap.
- **The clamp acted, but changed no outcome.** It changed 52–54 % (BC-MLP-chunk) and
  64–73 % (BC-MLP) of the executed actions on the Lift OOD sets, yet no clamped − unclamped
  success difference separated (0 of 15; largest |Δ| 0.7 pp). On Can every BC outcome was
  unchanged.
- **Verdicts.** Every between-family OOD difference and drop contrast keeps its frozen
  verdict.
- **H2-c.** Clamped − unclamped BC-MLP-chunk on `robustness_base` = +0.0 pp on both tasks,
  so H2-c parity stands as measured.
- **Interpretation (BC side).** Bounding BC's executed actions to the training range leaves
  BC's outcomes unchanged, so the H4 gaps are not caused by BC executing out-of-range actions.
- **Not tested (diffusion side).** The sampler clips x̂₀ at every DDIM step. Whether that
  in-sampler clipping contributes to diffusion's own robustness needs a `clip_sample=False`
  arm, which was not run. KI-052 is half-resolved.
- **Scope.** One bound, three seeds, two perturbations, robustness starts only. The frozen
  numbers are unchanged.

**STOP checks.** On the Session 13 tree (code as `ab02469`; documents only after `c2a9ae6`), host, no container running: `make lint` clean; full suite alone **550 passed** (8:28; the 526 of the baseline plus 24 new X1 tests; `logs/session13/stop_full_suite.log`); `make freeze-check` unchanged ("freeze committed; frozen code unchanged", also before every code change); `tests/research_integrity` 50 passed; `make release-check` PASS (136 disclosed local-path files, D-100). No frozen file, freeze, result or executed outcome changed; nothing was pushed, tagged, released or deployed.

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
| 8. STOP: docs, checks, commit, completion summary | DONE: STOP checks and publication record below | this file, `docs/HANDOFF.md` |

**STOP checks.** On `b4bd070` (release candidate), host: `make lint` clean; full suite alone
526 passed (8:32, `logs/session12/stop_full_suite.log`); `make freeze-check` unchanged; both
gates open; `make paper-check` pass; `make release-check` PASS; `make web-build`, `web-test`
49/49, `web-e2e` 7/7. After the test-harness fix for KI-061 (`ab17fb2`, `tests/conftest.py`
only): `make lint` clean; full suite alone 526 passed (8:20,
`logs/session12/stop_full_suite_2.log`); `make freeze-check` unchanged; `tests/research_integrity`
45 passed; `make release-check` PASS. CI on `ee1e574` (fresh checkouts, Linux aarch64): all
three workflows pass (publication record).

**Publication record** (each row verified before it was written; owner authorization
D-099, D-101, D-102):

| Step | Verified state | Evidence |
|---|---|---|
| Repository | public, `https://github.com/zackholt05-tech/roboforge`; `main` pushed (first `b4bd070`, then each later commit, checked with `git ls-remote`) | `gh repo view`; audit §7 |
| CI, Linux x86_64, `b4bd070` | research-integrity pass (run 42755805851); python-tests fail, triton segfault (42755805869); web-build fail, mesh check 7.45e-9 m (42755806070) | KI-061 |
| CI, Linux aarch64, `e11ad04` | research-integrity pass (42756520065); web-build pass (42756520080); python-tests fail, same triton segfault (42756519847) | KI-061 root cause |
| CI, Linux aarch64, `ee1e574` (after the harness fix) | python-tests pass (42758938590: lint; 303 passed, 12 skipped: 9 without the processed Lift data, 2 without `runs/`, 1 needs MPS); research-integrity pass (42758938394); web-build pass (42758938356) | D-101 amendment 2 |
| Observatory | GitHub Pages from branch `gh-pages`, orphan commit `c14e041` (Zackary Holt), staged by `make pages-bundle` from clean `b4bd070`; build `built`; `https://zackholt05-tech.github.io/roboforge/` | `gh api …/pages` |
| Live site check | all 186 files listed in `deploy.json` fetched from the live URL with equal SHA-256; the site's E2E run against the live URL 7/7 | `logs/session12/pages_live_e2e.log` |
| External links after publication | 19/19 answered with HTTP < 400 | `reports/session12/release_check_network.json` |
| CI, Linux aarch64, `7b394fa` (the tagged commit) | python-tests pass (42760766015), research-integrity pass (42760766276), web-build pass (42760766117) | |
| Tag `v1.0.0` | annotated tag object `82f3219` on `7b394fa`, tagger Zackary Holt <zackholt05@gmail.com>, no trailer; pushed (`git ls-remote`) | `git cat-file -p v1.0.0` |
| GitHub release | `https://github.com/zackholt05-tech/roboforge/releases/tag/v1.0.0`, published 2026-11-10 05:24 UTC, not a draft or prerelease; notes written by hand (no generated notes, no byline); asset `main.pdf` (878,204 bytes), GitHub digest and the SHA-256 of the downloaded copy both `c648e7532e7112cd8bcb3b4e0f118342d6eb97dffd94d3bd1e5448c81b50939f`, equal to the committed `paper/main.pdf` | `gh release view`, `gh release download` |

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
4. Done (Session 12): the Observatory is hosted on GitHub Pages at no cost (owner choice,
   D-099, D-102).
5. Later, not approved: Can for the RGB study (about $15, COMPUTE_PLAN §7.4). Any further
   Level 2 work, such as point-cloud augmentation, ego-centric frames or Can, would be a new,
   separately preregistered study.
