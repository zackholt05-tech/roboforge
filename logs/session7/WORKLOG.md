# Session 7 work log (running notes; logs/ is outside the dirty check)

## State at 2026-11-08 00:10 UTC
- main HEAD 96aa972 (plan + search space), clean; Can suite running since 23:26:38 UTC
  (`make suite`, plan configs/experiments/level1_core_can.yaml, sha 881b4fb7…).
- s7-dev branch (worktree ../roboforge-s7dev), commits on top of 96aa972:
  2859d04 protocol §14 prereg (+ second amendment), e9488f4 robustness starts,
  acb24f2 evaluator variants, bfe7c46 taxonomy, 0332650 bootstrap, d9c8683 held-out tooling.
  Merge (fast-forward) into main only after the suite ends.
- Protocol amendment commit time (2859d04 / e9488f4) must be recorded vs the first
  robustness run (none yet). The §14 amendment was committed 2026-11-07 ~23:31 UTC; first
  Can validation evaluation launched 23:42:15 UTC (not governed by §14).

## Timeline estimate (estimate, not measured)
- Eval capacity (4 containers, ~14-15 min per Can evaluation under load) binds: 480 Can
  validation evaluations ≈ 30 h. Diffusion seed-0 training ≈ 6 h (4.6 steps/s under load).
- Queue priority: diffusion evaluations first (family order), so diffusion chosen ≈ +12 h,
  its replicates trained ≈ +4 h, all done ≈ 2026-11-09 06:00 UTC (estimate).
- After the suite: robustness states (container), Lift 36 + Can 24 robustness evals, A1 6,
  A2 6, A3 (3 trainings + 60 val evals + 3 robustness), Can latency (nothing else running),
  rehearsal, freeze, held-out 26 evals, analysis.

## Disk plan
- 21 GB free at 00:05 UTC. Suite needs ≈ 15 GB more (runs ≈ 300 MB each, Can eval ≈ 24 MB).
- After merge: remove the s7-dev worktree (.venv 1.9 GB) and consider logs/tmp (1.1 GB,
  gitignored scratch from Sessions 4-7). Guard SIGINTs the suite below 3 GB.

## Checks done
- Taxonomy on Can bc_mlp validation records (steps 5k-20k): one label per episode,
  successes equal the summaries.

## Pending before the freeze
- Can taxonomy dry run on all Can validation records (§14.5 requires it before the freeze).
- Integrity tests `test_freeze_file_does_not_exist_yet` and
  `test_loading_final_test_states_is_refused` change in the freeze commit.
- Test of disjointness of robustness_base from the held-out manifests (hash-level).
- Container attestation end to end (lock_status only, no state read) before the freeze.

## 2026-11-08 00:30 UTC: s7-dev now at cef11a3 (D-067, D-068, D-069 + rehearsal oracle)
Advisor review fixed: infra/ mount, whole-package code hash, analysis writes last,
incomplete-family rule, host-wide container slots (logs/container_slots), disk check.

## Post-suite sequence (in order)
1. Suite `end` event → `git merge --ff-only s7-dev` → lint/test → THEN commit suite evidence (reports/session7, logs/session7 suite
   files; NOT freeze_prep/) → `make test` again.
2. Free disk: remove worktree (git worktree remove ../roboforge-s7dev; its runs/ holds only
   symlinks), consider logs/tmp after grepping docs for references.
3. `make figures SUITE_SUMMARY=reports/session7/suite/level1_core_can.json LATENCY=...` after
   latency; `make test-config TEST_TASK=can` (commit the record).
4. Can latency, NOTHING else running: `make benchmark-policies RUNS="<seed-0 run of each
   chosen Can config>"` → reports/session7/cost_latency/<platform>.json. Record host load.
5. Can taxonomy dry run on all Can validation records (one label each; §14.5).
6. `make robustness-states` (container) → commit suites; `make check-robustness-states`.
7. A3: write configs/models/diffusion_unet_to1_can.yaml (chosen Can diffusion optimizer +
   horizons.obs 1) and configs/experiments/ablation_a3_can.yaml (one trial,
   replicates_with_tuning: true); commit; launch suite with SUITE_PLAN, --max-train 3.
   Simultaneously `run_evals.py --batches robustness a1 a2` (slots keep total ≤ 4).
8. After A3: `run_evals.py --batches a3 --a3-summary reports/session7/suite/ablation_a3_can.json`;
   `analyze_robustness.py`.
9. Rehearsal: draft freeze (main, relative paths) in logs/session7/freeze_prep/; run_heldout
   --rehearsal-suite validation --draft-freeze ...; check_rehearsal.py; analyze_heldout
   --rehearsal-dir ... --draft-freeze ... with the same --latency args.
10. Freeze commit: docs/FINAL_TEST_FREEZE.md (write_freeze_block --document) + post-freeze
    lock tests (logs/session7/freeze_prep/test_final_test_lock.post_freeze.py).
11. In container: es.lock_status(task) for both tasks (no state read) → `make eval-final`
    → `make analyze-final`.
12. LEVEL1_COMPLETE.md, docs, ledger, Session 8 prompt, STOP.

## 2026-11-08 05:2x UTC: suite crash and restart (D-070, KI-036)
- run_suite.py died 02:14:19 UTC (JSONDecodeError reading a half-written lr3e4 summary); no
  suite_error/end event, so the watcher never fired; disk guard + renice loop exited with it.
- Orphans finished fine: 6 trainings complete (bc_rnn t3/t4 ~02:59, diffusion x4 ~04:15),
  4 lr3e4 evals (5k-20k) audit PASS n=50. 44 summaries = 44 val dirs. Nothing lost.
- Fix committed on main f0715c4 (run_suite.py + test + D-070/KI-036 only; train/eval code
  = 96aa972). Run-1 logs kept as *.run1.*. Restarted 05:21 UTC (events.jsonl line 114).
- s7-dev can no longer fast-forward: after the suite, REBASE s7-dev onto main (conflicts:
  run_suite.py — keep the D-070 read guards AND the slot/disk/replicates logic; end of
  DECISIONS.md — order D-063..D-069 before D-070). Re-run make lint/test after.
- Ledger: record the gap (02:14 -> restart), the 10 jobs without finished events, two git SHAs.

## 2026-11-08 07:40-08:10 UTC
- Diffusion chosen 07:37:12 UTC: trial 2 (cos), step 65k, 48/50 (trials 1, 4 also 48/50; tie on
  mean first-success step 110.46 vs 110.58 / 112.75). Replicates seed1/seed2 training.
- 124 Can validation summaries at 07:37 (80 diffusion evals took 05:21-07:37, ~35/h, no
  training load).
- s7-dev: MERGED main (not rebased, keeps 2859d04's original SHA/time) -> c42b0bf; drain
  fallback also releases slots. make lint pass, make test 363 passed 4 skipped.
  6b13f4f: A3 config + plan (D-071), test that only horizons.obs and name differ.
- After the suite: fast-forward main to s7-dev (main is an ancestor of s7-dev).
  ORDER (corrected 14:25 UTC): `git merge --ff-only s7-dev` FIRST, then lint/test, THEN commit
  the suite evidence on top (an evidence commit first would make main no longer an ancestor).
  Checked 14:25: no path in `git diff --name-only f0715c4 s7-dev` is untracked/modified in main.
  Keep logs/session7/freeze_prep/ OUT of the evidence commit (names the held-out suites);
  review the plan_resolved.json diff (rewritten by the restart); `make test` again after.

## 2026-11-08 14:18 UTC: all four Can choices made
- bc_rnn chosen 14:18:07: trial 2 (cos), step 30k, 50/50 (only 50/50 trial; t1 49 @35k,
  t3 48 @80k, t4 49 @45k). Replicates s7_bc_rnn_cos_can_seed1_d34c6b99 / seed2_8452a37a
  training; 440/480 Can validation summaries, eval queue empty. Remaining: those 2 runs x 20.
- A3 again NOT started early (advisor 14:2x): from main it would record f0715c4 (no §14, D-071,
  A3 config) and bypass the orchestrator's events; run IDs would match (checked) but the
  provenance gap is not worth ~1-2 h. Launch A3 via the suite after the fast-forward.
- Choices: diffusion t2 cos 65k 48/50; bc_mlp t2 cos 55k 37/50; bc_mlp_chunk t4 wd0.1 95k 50/50;
  bc_rnn t2 cos 30k 50/50.
- A3 training NOT started early: main's tree must not change during the suite, and the
  worktree is dirty by the untracked data/ symlinks (manifests would say dirty).
- Taxonomy dry run 14:3x UTC (§14.5; s7-dev taxonomy code, taxonomy sha 5a5d998f…): 440 Can
  validation summaries, 22,000 episodes, one label each, taxonomy successes == summary successes
  for all 440, 0 engineering_error. Totals: success 17020, gripper_never_closed 1842,
  no_grasp_drift 1354, no_grasp_near 411, timeout_holding 184, in_goal_bin_not_success 505,
  dropped 684. Output logs/session7/analysis/taxonomy_dryrun_can_validation.json. Re-run on the
  final 40 bc_rnn replicate summaries after the suite.
- bc_rnn replicates train at ~33 steps/s (13.3k steps in 397 s): suite end ~15:45-16:15 UTC (estimate).

## 2026-11-08 15:54-16:40 UTC: suite end -> robustness, A3, rehearsal running
- Suite `end` 15:54:11 UTC, exit 0, failed []. bc_rnn seed2 selected 35k 49/50 (seed1 25k 50/50).
- main fast-forwarded to s7-dev (6b13f4f); lint pass; make test 365 passed 3 skipped (twice).
- dadd533 suite evidence (480 summaries, selection, tuning, suite logs, taxonomy dry runs).
  Taxonomy dry run on all 480: 24,000 episodes, problems 0, engineering_error 0; totals success
  18807, gripper_never_closed 1881, no_grasp_drift 1397, no_grasp_near 416, timeout_holding 206,
  in_goal_bin_not_success 557, dropped 736.
- Worktree ../roboforge-s7dev removed (symlinks deleted first); 17 GiB free.
- c380a5e Can latency (16:13-16:15 UTC, 0 containers, load 2.7-4.1), figures, can_test_configuration
  (sha 116f10a4…). Medians: diffusion 37.71 ms, bc_mlp 1.08, bc_mlp_chunk 1.42, bc_rnn 9.00.
- 5d9b8c0 robustness starts (container, 0 rejected draws, disjoint). Native check PASS (1.1e-16).
  Container --check first run FAILED can/ood_arm_offset "not bitwise": tuple-vs-list contact_pairs
  in the comparison (KI-037), fixed 8a19943 (same_manifest + test); rerun 5/5 bitwise (005bc94).
- 16:36:51 UTC A3 suite (plan sha c44cfdc2…, --max-train 3, git 005bc94 clean); run IDs as
  pre-recorded. 16:37:11 UTC run_evals robustness a1 a2 (72 jobs). FIRST ROBUSTNESS ROLLOUT
  16:37:11 UTC (the §14 amendment 2859d04 was committed 2026-11-07 ~23:31 UTC).
- 16:39:25 UTC rehearsal: run_heldout --rehearsal-suite validation --draft-freeze
  logs/session7/freeze_prep/draft_freeze.md (= write_freeze_block --document 2026-11-09 from 005bc94).
- Watchers: bk4lvhwcs (A3 + run_evals), bi8dcq4by (rehearsal).
- Freeze waits for (2) to finish (prompt order; and no tracked edits while launchers run).

## 2026-11-08 17:40-17:50 UTC: rehearsal PASS
- Rehearsal 26/26 finished 17:40:23 UTC, failed []. check_rehearsal.py PASS: all 24 frozen
  checkpoints reproduce their validation evaluations episode by episode; random_uniform 0/50 per
  task, audits pass, no errors (logs/session7/rehearsal/check_rehearsal.{json,log}).
- analyze_heldout --rehearsal-dir: lift complete, can complete -> logs/session7/rehearsal/heldout/results
  (labelled REHEARSAL; validation states; not test results).
- Disjointness vs held-out: tests/research_integrity/test_robustness_disjoint.py compares
  robustness hashes with validation AND held-out open manifests: PASS (2 tests).
- Container lock_status attestation: at freeze time (no free slot now; Gate C).
- No commits until run_evals and the A3 suite exit (HEAD must not move under running launchers).
- A3's 60 validation summaries will land in reports/session7/validation (540 files): filter
  by run ID (`_to1_`) in any later glob.

## 2026-11-08 19:15-22:45 UTC: robustness, A1-A3 done; KI-038
- run_evals robustness+a1+a2 ended 19:15:24 UTC: 72/72 ok, 0 failed (from 005bc94, clean).
- analyze_robustness.py failed at once: KeyError 'step' (loader read s["step"], the writer stores
  checkpoint.step; no test). KI-038. A3 was still running, so the tracked file stayed unchanged;
  a fixed copy in the scratchpad confirmed the analysis completes. The test draft failed on the
  old line and passed on the fix.
- A3 suite ended 22:24:54 UTC, exit 0, failed none; chosen trial 1; selections seed0 95k 49/50,
  seed1 100k 49/50, seed2 90k 49/50 (development numbers). 60 validation evals clean at 005bc94.
- run_evals --batches a3: 3/3 ok 22:25-22:30 UTC (49, 46, 46 of 50 on robustness_base).
- All launchers exited; make lint + make test (371 passed); fix committed 565c6e4; the final
  analysis (with A3) ran from 565c6e4 clean -> reports/session7/robustness/.
- Rehearsal episode records (218 MB under logs/session7/rehearsal/heldout/runs/) and
  freeze_prep/ are gitignored, not committed.

## 2026-11-08 22:30-24:00 UTC: freeze, held-out test, STOP
- Container attestation before the freeze: locked (both tasks). The freeze was generated from
  a92c74d with --document 2026-11-08 and differs from the rehearsed draft only in its date.
- Lock-dependent test audit (advisor): test_env_adapter's seed-lock test would have built a
  held-out start once the lock opened (KI-039). It was fixed in the freeze commit bf0a0fa.
- make freeze-check OK, lint OK, test 372 passed; container attestation after: unlocked,
  freeze blob bfdb4b77 = HEAD blob.
- make eval-final once, 22:50:42-23:48:50 UTC: 26/26 ok, failed none. make analyze-final once:
  complete for both tasks. Committed 97cc48a.
- STOP docs: LEVEL1_COMPLETE, STATUS, HANDOFF (with the Session 8 prompt), DECISIONS D-072,
  KNOWN_ISSUES KI-039, LEDGER E7-02-E7-15 with cost and tuning accounting, COMPUTE_PLAN.
  Final lint, freeze-check and test (372) on the STOP tree.
