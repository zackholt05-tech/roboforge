<!-- parsable status header -->
```yaml
CURRENT_SESSION: 6
SESSION_STATUS: PASS            # Lift main suite complete (24 runs, 480 canonical evaluations, 0 failures); checkpoints selected; Lift test configuration recorded, not frozen
NEXT_SESSION: 7
BLOCKERS: none for Session 7    # open items carried forward: KI-001, KI-008, KI-011, KI-021, KI-023, KI-024, KI-028, KI-033, KI-034, KI-035
LAST_VERIFIED_COMMIT: 4721c01   # lint clean, 295 tests; later commits are docs/evidence only
PROTOCOL_COMMITS: {v1: 8b08b95, diffusion_amendment: 3aed462, main_budget_amendment: ec9626b, lift_search_space: 6dce01b}
EVIDENCE_COMMITS: {lift_suite_code: 6dce01b, figures_code: [1280b5d, 4721c01], lift_suite_evidence: "the Session 6 STOP commit, the child of 4721c01"}   # every run and evaluation manifest dirty=false at 6dce01b
EVALUATION_PLATFORM: linux-aarch64-container-1f685638f28c   # D-017, configs/eval/platform.yaml
FINAL_TEST: LOCKED              # docs/FINAL_TEST_FREEZE.md does not exist; no test state loaded; Lift held-out schedule referenced by hash only
RESOURCE_SPEND: {usd: 0.148, gpu_hours: 0.09, session6_usd: 0, notes: "Session 6: local CPU 33.25 core-h training, 26.92 container-h evaluation, $0. Modal total unchanged (Gate A, A-01 closed)"}
APPROVED_LIMIT: {gate_c: "local CPU only, $0", modal: "none until a new approval row"}
UPDATED: 2026-11-07
```

# RoboForge Status

## Session 6 acceptance criteria (MASTER_SPEC Session 6)
| Criterion | Status | Evidence |
|---|---|---|
| Prerequisites re-verified at `273f86a` | DONE: all PASS, `data/` clean, 281 tests, smoke macOS and container, image ID equal, 4 pilot runs hash-checked | LEDGER E6-01 |
| BC-MLP, BC-RNN, BC-MLP-chunk and diffusion trained on Lift PH with the approved seeds and budget; failures tracked | DONE: 24 runs (4 trials × 4 families at seed 0, then seeds 1–2 of each choice), 100k steps, batch 100, checkpoints every 5k. 24/24 `complete`, 0 failed, none retried | E6-04–E6-06, `reports/session6/suite/level1_core_lift.json` |
| At least one real converged training record per core model | DONE: training loss flattened in every chosen run (90–100k mean within 1–5 % of 80–90k) | `runs/<id>/metrics.jsonl`, figure `training_curves.png` |
| Checkpoints selected on validation by the predeclared rule; bounded, equal search on validation only | DONE: §7 on all 20 checkpoints × 50 validation starts per run; ≤ 4 configurations per family, identical across families, predeclared and committed (`6dce01b`) before the first run | E6-05, `reports/session6/{selection,tuning}/`, D-055 |
| Plots: training/validation curves, seed spread, action histograms, validation success and latency, labelled validation-only | DONE: every figure and the table carry the validation-only footer, source and git revision | E6-13, `reports/session6/figures/`, D-057 |
| Real successes and failures inspected | DONE: per-start failures, post-hoc failure categories, clipping, receding-horizon checks; one success and one failure montage per family by a fixed rule | E6-09, E6-10, E6-13 |
| Lift test configuration recorded (configs, checkpoint hashes, initial-state schedule by hash) | DONE, **not frozen**: 12 checkpoints; held-out schedule by manifest hash only | E6-12, D-058, `reports/session6/lift_test_configuration.json` |
| All-in cost record | DONE: **$0** (33.25 CPU core-h, 26.92 container-h, 9.80 h elapsed) | LEDGER Session 6, COMPUTE_PLAN §2 |
| No held-out test cherry-picking | DONE: held-out suite never loaded; contamination check clean; `FINAL_TEST_FREEZE.md` absent | E6-11 |
| Committed data, eval suites, `uv.lock` and dependencies unchanged | DONE | `git diff d956292 -- uv.lock pyproject.toml data/ configs/eval` empty |

## Lift validation results (development numbers: selection used the same 50 validation starts)
Selected checkpoint of the chosen configuration of each family, per seed (successes of 50,
selected step):

| Family | Chosen configuration | Seed 0 | Seed 1 | Seed 2 |
|---|---|---|---|---|
| BC-MLP | trial 4, weight decay 0.1 | 49 (15k) | 48 (5k) | 49 (5k) |
| BC-RNN | trial 4, weight decay 0.1 | 50 (15k) | 50 (10k) | 50 (5k) |
| BC-MLP-chunk | trial 2, cosine schedule | 50 (15k) | 49 (5k) | 50 (10k) |
| Diffusion U-Net | trial 3, lr 3e-4 | 50 (100k) | 50 (95k) | 50 (80k) |

These are **not** held-out results. Lift validation is saturated (KI-034): a 50/50 score has
a Wilson 95 % interval of [0.929, 1.000], so these numbers do not rank BC-RNN, BC-MLP-chunk
and diffusion. Full table: `reports/session6/figures/validation_table.md`.

## Owner decisions pending
1. Optional: set a Modal **workspace budget** (the only hard spending cap) before any future
   paid session.
2. Optional: a future approved L4 probe to settle EGL vs OSMesa (KI-001). It is not needed
   for Level 1.
3. Optional: prune the Docker build cache if disk space is needed (KI-008; 29 GB free at the
   end of Session 6).
