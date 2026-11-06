<!-- parsable status header -->
```yaml
CURRENT_SESSION: 3
SESSION_STATUS: PASS            # protocol v1 committed first; BC-MLP + BC-RNN trained, reloaded, rolled out on Lift
NEXT_SESSION: 4
BLOCKERS: none for Session 4    # open items carried forward: KI-001, KI-008, KI-011, KI-021, KI-023, KI-024
LAST_VERIFIED_COMMIT: a37b92d   # lint clean, 231 tests; code since 969e14f: Makefile + scripts/diagnose_actions.py only
PROTOCOL_COMMIT: 8b08b95        # docs/RESEARCH_PROTOCOL.md v1, committed before any reported run
EVIDENCE_COMMITS: {training_and_validation: 969e14f, diagnostics_cost_equivalence: a37b92d}   # all dirty=false
EVALUATION_PLATFORM: linux-aarch64-container-1f685638f28c   # D-017, configs/eval/platform.yaml
FINAL_TEST: LOCKED              # docs/FINAL_TEST_FREEZE.md does not exist; no test state loaded
RESOURCE_SPEND: {usd: 0.00, gpu_hours: 0, notes: "local CPU only; no remote compute"}
UPDATED: 2026-11-06
```

# RoboForge Status

## Session 3 acceptance criteria (MASTER_SPEC Session 3)
| Criterion | Status | Evidence |
|---|---|---|
| Prerequisites re-verified at `0322a15` | DONE: all PASS, `data/` clean, 185 tests, smoke macOS 33/33, container 17/17 + 16/16, image ID equal | `logs/session3/prereq/`, LEDGER E3-01 |
| KI-007 re-checked inside the repo before training code | DONE: not blocked; RESOLVED | `logs/session3/prereq/ki007_probe.log`, E3-02 |
| `docs/RESEARCH_PROTOCOL.md` committed before any reported run | DONE at `8b08b95`. It fixes hypotheses, tasks, observation privilege, splits, the paired schedule, success, the selection rule (§7), matched budgets (§6.3) and the bootstrap plan (§9) | E3-04 |
| Shared policy/trainer/checkpoint layer | DONE: strict configs, seeded sampler, masked MSE, gradient-norm logging, `weights_only` checkpoints, immutable manifests, `LearnedPolicy`, validation eval, selection | D-032–D-037; `src/roboforge/{models,train,eval}/` |
| BC-MLP and BC-GRU/LSTM | DONE: BC-MLP 2 × 1024 (1,077,255 params) and BC-RNN LSTM 2 × 296 (1,080,703, +0.3 %); plus the BC-MLP-chunk control (1,204,336). GMM not implemented (optional) | D-033; `reports/session3/param_counts.json` |
| Tiny-batch and tiny-episode overfit tests | DONE: MSE ×1.7e-6 to ×0.015 on real Lift windows | `tests/integration/test_train_real_data.py`, `reports/session3/overfit_lift.json`, E3-06 |
| Forward/backward/gradient-norm tests | DONE: every parameter gets a finite, non-zero gradient; RNN masking; shapes | `tests/unit/test_bc_models.py` |
| CPU train/eval smoke | DONE: `make train-smoke` PASS | E3-05 |
| Checkpoint save/reload equivalence (identical outputs, identical closed-loop `physics_sha256` on one platform) | DONE: reload check bitwise in every manifest; two loads, a fresh process and the recorded eval give identical `physics_sha256` on the container (selected and last checkpoints) and on macOS; bitwise resume | `reports/session3/checkpoint_equivalence/`, E3-10 |
| Both baselines consume the actual demonstration data | DONE: `SequenceDataset.from_committed` only; manifests carry the committed hashes; windows tested equal to the dataset | `tests/integration/test_train_real_data.py`, run manifests |
| Real Lift rollouts in the pinned simulator; validation-only, no test contamination | DONE: 18 canonical evaluations, 900 episodes, all `eval_validation`, 0 errors, audits pass | `reports/session3/validation/`, E3-08, E3-13 |
| Initial closed-loop Lift behavior + clipping/sign/coordinate diagnosis | DONE: controller axes/signs correct; failures are grasp failures with out-of-workspace drift; clipping is a symptom (KI-021); gripper clipping harmless (KI-022) | `reports/session3/diagnostics/actions.json`, E3-11 |
| True parameter counts + matched-capacity configurations | DONE: primary tier ±0.3 %; tiers at 0.27 M / 1.08 M / 4.3 M for MLP, LSTM, GRU, chunk | `reports/session3/param_counts.json` |
| Baseline training cost and inference latency | DONE: train BC-MLP 146 s, BC-RNN 983 s, BC-MLP-chunk 155 s (30k steps, CPU); container latency 0.91 / 8.6 / 1.26 ms per decision | `reports/session3/cost_latency/`, E3-07, E3-12 |
| No paid compute, no publication, committed data and eval suites unchanged, `uv.lock` unchanged | DONE | `git diff 0322a15 -- uv.lock data/` empty |

## Pilot validation results (development numbers, one seed, Lift validation suite)
| Family | Successes at 5k / 10k / 15k / 20k / 25k / 30k steps | Selected (§7) |
|---|---|---|
| BC-MLP | 47 / 41 / 45 / 41 / 42 / 42 of 50 | step 5000 (47/50) |
| BC-RNN | 50 / 50 / 50 / 48 / 49 / 49 | step 15000 (50/50) |
| BC-MLP-chunk | 49 / 50 / 49 / 48 / 48 / 46 | step 10000 (50/50) |

These are not results. They come from one training seed on the in-distribution validation
suite, and selection used that same suite.

## Owner decisions pending
1. None needed for Session 4, which is local CPU only. Any paid compute needs approval first
   (Session 5).
2. Optional: prune the Docker build cache if disk space is needed (KI-008; 55 GB free).
