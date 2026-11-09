<!-- parsable status header -->
```yaml
CURRENT_SESSION: 7
SESSION_STATUS: PASS            # Level 1 complete and frozen (docs/LEVEL1_COMPLETE.md): Can suite, robustness + 3 ablations, freeze, held-out test run once
NEXT_SESSION: 8
BLOCKERS: none for Session 8    # Gate D (RGB) needs an owner budget decision after the feasibility pilot; open items: KI-001, KI-008, KI-011, KI-021, KI-023, KI-024, KI-028, KI-033, KI-034, KI-035
LAST_VERIFIED_COMMIT: bf0a0fa   # last code/test change: lint clean, 372 tests; later commits evidence/docs only, re-verified at the STOP commit (LEDGER E7-15)
PROTOCOL_COMMITS: {v1: 8b08b95, diffusion_amendment: 3aed462, main_budget_amendment: ec9626b, lift_search_space: 6dce01b, can_search_space: 96aa972, section14: 2859d04}
EVIDENCE_COMMITS: {can_suite: dadd533, can_latency_figures_config: c380a5e, robustness_states: [5d9b8c0, 005bc94], robustness_ablations_rehearsal: a92c74d, freeze: bf0a0fa, heldout_results: 97cc48a}
EVALUATION_PLATFORM: linux-aarch64-container-1f685638f28c   # D-017, configs/eval/platform.yaml
FINAL_TEST: TESTED_ONCE         # docs/FINAL_TEST_FREEZE.md committed at bf0a0fa; 26 held-out evaluations at bf0a0fa; results reports/session7/results/ (97cc48a); never rerun
RESOURCE_SPEND: {usd: 0.148, gpu_hours: 0.09, session7_usd: 0, notes: "Session 7: local CPU 46.19 core-h training (27 runs), 64.25 container-h evaluation (667 evaluations), $0. Modal total unchanged (Gate A, A-01 closed)"}
APPROVED_LIMIT: {gate_c: "local CPU only, $0 (Level 1 main suite; done)", gate_d_rgb: "not yet requested", modal: "none until a new approval row"}
UPDATED: 2026-11-08
```

# RoboForge Status

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
1. **Gate D (Session 8, RGB visual policies):** a budget decision after the measured
   feasibility pilot. It can be local CPU at $0 or a capped Modal approval row.
2. Optional: set a Modal **workspace budget** (the only hard spending cap) before any paid
   session.
3. Optional: prune the Docker build cache (KI-008). 22 GB was free at the end of Session 7,
   and RGB datasets may need more.
