<!-- parsable status header -->
```yaml
CURRENT_SESSION: 2
SESSION_STATUS: PASS            # simulator adapter, eval suites, recorder, three.js export; GO for Session 3
NEXT_SESSION: 3
BLOCKERS: none for Session 3    # open items carried forward: KI-001, KI-007, KI-008, KI-011, KI-019, KI-020
LAST_VERIFIED_COMMIT: b739e99be81803b5e596ed8e8357633780e6a4d3   # lint, 185 tests, container smoke Lift 17/17 + Can 16/16
# Code changes 283a811 -> b739e99: infra/containers/run_sim_cpu.sh only (ROBOFORGE_EVIDENCE_SESSION selects
# the writable reports/logs session dirs; physics path unchanged). Verify: git diff 283a811 b739e99 --stat
EVIDENCE_COMMIT: 283a8115d63b8698503af607a67f6a06c3676745   # every Session 2 report regenerated from this clean commit
EVALUATION_PLATFORM: linux-aarch64-container-1f685638f28c   # D-017, configs/eval/platform.yaml
FINAL_TEST: LOCKED              # docs/FINAL_TEST_FREEZE.md does not exist
RESOURCE_SPEND: {usd: 0.00, gpu_hours: 0, notes: "no remote compute used"}
UPDATED: 2026-11-06
```

# RoboForge Status

## Session 2 acceptance criteria (MASTER_SPEC Session 2)
| Criterion | Status | Evidence |
|---|---|---|
| Prerequisites re-verified (setup, download, check-data, build-data, `data/` clean, test, smoke) | DONE at `4db6526` | `logs/session2/prereq/`, EXPERIMENT_LEDGER E2-01 |
| Simulator wrapper: reset from exact snapshot, action stepping + clipping, reward, success, truncation, cameras, robot/object world transforms | DONE | D-024; `src/roboforge/sim/env_adapter.py`; `tests/simulator/test_env_adapter.py`; smoke adapter checks |
| Demo replay on ≥ 10 sampled demos/task with divergence summary and video inspection; controller bounds | DONE: 12/task, all gates PASS, 0 clip events | D-029; `reports/session2/replay/` (E2-10) |
| Seed reproducibility: two fresh processes, same controlled seed, matching trajectory within declared tolerance | DONE on the canonical container (tolerance 0.0): fresh processes restoring the stored snapshot give identical `physics_sha256`, and a fresh process regenerating the start from its seed reproduces the stored start and trajectory bitwise. On macOS only snapshot restores are bitwise; seed regeneration there diverges by up to 0.144 (Lift) / 1.49 (Can), which is why evaluation always restores snapshots (KI-013) | `reports/session2/determinism/` (E2-13, E2-14) |
| Environment receives a valid dynamic-dimension action | DONE: action dim read from the committed `ActionSpec` (7) and checked against the env; wrong shapes/NaN/inf rejected, out-of-bounds clipped and reported | `tests/simulator/test_env_adapter.py`; smoke `adapter_valid_dynamic_dim_action` |
| Eval initial-state generator, validation vs final test, episode limits, success definition | DONE: 50 + 50 starts/task, hashed; 400-step horizon; first-success rule | D-025, D-026; `data/eval_states/`; RESEARCH_PROTOCOL §5 |
| Final-test states saved, hashed and blocked until `FINAL_TEST_FREEZE.md` | DONE: loading or resetting to the test seed block raises; enforced by tests | `tests/research_integrity/test_final_test_lock.py` |
| Test metadata identifies train/validation/test partitions | DONE: partition label in every snapshot, record and report; `partitions.json` per task | D-025; `data/eval_states/*/partitions.json` |
| Random/no-op policies + demo-replay instrumentation with non-autonomous labels | DONE | D-027; `src/roboforge/eval/policies.py` |
| Random policy produces correctly recorded closed-loop outcomes | DONE on the canonical container: random and no-op 0/50 on both tasks, every episode truncated at 400, 0 errors, 0 clips; re-simulation audits pass | D-028; `reports/session2/sanity_eval/` (E2-15); `tests/simulator/test_rollout.py` |
| Full rollout recorder with frame-indexed kinematics and frames | DONE | D-028; `src/roboforge/eval/rollout.py` |
| One real clip exported; MuJoCo → three.js conversion verified with reference frames | DONE: Lift `demo_0` clip; FK 1e-15, projection 1e-13 px (3e-13 px in three.js r187), masks contain 100 % of projected vertices | D-030; `reports/session2/web_diagnostic/` (E2-16) |
| KI-014 (markers) and KI-016 (XML) resolved; evaluation model chosen | DONE | KNOWN_ISSUES; `reports/session2/sim_contract_*.json` (E2-11) |
| Evaluation platform (D-017) and termination rule decided | DONE: pinned Linux aarch64 container; first success | D-017, D-026 |
| `make smoke-sim` and simulator tests pass | DONE: smoke Lift 17/17, Can 16/16; 185 tests passed | `logs/session2/{smoke_sim,test}.log` (E2-08) |
| No training, no paid compute, no publication; committed data unchanged | DONE | `git status --short data/` empty after regeneration |

## Owner decisions pending
1. Optional: prune the Docker build cache if disk space is needed (KI-008; 63 GB free
   at Session 2 end).
2. Nothing else is needed for Session 3. Session 3 trains small models on local CPU only.
   Any paid compute needs approval first.
