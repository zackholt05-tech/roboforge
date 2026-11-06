# Experiment Ledger

Chronological record of every executed experiment or engineering check. Results are
recorded as observed, including failures. **No research experiments (model training or
policy evaluation) have been run.** Session 1 ran engineering checks only.

| ID | Date | Session | Type | What | Outcome | Evidence | Cost |
|---|---|---|---|---|---|---|---|
| E0-01 | 2026-11-05 | 0 | engineering | Download Lift/Can PH low_dim + raw from HF @74fa018 | 4/4 SHA-256 match | `logs/session0/download_sha256.log` | $0 |
| E0-02 | 2026-11-05 | 0 | engineering | Env construction with mujoco 3.17.0 (unpinned) | **FAIL**: AssertionError in robosuite binding_utils | this ledger, KI-005 | $0 |
| E0-03 | 2026-11-05 | 0 | engineering | Replay sweep, 3 demos × mujoco {3.2.6, 3.2.7, 3.3.0} | all succeed; 3.2.6 ≡ 3.2.7 | DECISIONS D-002 | $0 |
| E0-04 | 2026-11-05 | 0 | engineering | Full open-loop replay, 200 demos/task, mujoco 3.2.7 | Lift 199/200, Can 190/200 | `reports/feasibility/data_report_*.json` | $0 |
| E0-05 | 2026-11-05 | 0 | engineering | Full open-loop replay, mujoco 3.3.0 | Lift 199/200, Can 193/200 | `reports/feasibility/mujoco_version_comparison.json` | $0 |
| E0-06 | 2026-11-05 | 0 | engineering | Smoke sim Lift/Can (seeded reset, random closed-loop steps, success, horizon, render, 2-process determinism) | all checks PASS | `reports/feasibility/smoke_sim_*.json` | $0 |
| E0-07 | 2026-11-05 | 0 | engineering | Observation consistency probe | Lift exact; Can relative blocks lagged by one update (KI-003) | `reports/feasibility/obs_consistency.json` | $0 |
| E0-08 | 2026-11-05 | 0 | engineering | Docker linux/aarch64 build with `uv sync --frozen` | build OK; container run **FAILED** (Docker storage I/O error after disk filled) | `logs/session0/docker_build_sim_cpu.log`, KI-001 | $0 |
| E0-09 | 2026-11-05 | 0 | engineering | Upstream `download_datasets.py --dry_run` | resolves same 4 files | `logs/session0/upstream_download_datasets_dry_run.log` | $0 |
| E0-10 | 2026-11-05 | 0 | engineering | After the owner-approved Docker Desktop restart: first container runs of `smoke_sim`/`verify_data` with `--out-dir /out` | **FAIL**: frames rendered, then `IndexError` recording paths (`out_dir.parents[k]`); `verify_data` frames ignored `--out-dir`; one run killed by a Docker Desktop self-update (29.2.1 → 29.9.2). Fixed in `e0b26ae`, `a9c6793`. Log overwritten by E0-11. | this ledger | $0 |
| E0-11 | 2026-11-05 | 0 | engineering | Linux aarch64 container, OSMesa: smoke Lift/Can + `verify_data --replay-demos all --render` at `a9c6793` | smoke 21/21, integrity 12/12 PASS; Lift 199/200, Can 190/200; frames correct | `reports/feasibility/linux_aarch64_osmesa/`, `logs/session0/docker_linux_aarch64_osmesa_after_restart.log` | $0 |
| E0-12 | 2026-11-05 | 0 | engineering | macOS vs Linux evidence comparison | same outcomes and failing demos; trajectories not bit-identical (KI-013) | `reports/feasibility/linux_aarch64_osmesa/platform_comparison.json` | $0 |
| E0-13 | 2026-11-05 | 0 | engineering | macOS evidence regenerated at `a9c6793` | identical to the `29645cf` run except timing fields; 20/20 frames byte-identical | `reports/feasibility/*.json`, `logs/session0/` | $0 |

| E1-01 | 2026-11-06 | 1 | engineering | Prerequisites at `833277d`: `make setup`, `make download`, `make test`, `make smoke-sim` | setup OK; 4/4 SHA-256 verified (no download); 56/56 tests; smoke Lift 11/11, Can 10/10 | `logs/session1/prereq/` | $0 |
| E1-02 | 2026-11-06 | 1 | engineering | Prototype state→feature function vs stored obs and robosuite (1 demo/task) | 0.0 on every Lift block and Can robot/world blocks; corrected Can rel block = robosuite with fresh cache (0.0) | this ledger (scratch prototype) | $0 |
| E1-03 | 2026-11-06 | 1 | engineering | Closed-loop purity probe: features after `env.step` vs restored state | `sim.data` after `env.step` is one substep stale (max 5.9e-4 over positions in m and quaternion components); computing on scratch `MjData` from the state fixes it (KI-015) | `tests/simulator/test_features_closed_loop.py` | $0 |
| E1-04 | 2026-11-06 | 1 | engineering | Same-step Can relative quaternion sign audit | 446 within-episode flips (stored: 598): rel quat dropped (D-010) | `reports/data_audit.json` | $0 |
| E1-05 | 2026-11-06 | 1 | engineering | Full build, all 400 demos (`scripts/build_data.py`) | all `equal` features 0.0 vs dataset; Can corrected block (robosuite-formula version, superseded by E1-11) 0.0 vs same-step reference; stored block = lag formula (2.2e-16), off by ≤ 0.861 m (t=0) / 0.091 m (t>0); 0 flagged episodes; 400/400 final states succeed | `data/manifests/tasks/*.json` | $0 |
| E1-06 | 2026-11-06 | 1 | engineering | Determinism: `--check` rebuild + second build compared byte-for-byte | PASS; split, normalization and task manifests byte-identical | this ledger | $0 |
| E1-07 | 2026-11-06 | 1 | engineering | Default robosuite model vs dataset model | default adds 2 non-colliding mocap markers (KI-014); `get_xml()` round trip changes features by 1.1e-7 (KI-016) | KNOWN_ISSUES | $0 |
| E1-08 | 2026-11-06 | 1 | engineering | Spec change: T_o removed from `ObservationSpec` (`cb9b6a3`) | obs-spec and normalization hashes changed; processed content and split hashes unchanged; check PASS | `data/manifests/tasks/*.json` | $0 |
| E1-09 | 2026-11-06 | 1 | engineering | KI-007 probe: write a scratch file (outside the repo) with `backward()` + `optimizer.step()` | not blocked; not re-checked inside the repo | KNOWN_ISSUES KI-007 | $0 |
| E1-10 | 2026-11-06 | 1 | engineering | `build_data.py --check` at `cb9b6a3` in the Linux aarch64 image (OSMesa, read-only mounts) | Lift PASS (bit-identical); **Can FAIL**: processed and normalization hashes differ. Cause: robosuite `quat2mat` is float32 + numba (KI-018) | `logs/session1/docker_linux_aarch64_check_data_cb9b6a3_FAIL.log` | $0 |
| E1-11 | 2026-11-06 | 1 | engineering | Can relative position evaluated in float64 from MuJoCo `xmat` (`c69551f`), rebuilt and re-checked on macOS and Linux | both tasks PASS on both platforms; feature vs float64 reference 7.2e-16, vs robosuite float32 formula 1.4e-7 m; Can hashes changed, Lift unchanged | `logs/session1/docker_linux_aarch64_check_data.log` | $0 |
| E1-12 | 2026-11-06 | 1 | engineering | Clean-tree evidence at `c69551f`: `make test`, `make lint`, `make smoke-sim SMOKE_OUT=logs/session1/smoke_sim`, `make verify-data VERIFY_OUT=logs/session1/verify_data`, `make audit-data`, Linux `--check` | 123 passed; lint clean; smoke 21/21; integrity 12/12 PASS, replay Lift 199/200, Can 190/200, `check-data` PASS ×2; audit written (dictionary unchanged); Linux check PASS ×2; all reports stamped `c69551f`, `dirty: false`; 20/20 re-rendered frames byte-identical to `reports/feasibility/frames` (duplicates not committed, D-023) | `logs/session1/`, `reports/data_audit.json` | $0 |

Owner approvals for paid compute: **none requested, none granted.**
