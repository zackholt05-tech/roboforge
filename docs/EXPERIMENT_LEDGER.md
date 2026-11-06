# Experiment Ledger

Chronological record of every executed experiment or engineering check. Results are
recorded as observed, including failures. **No research experiments (model training or
policy evaluation) have been run.**

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

Owner approvals for paid compute: **none requested, none granted.**
