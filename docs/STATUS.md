<!-- parsable status header -->
```yaml
CURRENT_SESSION: 1
SESSION_STATUS: PASS            # canonical Lift/Can data pipeline; GO for Session 2
NEXT_SESSION: 2
BLOCKERS: none for Session 2    # open items carried forward: KI-001, KI-007, KI-013, KI-014, KI-016
LAST_VERIFIED_COMMIT: c69551fc02f00841a05cbe2352539f1c6aca400c   # all Session 1 evidence regenerated from this clean tree
# The commit after LAST_VERIFIED_COMMIT adds evidence (logs/session1, reports/data_audit*) and docs only.
# Verify with: git diff c69551f HEAD -- src scripts configs tests data Makefile
RESOURCE_SPEND: {usd: 0.00, gpu_hours: 0, notes: "no remote compute used"}
UPDATED: 2026-11-06
```

# RoboForge Status

## Session 1 acceptance criteria (MASTER_SPEC Session 1)
| Criterion | Status | Evidence |
|---|---|---|
| Prerequisites re-verified (`make setup/download/test/smoke-sim`) | DONE | `logs/session1/prereq/`, EXPERIMENT_LEDGER E1-01 |
| Fetcher: resumable downloads, integrity checks, no re-download of verified files | DONE | `src/roboforge/data/fetch.py`; `tests/unit/test_fetch_resume.py` (7 tests) |
| Typed `TaskSpec` / `ActionSpec` / `ObservationSpec` read from the actual HDF5 | DONE | `src/roboforge/data/specs.py`; `data/manifests/tasks/*.json` |
| KI-003 resolved: one state→feature function for dataset windows and closed-loop eval | DONE | D-010; `src/roboforge/data/features.py`; `tests/simulator/test_features_closed_loop.py` |
| Features equal robosuite on Lift (all) and Can world-frame/robot blocks | DONE: max \|Δ\| 0.0 on all 400 demos | `reports/data_audit.json` → `feature_vs_dataset_max_abs` |
| Erroneous or missing episodes flagged, never deleted | DONE (0 flagged) | `src/roboforge/data/sanity.py`; split manifests `flagged: {}` |
| Deterministic episode-level split manifests (upstream masks, all IDs) | DONE | D-019; `data/splits/{lift,can}_ph.json` |
| Window sampler with T_o/T_p and padding masks | DONE | D-022; `src/roboforge/data/sequence.py` |
| Train-only reversible normalization, saved stats + hashes | DONE | D-020; `data/normalization/*.json` |
| Transformed-data checksums; same source → identical splits/hashes | DONE on macOS arm64 and Linux aarch64 | `make check-data`; `logs/session1/docker_linux_aarch64_check_data.log` |
| Dataset audit summaries + explorer figures | DONE | `reports/data_audit.json`, `reports/data_audit/*.png` |
| `docs/DATA_DICTIONARY.md`, `docs/DATA_PROVENANCE.md` | DONE (dictionary generated) | `make audit-data` |
| Tests: no future leakage, window boundaries, normalizer inversion, split determinism, one-batch forward pass on real data | DONE: 123 passed | `logs/session1/pytest.log` |
| `make download`, `make verify-data` and inspection scripts pass on real Lift and Can | DONE | `logs/session1/verify_data.log` (12/12 integrity PASS, `check-data` PASS) |
| Same observation spec for every model family | DONE (by construction: all data flows through `SequenceDataset`; T_o is per-run) | D-018, D-022 |
| No training, no evaluator beyond data needs, no paid compute, no publication | DONE | — |

## Owner decisions pending
1. Optional: prune the Docker build cache (27.5 GB reclaimable at Session 0) if disk space
   is needed (KI-008; 33 GB free at Session 1 start).
2. Nothing else is needed for Session 2.
