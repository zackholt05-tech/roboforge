<!-- parsable status header -->
```yaml
CURRENT_SESSION: 0
SESSION_STATUS: PASS            # GO for Session 1; see reports/feasibility/STACK_REPORT.md §10
NEXT_SESSION: 1
BLOCKERS: none for Session 1    # non-blocking open items: KI-001, KI-003, KI-007, KI-008
LAST_VERIFIED_COMMIT: 29645cf1cf7f5bcb248b56e9585b61be37ad7c0e   # evidence regenerated from this clean tree; committed in the next commit
CODE_COMMIT: 29645cf1cf7f5bcb248b56e9585b61be37ad7c0e
RESOURCE_SPEND: {usd: 0.00, gpu_hours: 0, notes: "no remote compute used"}
UPDATED: 2026-11-05
```

# RoboForge Status

## Session 0 acceptance criteria (user brief §7 / MASTER_SPEC Session 0)
| Criterion | Status | Evidence |
|---|---|---|
| Master spec reviewed in full | DONE | audit findings in DECISIONS.md / KNOWN_ISSUES.md |
| Technical assumptions checked against authoritative sources | DONE | upstream source at pinned commits, PyPI/GitHub/HF APIs, Modal docs (D-001…D-005) |
| Compatible dependency stack selected | DONE | `pyproject.toml`, `uv.lock`; STACK_REPORT §1 |
| Datasets investigated, accessibility documented | DONE | DATA_PROVENANCE.md; Lift + Can downloaded, Square metadata |
| Dataset structure inspected programmatically | DONE | `reports/feasibility/data_report_*.json` |
| Simulation environment tested | DONE (macOS) | `smoke_sim_*.json` |
| Minimal rollout attempted | DONE | random closed-loop + 400 demo replays |
| Rendering tested | DONE on macOS; **INCOMPLETE on Linux** | frames/; KI-001 |
| Repository structure established | DONE | README layout |
| Automated tests + reproducible setup | DONE | `make setup/test/smoke-sim/download/verify-data` |
| Research and evaluation conventions documented | DONE (DRAFT) | RESEARCH_PROTOCOL.md |
| Modal training architecture proposed | DONE (proposal, nothing executed) | COMPUTE_PLAN.md |
| Git configuration and authorship verified | DONE | DECISIONS D-012; commit metadata in HANDOFF.md |
| Handoff prepared | DONE | HANDOFF.md |

## Owner decisions pending
1. LICENSE for RoboForge's own code (D-013).
2. Docker Desktop restart or purge, to complete the Linux render check (KI-001); optional
   before Session 5.
3. Confirm after a Claude Code restart that the plugin hooks are off (KI-007).
