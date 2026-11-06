# Session 1 command evidence

All runs except `prereq/` and the `*_cb9b6a3_FAIL.log` file were produced from the clean tree
at `c69551f` (see `run_header.log` for the sequence and exit codes):

- `pytest.log`, `lint.log`: `make test`, `make lint`;
- `smoke_sim.log`, `smoke_sim/`: `make smoke-sim SMOKE_OUT=logs/session1/smoke_sim`;
- `verify_data.log`, `verify_data/`: `make verify-data VERIFY_OUT=logs/session1/verify_data`
  (includes `check-data`);
- `audit_data.log`: `make audit-data` (writes `reports/data_audit.json`, `reports/data_audit/`,
  `docs/DATA_DICTIONARY.md`);
- `docker_linux_aarch64_check_data.log`: `scripts/build_data.py --check` in the Linux aarch64
  image with read-only mounts (PASS);
- `docker_linux_aarch64_check_data_cb9b6a3_FAIL.log`: the same check at `cb9b6a3`, where Can
  failed (KI-018);
- `prereq/`: smoke-sim reports from the Session 1 prerequisite run at `833277d`;
- `post_599e326/`: lint, `make test` (124 passed), `make build-data` and `make check-data`
  re-run after the write-protection change. Only `docs/DECISIONS.md` was uncommitted then.

The frames that `smoke_sim` and `verify_data` rendered (2 + 18 PNGs) were byte-identical to
`reports/feasibility/frames/`, so they were deleted rather than committed twice (D-023). The
JSON reports still name the original paths.
