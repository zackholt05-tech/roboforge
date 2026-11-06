# RoboForge developer commands. Only targets that work today are defined; later
# sessions add training/eval/web/paper targets as they are implemented (MASTER_SPEC §6.5).
# Every target runs through the locked uv environment (uv.lock).

UV ?= uv
RUN := $(UV) run --frozen
TASKS ?= lift can
# verify_data.py evidence directory; the default keeps Session 0's location.
VERIFY_OUT ?= reports/feasibility
# smoke_sim.py evidence directory; the default keeps Session 0's location.
SMOKE_OUT ?= reports/feasibility

.PHONY: help setup test test-fast lint smoke-sim download verify-data probe-obs build-data \
	check-data audit-data

help:  ## List available targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-12s %s\n", $$1, $$2}'

setup:  ## Create .venv from uv.lock (Python 3.11)
	$(UV) sync --frozen

test:  ## All tests (dataset tests skip with a reason if data is not downloaded)
	$(RUN) pytest

test-fast:  ## Unit tests only (no simulator, no dataset)
	$(RUN) pytest tests/unit

lint:  ## Ruff lint
	$(RUN) ruff check src scripts tests

smoke-sim:  ## Real Lift + Can reset/step/success/render checks (no dataset needed)
	@for t in $(TASKS); do $(RUN) python scripts/smoke_sim.py --task $$t --out-dir $(SMOKE_OUT) \
		|| exit 1; done

download:  ## Fetch + SHA-256-verify the manifest datasets (TASKS="lift can")
	$(RUN) python scripts/download_data.py --tasks $(TASKS)

verify-data:  ## HDF5 checks, low_dim==raw, obs/state alignment, replay of all demos, split contract
	$(RUN) python scripts/verify_data.py --tasks $(TASKS) --replay-demos all --render \
		--out-dir $(VERIFY_OUT)
	$(MAKE) check-data TASKS="$(TASKS)"

build-data:  ## Build processed data + committed split/normalization/task manifests (Session 1)
	$(RUN) python scripts/build_data.py --tasks $(TASKS)

check-data:  ## Rebuild in memory; fail unless committed data artifacts match exactly
	$(RUN) python scripts/build_data.py --tasks $(TASKS) --check

audit-data:  ## reports/data_audit.json + figures + docs/DATA_DICTIONARY.md (requires build-data)
	$(RUN) python scripts/audit_data.py --tasks $(TASKS)

probe-obs:  ## Relative-pose observation consistency probe (KNOWN_ISSUES KI-003)
	$(RUN) python scripts/probe_obs_consistency.py --tasks $(TASKS) --demos 50
