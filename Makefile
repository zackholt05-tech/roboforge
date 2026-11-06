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

# Session 2 evidence directory and the canonical evaluation platform wrapper (D-017).
S2_OUT ?= reports/session2
IN_SIM := infra/containers/run_sim_cpu.sh

.PHONY: help setup test test-fast lint smoke-sim download verify-data probe-obs build-data \
	check-data audit-data eval-states check-eval-states replay-demos check-sim-contract \
	check-determinism sanity-eval sanity-eval-canonical export-clip web-contract in-sim

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

build-data:  ## Build processed data (gitignored); fails unless committed artifacts match
	$(RUN) python scripts/build_data.py --tasks $(TASKS)

check-data:  ## Rebuild in memory; fail unless committed data artifacts match exactly
	$(RUN) python scripts/build_data.py --tasks $(TASKS) --check

audit-data:  ## reports/data_audit.json + figures + docs/DATA_DICTIONARY.md (requires build-data)
	$(RUN) python scripts/audit_data.py --tasks $(TASKS)

probe-obs:  ## Relative-pose observation consistency probe (KNOWN_ISSUES KI-003)
	$(RUN) python scripts/probe_obs_consistency.py --tasks $(TASKS) --demos 50

# ---------------------------------------------------------------- Session 2: simulator apparatus
eval-states:  ## Generate eval suites ON THE CANONICAL CONTAINER (refuses to overwrite differing suites)
	ROBOFORGE_WRITE_EVAL_STATES=1 $(IN_SIM) python scripts/generate_eval_states.py --tasks $(TASKS)

check-eval-states:  ## Regenerate eval suites in memory; bitwise on the generating platform, 1e-12 elsewhere
	$(RUN) python scripts/generate_eval_states.py --tasks $(TASKS) --check

replay-demos:  ## Replay >= 10 sampled demos/task through RoboForgeEnv; divergence, montage, video
	$(RUN) python scripts/replay_demos.py --tasks $(TASKS) --video --out-dir $(S2_OUT)/replay

check-sim-contract:  ## KI-014 markers, purity, episode isolation, model diffs (reports/session2)
	$(RUN) python scripts/check_sim_contract.py --tasks $(TASKS) --out-dir $(S2_OUT)

check-determinism:  ## Two fresh processes, same controlled seed -> bitwise-identical trajectories
	$(RUN) python scripts/check_determinism.py --tasks $(TASKS) --out-dir $(S2_OUT)/determinism

sanity-eval:  ## Random + no-op policies on the validation suite (this platform; development only)
	$(RUN) python scripts/sanity_eval.py --tasks $(TASKS) --out-dir $(S2_OUT)/sanity_eval

sanity-eval-canonical:  ## Random + no-op on the validation suite on the canonical container (D-017)
	$(IN_SIM) python scripts/sanity_eval.py --tasks $(TASKS) --out-dir $(S2_OUT)/sanity_eval

export-clip:  ## Export one real Lift clip (MuJoCo + three.js frames) with coordinate verification
	$(RUN) python scripts/export_clip.py --task lift --out-dir $(S2_OUT)/web_diagnostic

web-contract:  ## Check the exported clip with the real three.js library (node; pinned three@0.187.0)
	cd tests/web_contract && npm ci --no-audit --no-fund && npm run -s check

in-sim:  ## Run CMD on the canonical evaluation container, e.g. make in-sim CMD="python scripts/x.py"
	$(IN_SIM) $(CMD)
