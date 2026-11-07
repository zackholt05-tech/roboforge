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
	check-determinism sanity-eval sanity-eval-canonical export-clip web-contract in-sim \
	train-smoke train eval select param-counts overfit-check benchmark-policies \
	diagnose-actions check-ckpt-equivalence diffusion-reference export-denoising \
	modal-dry-run modal-probe-dry-run modal-sim fetch-run profile-rollouts \
	cost-estimate train-main suite suite-plan figures

help:  ## List available targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-12s %s\n", $$1, $$2}'

setup:  ## Create .venv from uv.lock (Python 3.11)
	$(UV) sync --frozen

test:  ## All tests (dataset tests skip with a reason if data is not downloaded)
	$(RUN) pytest

test-fast:  ## Unit tests only (no simulator, no dataset)
	$(RUN) pytest tests/unit

lint:  ## Ruff lint
	$(RUN) ruff check src scripts tests infra

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

# ---------------------------------------------------------------- Sessions 3-4: policies
# Training runs on local CPU (no remote compute). Closed-loop evaluation that is reported or
# enters checkpoint selection runs on the canonical container (D-017), writing evidence to
# reports/session$(EVIDENCE_SESSION) and records to runs/. TRAIN_RUN (not RUN, which is the uv
# command above) names a training run directory. EVIDENCE_SESSION follows
# ROBOFORGE_EVIDENCE_SESSION when that is set, else the current session (D-039); host and
# container commands both receive it, and the scripts derive their default output
# directories from it.
MODEL ?= bc_mlp
TASK ?= lift
SEED ?= 0
# Defaults bumped to Session 7 (DECISIONS D-060; Session 6: D-053; Session 5: D-048) so that no
# Session 7 command can write into Session 6 evidence or reuse the s6 run prefix by omission.
PREFIX ?= s7
EVAL_SET ?= validation
EVIDENCE_SESSION ?= $(if $(ROBOFORGE_EVIDENCE_SESSION),$(ROBOFORGE_EVIDENCE_SESSION),7)
IN_SIM_S := ROBOFORGE_EVIDENCE_SESSION=$(EVIDENCE_SESSION) $(IN_SIM)
RUN_S := ROBOFORGE_EVIDENCE_SESSION=$(EVIDENCE_SESSION) $(RUN)
SMOKE_DIR ?= logs/tmp/train_smoke

train-smoke:  ## CPU miniature: overfit checks + 200-step BC-MLP/BC-RNN/diffusion runs + 2-start dev evals
	$(RUN) python scripts/overfit_check.py --task lift --out-dir $(SMOKE_DIR)
	$(RUN) python scripts/train_smoke.py --out-dir $(SMOKE_DIR)

# SET="section.key=value ..." passes recorded --set overrides to scripts/train.py.
SET ?=
SET_ARGS = $(foreach s,$(SET),--set $(s))
# Main-suite budget (RESEARCH_PROTOCOL §6.3, §13 2026-11-07): the model configs still carry
# the 30k pilot budget, so main runs get it as recorded overrides (D-053).
MAIN_BUDGET := --set train.max_steps=100000 --set train.ckpt_every=5000

train:  ## Train one run on local CPU: make train MODEL=bc_mlp|bc_rnn|bc_mlp_chunk|diffusion_unet TASK=lift SEED=0
	$(RUN) python scripts/train.py --config configs/models/$(MODEL).yaml --task $(TASK) \
		--seed $(SEED) --prefix $(PREFIX) $(SET_ARGS)

train-main:  ## One run at the main budget (100k steps, checkpoints every 5k); SET may not touch train.*
	@case "$(SET)" in *train.*) echo "SET may not change the main budget (train.*)" >&2; exit 2;; esac
	$(RUN) python scripts/train.py --config configs/models/$(MODEL).yaml --task $(TASK) \
		--seed $(SEED) --prefix $(PREFIX) $(MAIN_BUDGET) $(SET_ARGS)

# The main suite (Session 6: Lift, Session 7: Can): train every predeclared trial, evaluate every
# checkpoint on the canonical container, select (§7), choose each family's configuration (§6.3),
# replicate. The default plan follows the current session (D-060).
SUITE_PLAN ?= configs/experiments/level1_core_can.yaml
CAFFEINATE := $(shell command -v caffeinate >/dev/null 2>&1 && echo caffeinate -i)
suite-plan:  ## Resolve SUITE_PLAN and print trial -> run ID (launches nothing)
	$(RUN_S) python scripts/run_suite.py --plan $(SUITE_PLAN) --dry-run

suite:  ## Run SUITE_PLAN end to end (Gate C limits: 8 training runs, 4 containers, 10 in all)
	ROBOFORGE_EVIDENCE_SESSION=$(EVIDENCE_SESSION) $(CAFFEINATE) $(RUN) python scripts/run_suite.py \
		--plan $(SUITE_PLAN)

# Validation-only figures, table and episode montages from a finished suite (D-057).
SUITE_SUMMARY ?= reports/session$(EVIDENCE_SESSION)/suite/$(basename $(notdir $(SUITE_PLAN))).json
LATENCY ?=
figures:  ## Validation-only figures, table and episode montages from SUITE_SUMMARY (LATENCY=<benchmark json>)
	$(RUN_S) python scripts/plot_validation.py --summary $(SUITE_SUMMARY) $(if $(LATENCY),--latency $(LATENCY))
	$(RUN_S) python scripts/render_episodes.py --summary $(SUITE_SUMMARY)

eval:  ## Closed-loop validation of every checkpoint of TRAIN_RUN=runs/<id> on the canonical container
	@if [ "$(EVAL_SET)" != "validation" ]; then \
	  echo "EVAL_SET=$(EVAL_SET) refused: only the validation suite is accessible before the freeze" >&2; \
	  exit 2; fi
	@test -n "$(TRAIN_RUN)" || (echo "set TRAIN_RUN=runs/<train_run_id>" >&2; exit 2)
	$(IN_SIM_S) python scripts/eval_policy.py --train-run $(TRAIN_RUN) \
		--reports-dir reports/session$(EVIDENCE_SESSION)

select:  ## Apply the RESEARCH_PROTOCOL §7 selection rule to TRAIN_RUN=runs/<id>
	$(RUN_S) python scripts/select_checkpoint.py --train-run $(TRAIN_RUN) \
		--reports-dir reports/session$(EVIDENCE_SESSION)

param-counts:  ## Exact parameter counts and capacity-matched tiers
	$(RUN_S) python scripts/param_counts.py

overfit-check:  ## Deliberate overfit of tiny batches / 2 demos on real Lift data
	$(RUN_S) python scripts/overfit_check.py --task lift

benchmark-policies:  ## Training cost + inference latency of RUNS="runs/<a> runs/<b>" (canonical container)
	$(IN_SIM_S) python scripts/benchmark_policies.py --train-runs $(RUNS)

diagnose-actions:  ## Controller axes/signs, teacher-forced errors, closed-loop clipping for RUNS=...
	$(RUN_S) python scripts/diagnose_actions.py --train-runs $(RUNS)

check-ckpt-equivalence:  ## Reload equivalence (2 loads, fresh process, eval record, S2 sanity) for TRAIN_RUN=...
	$(RUN_S) python scripts/check_checkpoint_equivalence.py --write-sanity-reference
	$(IN_SIM_S) python scripts/check_checkpoint_equivalence.py --train-run $(TRAIN_RUN)

# ---------------------------------------------------------------- Session 4: diffusion
diffusion-reference:  ## Diffusion schedulers/U-Net/sampling/EMA vs diffusers 0.11.1 + robomimic d309eae
	$(RUN_S) python scripts/compare_diffusion_reference.py

export-denoising:  ## Real denoising traces of EVAL_RUN=runs/<eval_run_id> (canonical container)
	@test -n "$(EVAL_RUN)" || (echo "set EVAL_RUN=runs/<eval_run_id>" >&2; exit 2)
	$(IN_SIM_S) python scripts/export_denoising.py --eval-run $(EVAL_RUN)

# ---------------------------------------------------------------- Session 5: Modal + costs
# The Modal SDK lives outside the project venv (D-045): MODAL_PY is a Python that has it.
# Every modal-* target below is a DRY RUN (no server contact, nothing created). A paid launch
# needs `$(MODAL_PY) infra/modal_app.py ... --budget-approved <A-xx>` with an approval recorded
# in docs/EXPERIMENT_LEDGER.md (roboforge.compute.approval).
MODAL_PY ?= python3
MODAL_TIMEOUT_S ?= 1800
modal-dry-run:  ## Dry-run plan + upper-bound estimate of one Modal CPU training job (MODEL, TASK, SEED, PREFIX)
	$(MODAL_PY) infra/modal_app.py train --config configs/models/$(MODEL).yaml --task $(TASK) \
		--seed $(SEED) --prefix $(PREFIX) --timeout-s $(MODAL_TIMEOUT_S)

modal-probe-dry-run:  ## Dry-run plan + estimate of the Gate A GPU probe
	$(MODAL_PY) infra/modal_app.py probe

SIM_ROOT ?= logs/tmp/s5_modal_sim
SIM_PREFIX ?= s5sim
SIM_REPORT ?= reports/session$(EVIDENCE_SESSION)/modal_sim/report.json
modal-sim:  ## Local simulation of a Modal job: preemption, SIGKILL, deadline, duplicate, resume, retrieval
	$(RUN_S) python scripts/modal_sim.py --root $(SIM_ROOT) \
		--config configs/models/diffusion_unet.yaml --prefix $(SIM_PREFIX) \
		--set train.max_steps=2000 --set train.ckpt_every=500 --set train.val_every=500 \
		--set train.log_every=100 --sigint-at 650 --sigkill-at 1150 --deadline-s 30 \
		--install --report $(SIM_REPORT)

fetch-run:  ## Download RUN_ID from the Modal runs Volume and install it into runs/ (hash-checked)
	$(MODAL_PY) infra/modal_app.py fetch --run-id $(RUN_ID)

profile-rollouts:  ## Parallel canonical containers on validation starts: physics vs inference vs rendering
	$(RUN_S) python scripts/profile_parallel.py --levels 1 2 4 \
		--out-dir reports/session$(EVIDENCE_SESSION)/rollout_profile

cost-estimate:  ## Main-suite cost/time projection from the measured numbers (estimates)
	$(RUN_S) python -m roboforge.compute.estimate suite \
		--out reports/session$(EVIDENCE_SESSION)/cost/suite_projection.json
