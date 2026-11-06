<!-- parsable handoff header (values below are real, filled at the end of Session 0) -->
```yaml
project: roboforge-3d
current_session: 0
session_status: PASS
next_session: 1
last_verified_commit: a9c67938b516915fc4a1244e5b188dd0ae358775   # macOS + Linux container evidence regenerated from this clean tree
code_commit: 590812e19fb38abe7633c3782cc0d0d6a2e26cc9   # + compare_platform_evidence.py; evidence committed in the next commit
python: 3.11.15
sim_engine_version: mujoco 3.2.7
robosuite_revision: 51cc01785bab80ffeed20da15e67d7dd4140e76a   # v1.5.1
robomimic_revision: d309eaecc18acf4152a830a895a6984b8ac71b05   # v0.5.0+15
dataset_source: hf://robomimic/robomimic_datasets@74fa018461f479cd9fd15b924a16103012096203
dataset_sha256:
  lift_low_dim: 2067777cb8b532e9263dd09fd6448c41cc31224bb27be4a3b734010ae13eb540
  lift_raw: 68f2a8cfe2502de1224f0708fb82e46da07abce108c63e721d72fc1618272f37
  can_low_dim: 3f2eb92e0a5025d0095e866ac16cc8092d6a762abe27dec90dbaff9027282962
  can_raw: 86961df85af1b9c6b9d4182a3755a8a4db6d0660cd550e45cca1f7accdb6d73d
action_dim: 7            # OSC_POSE delta (3 pos + 3 axis-angle) + gripper, bounds [-1, 1], 20 Hz
success_signal: "robosuite env._check_success() (Lift: cube z > table + 0.04 m; Can: can in goal bin)"
training_spend_usd: 0
next_required_owner_approval: none for Session 1
license: Apache-2.0      # DECISIONS D-013; LICENSE + NOTICE
```

# Session 0 Handoff

Do not trust this file blindly: re-run the checks in "Exact starting instructions".

## 1. Session completed
Session 0 (feasibility, licenses, environment lock). **PASS / GO** for Session 1. Full
evidence is in `reports/feasibility/STACK_REPORT.md`.

## 2. Completed work
- Audited MASTER_SPEC against upstream sources. Versions were read from pinned upstream
  source, PyPI/GitHub/HF APIs and Modal docs.
- Locked a working stack (uv).
- Downloaded and verified Lift + Can PH (`low_dim` + `raw`).
- Implemented:
  - hash-verified fetcher;
  - HDF5 inspector;
  - MuJoCo backend selection;
  - env factory from dataset metadata;
  - exact state restore;
  - decomposed open-loop replay checker;
  - upright offscreen renderer;
  - minimal episode-bounded window loader.
- Smoke tests, the observation-consistency probe, a headless Linux Dockerfile, docs, and
  attribution.
- Ran the full smoke and data checks in the Linux aarch64 container with OSMesa. Compared
  the results with macOS (`scripts/compare_platform_evidence.py`). Fixed two `--out-dir`
  bugs found on the way (`e0b26ae`, `a9c6793`).

## 3. Files created
- Build / tooling: `pyproject.toml`, `uv.lock`, `Makefile`, `.gitignore`,
  `.gitattributes`, `.dockerignore`, `.env.example`.
- Top-level docs: `README.md`, `CITATION.cff`, `THIRD_PARTY_NOTICES.md`, `LICENSE`
  (Apache-2.0), `NOTICE`.
- `configs/simulator/{lift,can}.yaml` (generated from data; equality tested).
- `data/manifests/robomimic_v1.5_ph.json`.
- `src/roboforge/`:
  - `paths.py`, `logging_utils.py`, `provenance.py`;
  - `data/{fetch,inspect,windows}.py`;
  - `sim/{backend,factory,renderer,state_replay}.py`;
  - empty boundary packages `models/ train/ eval/ viz/ compute/`.
- `scripts/{download_data,smoke_sim,verify_data,probe_obs_consistency,compare_platform_evidence}.py`.
- `tests/{conftest.py, unit/*, simulator/test_lift_sim.py, integration/{test_datasets,test_verify_data_cli}.py}`.
- `infra/containers/Dockerfile.sim-cpu`.
- `docs/{STATUS,HANDOFF,DECISIONS,KNOWN_ISSUES,EXPERIMENT_LEDGER,RESEARCH_PROTOCOL,DATA_PROVENANCE,SIMULATOR_CONTRACT,REPRODUCIBILITY,COMPUTE_PLAN}.md`.
- `reports/feasibility/` (STACK_REPORT + JSON + frames), including
  `linux_aarch64_osmesa/` (Linux container reports, frames, platform comparison);
  `logs/session0/*.log`; `runs/README.md`.
- Outside the repo, by owner request: `~/.claude/settings.json` gained
  `env.ML_PIPELINE_ENFORCE=0`. The owner also disabled `cto-toolkit@synced`.

## 4. Dependencies
See the header and `uv.lock`. Key pins:
- mujoco 3.2.7;
- robosuite git@51cc017;
- robomimic git@d309eae;
- torch 2.15.0, numpy 2.4.6, h5py 3.16.0;
- diffusers 0.11.1, transformers 4.41.2, huggingface-hub 0.23.4.

Overrides: `egl-probe` Linux-only. Dev: pytest, ruff.

## 5. Dataset findings
- Lift: 200 demos / 9,666 steps / state 32-D / obs `object` 10-D.
- Can: 200 demos / 23,207 steps / state 71-D / obs `object` 14-D.
- Both: 7-D actions in [−1, 1]; upstream masks train/valid 180/20; `low_dim` ≡ `raw`.
- No image HDF5 exists for v1.5 PH. Regenerate from raw states when needed.
- Details: `docs/DATA_PROVENANCE.md`.

## 6. Simulator findings
- Reset, step, success, render, exact state restore and cross-process determinism all
  verified on macOS and in the Linux aarch64 container (OSMesa).
- Open-loop replay: Lift 199/200, Can 190/200. The controller is consistent; contact
  divergence is accepted (KI-004).
- **Can relative-pose observations are lagged** (KI-003).
- **robosuite seeding is global-RNG only** (KI-002).
- **Physics is not bit-identical across platforms** (KI-013): the same pass/fail outcomes,
  but Can trajectories differ by up to 5 cm. Evaluate each comparison on one platform (D-017).
- linux/x86_64 install and EGL unverified (KI-001).
- Details: `docs/SIMULATOR_CONTRACT.md`.

## 7. Tests executed (final run recorded in the evidence commit)
- `make test`: 56 passed (unit 35, simulator 12, integration 9).
- `make lint`: clean.
- `make smoke-sim`: Lift 11/11, Can 10/10 PASS.
- `make verify-data`: all integrity checks PASS for both tasks.
- `make probe-obs`: Lift exact, Can lag confirmed.
- Linux aarch64 container (OSMesa): smoke 21/21 and integrity 12/12 PASS; frames inspected.
- macOS evidence regenerated on `a9c6793`: identical to the previous run except timings;
  all 20 frames byte-identical.

## 8. Architectural decisions
`docs/DECISIONS.md` D-001…D-017. The most consequential for Session 1:
- **D-010:** Can observation features computed from MuJoCo state by one function for both
  training and evaluation.
- **D-008:** serialized evaluation states.
- The upstream-mask split default (RESEARCH_PROTOCOL §4).

## 9. Remaining risks and unresolved issues
`docs/KNOWN_ISSUES.md`. Highest:
- **KI-003:** must be resolved in the Session 1 observation spec.
- **KI-007:** confirm the hooks are off before Session 3 training code.
- **KI-013:** cross-platform physics differences; fix the evaluation platform in Session 2.
- **KI-001:** linux/x86_64 and EGL unverified; needed by Session 5/8.
- **KI-008:** disk space (36 GB free).
- **KI-011:** narrow Lift initial distribution, relevant to OOD claims.

## 10. Exact starting instructions for Session 1
Paste into a fresh Claude Code session:

```text
Implement RoboForge 3D SESSION 1 ONLY from MASTER_SPEC.md. Begin by reading all of
MASTER_SPEC.md, docs/STATUS.md, docs/HANDOFF.md, docs/DECISIONS.md, docs/KNOWN_ISSUES.md,
docs/EXPERIMENT_LEDGER.md, docs/RESEARCH_PROTOCOL.md, docs/DATA_PROVENANCE.md and
reports/feasibility/STACK_REPORT.md, and inspect the actual repository/test state. Do not
blindly trust prior handoff claims: first run `make setup`, `make download`, `make test`
and `make smoke-sim` and confirm they pass (git log --format=fuller -n 5 to confirm the
verified commits and identity Zackary Holt <zackholt05@gmail.com>).

Session 1 goal: the audited canonical data pipeline for Lift and Can (MASTER_SPEC Session 1):
typed ObservationSpec/ActionSpec/TaskSpec read from the actual HDF5; deterministic
episode-level split manifests in data/splits/ (default: robomimic upstream mask/train and
mask/valid, record all IDs; justify in DECISIONS.md if you change it); window sampler with
T_o/T_p, padding masks and train-only reversible normalization with saved stats + hashes;
dataset audit summaries (reports/data_audit.json, docs/DATA_DICTIONARY.md); tests for no
future leakage, window boundaries, normalizer inversion, split determinism and a one-batch
forward pass on real data. Resolve KNOWN_ISSUES KI-003 per DECISIONS D-010 (one RoboForge
state->feature function used for dataset windows and later closed-loop eval; verify it equals
robosuite on Lift and on Can's world-frame blocks), and apply the same observation spec to
every model family. Do not train policies, do not build the evaluator beyond what the data
pipeline needs, no paid compute, no remote publication.

Use the current verified simulator/data contracts. Run relevant tests and record commands,
real results and provenance. Never leak final-test states into model selection; never invent
results. Use Zackary Holt as the sole named RoboForge project author with the verified local
Git identity; no AI co-author trailers, while preserving external scientific and license
credit. At the SESSION 1 STOP boundary, update docs/STATUS.md, docs/HANDOFF.md,
docs/DECISIONS.md, docs/KNOWN_ISSUES.md and docs/EXPERIMENT_LEDGER.md, commit verified code
only, inspect commit metadata (git log --format=fuller -n 5), write the exact prompt for
SESSION 2, and STOP. Do NOT proceed automatically.
```
