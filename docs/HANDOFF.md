<!-- parsable handoff header (values below are real, filled at the end of Session 2) -->
```yaml
project: roboforge-3d
current_session: 2
session_status: PASS
next_session: 3
last_verified_commit: b739e99be81803b5e596ed8e8357633780e6a4d3   # lint, 185 tests, container smoke (runner change only since 283a811)
evidence_commit: 283a8115d63b8698503af607a67f6a06c3676745   # every Session 2 report regenerated from this clean commit
python: 3.11.15                       # development (macOS arm64); canonical platform below uses 3.11.18
sim_engine_version: mujoco 3.2.7
robosuite_revision: 51cc01785bab80ffeed20da15e67d7dd4140e76a   # v1.5.1
robomimic_revision: d309eaecc18acf4152a830a895a6984b8ac71b05   # v0.5.0+15
evaluation_platform:                  # D-017 (FIXED)
  platform_id: linux-aarch64-container-1f685638f28c
  image: roboforge-sim-cpu:session0
  image_id: sha256:1f685638f28c720fbec7b084b8e5c2d68073d8f637450d5f6e0a34f0d123071c
  runner: infra/containers/run_sim_cpu.sh
  python: 3.11.18
  mujoco_gl: osmesa
physics:                              # both tasks (SIMULATOR_CONTRACT.md)
  timestep_s: 0.002
  substeps_per_control: 25
  control_hz: 20
  integrator: Euler
  solver: Newton (iterations 100, tolerance 1e-8)
  cone: elliptic
  impratio: 20
  horizon_steps: 400
  termination: first_success          # D-026; success on current kinematics
eval_suite_content_sha256:            # data/eval_states/<task>/<suite>.json (D-025)
  lift_validation: 0b1368fae0d5e967955ad7b5c3a6191001bbad2758becad537412b45e71c8785
  lift_final_test: e02498651b9d5bb88b6aec0870c5e3277cadaaeaf0e325abf86c79e89b1f22da   # LOCKED
  can_validation: 857673382501c1b32880213cd8456a5197c33258ff2ab443e7a9b4700742fbfa
  can_final_test: 5f407748f5f261ef8fdfe4bec0d53c74f8e5ae064e53ff831e17a649528f7ff0   # LOCKED
eval_seeds: {validation: "100000 + i, i < 50", final_test: "900000 + i, i < 50 (locked)"}
final_test_locked_until: docs/FINAL_TEST_FREEZE.md   # does not exist (Session 7)
dataset_source: hf://robomimic/robomimic_datasets@74fa018461f479cd9fd15b924a16103012096203
dataset_sha256:
  lift_low_dim: 2067777cb8b532e9263dd09fd6448c41cc31224bb27be4a3b734010ae13eb540
  lift_raw: 68f2a8cfe2502de1224f0708fb82e46da07abce108c63e721d72fc1618272f37
  can_low_dim: 3f2eb92e0a5025d0095e866ac16cc8092d6a762abe27dec90dbaff9027282962
  can_raw: 86961df85af1b9c6b9d4182a3755a8a4db6d0660cd550e45cca1f7accdb6d73d
converter_version: lowdim_v1
processed_content_sha256:             # unchanged since Session 1
  lift: 74e70119993830798c8b4ed87f202f64d8d2c9d573c0f376d54c4130e52d63c8
  can: e6055a6f066f53b987ed27a7b5894f20ec7053fc9cdf3222b4ad79ed3834ea7b
split_sha256:
  lift: 73097265526c03dfb7535aa9bff50aadd285cbfb554e563636d66395909c331e
  can: 743779b0ad9cacd5bab85a74421f6fba4ace5e980f5979a4a3534c82f042b46f
normalization_sha256:
  lift: 253bfc7e63570bf3e8f8646a0094d1da4c5b729ba8f463b4955f01410b4812d6
  can: e0437cd7b8e48403366d19f7b0e6116e2224723b8a5240b2d7ba008403b3228d
observation_spec_sha256:
  lift: dfbd868e0ec11da76f66a71a9c3468e58590728ca95e93bbfb0a50ce182a7721
  can: 71e89629bc3447b02b4bb7ec2f470e4a72f4d44ec04a74f90b6da146f425b71c
action_spec_sha256: bcae0b7110c56e1bfdf8e2754c03ff95c8c9e77a34c56ef9ebae11e60d7fa77f   # identical for both tasks
task_manifest_sha256:
  lift: 2ac6070d18cc53f565ec11e67e01f6b5a91de7e268df6a491590b0e45f0528ac
  can: d8f85ecea43c0e616162079757598af6b06c0edd8f03a3c563c08b06dea47162
obs_dim: 19              # both tasks; privileged (exact object state)
action_dim: 7            # OSC_POSE delta (3 pos + 3 axis-angle) + gripper, bounds [-1, 1], 20 Hz
training_spend_usd: 0
next_required_owner_approval: none for Session 3 (local CPU training only; any paid compute needs approval)
license: Apache-2.0
```

# Session 2 Handoff

Do not trust this file blindly: re-run the checks in "Exact starting instructions".

## 1. Session completed
Session 2 (3D simulator adapter, faithful replay, closed-loop evaluator): **PASS**.
Session 3 can start.

## 2. Completed work
- **`RoboForgeEnv`** (`src/roboforge/sim/env_adapter.py`, D-024).
  - **Reset** only from an exact serialized snapshot. This clears all hidden state: solver
    warm start, ctrl, mocap, gripper integrator, controller goals, step counter.
  - **Actions:** `step` validates the shape `(7,)` and finiteness, or raises. It clips to
    the `ActionSpec` bounds and reports a `clip_mask`.
  - **Observations:** returned only via `FeatureExtractor.from_sim`.
  - **Success and reward:** sparse reward; success on current kinematics (`mj_kinematics`
    only; KI-019). The episode terminates at first success and truncates at 400 steps
    (D-026).
  - **Rendering:** shadow rendering from a scratch `MjData`. It equals robosuite's
    renderer except `frontview`, which is ≤ 1 grey level off on ≤ 23 px (256², macOS).
  - **Kinematics:** robot and object world transforms, cameras, and a body hierarchy for
    FK.
- **Snapshots and evaluation suites** (`snapshot.py`, `eval_states.py`, D-025).
  - **Snapshots** store the saved XML of the live model, with portable asset paths, plus
    the float64 state and hashes.
  - **Suites:** validation (seeds 100000 + i) and final test (900000 + i), 50 starts per
    task. They were generated on the canonical container.
  - **Final-test lock:** loading the final test, or resetting to its seed block, raises
    `FinalTestLockedError` until `docs/FINAL_TEST_FREEZE.md` exists, is committed and
    names the hash. Research-integrity tests enforce this.
  - **Partition index:** `data/eval_states/<task>/partitions.json` identifies
    `demo_train` / `demo_valid` / `eval_validation` / `eval_final_test`.
- **Policies and recorder** (`src/roboforge/eval/`, D-027, D-028).
  - **Policies:** `RandomPolicy`, `NoOpPolicy` and `DemoReplayPolicy` (labeled
    NON-AUTONOMOUS).
  - **Recorder:** `run_episode` writes a frame-indexed record with states, observations,
    kinematics, raw and executed actions, clip masks, success, decision/chunk indices,
    latency and optional RGB, plus `physics_sha256`.
  - **Audit:** `verify_record` re-simulates the episode and re-evaluates success.
  - **Policy errors** become recorded failures.
- **Demo replay** (`replay_compare.py`, `scripts/replay_demos.py`, D-029).
  - 12 demos per task, chosen by the declared rule.
  - Gates: bitwise equal to a bare robosuite loop, equal to Session 0 metrics, zero clips.
  - Output: divergence summary, montage, mp4s.
- **Model and contract checks** (`scripts/check_sim_contract.py`): KI-014 markers, purity,
  episode isolation, and dataset/eval/saved-XML model diffs.
- **Determinism** (`scripts/check_determinism.py`): 3 fresh processes per task, plus a
  macOS-vs-container comparison.
- **Sanity evaluation** (`scripts/sanity_eval.py`): random and no-op policies on the
  validation suite of the canonical platform.
- **three.js export** (`src/roboforge/viz/`, `scripts/export_clip.py`,
  `tests/web_contract/`, D-030). One real Lift demo-0 clip, verified against MuJoCo FK,
  pinhole projection and segmentation masks, and with the real three.js r187 in node.
- **Canonical platform** (D-017): `infra/containers/run_sim_cpu.sh` and
  `configs/eval/platform.yaml`. `provenance.evaluation_platform()` is stamped into every
  record.

## 3. Files created or changed
- **New:**
  - `src/roboforge/sim/{env_adapter,snapshot,eval_states,kinematics,shadow_render,replay_compare,model_compare}.py`;
  - `src/roboforge/eval/{policies,rollout}.py`, `src/roboforge/viz/{coords,clip_export}.py`;
  - `scripts/{generate_eval_states,replay_demos,check_sim_contract,check_determinism,sanity_eval,export_clip}.py`;
  - `configs/eval/{suites,platform}.yaml`, `infra/containers/run_sim_cpu.sh`;
  - `data/eval_states/{lift,can}/` (suites + `partitions.json`, 128 KB);
  - tests: `tests/unit/{test_coords,test_snapshot}.py`,
    `tests/simulator/{test_env_adapter,test_rollout}.py`,
    `tests/research_integrity/test_final_test_lock.py`, `tests/web_contract/` (node
    package pinned to `three@0.187.0` with a lockfile);
  - evidence: `reports/session2/`, `logs/session2/`.
- **Changed:**
  - `scripts/smoke_sim.py` (adapter checks);
  - `src/roboforge/provenance.py` (`evaluation_platform`, container git env);
  - `Makefile`, `pyproject.toml` (marker `web`), `.gitignore`, `README.md`;
  - docs: DECISIONS, KNOWN_ISSUES, EXPERIMENT_LEDGER, RESEARCH_PROTOCOL,
    SIMULATOR_CONTRACT, STATUS, HANDOFF.
- **Unchanged:** `uv.lock` and every Session 1 data artifact (`git status --short data/`
  stays empty after regeneration).

## 4. Simulator contract (Session 2 STOP list)
- **Physics and control:**
  - MuJoCo 3.2.7, `timestep = 0.002` s, Euler integrator;
  - Newton solver (100 iterations, tolerance 1e-8, 50 line-search iterations), elliptic
    cone, `impratio = 20`;
  - 25 substeps per 0.05 s control step (20 Hz), OSC_POSE delta controller.

  Full table: `docs/SIMULATOR_CONTRACT.md`.
- **Reset mechanism:** `InitialState` (portable saved XML + `[time, qpos, qvel]` float64 +
  hashes + partition), then `snapshot.restore`:
  1. `env.reset()`;
  2. `reset_from_xml_string(edit_model_xml(xml))`;
  3. `sim.reset()`;
  4. `set_state_from_flattened`;
  5. `sim.forward()`;
  6. an exact read-back check.

  Seeds only generate snapshots.
- **File schemas:**
  - eval suites `roboforge.eval_suite/1`: an open manifest + `.data.jsonl.xz`;
  - rollout records `roboforge.rollout/1`: `record.json` + `arrays.npz`, with frame
    k = state after k actions;
  - web clip `roboforge.web_clip/1`.
- **Transform schema:**
  - **MuJoCo world** is z-up, in metres. Quaternions are wxyz in records; observation
    quaternions stay xyzw (Session 1).
  - **Bodies** are categorized as robot (the `robot0_base` subtree), free_object or
    fixture. Mocap markers are excluded.
  - **Cameras** are stored as `cam_xpos`, `cam_xmat` (−z view, +y up) and `cam_fovy`, with
    `f = (H/2)/tan(fovy/2)`.
  - **three.js:** `(x, y, z) → (x, z, −y)`, and `q_three = q_C ⊗ q_mj` with
    `q_C = (cos(−π/4), sin(−π/4), 0, 0)`.

## 5. Executed commands (evidence run from clean `283a811`, `logs/session2/`)
| Command | Platform | Result |
|---|---|---|
| `make smoke-sim SMOKE_OUT=logs/session2/smoke_sim` | macOS | Lift 17/17, Can 16/16 PASS (6 new adapter checks each) |
| `make lint` / `make test` | macOS | clean / **185 passed** (79 s) |
| `make check-eval-states` | macOS | PASS: XML equal, states within 4.4e-16 of the container-generated suites |
| `run_sim_cpu.sh python scripts/generate_eval_states.py --check` | container | PASS: bitwise (0.0) for all 4 suites |
| `make eval-states` (regeneration), then `git status --porcelain data/` | container | byte-identical files; `data/` clean |
| `make replay-demos` | macOS | 12 demos/task; all gates PASS; replay success 11/12 per task |
| `make check-sim-contract` | macOS | checks A–G PASS for both tasks |
| `check_determinism.py` (3 fresh processes/task) | macOS + container | 7/7 checks PASS on both; `--compare`: hashes differ, outcomes equal |
| `make sanity-eval-canonical` | container | random and no-op 0/50 per task; 0 errors; audits pass |
| `make export-clip` / `make web-contract` | macOS / node | 5/5 and 3/3 PASS |

- **Diagnostics:**
  - `logs/session2/diag/`: RNG cross-platform probes, `uv sync --dry-run` in the image,
    shadow vs robosuite renders.
  - `logs/session2/evidence_chain.log`: every step, its exit code and duration.
- **Clean-tree audit:** all 18 git-stamped JSON reports carry `283a811`, `dirty: false`.
  The node `three_check.json` has no stamp of its own; it checks the stamped clip.
- **Runner change `b739e99`:**
  - change: `run_sim_cpu.sh` gained `ROBOFORGE_EVIDENCE_SESSION`; the physics path is
    unchanged (`git diff 283a811 b739e99 --stat`: 1 file);
  - re-verified at `b739e99` (`logs/session2/post_b739e99/`): mount audit, lint clean,
    185 passed, container smoke Lift 17/17 and Can 16/16.

Prerequisites at `4db6526` (`logs/session2/prereq/`): setup OK, 4/4 hashes verified,
`check-data` and `build-data` PASS, `git status --short data/` empty, 124 tests passed,
smoke Lift 11/11 and Can 10/10.

## 6. Key findings
- **robosuite's generator XML is not the model that runs.**
  - Lift samples the cube size at every reset, and PickPlace edits body poses at runtime.
  - Snapshots therefore store `mj_saveLastXML`, which is also the robomimic convention.
  - KI-016 resolved: the saved XML *is* the evaluation model.
- **Success must use current kinematics, and it must not use `mj_forward`.**
  - Calling `mj_forward` on live data changes later Can steps by up to 0.587 (KI-019).
  - robosuite's own success is evaluated on stale kinematics. It is recorded, but never
    scored.
- **Cross-platform chaos (KI-013).** numpy's legacy `randn`/`sin` differ by 1 ulp between
  macOS and the container.
  - On macOS a start regenerated from its seed is off by 2.8e-17–4.4e-16.
  - The same 400 random actions from that start then end up to 0.144 (Lift) and 1.49 (Can)
    apart.
  - Hence: one canonical platform (D-017), stored snapshots only, never re-seeding.
- **KI-014 resolved: the markers are harmless.** They change 0.0 in dynamics and 0 pixels
  in renders, so the default robosuite model is kept.
- **Open-loop demo replay drifts (KI-004).**
  - Task-object error p95: Lift 25.7 mm, Can 80.9 mm.
  - The adapter adds nothing: it is bitwise equal to a bare robosuite loop.
- **The random and no-op baselines never succeed** in 400 steps on 50 validation starts per
  task. This is the H1 sanity floor.
- **Canonical throughput** with full recording: Lift ~140 and Can ~60 control steps/s
  (container, OSMesa, no rendering).
  - A 50-episode × 400-step Can validation sweep takes ~5.5 min per policy.
- **Running `uv` inside the image would mutate it (KI-020).** The runner uses the image
  venv directly.

## 7. Decisions
- D-017 is now FIXED (canonical container platform).
- New: D-024 (adapter), D-025 (snapshots, suites, lock), D-026 (first-success
  termination), D-027 (sanity policies), D-028 (record schema), D-029 (replay sample),
  D-030 (three.js conversion), D-031 (evidence location).

## 8. Outstanding issues (none blocks Session 3)
- **KI-007:** re-check inside the repository that a file with `loss.backward()` and
  `optimizer.step()` is not blocked by the ml-pipeline hook. Session 3 needs training code.
- **KI-001:** linux/x86_64 and EGL are unverified. The canonical platform is aarch64; any
  remote rollout is a different platform (D-017).
- **KI-020:** never `uv sync` or `uv run` inside the pinned image; use
  `run_sim_cpu.sh`.
- **KI-019:** never call `mj_forward` on live data during evaluation.
- **KI-011:** Lift starts are narrow.
  - Each Lift validation start's cube lies 0.5–4.1 mm (median 1.8 mm) in x/y from the
    nearest demonstration start.
  - For Can it is 1.7–21 mm (median 8.4 mm).
  - Source: `reports/session2/eval_states_{lift,can}.json`.
  - The validation suite is therefore in-distribution by construction.
- **KI-008:** 63 GB free.

## 9. Budget
Spent $0.00, with no remote compute. No approval is needed for Session 3, which uses local
CPU only. Any paid compute needs owner approval first.

## 10. Exact starting instructions for Session 3
Paste into a fresh Claude Code session:

```text
Implement RoboForge 3D SESSION 3 ONLY from MASTER_SPEC.md. Begin by reading all of
MASTER_SPEC.md, docs/STATUS.md, docs/HANDOFF.md, docs/DECISIONS.md, docs/KNOWN_ISSUES.md,
docs/EXPERIMENT_LEDGER.md, docs/RESEARCH_PROTOCOL.md, docs/SIMULATOR_CONTRACT.md,
docs/DATA_DICTIONARY.md, docs/DATA_PROVENANCE.md and docs/COMPUTE_PLAN.md, and inspect the
actual repository/test state. Do not blindly trust prior handoff claims: first run
`make setup`, `make download`, `make check-data` (in-memory rebuild; writes nothing),
`make build-data` (writes the gitignored processed files that training reads; refuses if
the committed artifacts differ), `make check-eval-states`, confirm
`git status --short data/` is empty, then run `make lint`, `make test`,
`make smoke-sim SMOKE_OUT=logs/session3/prereq` and, on the canonical container,
`ROBOFORGE_EVIDENCE_SESSION=3 make in-sim CMD="python scripts/smoke_sim.py --task lift --out-dir logs/session3/prereq/container"`
and the same with `--task can`, and confirm they pass; confirm `docker image inspect -f '{{.Id}}' roboforge-sim-cpu:session0`
equals the image_id in configs/eval/platform.yaml; run git log --format=fuller -n 5 to
confirm the verified commits and identity Zackary Holt <zackholt05@gmail.com>. Before
writing any training code, re-check KI-007 inside the repository (a file containing
loss.backward() and optimizer.step() must not be blocked); if a hook blocks it, STOP and
report to the owner instead of working around it.

Session 3 goal (MASTER_SPEC Session 3): research protocol and deterministic/recurrent
baselines. Finalize and commit docs/RESEARCH_PROTOCOL.md (hypotheses, tasks, observation
privilege, data splits, paired evaluation schedule, success definition, checkpoint-selection
rule, matched tuning budgets and planned uncertainty analysis) BEFORE any reported run.
Implement a shared policy/trainer/checkpoint layer plus BC-MLP and BC-GRU or BC-LSTM
(optional GMM only after the core checks); tiny-batch and tiny-episode overfit tests; full
forward/backward/gradient-norm tests; a CPU train/eval smoke; checkpoint save/reload
equivalence (identical outputs, identical closed-loop record physics_sha256 on the same
platform). Run validation-only training runs of limited length, gather initial closed-loop
Lift behavior and diagnose action clipping/sign/coordinate problems; compute true parameter
counts and matched-capacity configurations; measure baseline training cost and inference
latency. Acceptance: both learned baselines consume the actual demonstration data, decrease
loss on a small controlled overfit test, load from saved checkpoints, and perform real Lift
rollouts in the pinned simulator; validation results are logged without test contamination.

Keep the Session 1 and Session 2 contracts (DECISIONS D-010, D-017..D-031): training data
come only from roboforge.data.sequence.SequenceDataset.from_committed (hash-checked,
train-only normalization; validation demos for offline loss only); policy observations come
only from RoboForgeEnv (FeatureExtractor.from_sim after every step), and policy outputs are
decoded with ActionCodec; learned policies implement roboforge.eval.policies.Policy
(PolicyInfo with learned=True, action chunks allowed) and are run with
roboforge.eval.rollout.run_episode, which records every episode and audits with
verify_record; episodes reset only from stored snapshots and end at first success or 400
steps (D-026); model and checkpoint selection use ONLY the eval_validation suite
(load_suite(task, "validation")) — never load, name, summarize or reset to final-test states,
and do not create docs/FINAL_TEST_FREEZE.md. Every closed-loop rollout that is reported or
enters checkpoint selection runs on the canonical platform via infra/containers/run_sim_cpu.sh with
ROBOFORGE_EVIDENCE_SESSION=3 (D-017; never uv sync/uv run inside the image, KI-020); native macOS rollouts are labeled
development-only and never pooled with canonical results. Never call mj_forward on live
evaluation data (KI-019); never use robosuite transform helpers for precision-critical inputs
or metrics (KI-018). Checkpoints and anything else a canonical container rollout must read go
under runs/<run_id>/ (the container sees only runs/, read-only src/configs/data and the
selected session's reports/logs directories). Every run writes an immutable
runs/<run_id>/manifest.json with git SHA + dirty flag, environment snapshot, platform, dataset/split/normalization/spec hashes, full
config, seeds, parameter count, checkpoint hashes and wall/CPU time. Train on local CPU only;
no paid or remote compute and no remote publication without explicit owner approval; do not
change committed data artifacts or eval suites without a recorded decision; do not add
dependencies or change uv.lock without a recorded decision (the pinned image's venv is
frozen, so a change forces an image rebuild, which is a new canonical platform under D-017
and invalidates every earlier canonical rollout).

Use the current verified simulator/data contracts. Run relevant tests and record commands,
real results and provenance. Never leak final-test states into model selection; never invent
results. Use Zackary Holt as the sole named RoboForge project author with the verified local
Git identity; no AI co-author trailers, while preserving external scientific and license
credit. At the SESSION 3 STOP boundary, update docs/STATUS.md, docs/HANDOFF.md,
docs/DECISIONS.md, docs/KNOWN_ISSUES.md and docs/EXPERIMENT_LEDGER.md (baseline configs,
run IDs, known action/controller quirks and open tuning decisions), commit verified code
only, inspect commit metadata (git log --format=fuller -n 5), write the exact prompt for
SESSION 4, and STOP. Do NOT proceed automatically.
```
