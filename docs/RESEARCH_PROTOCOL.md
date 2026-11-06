# Research Protocol — DRAFT (Session 0)

> **Status: DRAFT. Not frozen.** This document fixes the conventions established in
> Session 0. It marks every item as **ESTABLISHED** (backed by Session 0 evidence or a
> spec requirement) or **TO FINALIZE (Session N)**. It must be completed and committed in
> Session 3, before any reported runs. It becomes binding for the final test only through
> `docs/FINAL_TEST_FREEZE.md` (Session 7). No results exist yet.

## 1. Research question (ESTABLISHED, MASTER_SPEC §1.1)
Can a conditional diffusion action-sequence policy match or outperform deterministic and
recurrent behavior cloning on closed-loop 3D robotic manipulation, especially under
variation in object placement and initial robot state? A null or negative answer is a valid
outcome.

## 2. Hypotheses (ESTABLISHED as prespecified, not assumed true)
- **H1** Trained BC, BC-RNN and diffusion policies beat a random-action sanity baseline in
  true task success.
- **H2** Diffusion's closed-loop success may exceed deterministic BC. Tested with paired
  initial states and uncertainty intervals.
- **H3** History-conditioned models (BC-RNN; diffusion with T_o ≥ 2) may beat one-frame BC.
  The *history* benefit must be separated from the *generative-model* benefit, e.g. with a
  diffusion T_o=1 ablation and BC with history.
- **H4** Robustness differences under implemented, physically valid perturbations only.
- **H5** (Level 2, optional) Point clouds vs RGB under matched protocols.

## 3. Tasks, data, modalities
| Item | Value | Status |
|---|---|---|
| Primary task | robosuite `Lift`, robomimic PH, 200 demos, 36–64 steps (mean 48.3), 9,666 samples | ESTABLISHED |
| Secondary task | robosuite `PickPlaceCan`, robomimic PH, 200 demos, 82–151 steps (mean 116.0), 23,207 samples | ESTABLISHED (data + sim verified) |
| Robot / control | Panda, OSC_POSE delta controller, 20 Hz control (25 physics substeps of 2 ms) | ESTABLISHED (dataset `env_args`) |
| Action | 7-D in [−1, 1]: Δposition (scaled to ±0.05 m), Δorientation axis-angle (±0.5 rad), gripper (−1 open / +1 close; data values exactly ±1) | ESTABLISHED (data + `env.action_spec`) |
| Level 1 state observation (Lift) | `robot0_eef_pos`(3), `robot0_eef_quat`(4, xyzw), `robot0_gripper_qpos`(2), `object`(10 = cube pos, cube quat, gripper→cube) = 19-D | DEFAULT; TO FINALIZE (Session 1) |
| Level 1 state observation (Can) | same robot keys + Can pose features computed from MuJoCo state (KI-003 / D-010) | TO FINALIZE (Session 1) |
| Privilege | State observations include exact simulator object poses = **privileged**. Results must say so and never be compared causally with RGB-only results. | ESTABLISHED |
| Operator metadata | None in the HDF5 files. robomimic describes PH as collected by a proficient operator; verify against the paper. Make no held-out-operator claims. | TO FINALIZE (Session 1) |

## 4. Demonstration split (leakage prevention)
- **ESTABLISHED:**
  - Splits are by whole episode; no window crosses an episode boundary (tested).
  - Normalization is fit on the training episodes only.
  - Policy inputs come only from `obs`, never `next_obs`.
  - Observations at index > t never enter a decision at t (tested).
- **DEFAULT, to finalize in Session 1:**
  - Use robomimic's upstream `mask/train` (180) and `mask/valid` (20) episode lists. They
    are disjoint, documented, identical in `low_dim` and `raw`, and comparable with prior
    work.
  - Alternative: a deterministic SHA-256 hash split with a seed locked in config.
  - Record whichever is chosen, with all episode IDs, in `data/splits/`.
- Validation demos support offline loss curves only. **Model selection uses closed-loop
  validation rollouts** (§6).

## 5. Evaluation initial states and seeds
- **ESTABLISHED:** initial conditions are serialized MuJoCo states (model XML + flattened
  `[time, qpos, qvel]`), restored identically for every model. robosuite seeding cannot be
  relied on (KI-002).
- **ESTABLISHED:** all rollouts in one comparison run on one platform and container image.
  Physics is not bit-identical across platforms (KI-013, D-017).
- **DEFAULT seed conventions (finalized and generated in Session 2):**
  - Training seeds: `{0, 1, 2}` per algorithm × task (spec §3.4 target: 3 seeds).
  - Validation start states: global seeds `100000 + i`, `i ∈ [0, 50)`.
  - Final-test start states: global seeds `900000 + i`, `i ∈ [0, 50)`. These are generated
    and hashed in Session 2. Script access stays blocked until `FINAL_TEST_FREEZE.md` exists.
  - Stochastic-policy noise seeds: derived deterministically from
    `(policy_seed, state_id, repeat)`, and never reused across models in a way that
    duplicates trials.
- Horizon: 400 control steps for Lift and Can, the robomimic registry rollout horizon. An
  episode ends at first success or at the horizon. **TO FINALIZE (Session 2):** decide
  whether success must persist.

## 6. Model comparison and selection
- **Families (ESTABLISHED):**
  - BC-MLP (deterministic).
  - BC-RNN (GRU or LSTM, hidden state reset per episode).
  - Conditional diffusion with a 1D temporal U-Net, DDPM/DDIM, receding horizon.
  - Optional GMM-BC only if it can be budgeted credibly.
- **Fairness:** same splits, observation spec, normalization, action clipping, horizon,
  controller and evaluation states. Matched hyperparameter-search budgets (number of trials)
  and approximate parameter parity, with parameter counts reported.
- **Checkpoint selection rule (DEFAULT, finalize in Session 3 before training):**
  - Evaluate saved checkpoints at a fixed cadence on the validation start states.
  - Select the highest validation success rate; on a tie, the earlier checkpoint.
  - The same rule applies to every family.
- **Horizons (DEFAULT, validate in Session 4):** T_o=2, T_p=16, T_a=8. Lift demos are 36–64
  steps long, so a 16-step chunk spans about a quarter to a half of an episode.

## 7. Metrics
- **Primary:** success rate before the horizon over paired held-out initial states, with
  numerator/denominator, per-seed values and confidence intervals.
- **Secondary:** steps to first success, action smoothness/jerk (per 0.05 s step), action
  saturation rate, inference latency, training GPU-hours and dollars, failure categories.
- **Diffusion-specific:** denoising steps, samples per decision, plan diversity, clipping
  frequency.
- **Uncertainty (DEFAULT):**
  - Paired bootstrap over initial states, nested within training seeds (hierarchical),
    with a fixed bootstrap seed. Per-seed summaries are reported alongside.
  - Time steps within a rollout are never treated as independent observations.
  - Deterministic policies are evaluated once per (seed, state).

## 8. Tracking, provenance, artifacts (ESTABLISHED structure; implemented from Session 3)
- Every run gets an immutable `runs/<run_id>/manifest.json` containing:
  - git SHA + dirty flag;
  - `roboforge.provenance.environment_snapshot()` (Python, platform, package versions,
    MUJOCO_GL);
  - dataset SHA-256 + split manifest hash;
  - observation/action/task specs, full config, seeds;
  - normalization stats hash, checkpoint hashes, the selection rule and the chosen
    checkpoint;
  - wall/GPU/CPU time, cost, errors and restarts.
- Results tables are generated from run manifests, never typed by hand.

## 9. Expected limitations (to carry into the paper)
- Simulation only, no sim-to-real claims.
- Privileged state observations.
- One operator-quality tier (PH) and 200 demos per task.
- Narrow Lift initial distribution (KI-011).
- Imperfect replay fidelity (KI-004).
- Can relative-pose sensor artifact in upstream observations (KI-003).
- v1.5 datasets ≠ CoRL 2021 benchmark (KI-012).
- Limited tuning budget; compute and cost constraints.
