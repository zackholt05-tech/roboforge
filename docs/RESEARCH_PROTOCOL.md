# Research Protocol — v1 (finalized in Session 3, before any reported run)

> **Status: FINALIZED for model development (Sessions 3–7).** Committed before the first
> reported training or evaluation run of RoboForge. It binds every reported Level 1 run.
> - Items marked **TO FINALIZE (Session N)** may be filled in only by that session, before
>   its first reported run, and only as a dated amendment in §13.
> - Any other change also needs a dated amendment in §13 with its reason, committed before
>   the runs it affects. Runs made before an amendment keep the protocol version they ran
>   under.
> - The final test becomes accessible only through `docs/FINAL_TEST_FREEZE.md` (Session 7).
>   That document fixes the exact checkpoints, test schedule and analysis.
> - No research results exist at the time of writing. Session 0–2 rollouts are engineering
>   controls.

## 1. Research question (MASTER_SPEC §1.1)
Can a conditional diffusion action-sequence policy match or outperform deterministic and
recurrent behavior cloning on closed-loop 3D robotic manipulation, especially under
variation in object placement and initial robot state? A null or negative answer is a valid
outcome and will be reported as such.

## 2. Hypotheses and how each is tested (prespecified, not assumed true)
| ID | Hypothesis | Comparison (same task, same paired states) | Status |
|---|---|---|---|
| H1 | Trained BC-MLP, BC-RNN and diffusion beat a non-learned random-action baseline in task success | each family's selected checkpoints vs `random_uniform` (0/50 on both validation suites, Session 2) | primary |
| H2 | Diffusion's closed-loop success may exceed deterministic BC | diffusion vs BC-MLP and diffusion vs BC-RNN, each at its primary configuration (§6) | **primary comparison** |
| H2-c | Any diffusion advantage is due to the generative model, not to action chunking | diffusion vs **BC-MLP-chunk** (identical T_o, T_p, T_a; §6.2) | secondary (required before attributing an H2 effect to the generative model) |
| H3 | Observation history helps; separate the *history* benefit from the *generative* benefit | BC-RNN vs BC-MLP (history, deterministic head); diffusion T_o = 1 vs T_o = 2 ablation (Session 7) | secondary |
| H4 | Robustness differs under implemented, physically valid perturbations | only on perturbation sets defined, validated and frozen before the main robustness runs (Session 7) | secondary |
| H5 | (Level 2, optional) point clouds vs RGB under a matched protocol | separate preregistration (`docs/LEVEL2_GO_NO_GO.md`) | out of scope here |

Everything not in this table is **exploratory** and labeled so.

## 3. Tasks, data, modalities
| Item | Value | Status |
|---|---|---|
| Primary task | robosuite `Lift`, robomimic PH v1.5, 200 demos, 36–64 steps (mean 48.3), 9,666 samples | ESTABLISHED |
| Secondary task | robosuite `PickPlaceCan`, robomimic PH v1.5, 200 demos, 82–151 steps (mean 116.0), 23,207 samples | ESTABLISHED |
| Robot / control | Panda, OSC_POSE delta controller in the world frame, 20 Hz control (25 physics substeps of 2 ms) | ESTABLISHED |
| Action | 7-D in [−1, 1]: Δposition (±0.05 m), Δorientation axis-angle (±0.5 rad), gripper (−1 open / +1 close; data values exactly ±1; the simulator integrates `sign(command)`) | ESTABLISHED |
| Observation (Lift, 19-D) | `robot0_eef_pos` 3, `robot0_eef_quat` 4 (xyzw), `robot0_gripper_qpos` 2, `cube_pos` 3, `cube_quat` 4, `gripper_to_cube_pos` 3 (D-018) | ESTABLISHED |
| Observation (Can, 19-D) | same robot keys + `Can_to_robot0_eef_pos` 3 (same-step, float64) + `Can_pos` 3 + `Can_quat` 4. Not robomimic's 23-D default (KI-003, D-010) | ESTABLISHED |
| Operator metadata | none (one teleoperation channel per demo). No held-out-operator claims | ESTABLISHED |

**Observation privilege (binding).** Every Level 1 core family (BC-MLP, BC-MLP-chunk,
BC-RNN, diffusion) receives exactly the same 19-D vector per step. It contains exact
simulator object poses, so all Level 1 core results are **privileged-state** results. They
must never be compared causally with camera-based policies.
- No model receives anything else: no `next_obs`, time index, step counter, episode ID,
  demonstration ID, success flag, reward, or any statistic other than the committed
  train-only normalizer.
- During training, inputs come only from `SequenceDataset.from_committed`.
- During evaluation, inputs come only from `RoboForgeEnv` (`FeatureExtractor.from_sim` after
  every step).

## 4. Demonstration split and leakage prevention (ESTABLISHED, Session 1)
- Split by whole episode with robomimic's upstream masks: `train` 180, `valid` 20,
  identical ID lists for both tasks (`data/splits/{lift,can}_ph.json`, D-019). No window
  crosses an episode boundary.
- Normalization: per-dimension min-max to [−1, 1], fit on the training episodes' per-step
  arrays only (D-020). Model actions are decoded by inverting it, then clipping to the action
  bounds (`ActionCodec`). The same decode and clip path is used for every family.
- Windows (D-022): `obs_history[k] = obs[clip(t − T_o + 1 + k)]` with `obs_valid` marking
  pre-start padding; `future_action[k] = actions[t + k]` with `valid_mask` marking
  post-end padding, which every loss masks. Observations at index > t are never read.
- Validation demos give **offline losses only**. They never select a model. Model and
  checkpoint selection use closed-loop rollouts on the validation simulator suite (§7).
- There is no demonstration test split. The final test uses held-out simulator initial
  states (§5).

## 5. Evaluation states, platform, episodes and the paired schedule
- **Initial states** are serialized MuJoCo snapshots: saved model XML + `[time, qpos, qvel]`.
  They are restored exactly for every model; evaluation never re-seeds (D-008, D-025).
- **Suites** (`data/eval_states/`, generated on the canonical container):

  | Task | Suite | Partition | Seeds | n | Content SHA-256 |
  |---|---|---|---|---|---|
  | Lift | validation | `eval_validation` | 100000 + i | 50 | `0b1368fa…` |
  | Lift | held-out test | `eval_final_test` | 900000 + i | 50 | `e0249865…` (locked) |
  | Can | validation | `eval_validation` | 100000 + i | 50 | `85767338…` |
  | Can | held-out test | `eval_final_test` | 900000 + i | 50 | `5f407748…` (locked) |

  The held-out test states cannot be loaded, and their seed block cannot be reset to, until
  `docs/FINAL_TEST_FREEZE.md` is committed and names each suite's content hash. Until then
  no code may load, name, summarize or reset to them (`tests/research_integrity/`).
- **Episode rule (D-026):**
  - horizon 400 control steps;
  - the episode ends at the **first success** (task predicate on the post-step state's
    current kinematics), or is truncated at the horizon;
  - a post-success `hold_steps` persistence diagnostic may be reported separately and never
    changes the primary metric.
- **Platform (D-017):** every rollout that is reported, enters checkpoint selection, or
  enters a comparison runs on the canonical Linux aarch64 container
  `roboforge-sim-cpu:session0` (`sha256:1f685638…`) through
  `infra/containers/run_sim_cpu.sh`.
  - Native macOS rollouts are **development-only**. They are labeled with their platform ID
    and never pooled with, or substituted for, canonical results.
  - Policy inference in rollouts runs on CPU with one torch thread, so that it is
    deterministic.
- **Paired schedule:**
  - *Validation (development, checkpoint selection):* every evaluated checkpoint of every
    family runs **all 50** validation starts of its task. A deterministic policy runs once
    per start.
  - *Held-out test (Session 7, after the freeze):* every selected checkpoint (family × task
    × training seed) runs all 50 held-out starts, once per start if deterministic. All
    models see the same starts, so every comparison is paired by `state_id`.
  - *Stochastic policies* (diffusion): the noise seed is derived from
    `(policy_seed, state_id, repeat)`, with R = 1 repeat by default (TO FINALIZE, Session 4).
    Repeats of a deterministic policy are never added, because they would be duplicates.
- **Expected saturation (KI-011).** Lift validation starts lie 0.5–4.1 mm (median 1.8 mm)
  in x/y from the nearest demonstration start, so the Lift validation suite is
  in-distribution by construction. Published BC results on Lift PH low-dim are near 100 %.
  Many Lift checkpoints are therefore expected to score 50/50, and Lift may not separate the
  families. **Can is expected to be the discriminating task.** Spatial claims need the
  separately defined perturbation sets (H4).

## 6. Model families, configurations and fairness
### 6.1 Primary configurations
All families share: the 19-D observation, the 7-D normalized action, masked MSE in normalized
action space, float32 on CPU, and the decode/clip path of §4. Parameter counts are exact
(`scripts/param_counts.py`).

| Family | Network | Horizons (T_o / T_p / T_a) | Params (Lift/Can, 19-D in) | Status |
|---|---|---|---|---|
| **BC-MLP** | MLP 2 × 1024, ReLU, linear output (robomimic BC) | 1 / 1 / 1 | 1,077,255 | DEFAULT (Session 3) |
| **BC-RNN** | LSTM 2 layers, hidden 296, linear head on the last valid step | context L = 10 / 1 / 1 | 1,080,703 | DEFAULT (Session 3) |
| **BC-MLP-chunk** | MLP 2 × 1024, ReLU, linear output | 2 / 16 / 8 | 1,204,336 | DEFAULT (H2-c control) |
| **Diffusion** | conditional 1D temporal U-Net, DDPM/DDIM | 2 / 16 / 8 | TO FINALIZE (Session 4) | TO FINALIZE (Session 4) |

- **BC-RNN inference** (deviation from robomimic, recorded):
  - At every decision the LSTM re-runs from a zero state over the last ≤ L observations,
    exactly as in training, where each window starts from zero.
  - Pre-start padded steps (`obs_valid = False`) are skipped: the state stays at zero until
    the first valid step.
  - No hidden state persists between decisions or between episodes.
  - robomimic instead carries the hidden state for up to 10 steps and then resets it. RoboForge
    chooses exact train/inference consistency.
- **Receding horizon:** a chunked policy executes the first T_a of its T_p predicted actions,
  then observes again. It sees every intermediate observation through the rollout runner's
  `observe` hook.
- **Capacity matching:**
  - In the primary tier, BC-MLP and BC-RNN lie within ±5 % of 1.08 M parameters (actual
    difference +0.3 %).
  - BC-MLP-chunk keeps BC-MLP's hidden layers, so that it differs from BC-MLP only in its
    input and output horizons. Its larger input and output layers give 1.20 M parameters
    (+11.8 %).
  - Diffusion's count is reported in Session 4. If it exceeds 2× the BC budget, a
    capacity-matched BC-MLP and BC-RNN at diffusion's count are added to the comparison
    (MASTER_SPEC §5.7).
- **GMM-BC** (optional, spec §5.4): not run. It would need its own matched budget; adding it
  later is an amendment.

### 6.2 Training defaults (identical for every BC family; DEFAULT, subject to §6.3 tuning)
- AdamW, lr 1e-4, betas (0.9, 0.999), weight decay 0, constant learning rate.
- Batch 100. Masked MSE averaged over valid action steps and dimensions.
- No gradient clipping; the global gradient norm is logged every step. No EMA.
- One torch thread, CPU, float32.
- The training seed `s` seeds torch (initialization) and an independent numpy generator
  `default_rng([s, 1])` for batch sampling. The sampler concatenates one seeded random
  permutation of all training windows per epoch and takes consecutive batches of 100 from
  that stream (a batch may straddle two epochs), so every window is seen equally often.
- The offline validation loss over all `valid` windows is logged every 1,000 steps. It is
  never used for selection.

### 6.3 Matched budgets
- **Training budget:** within one comparison, every family uses the same batch size, number
  of gradient steps and checkpoint cadence, so every family sees the same number of training
  samples. Wall-clock and CPU time are reported per run.
  - *Session 3 pilot:* 30,000 steps, checkpoints every 5,000 steps (6 checkpoints).
  - *Main runs (Session 6):* DEFAULT 100,000 steps, checkpoints every 10,000 (10
    checkpoints). Session 5 must confirm this from measured costs before any main run; it
    stays identical across families.
- **Tuning budget:** at most **4 configurations per family per task**, each trained with
  seed 0 and selected by the rule of §7.
  - The search space of each family is written in `docs/EXPERIMENT_LEDGER.md` before its
    first tuning run.
  - The **Session 3 pilots count as trial 1** for BC-MLP and BC-RNN on Lift, so diffusion
    later receives the same number of trials.
  - The chosen configuration is then trained with seeds 1 and 2 (seed 0 is reused).
- **Seeds:** 3 training seeds `{0, 1, 2}` per family × task (spec §3.4 target), unless an
  owner-approved, precommitted reduction is recorded.

## 7. Checkpoint-selection rule (binding, identical for every family)
1. A training run saves checkpoints at steps `k · C`, `k = 1..K` (cadence C and count K from
   §6.3). The last checkpoint is the final step.
2. Every saved checkpoint runs all 50 validation starts of its task on the canonical platform
   (§5). If cost ever forbids evaluating all K, a fixed stride is declared in §13 *before*
   training, identically for every family.
3. Select the checkpoint with the **highest validation success count**. Ties go to the
   **lower mean first-success step** over its successful episodes; remaining ties go to the
   **earlier checkpoint**.
4. Offline losses, gradient norms and clip rates are logged but never select.
5. The selection artifact (`reports/session<N>/selection/<train_run_id>.json`) records the
   rule, every checkpoint's SHA-256 and validation result (with eval run IDs), and the chosen
   checkpoint's hash.
   - The eval script verifies each checkpoint's SHA-256 against the training manifest before
     loading it. The hash travels into every rollout record (`PolicyInfo.params`).

Selecting the maximum over K checkpoints on the same 50 starts biases validation success
upward. Validation numbers are therefore development numbers only; the held-out test is the
unbiased estimate.

## 8. Metrics
- **Primary:** success before the 400-step horizon (first-success rule), as numerator /
  denominator per checkpoint, per seed and per family × task.
- **Secondary:** steps to first success; action saturation and clip rate per dimension
  (decoded actions outside [−1, 1] before clipping); gripper command sign changes; action
  smoothness (mean |Δa| per 0.05 s step); inference latency (ms per decision, one thread,
  canonical container); training wall-clock and CPU seconds; failure categories (Session 7,
  defined before the test).
- **Diffusion-specific** (Session 4): denoising steps, samples per decision, plan diversity,
  clip frequency.

## 9. Planned uncertainty analysis (fixed before any result)
- **Per checkpoint:** k / n with a Wilson 95 % interval (n = 50).
- **Per family × task:** the mean success over training seeds, with every per-seed value
  listed.
- **Hierarchical paired bootstrap:**
  - B = 10,000 replicates with bootstrap seed `20261106`.
  - Each replicate resamples the training seeds with replacement, independently per family,
    and the 50 initial states with replacement, in one draw shared by every family so that
    pairing is kept.
  - Statistic: mean success over the resampled seeds × states. Report 95 % percentile
    intervals for each family and for each paired difference, computed on the same
    replicates.
  - With 3 seeds the seed-level resampling is coarse; the report says so and lists the
    per-seed values.
- **Units of analysis:**
  - Time steps within a rollout are never treated as independent.
  - Repeated stochastic samples at one (seed, state) are averaged before the bootstrap, so
    they never count as extra states.
- **Claims:** no p-value threshold decides "better". Report absolute differences, intervals,
  practical relevance and cost/latency. H2 is the primary comparison per task; the others are
  secondary. Tasks are reported separately; no pooled Lift + Can headline.

## 10. Final-test protection (binding)
- The locked suite is read only by `src/roboforge/sim/eval_states.py` after the lock check.
  Only that module, `snapshot.py` and `scripts/generate_eval_states.py` may contain its
  name (research-integrity test).
- Evaluation CLIs before the freeze accept only the validation suite.
- No validation result, tuning decision or checkpoint choice may be revisited after a test
  run. Any later test access is labeled exploratory or replication (MASTER_SPEC §3.2).

## 11. Tracking and provenance
- **Training run:** `runs/<run_id>/` is never reused.
  - Its `manifest.json` is written once, with exclusive create, at the end of the run, or
    with `status: failed` and the error. It holds:
    - git SHA + dirty flag, the environment snapshot and the platform;
    - dataset, processed, split, normalization and spec hashes;
    - the full resolved config and its hash, seeds and thread counts;
    - parameter counts;
    - every checkpoint's step and SHA-256;
    - wall and CPU time, and metric file hashes.
  - Checkpoints (`runs/<run_id>/checkpoints/step_<N>.pt`) hold only tensors and
    JSON-compatible values and load with `torch.load(weights_only=True)` and a strict
    `load_state_dict`. They contain model, optimizer and scheduler state, normalizers,
    config, hashes, the global step and RNG states.
- **Evaluation run:** `runs/<eval_run_id>/` holds the episode records and an immutable
  manifest that references the training run ID and checkpoint SHA-256. Its summary goes to
  `reports/session<N>/validation/`.
- **Selection:** §7.5.
- Results tables are generated from these files, never typed by hand. Every report carries
  the git SHA, dirty flag and platform.

## 12. Expected limitations (carried into the paper)
- Simulation only; no sim-to-real claims.
- Privileged state observations (§3).
- One demonstrator-quality tier (PH), 200 demos per task.
- Narrow, in-distribution Lift starts (KI-011) and a probable Lift ceiling (§5).
- Imperfect open-loop replay fidelity (KI-004).
- Corrected 19-D Can input, not comparable with robomimic's 23-D numbers (KI-003, D-010).
- v1.5 datasets ≠ CoRL 2021 benchmark (KI-012).
- One evaluation platform (Linux aarch64 container); other platforms are unverified
  (KI-001, KI-013).
- A limited, matched tuning budget (§6.3) and 3 training seeds.

## 13. Amendments
| Date | Section | Change | Reason | Runs affected |
|---|---|---|---|---|
| 2026-11-06 | all | v1 finalized from the Session 0–2 draft | Session 3 requirement: finalize before any reported run | none (no reported run existed) |
