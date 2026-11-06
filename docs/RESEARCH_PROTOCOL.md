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
    `(policy_seed, state_id, repeat)` with `policy_seed = 0` and R = 1 repeat (finalized in
    §6.4 by the amendment of 2026-11-06). Repeats of a deterministic policy are never added,
    because they would be duplicates.
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
| **Diffusion** | flatten encoder + conditional 1D temporal U-Net (64, 128), ε-prediction, K = 100 `squaredcos_cap_v2`, DDIM-10 sampling, EMA (§6.4) | 2 / 16 / 8 | 1,092,551 | DEFAULT (Session 4, §6.4) |

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
    (MASTER_SPEC §5.7). **Result (Session 4):** 1,092,551 parameters, +1.4 % vs BC-MLP, so
    the rule is not triggered and no capacity-matched configurations are added.
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

### 6.4 Diffusion family (FINALIZED 2026-11-06, Session 4, amendment §13)
Every value below was fixed before the first reported diffusion run. Code:
`src/roboforge/models/{diffusion,unet1d,noise_schedule}.py`, `src/roboforge/train/ema.py`;
config `configs/models/diffusion_unet.yaml`.

| Item | Value |
|---|---|
| Observation encoder | none with parameters: the normalized `T_o × 19` window is flattened into a 38-D global condition (robomimic's low-dim path). Padding repeats the first observation (D-022); `obs_valid` adds nothing. This is exactly BC-MLP-chunk's input |
| Noise-prediction network | conditional 1D temporal U-Net (Diffusion Policy `ConditionalUnet1D`, adapted from robomimic `d309eae`, MIT): down widths **(64, 128)**, i.e. 2 encoder stages (T_p 16 → 8), a bottleneck of 2 residual blocks at width 128, 1 decoder stage (8 → 16); kernel 5; GroupNorm with 8 groups; Mish. Every residual block gets FiLM scale and bias from `[timestep embedding ‖ 38-D condition]`. Timestep embedding: 64-D sinusoidal → Linear 256 → Mish → Linear 64 |
| Parameters | **1,092,551** (exact; `reports/session4/param_counts.json`): +1.4 % vs BC-MLP (1,077,255), +1.1 % vs BC-RNN, −9.3 % vs BC-MLP-chunk (1,204,336) |
| Capacity rule (§6.1) | **not triggered**: 1,092,551 ≤ 2 × 1,077,255 = 2,154,510. No capacity-matched BC configurations are added |
| Forward process | DDPM, K = **100** training timesteps, `squaredcos_cap_v2` betas (abar_0 = 0.99937, abar_99 = 2.4e-7), `A_t = sqrt(abar_t) A_0 + sqrt(1 − abar_t) ε` on the normalized chunk |
| Prediction target | **ε** (noise) |
| Loss | masked MSE between ε̂ and ε over the valid (non-padded) action steps and all 7 dimensions (the same masked MSE of §6.1). Per sample, `t ~ U{0..99}` and `ε ~ N(0, I)` come from the run's own generator, seeded `SeedSequence([s, 2])`. It is checkpointed, so resuming is bitwise |
| Sampler | **DDIM**, **10** inference steps (timesteps 90, 80, …, 0), η = 0 (deterministic given the initial noise), `set_alpha_to_one`. `clip_sample`: x̂_0 is clipped to [−1, 1] at every step, so the final chunk lies in the normalized range. **1 sample per decision** |
| Why DDIM-10 | measured on the canonical container (`reports/session4/latency_pre_amendment/`, untrained 200-step checkpoint, 1 thread, median of 60 decisions): DDIM-10 takes 42 ms per decision, DDIM-16 67 ms and DDPM-100 436 ms (natively 11 / 16 / 99 ms). DDPM-100 would make every canonical evaluation about 5–10× longer for the main suite (Sessions 6–7, Lift and Can). The training objective is identical for both samplers. DDPM-100 (robomimic's default) stays a declared alternative for the denoising-steps ablation (Session 7) |
| EMA | yes: diffusers 0.11.1 `EMAModel` semantics, power 0.75, inv_gamma 1, max decay 0.9999, updated after every optimizer step (as in robomimic `d309eae`). **The EMA weights are the evaluated and selected network**; the trained weights are checkpointed as well. BC families have no EMA (§6.2). EMA belongs to the reference recipe; the EMA ablation is listed for Session 7 |
| Horizons | T_o = 2, T_p = 16, T_a = 8, identical to BC-MLP-chunk (H2-c). A decision at frame t predicts `a[t .. t+15]` (D-022) and executes `a[t .. t+7]`, then replans. robomimic instead predicts `a[t−1 .. t+14]` and executes from index 1. Both execute `a[t]` first; RoboForge keeps BC-MLP-chunk's window (recorded deviation) |
| Optimization | identical to §6.2: AdamW, lr 1e-4, betas (0.9, 0.999), weight decay 0, constant LR, batch 100, no gradient clipping (norm logged), 1 torch thread, CPU, float32. **Deviation from robomimic's diffusion defaults** (cosine LR with 500 warm-up steps, L2 1e-6), chosen to keep the optimizer matched across families |
| Budget | pilot: 30,000 steps, checkpoints every 5,000 (the Session 3 pilot budget, unchanged). Main runs: the BC main budget of §6.3 |
| Offline metrics | logged on the EMA network every 1,000 steps: fixed-draw noise-prediction MSE (train and valid) and the MSE of seeded sampled chunks vs the demonstrated chunks (valid only). Never used for selection (§7.4) |
| Selection | §7 unchanged, on the EMA network of every checkpoint |

**Stochastic evaluation (fills §5).**
- **Noise-seed rule:** a stochastic policy starts each episode with a new `torch.Generator`
  seeded with `SeedSequence([policy_seed, uint64(sha256(state_id)[:8], little-endian),
  repeat]).generate_state(1, uint64)[0]` (`roboforge.eval.learned.noise_seed`). Every decision
  of the episode consumes it in order: first the initial noise `A_K`, then any sampler noise
  (none for DDIM with η = 0). The global RNG is never read. An episode's actions therefore
  depend only on the checkpoint, the seed triple and the observations, never on episode order.
- **policy_seed = 0** for every evaluation: every checkpoint, every training seed, validation
  and the held-out test. Checkpoints of one run are thus compared under common random numbers.
- **R = 1** (repeat 0) for validation, checkpoint selection and the held-out test: one rollout
  per (checkpoint, state), like the deterministic families. Repeats r ≥ 1 may only be added by
  an amendment before `docs/FINAL_TEST_FREEZE.md`; they would be averaged per (seed, state)
  before the bootstrap (§9).
- Plan diversity is measured from repeated samples at fixed conditions in the denoising export
  (8 extra seeds `1,000,000 + i`), never by adding rollouts.
- Each rollout record stores its seed (`policy_episode`). The evaluation run ID ends in
  `__ps0r0`.

**Tuning accounting.** The Session 4 diffusion Lift pilot is **trial 1 of 4** of the diffusion
family on Lift (§6.3).

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
- **Diffusion-specific** (finalized in §6.4): denoising steps (10, DDIM) and samples per
  decision (1), both fixed; inference latency per decision on the canonical container; plan
  diversity (per-step, per-dimension standard deviation of 8 extra samples at fixed exported
  conditions, normalized units); clip frequency of the decoded actions, motion dimensions
  separately from the gripper (KI-022).

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
| 2026-11-06 | §6.3 | The Session 3 BC-MLP-chunk Lift pilot also counts as **trial 1 of BC-MLP-chunk** on Lift | It was run and reported alongside the BC-MLP and BC-RNN pilots; every reported run must count against its family's tuning budget | accounting only: no rule or result changes |
| 2026-11-06 | §5, §6.1, §6.4 (new), §8 | **Session 4: every diffusion item marked TO FINALIZE is fixed**: flatten observation encoder; 1D U-Net (64, 128) with 64-D timestep embedding and FiLM, 1,092,551 parameters (capacity rule not triggered); DDPM forward process K = 100, `squaredcos_cap_v2`, ε-prediction, masked MSE; DDIM-10 sampler, η = 0, `clip_sample`, 1 sample per decision; EMA (power 0.75) as the evaluated network; T_o/T_p/T_a = 2/16/8; BC optimizer and budgets; noise-seed rule with `policy_seed = 0`, R = 1; the Session 4 Lift pilot is diffusion's tuning trial 1 | MASTER_SPEC Session 4 requires these values before the first reported diffusion run. Sampler chosen from latency measured on the canonical container (DDIM-10 42 ms vs DDPM-100 436 ms per decision); architecture sized to the 1.08 M BC budget within the spec's bounds (width 64–256, 2–3 stages) | none: committed before the first diffusion pilot run |
