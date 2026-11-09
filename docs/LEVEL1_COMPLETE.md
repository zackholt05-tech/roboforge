# Level 1 Research Freeze: behaviour cloning vs diffusion policies on robosuite Lift and Can

Frozen on **2026-11-08** at the end of Session 7 (MASTER_SPEC Session 7). This document records
what was actually executed, what the held-out test found, what was left out, and why. It is the
complete Level 1 scientific unit.

**Hard boundary (MASTER_SPEC Session 7 STOP).** Nothing after this point may change the core
test selection, labels or findings below. That includes later RGB, point-cloud or other Level 2
results. Later work is reported separately and never re-ranks these results.

Every number here is copied from a committed machine-readable record named next to it. Where
this prose and a record differ, the record is authoritative:

| Content | Record | SHA-256 | Commit |
|---|---|---|---|
| Held-out test | `reports/session7/results/results.json` | `b8b097c3…` | `97cc48a` |
| Generated tables | `reports/session7/results/tables.md` | — | `97cc48a` |
| Freeze | `docs/FINAL_TEST_FREEZE.md` | `6a3c6874…` | `bf0a0fa` |
| Robustness and ablations | `reports/session7/robustness/analysis.json` | `8cca66a8…` | `a92c74d` |

## 1. Question and design

- **Question.** In simulated 3D manipulation with privileged low-dimensional state, does a
  diffusion policy achieve higher closed-loop task success than deterministic behaviour cloning
  trained on the same demonstrations? If so, is the gain due to the generative model or to
  action chunking? (RESEARCH_PROTOCOL §1, H1–H4.)
- **Tasks:** robosuite v1.5.1 **Lift** and **PickPlaceCan** (Can). MuJoCo 3.2.7, OSC_POSE
  control, 7-D actions, 19-D state observations, 400-step horizon, an episode ends at the first
  success.
- **Data:** robomimic v1.5 proficient-human (PH) low-dim demonstrations, 180 train and 20 valid
  demonstrations per task (`data/manifests/robomimic_v1.5_ph.json`).
  - Lift: `low_dim_v15.hdf5`, SHA-256 `2067777c…`.
  - Can: `low_dim_v15.hdf5`, SHA-256 `3f2eb92e…`.
  - The demonstration data, the validation and held-out suites, `uv.lock` and the dependencies
    are unchanged since Session 2. Session 7 added only the preregistered robustness start sets
    (`data/eval_states/<task>/{robustness_base,ood_*}`) and `configs/eval/{robustness,failure_taxonomy}.yaml`
    (D-063, §14).
- **Families**, all at about 1.1 M parameters:

  | Family | Model | Horizons | Parameters |
  |---|---|---|---|
  | BC-MLP | MLP 2 × 1024 | T_o = 1, T_a = 1 | 1,077,255 |
  | BC-RNN | LSTM 2 × 296 | sequence length 10 | 1,080,703 |
  | BC-MLP-chunk (the H2-c control) | MLP 2 × 1024 | T_o = 2, T_p = 16, T_a = 8 | 1,204,336 |
  | Diffusion U-Net | ConditionalUnet1D, EMA weights, DDIM-10 sampler | T_o = 2, T_p = 16, T_a = 8 | 1,092,551 |

- **Budget and selection:**
  - Every run: 100,000 steps, batch 100, checkpoints every 5,000 (20 per run).
  - Every checkpoint was evaluated on 50 validation starts, and §7 selected one per run.
  - At most 4 configurations per family and task were tried at seed 0, identical across
    families and predeclared. The chosen configuration was then trained with seeds 1 and 2.
  - 3 seeds per family and task, 24 runs per task.
  - Lift ran in Session 6 (D-058) and Can in Session 7.
- **Held-out test:** 50 stored held-out starts per task, R = 1 rollout per checkpoint and start,
  policy seed 0, and the canonical container `linux-aarch64-container-1f685638f28c` (image
  `sha256:1f685638…`).
  - Run once from the committed freeze; analysed once with the §9 hierarchical paired bootstrap
    (B = 10,000, seed 20261106, paired by start, seeds resampled).
  - "Separated" means the 95 % interval excludes 0. "Not separated" is a reportable outcome,
    not evidence of equality.

## 2. Held-out results (primary; `reports/session7/results/tables.md`)

Engineering status: **complete for both tasks**.
- All 26 frozen evaluations ran once from `bf0a0fa` (clean), 22:50–23:49 UTC 2026-11-08.
- Every evaluation has 50 episodes, 0 errors and a passing re-simulation audit.
- No summary is missing, no issue is open and no engineering rerun was made.

### Family success, % of 3 seeds × 50 held-out starts (95 % CI; per seed)

| Family | Lift | Can |
|---|---|---|
| BC-MLP | 90.7 [83.3, 96.7] (88, 92, 92) | 76.7 [62.7, 90.0] (68, 88, 74) |
| BC-RNN | 100.0 [100.0, 100.0] (100, 100, 100) | 96.0 [89.3, 100.0] (92, 100, 96) |
| BC-MLP-chunk | 98.0 [94.0, 100.0] (98, 100, 96) | 96.0 [88.7, 100.0] (96, 92, 100) |
| Diffusion U-Net | 97.3 [93.3, 100.0] (98, 96, 98) | 95.3 [89.3, 100.0] (92, 98, 96) |
| random_uniform | 0.0 (0/50) | 0.0 (0/50) |

### Declared paired differences in success (percentage points, 95 % CI)

| Hypothesis | Difference | Lift | Can |
|---|---|---|---|
| H1 | each family − random | separated (+) for all four: +90.7 … +100.0 | separated (+) for all four: +76.7 … +96.0 |
| H2 | diffusion − BC-MLP | +6.7 [−0.7, +14.7], **not separated** | **+18.7 [+4.7, +32.7], separated** |
| H2 | diffusion − BC-RNN | −2.7 [−6.7, 0.0], **not separated** | −0.7 [−8.0, +6.7], **not separated** |
| H2-c | diffusion − BC-MLP-chunk | −0.7 [−5.3, +4.7], **not separated** | −0.7 [−8.7, +7.3], **not separated** |
| H3 | BC-RNN − BC-MLP | **+9.3 [+3.3, +16.7], separated** | **+19.3 [+5.3, +34.0], separated** |

### Secondary outcome: censored episode length (steps; failures count 400; declared in the freeze)

| Hypothesis | Difference | Lift | Can |
|---|---|---|---|
| H2 | diffusion − BC-MLP | −28.9 [−59.4, −1.6], separated (−) | −56.0 [−98.0, −14.7], separated (−) |
| H2 | diffusion − BC-RNN | +11.0 [+1.1, +25.1], separated (+) | +2.1 [−18.5, +22.7], not separated |
| H2-c | diffusion − BC-MLP-chunk | +2.8 [−15.6, +20.0], not separated | +9.5 [−13.0, +33.3], not separated |
| H3 | BC-RNN − BC-MLP | −39.9 [−67.8, −16.2], separated (−) | −58.1 [−100.3, −16.7], separated (−) |

A negative difference means fewer steps, so A is faster or fails less. On Lift, BC-RNN
finishes about 11 steps sooner than diffusion: mean 42.7 against 53.8.

### What the test supports

1. **H1 holds** on both tasks. Every learned family beats the random baseline by a wide
   margin.
2. **H2, against BC-MLP.**
   - Diffusion has higher success than BC-MLP on Can (+18.7 pp, separated). On Lift the
     difference is not separated in success (+6.7 pp), but the secondary episode length
     favours diffusion.
   - **H2-c qualifies this.** Diffusion and BC-MLP-chunk are not separated on either task,
     in success or in episode length. These data therefore do **not** attribute the
     advantage over BC-MLP to the generative model rather than to observation history plus
     action chunking. RESEARCH_PROTOCOL makes H2-c a requirement before any such
     attribution.
3. **H2, against BC-RNN.** Diffusion and BC-RNN are not separated in success on either
   task. On Lift, BC-RNN completes the task sooner than diffusion (secondary outcome).
4. **H3 (history) holds** for the deterministic models. BC-RNN beats BC-MLP in success on
   both tasks, and in episode length. The diffusion T_o ablation (A3, §3) found no separated
   effect of history within diffusion on Can.
5. **Lift is near ceiling.** Three families reach 97–100 %, so Lift separates only BC-MLP
   from the others. Can is harder and gives the only separated diffusion − BC result.

### Failure analysis, held-out episodes over all seeds (`reports/session7/results/taxonomy.json`; taxonomy SHA-256 `5a5d998f…`)

| Task | Family | Successes / 150 | Failures by category |
|---|---|---|---|
| Lift | BC-MLP | 136 | no grasp, drifted 12; dropped 2 |
| Lift | BC-RNN | 150 | none |
| Lift | BC-MLP-chunk | 147 | no grasp, drifted 3 |
| Lift | diffusion | 146 | no grasp, near 3; dropped 1 |
| Can | BC-MLP | 115 | gripper never closed 11; no grasp, drifted 8; dropped 7; no grasp, near 4; timeout while holding 3; in goal bin but not success 2 |
| Can | BC-RNN | 144 | in goal bin but not success 3; gripper never closed 1; no grasp, drifted 1; dropped 1 |
| Can | BC-MLP-chunk | 144 | no grasp, near 4; gripper never closed 1; in goal bin but not success 1 |
| Can | diffusion | 143 | in goal bin but not success 3; no grasp, drifted 2; dropped 2 |

`engineering_error` is 0 everywhere. BC-MLP's failures are mostly grasp failures: it never
closes the gripper, or it closes it away from the object. The categories are labels of
outcomes, not causes.

### Negative results, kept apart (freeze §4)

- **Engineering-incomplete: none.** No missing evaluation, no partial run, no failed audit
  and no engineering rerun.
- **Task failures:** the failure-analysis table above, one category per failed episode.
- **Statistical uncertainty:** the "not separated" rows above. With 3 seeds × 50 starts per
  family, differences of a few percentage points near the ceiling cannot be resolved. They
  are reported as not separated, not as equal.

### Example episodes (frozen rule)

Per task and family, among the frozen checkpoints in seed order, the rule picks the first
success and the first failure by sorted `state_id`. They are listed in `results.json`
(`tasks.<task>.examples`) by evaluation ID and state ID. Their episode records are under
`runs/<eval_run_id>/` (gitignored). Rendering them to video is a presentation step (§5).

## 3. Robustness and ablations (validation regime: fresh, preregistered starts; nothing selects)

Preregistration:
- RESEARCH_PROTOCOL §14 was committed 2026-11-07 (`2859d04`) before any robustness, ablation
  or test rollout.
- The starts were generated from stored snapshots: 5 sets × 50, seeds 300000+.
- They were validated and frozen, and the container check reproduced them bitwise.
- They are disjoint from validation, the held-out starts and each other.
- Results: `reports/session7/robustness/tables.md`, from `565c6e4`.

### H4: success under perturbation, % of 3 seeds × 50 starts (95 % CI in the table file)

| Family | Lift base | Lift cube shift | Lift arm offset | Can base | Can arm offset |
|---|---|---|---|---|---|
| BC-MLP | 98.0 | 52.7 | 22.7 | 69.3 | 30.0 |
| BC-RNN | 100.0 | 96.0 | 50.0 | 96.7 | 53.3 |
| BC-MLP-chunk | 100.0 | 54.7 | 48.7 | 95.3 | 79.3 |
| Diffusion U-Net | 99.3 | 85.3 | 88.0 | 92.0 | 71.3 |

**Every family drops under every perturbation.** The drops are separated except BC-RNN on
the Lift cube shift (+4.0 pp).

**Lift arm offset: diffusion degrades least.** Its drop of 11.3 pp is smaller than every BC
family's: 75.3, 50.0 and 51.3 pp. Every difference in drop is separated.

**Lift cube shift:** BC-RNN is best (96.0 %). Diffusion (85.3 %) and BC-RNN are not
separated.

**Can arm offset: diffusion (71.3 %) and BC-MLP-chunk (79.3 %) are not separated.**
Diffusion beats BC-MLP and BC-RNN in OOD success and in drop, all separated. BC-MLP-chunk
against BC-MLP or BC-RNN is not a declared comparison (its drop is 16.0 pp). H4 therefore
holds: robustness differs between families. The ordering depends on the perturbation, and on
Can the chunked deterministic control is not separated from diffusion.

### Ablations on `robustness_base`

| Ablation | Lift | Can |
|---|---|---|
| A1: primary (EMA) − raw trained weights | +0.7 [+0.0, +2.7], not separated | −2.7 [−7.3, +0.7], not separated |
| A2: primary (DDIM-10) − DDPM-100 | −0.7 [−4.0, +0.0], not separated | −3.3 [−10.0, +3.3], not separated |
| A3: T_o = 2 − T_o = 1 (Can only, separate runs) | — | −2.0 [−8.7, +4.0], not separated |

**Bias disclosure (§14.4):** A1 and A2 use checkpoints selected with the primary arm. A3's
T_o = 1 runs were selected by §7 on validation: 49/50 for all three seeds.

**Finding:** within this budget, EMA, the sampler (10 DDIM steps against 100 DDPM steps)
and the second observation frame make no separated difference to diffusion's success.

### Selection optimism (descriptive, not a preregistered test)

The validation numbers were used for selection, so they are optimistic. Fresh draws from the
same start distribution (`robustness_base`) and the held-out test both come out lower:

| Family | Can validation | Can fresh (`robustness_base`) | Can held-out | Lift validation | Lift held-out |
|---|---|---|---|---|---|
| BC-MLP | 82.7 | 69.3 | 76.7 | 97.3 | 90.7 |
| BC-RNN | 99.3 | 96.7 | 96.0 | 100.0 | 100.0 |
| BC-MLP-chunk | 100.0 | 95.3 | 96.0 | 99.3 | 98.0 |
| Diffusion U-Net | 98.7 | 92.0 | 95.3 | 100.0 | 97.3 |

Validation success is the selected checkpoints' score on the 50 validation starts that also
selected them. Development numbers only: `reports/session6/` and `reports/session7/`
`suite/*.json`.

## 4. Cost and latency

| Item | Value |
|---|---|
| Level 1 compute bill | **$0.148** in all (Session 5 Modal GPU smoke, Gate A) |
| Training and evaluation (Sessions 6–7) | **$0** (Gate C: local CPU only) |
| Training | 1 torch thread per run; Session 6: 33.25 CPU core-h; Session 7: 46.19 CPU core-h, of which A3 took 12.07 |
| Evaluation | canonical container; Session 6: 26.92 container-h; Session 7: 64.25 container-h, of which the held-out test took 1.83 |

Latency is the median per `act` call on the canonical container, nothing else running
(`reports/session{6,7}/cost_latency/`):

| Family | Lift (ms) | Can (ms) | Training CPU-h per run, Lift / Can |
|---|---|---|---|
| BC-MLP | 0.91 | 1.08 | 0.17–0.22 / 0.16–0.22 |
| BC-RNN | 9.04 | 9.00 | 0.93–1.20 / 0.85–1.27 |
| BC-MLP-chunk | 1.66 | 1.42 | 0.17–0.18 / 0.18 |
| Diffusion U-Net | 36.95 | 37.71 | 3.36–4.42 / 3.98–4.31 |

**Diffusion costs about 19–26× BC-MLP-chunk's training CPU time and about 22–27× its
median per-decision latency, for success that is not separated from BC-MLP-chunk on either
task.**

## 5. Omissions and reasons

- **The Lift cube shift has no Can counterpart, by design (§14.1).** Can's can is already
  sampled over the whole source bin with random yaw, so a placement shift would leave the bin
  and change the task. Can is therefore perturbed only in the robot's initial joints.
- **Square was not run.** It is MASTER_SPEC's optional P2 stretch task. Its data is listed in
  the dataset manifest but was never downloaded by default, and Level 1 is Lift and Can.
- **One rollout per start (R = 1)** for the held-out test (§6.4, freeze). Policy-noise
  variance is not estimated separately; seeds and starts are resampled by the bootstrap.
- **No GPU training.** Gate C chose local CPU only, $0 (D-051). EGL against OSMesa rendering
  on Modal is still pending (KI-001, KI-033) and is not needed for state-based Level 1.
- **The example episodes are identified by the frozen rule but not rendered** in Session 7.
  Rendering is presentation only and cannot change any result.
- **Lift validation was saturated** (KI-034). That limited how informative the Lift
  configuration choice could be. The held-out test is less saturated for BC-MLP but still
  near ceiling for the others.

## 6. Exploratory follow-ups (separate from the findings above; none changes them)

1. **A task harder than Lift** is needed to separate diffusion, BC-RNN and BC-MLP-chunk.
   Candidates are Square (data present) or Can with more perturbation. This is a new,
   separately preregistered study.
2. **Selection optimism** (§3) suggests reporting fresh-start validation next to selection
   validation in future suites.
3. **Diffusion's robustness on the Lift arm offset** stands out against the chunked control's
   failure there, while on Can the two match. A mechanism study is a follow-up, not a
   finding.
4. **RGB-conditioned policies** are Session 8, under their own feasibility gate, budget
   approval and RGB experiment freeze.

## 7. Provenance

| Item | Value |
|---|---|
| Frozen code | SHA-256 of every file, in the freeze block (`docs/FINAL_TEST_FREEZE.md` §6–7); `make freeze-check` passes at the STOP commit |
| Held-out schedules | Lift `e02498651b9d5bb88b6aec0870c5e3277cadaaeaf0e325abf86c79e89b1f22da`; Can `5f407748f5f261ef8fdfe4bec0d53c74f8e5ae064e53ff831e17a649528f7ff0` |
| Test configurations | `reports/session6/lift_test_configuration.json` (`38ac3c29…`, D-058); `reports/session7/can_test_configuration.json` (`116f10a4…`). Checkpoint hashes are in the freeze. |
| Protocol | `docs/RESEARCH_PROTOCOL.md`: v1 `8b08b95`; amendments `3aed462`, `ec9626b`, `6dce01b`, §14 `2859d04` |
| Lock | Container attestation: locked before the freeze commit, unlocked after it (`logs/session7/analysis/lock_status_*`). The freeze differs from the rehearsed draft only in its date (D-072). |
| Ledger | Every run, including failed and engineering attempts, with costs: `docs/EXPERIMENT_LEDGER.md` E6-*, E7-* |
| Authorship | Zackary Holt. External scientific and licence credit: Diffusion Policy (Chi et al.), robomimic (Mandlekar et al.), diffusers (Hugging Face), robosuite and MuJoCo. |
