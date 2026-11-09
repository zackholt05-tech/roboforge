<!-- parsable handoff header (values below are real, filled at the end of Session 7) -->
```yaml
project: roboforge-3d
current_session: 7
session_status: PASS
next_session: 8
last_verified_commit: bf0a0fa   # last code/test change; lint clean, 372 tests; later commits are evidence/docs only (re-verified at the STOP commit, LEDGER E7-15)
protocol_commits: {v1: 8b08b95, diffusion_amendment: 3aed462, main_budget_amendment: ec9626b, lift_search_space: 6dce01b, can_search_space: 96aa972, section14_preregistration: 2859d04}
freeze_commit: bf0a0fa          # docs/FINAL_TEST_FREEZE.md, sha256 6a3c6874c4bc1857efe46ccb05c0a1fa68247671fc9a96f5f54ce079aa9153e9, git blob bfdb4b77d478bdb1908b34b96e1dad70eb88147c
heldout_results_commit: 97cc48a # reports/session7/results/results.json sha256 b8b097c38765c588711615917e3cfd5be6847e63083ddebe92f0f1488672408c; RUN ONCE, never rerun
level1_freeze: docs/LEVEL1_COMPLETE.md   # HARD SCIENTIFIC BOUNDARY: later RGB/point-cloud results never modify Level 1 selection, labels or findings
python: 3.11.15                       # development (macOS arm64); canonical platform uses 3.11.18
sim_engine_version: mujoco 3.2.7
robosuite_revision: 51cc01785bab80ffeed20da15e67d7dd4140e76a   # v1.5.1
robomimic_revision: d309eaecc18acf4152a830a895a6984b8ac71b05
diffusers: 0.11.1
torch: 2.15.0 (macOS) / 2.15.0+cu132 (container and Modal)
evaluation_platform: {platform_id: linux-aarch64-container-1f685638f28c, image_id: "sha256:1f685638f28c720fbec7b084b8e5c2d68073d8f637450d5f6e0a34f0d123071c", runner: infra/containers/run_sim_cpu.sh}
dataset_and_specs: "demonstrations, validation/held-out suites, uv.lock, pyproject unchanged since Session 2; Session 7 added data/eval_states/<task>/{robustness_base,ood_*} and configs/eval/{robustness,failure_taxonomy}.yaml (D-063, §14)"
raw_rgb_source: "data/raw/robomimic_v1.5/{lift,can}/ph/demo_v15.hdf5 downloaded (states + model XML per demo; no images); Square not downloaded"
main_suite:
  families: [bc_mlp, bc_rnn, bc_mlp_chunk, diffusion_unet]
  tasks: {lift: "Session 6 (24 runs, 480 evaluations)", can: "Session 7 (24 runs, 480 evaluations)"}
  seeds: [0, 1, 2]
  budget: "100,000 steps, batch 100, checkpoints every 5,000 (K = 20), every checkpoint on 50 validation starts"
  tuning: "4 of 4 configurations used per family on Lift and on Can; none remains"
  compute: "local CPU only, 1 torch thread, <= 8 concurrent training runs; canonical container <= 4 in parallel; $0 (Gate C, D-051)"
  can_suite: {plan: configs/experiments/level1_core_can.yaml, plan_sha256: 881b4fb7fd72a33866ad4924db3b72cbe4270a212b03a8cc958efc6ba8a89e3c, summary: reports/session7/suite/level1_core_can.json, runs: 24, evaluations: 480, failed: 0}
baseline_configs:
  bc_mlp:         {horizons: "T_o=1 T_p=1 T_a=1", model: "MLP 2x1024 ReLU", params: 1077255}
  bc_rnn:         {horizons: "L=10 T_p=1 T_a=1", model: "LSTM 2x296, sliding window", params: 1080703}
  bc_mlp_chunk:   {horizons: "T_o=2 T_p=16 T_a=8", model: "MLP 2x1024 ReLU", params: 1204336}
  diffusion_unet: {horizons: "T_o=2 T_p=16 T_a=8", model: "flatten encoder + ConditionalUnet1D (64,128)", params: 1092551, sampler: "DDIM-10", ema: "power 0.75"}
lift_test_configuration:              # FROZEN (bf0a0fa); reports/session6/lift_test_configuration.json (sha256 38ac3c29...)
  held_out_schedule_manifest_sha256: e02498651b9d5bb88b6aec0870c5e3277cadaaeaf0e325abf86c79e89b1f22da   # n = 50; hash only, never loaded
  bc_mlp:   # chosen trial 4 (optim.weight_decay=0.1)
    s6_bc_mlp_wd1e1_lift_seed0_c39e4980: {seed: 0, selected_step: 15000, validation: 49/50, checkpoint_sha256: e68bc5e05c5b5f78c0e11c33401ed6feee361be44a629ff1fac394949cd7ef54}
    s6_bc_mlp_wd1e1_lift_seed1_628a1e7a: {seed: 1, selected_step: 5000, validation: 48/50, checkpoint_sha256: 5454cf8081b88319bcb91e35d8e69eb8fe0dbef9973c21454b690309dd9995ea}
    s6_bc_mlp_wd1e1_lift_seed2_67c90df6: {seed: 2, selected_step: 5000, validation: 49/50, checkpoint_sha256: 81a69c4c31d1446e5915977831338244e71dda7721697ef62230488548025d1e}
  bc_rnn:   # chosen trial 4 (optim.weight_decay=0.1)
    s6_bc_rnn_wd1e1_lift_seed0_166e9b0a: {seed: 0, selected_step: 15000, validation: 50/50, checkpoint_sha256: fb71d7c60f72ab68c0906d08837d51bbeeb06fbc253de0d5544c255d2d767b1b}
    s6_bc_rnn_wd1e1_lift_seed1_5221b93e: {seed: 1, selected_step: 10000, validation: 50/50, checkpoint_sha256: 30269f1f3b179640687e2c969e8c05095e1c73d1ddbcebbcd30ee5f627511508}
    s6_bc_rnn_wd1e1_lift_seed2_3368e1df: {seed: 2, selected_step: 5000, validation: 50/50, checkpoint_sha256: 1334302f18c9b6a82fa99e676965f5cb54896220b6c8a47b505e727b902dc9b9}
  bc_mlp_chunk:   # chosen trial 2 (optim.schedule=cosine)
    s6_bc_mlp_chunk_cos_lift_seed0_445dbbb5: {seed: 0, selected_step: 15000, validation: 50/50, checkpoint_sha256: c72c4d182450e63456ce53d1db6f35a95531ecab79181602956e6feaaeb03d54}
    s6_bc_mlp_chunk_cos_lift_seed1_c5955411: {seed: 1, selected_step: 5000, validation: 49/50, checkpoint_sha256: 9a46001eab065a990e80270f37e622a4c86664aec75517ac5acbab018dc491bb}
    s6_bc_mlp_chunk_cos_lift_seed2_98c39779: {seed: 2, selected_step: 10000, validation: 50/50, checkpoint_sha256: b7c9aa48bbd063f4466cc9f855542fa7408d181a13a7366aed6c75485ab2fe21}
  diffusion_unet:   # chosen trial 3 (optim.lr=3.0e-4)
    s6_diffusion_unet_lr3e4_lift_seed0_2f635398: {seed: 0, selected_step: 100000, validation: 50/50, checkpoint_sha256: 7e48766542774f52a5c9e73ca2a2776d95981188ed30d65188746479c55f0f38}
    s6_diffusion_unet_lr3e4_lift_seed1_50ae45de: {seed: 1, selected_step: 95000, validation: 50/50, checkpoint_sha256: f26868ea80e307200e92de962493797add56e97d0a1c38519383de64baeb81f7}
    s6_diffusion_unet_lr3e4_lift_seed2_427ac4f5: {seed: 2, selected_step: 80000, validation: 50/50, checkpoint_sha256: b52c30177478e6c8fed6eabd220c08fb839974550da925cb751b341a537f5587}
can_test_configuration:               # FROZEN (bf0a0fa); reports/session7/can_test_configuration.json (sha256 116f10a4...)
  held_out_schedule_manifest_sha256: 5f407748f5f261ef8fdfe4bec0d53c74f8e5ae064e53ff831e17a649528f7ff0   # n = 50; tested once in Session 7
  bc_mlp:   # chosen trial 2 (optim.schedule=cosine)
    s7_bc_mlp_cos_can_seed0_ae8152e3: {seed: 0, selected_step: 55000, checkpoint_sha256: b8fa2d06c19ae9d4830cf6b2a50b272e6fc91f3ab8668c68aae17092a16dc081}
    s7_bc_mlp_cos_can_seed1_00564355: {seed: 1, selected_step: 15000, checkpoint_sha256: cbf55c0d0656506c8457519024dee360d52f06abf3e92a485a2ef38ee03a34e2}
    s7_bc_mlp_cos_can_seed2_7359a97f: {seed: 2, selected_step: 55000, checkpoint_sha256: 3c1535918fb744fca9594da6e11220922db759f72fa09c984ec00006f0e4647a}
  bc_rnn:   # chosen trial 2 (optim.schedule=cosine)
    s7_bc_rnn_cos_can_seed0_5ff5ce91: {seed: 0, selected_step: 30000, checkpoint_sha256: 5085d716f285631e749a9439151c3aafab6fb6c67baae78800b3a9a0014e6137}
    s7_bc_rnn_cos_can_seed1_d34c6b99: {seed: 1, selected_step: 25000, checkpoint_sha256: a69eef409a35adbcd6e393e3397fb5603670a254d1d2b05b6d68e677d39fd23a}
    s7_bc_rnn_cos_can_seed2_8452a37a: {seed: 2, selected_step: 35000, checkpoint_sha256: 7cda91e6881c5d9a71a8c4b4fc5e56b6d527887b049af91eb481d700e164a148}
  bc_mlp_chunk:   # chosen trial 4 (optim.weight_decay=0.1)
    s7_bc_mlp_chunk_wd1e1_can_seed0_0c2f9d4b: {seed: 0, selected_step: 95000, checkpoint_sha256: 1968ad1148e4f20f82fffd6dfa57196f82ee3c7544e4e8c6eea5af2d7b367cd9}
    s7_bc_mlp_chunk_wd1e1_can_seed1_92a1f312: {seed: 1, selected_step: 85000, checkpoint_sha256: 3b306ad736f3b2df7e53928599b17125f834c1260224dbde97d57cdb23dc4fb2}
    s7_bc_mlp_chunk_wd1e1_can_seed2_63847bb4: {seed: 2, selected_step: 30000, checkpoint_sha256: ebbff8e38a713c584cbef4058534d0377d108701457d79ac25814bc7ff24552f}
  diffusion_unet:   # chosen trial 2 (optim.schedule=cosine)
    s7_diffusion_unet_cos_can_seed0_99251da3: {seed: 0, selected_step: 65000, checkpoint_sha256: 41dd66ddaf7b54d7114cbfda8058ede8270abbd83eefe715d679aa8645c35c1d}
    s7_diffusion_unet_cos_can_seed1_51f8481d: {seed: 1, selected_step: 100000, checkpoint_sha256: a687d9fbadf65410cc768ee63b901c248099d9de1b13bf65384433df58f74fab}
    s7_diffusion_unet_cos_can_seed2_9b0fde56: {seed: 2, selected_step: 90000, checkpoint_sha256: 49d73c93ed0a70b79a433892808468005004f89f8c923685c4cb342e3a456e27}
ablation_a3_runs: [s7_diffusion_unet_cos_to1_can_seed0_4b9d434e, s7_diffusion_unet_cos_to1_can_seed1_f9e76b7e, s7_diffusion_unet_cos_to1_can_seed2_5fa43638]   # T_o = 1; selected 95k/100k/90k, 49/50 each (validation, development)
heldout_results:                      # family success %, 3 seeds x 50 held-out starts (reports/session7/results/tables.md)
  lift: {bc_mlp: 90.7, bc_rnn: 100.0, bc_mlp_chunk: 98.0, diffusion_unet: 97.3, random_uniform: 0.0}
  can:  {bc_mlp: 76.7, bc_rnn: 96.0, bc_mlp_chunk: 96.0, diffusion_unet: 95.3, random_uniform: 0.0}
modal: {gate_a: done, approval: "A-01 ($2.00 cap) closed", billed_usd: 0.148, used_in_sessions_6_7: none, egl_vs_osmesa: PENDING (KI-001, KI-033)}
final_test: "TESTED ONCE (Session 7). The lock is open because the freeze is committed; that is not permission: no further held-out rollout of any Level 1 configuration, ever, and no Session 8 read of final_test until docs/RGB_EXPERIMENT_FREEZE.md is committed and the lock is re-gated on it"
disk: "22 GB free at the end of Session 7; runs/ 26 GB (KI-008: Docker build cache prune is an owner decision)"
next_required_owner_approval: "Session 8: Gate D (RGB visual policies) budget approval before any RGB main training; any Modal/GPU use needs a new approval row with a cap"
```

# Session 7 Handoff

Do not trust this file blindly: re-run the checks in "Exact starting instructions".

## 1. Session completed
Session 7 (MASTER_SPEC: Level 1 Can, paired held-out testing, robustness, scientific freeze).
Status **PASS**. Level 1 is complete and frozen in `docs/LEVEL1_COMPLETE.md`.
- **Can main suite:** 24 runs and 480 validation evaluations.
- **Robustness and ablations:** H4 plus A1–A3 on preregistered fresh starts, 135
  evaluations, validation regime.
- **Freeze:** committed before any held-out rollout (`bf0a0fa`).
- **Held-out test:** run once, 26 evaluations, then analysed once (`97cc48a`).

All work was local at $0. No run failed.

## 2. Completed work
- **Can main suite** (E7-02–E7-04):
  - Started from `96aa972`. It crashed at 02:14 UTC (KI-036) and was fixed in D-070
    (`f0715c4`), then restarted.
  - 24/24 runs complete, 480/480 evaluations, failed none.
  - Chosen trials: BC-MLP 2, BC-RNN 2, BC-MLP-chunk 4 and diffusion 2.
- **Can latency, figures and test configuration** (E7-06, E7-07):
  - `reports/session7/cost_latency/`, `figures/` and `can_test_configuration.json`.
- **Preregistration, RESEARCH_PROTOCOL §14** (`2859d04`):
  - the robustness starts (`5d9b8c0`; container check bitwise after KI-037, `005bc94`);
  - the failure taxonomy, D-065;
  - the bootstrap, D-066;
  - the held-out path with its rehearsal, D-067 and D-069.
- **Robustness (H4) and ablations A1, A2 and A3** (E7-09–E7-11):
  - Evaluated from `005bc94` and analysed from `565c6e4` after KI-038.
- **Rehearsal** of the held-out path on the validation suite (E7-12): PASS.
- **Freeze** (E7-13, D-072, KI-039):
  - Container attestation: locked before the freeze, unlocked after it.
  - The freeze is identical to the rehearsed draft except for its date.
- **Held-out test** once, then the frozen analysis once (E7-14).
- **Records:** `docs/LEVEL1_COMPLETE.md`, plus all the STOP docs.

## 3. Files created or changed (Session 7)
- **Code:**
  - Can plan: `configs/experiments/level1_core_can.yaml`.
  - A3 plan and config: `ablation_a3_can.yaml`, `configs/models/diffusion_unet_to1_can.yaml`.
  - `src/roboforge/sim/robustness.py`, `configs/eval/{robustness,failure_taxonomy}.yaml`.
  - `src/roboforge/eval/{taxonomy,statistics,freeze,heldout}.py` and inference variants.
  - Scripts: `generate_robustness_states`, `run_evals`, `analyze_robustness`,
    `failure_taxonomy`, `write_freeze_block`, `run_heldout`, `eval_heldout`,
    `analyze_heldout`, `check_rehearsal`.
  - Changes: the D-070 guards in `run_suite.py`; the host-wide container slots
    (`roboforge.slots`).
  - Tests for each of these.
- **Data:** `data/eval_states/<task>/{robustness_base,ood_*}`, 5 sets × 50 (D-063).
- **Docs:**
  - `docs/FINAL_TEST_FREEZE.md` and `docs/LEVEL1_COMPLETE.md`;
  - RESEARCH_PROTOCOL §14;
  - DECISIONS D-060–D-072;
  - KNOWN_ISSUES KI-036–KI-039;
  - EXPERIMENT_LEDGER E7-01–E7-15, with cost and tuning accounting;
  - COMPUTE_PLAN, STATUS and HANDOFF.
- **Evidence:**
  - `reports/session7/`: suite, validation, selection, tuning, cost_latency, figures,
    robustness, heldout and results.
  - `logs/session7/`: suite, robustness, rehearsal, heldout, analysis and `WORKLOG.md`.
  - Checkpoints and episode records stay in `runs/` (gitignored).

## 4. Executed commands and results
See EXPERIMENT_LEDGER E7-01–E7-15. Every reported rollout ran on the canonical container
through `infra/containers/run_sim_cpu.sh` from a clean commit:
- **Training:** 27 runs, all `complete`.
- **Evaluations:** 667 in all, with 0 errors and every audit passing.
  - 480 Can validation;
  - 135 robustness and ablation;
  - 26 rehearsal;
  - 26 held-out.
- **Final checks:** `make lint` is clean and `make test` gives **372 passed** at `bf0a0fa`.
  They were re-run at the STOP commit (E7-15).

## 5. Key findings: held-out test, frozen protocol (`docs/LEVEL1_COMPLETE.md` §2–§3)
- **Success, % of 3 seeds × 50 held-out starts:**

  | Family | Lift | Can |
  |---|---|---|
  | BC-MLP | 90.7 | 76.7 |
  | BC-RNN | 100.0 | 96.0 |
  | BC-MLP-chunk | 98.0 | 96.0 |
  | Diffusion | 97.3 | 95.3 |
  | Random | 0 | 0 |

- **H1:** separated for every family on both tasks.
- **H2, diffusion − BC-MLP:** on Can +18.7 pp [+4.7, +32.7], separated. On Lift +6.7 pp,
  not separated in success, but separated in the secondary episode length.
- **H2, diffusion − BC-RNN:** not separated on either task.
- **H2-c, diffusion − BC-MLP-chunk:** not separated on either task, so the data do not
  attribute any diffusion advantage to the generative model.
- **H3, BC-RNN − BC-MLP:** separated on both tasks, +9.3 and +19.3 pp. A3 (T_o within
  diffusion) is not separated.
- **H4:** every family drops under every OOD set.
  - Diffusion degrades least on the Lift arm offset: 88.0 %, against 22.7–50.0 % for the BC
    families.
  - On the Can arm offset, diffusion (71.3 %) and BC-MLP-chunk (79.3 %) are not separated.
- **A1 (EMA) and A2 (DDIM-10 vs DDPM-100):** not separated on either task.
- **Selection optimism:** validation scores exceed fresh-start and held-out scores. On Can the
  gap is up to 13.4 pp on fresh starts and up to 6.0 pp on the held-out starts (BC-MLP;
  descriptive).

## 6. Open decisions (Session 8)
- **Gate D (RGB visual policies):** the owner must approve a budget before any RGB main
  training, from a measured pilot.
  - Local CPU only at $0, or a capped Modal approval row.
- **Resolution 84×84 or 128×128:** from the alignment and throughput pilot (MASTER_SPEC §7.2).
- **Encoder regime:** from scratch, pretrained and frozen, or fine-tuned. Shared across
  visual BC and visual diffusion.
- **Disk:** RGB datasets may not fit in the 22 GB free. Pruning the Docker build cache
  (KI-008) is an owner decision.

## 7. Decisions
D-060–D-072 (Session 7). D-072 records how the freeze and the held-out run were executed.

## 8. Outstanding issues (none blocks Session 8)
- **KI-001 / KI-033:** EGL vs OSMesa on Modal is still unsettled. OSMesa works in the
  canonical container; check it for RGB throughput.
- **KI-008:** disk and the Docker build cache.
- **KI-034:** Lift is near ceiling.
- **KI-035:** a shared host inflates wall times.
- **KI-028:** R = 1.
- All Session 7 issues (KI-036–KI-039) are FIXED.

## 9. Budget
- **Session 7:** $0. 46.19 CPU core-h of training and 64.25 container-h of evaluation.
- **Level 1 in all:** $0.148, from Session 5 Modal Gate A.
- No approval row is open. Any Modal or GPU use needs a new row with a cap first.

## 10. Exact starting instructions for Session 8
Paste into a fresh Claude Code session:

```text
Implement RoboForge 3D SESSION 8 ONLY from MASTER_SPEC.md (Level 1 RGB-conditioned policy,
separate feasibility/budget gate; MASTER_SPEC §7.2 and Session 8). Begin by reading all of
MASTER_SPEC.md, docs/STATUS.md, docs/HANDOFF.md, docs/LEVEL1_COMPLETE.md,
docs/FINAL_TEST_FREEZE.md, docs/DECISIONS.md (through D-072), docs/KNOWN_ISSUES.md
(especially KI-001, KI-008, KI-018, KI-019, KI-020, KI-022, KI-033, KI-034, KI-035, KI-039),
docs/EXPERIMENT_LEDGER.md (Sessions 6-7), docs/RESEARCH_PROTOCOL.md (all amendments
binding, §14 included), docs/SIMULATOR_CONTRACT.md, docs/DATA_DICTIONARY.md and
docs/COMPUTE_PLAN.md, and inspect the actual repository/test state. Do not blindly trust
prior handoff claims: first run `make setup`, `make download`, `make check-data`,
`make build-data`, `make check-eval-states`, confirm `git status --short data/` is empty,
then `make lint`, `make test`, `make freeze-check`,
`make smoke-sim SMOKE_OUT=logs/session8/prereq`, and on the canonical container
`ROBOFORGE_EVIDENCE_SESSION=8 make in-sim CMD="python scripts/smoke_sim.py --task lift --out-dir logs/session8/prereq/container"`
and the same with `--task can`; confirm
`docker image inspect -f '{{.Id}}' roboforge-sim-cpu:session0` equals the image_id in
configs/eval/platform.yaml; confirm the Level 1 freeze is intact:
`git diff 97cc48a -- docs/FINAL_TEST_FREEZE.md reports/session6/lift_test_configuration.json reports/session7/can_test_configuration.json reports/session7/heldout reports/session7/results`
is empty and `git log --oneline -- docs/LEVEL1_COMPLETE.md` lists exactly one commit (the
Session 7 STOP commit; report, do not repair, if not); run git log --format=fuller -n 5 to confirm the
verified commits and the identity Zackary Holt <zackholt05@gmail.com>. Pass
EVIDENCE_SESSION=8 and PREFIX=s8 (or bump the Makefile defaults as a recorded change) so
Session 8 never writes into Session 6-7 evidence.

HARD SCIENTIFIC BOUNDARY (Session 7 STOP): Level 1 state-based results are frozen. Never
rerun make eval-final or make analyze-final, never re-evaluate any Level 1 configuration on
the held-out suites, never edit reports/session6-7 evidence, the freeze or LEVEL1_COMPLETE,
and never let RGB results modify core selection, labels or findings. RGB results are a
separately labelled observation-modality study (MASTER_SPEC §3.2 item 5); state-based Level 1
is an oracle-style comparator because it sees privileged object state.
The held-out lock is now OPEN for every caller (the Level 1 freeze is committed, so
eval_states.lock_status returns unlocked): that is NOT permission. No Session 8 code may
load, render, reset to or otherwise read the final_test suite or its seeds for any purpose
until docs/RGB_EXPERIMENT_FREEZE.md is committed; before any RGB held-out work, re-gate the
lock on that file (or on a new, separately locked RGB suite) with a test, as a recorded
decision, and decide in the RGB freeze whether the RGB test reuses the Level 1 held-out
states or a new suite. Never prune or move the 24 frozen run directories, the A3 runs or
runs/*__heldout* (gitignored; the only copies behind the freeze's checkpoint hashes and
the results' taxonomy), even if disk runs short.

Session 8 goal (MASTER_SPEC Session 8), in this order:
(1) Feasibility, no training yet: from data/raw/robomimic_v1.5/{lift,can}/ph/demo_v15.hdf5
(per-demo `states` and `model_file`; no images in the file), restore raw simulator states
through RoboForgeEnv-compatible code (never mj_forward on live evaluation data, KI-019;
never robosuite transform helpers for precision-critical values, KI-018), render RGB
headless with the exact camera name(s) and backend documented (OSMesa in the canonical
container; EGL on Modal is still unsettled, KI-001/KI-033), and verify exact time-step
alignment (task, demo/episode id, step_index, state hash, camera intrinsics/extrinsics,
model XML hash, frame hash). Create a small demonstrably aligned clip with ground-truth
actions, inspect a fixed random sample by eye, and add a code test comparing the projected
object position with its visible location (segmentation where available). Measure render
throughput and storage for 84x84 and 128x128 on this host. If raw data or rendering fails,
mark the RGB study NOT EXECUTED with details and reproducible evidence, publish that
feasibility note honestly, update the docs and STOP (no fake or untrained "visual control").
(2) Gate D budget proposal and owner approval: from the measured pilot, propose seeds x
steps x encoder/resolution x evaluation cost (local CPU at $0, or an exact capped Modal plan
with upper bounds), record it in docs/COMPUTE_PLAN.md and STOP to ask the owner; record the
owner's decision as an approval row in docs/EXPERIMENT_LEDGER.md before any main RGB
training. No paid or remote compute without that row.
(3) RGB experiment freeze, committed before the first RGB training run: a fair visual BC vs
visual diffusion comparison on Lift (optionally Can), RGB-only meaning robot proprioception
+ camera image and NO object ground-truth state (check that object_state never leaks
through normalisation or features), declared in ObservationSpec and every run manifest;
one shared encoder architecture and initialization (e.g. a compact ResNet), identical
augmentations, preprocessing, proprioceptive features, camera settings, budget, selection
rule (§7 on validation starts only) and evaluation states; the held-out RGB test initial
states, run count, bootstrap and reporting rules, written into a new docs/RGB_EXPERIMENT_FREEZE.md
(name it; do not edit docs/FINAL_TEST_FREEZE.md) before any RGB held-out rollout.
(4) Train and evaluate under that freeze and the approved budget; produce real
RGB-conditioned closed-loop results on the canonical container and representative failure
videos; if unsuccessful, publish that fact. Label every RGB number by modality, data and
compute budget, and compare visual-only to visual-only for architectural claims.

Compute: Gate C covered the Level 1 main suite (local CPU only, $0; 1 torch thread per run,
at most 8 concurrent training runs, at most 4 evaluation containers in parallel); Session 8
needs its own Gate D decision before main RGB training, and any Modal use needs a new owner
approval row first. Disk is tight (22 GB free at the end of Session 7; KI-008): measure RGB
storage before generating datasets. Track failed runs too. Measure latency on the canonical
container with nothing else running.

Keep the Session 1-7 contracts (DECISIONS D-010, D-017..D-072): training data only through
committed, hashed, versioned datasets (extend SequenceDataset.from_committed or add a
recorded RGB equivalent); policy observations only from RoboForgeEnv; outputs decoded with
ActionCodec; learned policies implement roboforge.eval.policies.Policy and run with
run_episode + verify_record; stochastic policies seeded by the protocol's noise-seed rule;
resets only from stored snapshots; first success or 400 steps; model and checkpoint
selection only on validation starts; held-out states read only through
src/roboforge/sim/eval_states.py and only under a committed freeze; every reported,
selection or test rollout on the canonical container via infra/containers/run_sim_cpu.sh
with ROBOFORGE_EVIDENCE_SESSION=8 (never uv sync/uv run inside the image, KI-020). Native
macOS rollouts are development-only; checkpoints and anything a container rollout reads
live under runs/<run_id>/; every run writes an immutable manifest with git SHA + dirty flag
(launch reported runs from a clean committed tree; do not edit tracked files or commit
while they run), environment, platform, data/spec hashes, full config, seeds, parameter
count, checkpoint hashes and wall/CPU time; compare checkpoints by content, not file hash
(KI-029); report motion-dimension clipping separately from the gripper (KI-022); offline
validation MSE never selects (KI-024). Use the Make variable TRAIN_RUN (not RUN) for run
directories. No paid or remote compute and no remote publication without explicit owner
approval; do not change committed data artifacts, eval suites, uv.lock or dependencies
without a recorded decision (an image/rendering dependency needs one).

Use the current verified simulator/data/checkpoint/manifest contracts. Run relevant tests
and record commands, real results and provenance. Never leak held-out test states into model
selection; never invent results or costs (label estimates as estimates). Use Zackary Holt
as the sole named RoboForge project author with the verified local Git identity; no AI
co-author trailers, while preserving external scientific and license credit (Diffusion
Policy, robomimic, diffusers, robosuite, MuJoCo and any vision backbone used). At the
SESSION 8 STOP boundary (core Level 1 metrics remain frozen and unchanged), update
docs/STATUS.md, docs/HANDOFF.md, docs/DECISIONS.md, docs/KNOWN_ISSUES.md,
docs/EXPERIMENT_LEDGER.md (all runs, including failed ones, and the all-in cost record) and
docs/COMPUTE_PLAN.md; commit verified code only, inspect commit metadata
(git log --format=fuller -n 5), write the exact prompt for SESSION 9, and STOP. Do NOT
proceed automatically.
```
