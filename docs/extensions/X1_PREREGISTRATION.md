# X1 preregistration: BC with the diffusion sampler's action bound

**Label of every result: exploratory, after the freeze.** This file was written and committed
on 2026-11-10 (Session 13), before the first reported X1 rollout. That is the first clamped
evaluation on a robustness set; the commit order shows it. It governs X1 only, under DECISIONS
D-105, D-106 and D-107. Level 1, the RGB study and Level 2 stay frozen. No frozen file,
freeze, result, label or executed outcome changes. X1 adds an attribution; it never replaces a
frozen number.

## 1. Question

The diffusion sampler clamps its predicted clean action chunk to [−1, 1] in min–max normalized
space (`_Sampler._x0`, `clip_sample=True`). The last DDIM step returns that chunk unchanged.
Every diffusion action therefore lies inside each dimension's training range. The BC heads are
clipped only to the controller limits, ±1 (KI-052, audit A-6). Two frozen comparisons compare
the generative head *together with* this bound:
- **H4** (Lift arm offset): drop(diffusion) − drop(BC-MLP-chunk) = −40.0 pp [−54.0, −26.0],
  separated (−);
- **H2-c**: diffusion and BC-MLP-chunk are not separated on either task.

X1 asks: **when the frozen BC checkpoints get the same bound, does diffusion's advantage under
perturbation remain?**

## 2. Checkpoints (frozen; nothing is trained or reselected)

The 9 BC checkpoints per task of the frozen test configuration records:
- `reports/session6/lift_test_configuration.json`, SHA-256
  `38ac3c293bc5b2cd72b970b2044ed485a14800e7353b07e8956f7231d10961de`;
- `reports/session7/can_test_configuration.json`, SHA-256
  `116f10a45c243a0cecb6aabc9c48167507f290c70bf08f2f29df6bfc6184f690`.

These are the values the Session 7 robustness analysis recorded. Each checkpoint file is checked
against this SHA-256 by the host, against the training manifest in the container, and again when
it is loaded.

#### lift

| Family | Seed | Training run | Step | Checkpoint SHA-256 |
|---|---|---|---|---|
| bc_mlp | 0 | `s6_bc_mlp_wd1e1_lift_seed0_c39e4980` | 15000 | `e68bc5e05c5b5f78c0e11c33401ed6feee361be44a629ff1fac394949cd7ef54` |
| bc_mlp | 1 | `s6_bc_mlp_wd1e1_lift_seed1_628a1e7a` | 5000 | `5454cf8081b88319bcb91e35d8e69eb8fe0dbef9973c21454b690309dd9995ea` |
| bc_mlp | 2 | `s6_bc_mlp_wd1e1_lift_seed2_67c90df6` | 5000 | `81a69c4c31d1446e5915977831338244e71dda7721697ef62230488548025d1e` |
| bc_rnn | 0 | `s6_bc_rnn_wd1e1_lift_seed0_166e9b0a` | 15000 | `fb71d7c60f72ab68c0906d08837d51bbeeb06fbc253de0d5544c255d2d767b1b` |
| bc_rnn | 1 | `s6_bc_rnn_wd1e1_lift_seed1_5221b93e` | 10000 | `30269f1f3b179640687e2c969e8c05095e1c73d1ddbcebbcd30ee5f627511508` |
| bc_rnn | 2 | `s6_bc_rnn_wd1e1_lift_seed2_3368e1df` | 5000 | `1334302f18c9b6a82fa99e676965f5cb54896220b6c8a47b505e727b902dc9b9` |
| bc_mlp_chunk | 0 | `s6_bc_mlp_chunk_cos_lift_seed0_445dbbb5` | 15000 | `c72c4d182450e63456ce53d1db6f35a95531ecab79181602956e6feaaeb03d54` |
| bc_mlp_chunk | 1 | `s6_bc_mlp_chunk_cos_lift_seed1_c5955411` | 5000 | `9a46001eab065a990e80270f37e622a4c86664aec75517ac5acbab018dc491bb` |
| bc_mlp_chunk | 2 | `s6_bc_mlp_chunk_cos_lift_seed2_98c39779` | 10000 | `b7c9aa48bbd063f4466cc9f855542fa7408d181a13a7366aed6c75485ab2fe21` |

#### can

| Family | Seed | Training run | Step | Checkpoint SHA-256 |
|---|---|---|---|---|
| bc_mlp | 0 | `s7_bc_mlp_cos_can_seed0_ae8152e3` | 55000 | `b8fa2d06c19ae9d4830cf6b2a50b272e6fc91f3ab8668c68aae17092a16dc081` |
| bc_mlp | 1 | `s7_bc_mlp_cos_can_seed1_00564355` | 15000 | `cbf55c0d0656506c8457519024dee360d52f06abf3e92a485a2ef38ee03a34e2` |
| bc_mlp | 2 | `s7_bc_mlp_cos_can_seed2_7359a97f` | 55000 | `3c1535918fb744fca9594da6e11220922db759f72fa09c984ec00006f0e4647a` |
| bc_rnn | 0 | `s7_bc_rnn_cos_can_seed0_5ff5ce91` | 30000 | `5085d716f285631e749a9439151c3aafab6fb6c67baae78800b3a9a0014e6137` |
| bc_rnn | 1 | `s7_bc_rnn_cos_can_seed1_d34c6b99` | 25000 | `a69eef409a35adbcd6e393e3397fb5603670a254d1d2b05b6d68e677d39fd23a` |
| bc_rnn | 2 | `s7_bc_rnn_cos_can_seed2_8452a37a` | 35000 | `7cda91e6881c5d9a71a8c4b4fc5e56b6d527887b049af91eb481d700e164a148` |
| bc_mlp_chunk | 0 | `s7_bc_mlp_chunk_wd1e1_can_seed0_0c2f9d4b` | 95000 | `1968ad1148e4f20f82fffd6dfa57196f82ee3c7544e4e8c6eea5af2d7b367cd9` |
| bc_mlp_chunk | 1 | `s7_bc_mlp_chunk_wd1e1_can_seed1_92a1f312` | 85000 | `3b306ad736f3b2df7e53928599b17125f834c1260224dbde97d57cdb23dc4fb2` |
| bc_mlp_chunk | 2 | `s7_bc_mlp_chunk_wd1e1_can_seed2_63847bb4` | 30000 | `ebbff8e38a713c584cbef4058534d0377d108701457d79ac25814bc7ff24552f` |

The diffusion arm is not re-run. Its frozen Session 7 evaluations on the same sets already carry
the bound. Engineering check 2 shows that the clamp changes no diffusion action.

## 3. Suites (Session 7 robustness sets, RESEARCH_PROTOCOL §14.1–14.2; 50 starts each)

| Task | Set | Manifest content SHA-256 |
|---|---|---|
| lift | `robustness_base` | `b1ffb682c2c94128c7c773f8899edba42f04d8e2631f3ab63f7f7085cb05855d` |
| lift | `ood_cube_shift` | `c6083afc0212bf06f6eb98e7fc5522fae4ed8fae49bb89d8faf1e74d65d1b7ac` |
| lift | `ood_arm_offset` | `43c7c3d275b43880277e6ec762244b8600cf20bd717aac7451b032d10ebf0423` |
| can | `robustness_base` | `b91082eab3299e28ff028eb308da7c8166962a0ac3bd3fb44ac15ec26ba372ec` |
| can | `ood_arm_offset` | `32388f7907931f88822b4f23df1779f1bf5c51c94a5deb147d9a22f43f131690` |

- **45 evaluations:** Lift 9 × 3, Can 9 × 2.
- `robustness_base` is in distribution and disjoint from the held-out starts. It is evidence
  relevant to H2-c without revisiting the test.
- No held-out start and no validation start is stepped. The X1 code admits only the robustness
  sets (`tests/research_integrity/test_x1_scope.py`).

## 4. The clamp

- Each normalized action of the BC policy's executed chunk is clipped to [−1, 1], then decoded
  by the frozen `ActionCodec.decode`, which runs unchanged. This is the diffusion sampler's bound
  bit for bit.
- The statistics are the committed action normalizer of each task (mode `minmax`, fit on the
  training split). It is the only normalization any Level 1 checkpoint can load, so BC and
  diffusion share it:
  - Lift: `data/normalization/lift_ph_lowdim_v1.json`, content SHA-256
    `253bfc7e63570bf3e8f8646a0094d1da4c5b729ba8f463b4955f01410b4812d6`;
  - Can: `data/normalization/can_ph_lowdim_v1.json`,
    `e0437cd7b8e48403366d19f7b0e6116e2224723b8a5240b2d7ba008403b3228d`.
- In environment units the clamp is the training range [min_i, max_i]. The decoded ±1 differ
  from the training statistics by at most 1.1e-16.

| Dim | Lift clamp | Lift binds | Can clamp | Can binds |
|---|---|---|---|---|
| 0 dpos_x | [−1, 1] | no (= controller ±1) | [−1, 1] | no |
| 1 dpos_y | [−0.560000, 0.652000] | both sides | [−1, 1] | no |
| 2 dpos_z | [−1, 1] | no | [−1, 1] | no |
| 3 drot_x | [−0.150659, 0.118636] | both | [−0.556340, 0.729737] | both |
| 4 drot_y | [−0.140395, 0.305093] | both | [−1, 0.450037] | high side only |
| 5 drot_z | [−0.517975, 0.465158] | both | [−1, 0.745340] | high side only |
| 6 gripper | [−1, 1] | no | [−1, 1] | no |

- Where the clamp does not bind, it and the controller clip give bitwise-identical executed
  actions. **The gripper clamp is a no-op** on both tasks. Can's clamp is weaker by construction.
- **What the records keep:**
  - `raw_policy_action` is the network's own output;
  - `executed_action` is the clamped and decoded action;
  - `clip_mask` marks the controller clip or the clamp;
  - the per-episode clamp counts are in `policy_episode.x1_clamp`.

## 5. Execution

- `make x1-run` (`scripts/run_x1_clamp.py`) launches every job in its own canonical container
  (`infra/containers/run_sim_cpu.sh`, image `sha256:1f685638f28c…`), with
  `ROBOFORGE_EVIDENCE_SESSION=13`, ≤ 4 at a time, from this preregistration's commit on a clean
  tree.
- Each job (`scripts/eval_x1_clamp.py`): R = 1, BC is deterministic, first success or
  400 steps, 1 torch thread, the first 3 episodes re-simulated (`verify_record`).
- **Outputs:**
  - records: `runs/x1_clamp/<Session 7 eval ID>__x1-clamp/`;
  - summaries: `reports/session13/x1_clamp/evals/`;
  - events: `logs/session13/x1_clamp/events.jsonl`.
  No existing directory is written into.
- **Each evaluation runs once.** A fault outside the policy (container or host crash, out of
  disk) can stop an evaluation before it writes its summary. In that case the partial record
  directory is renamed to `<id>.incomplete-1` (never deleted), and the evaluation is rerun once
  from scratch with the incident recorded (§14.6). The rename is a manual, logged step: the
  orchestrator only refuses such a directory (`PARTIAL`) and never renames or deletes it. An error raised inside an episode is an
  outcome, `engineering_error`, and is never rerun.
- Isolation is checked after the run against `logs/session13/x1_checks/isolation_before.json`
  (D-107 check 3).

## 6. Analysis (`make x1-analyze`, `scripts/analyze_x1_clamp.py`, once)

**Inputs, through a hash chain.**
- The frozen H4 analysis, `reports/session7/robustness/analysis.json`, SHA-256
  `8cca66a8a8a8cecf0b734355167936e81983c821c625a23456523365409c9451`. The same value is pinned
  in `paper/generated/index.json`.
- The 60 Session 7 H4 summaries and the two configuration records, each read only after its
  SHA-256 equals the value recorded there.
- The 45 X1 summaries. Each must be:
  - clamped, canonical, complete (n = 50), audited, and from one clean commit that contains this
    file;
  - recorded with the frozen checkpoint and suite hashes.

**Bootstrap.** One §9 hierarchical paired bootstrap per task (B = 10,000, seed 20261106,
95 % percentile interval).
- The keys are the frozen `family:set` matrices, in the frozen order, then `<bc>+clamp:set`.
  Every key's seed group is its family.
- The random draws are therefore exactly those of the frozen analysis. The script stops unless
  every frozen family estimate, difference and contrast equals `analysis.json` exactly. A unit
  test confirmed this before any X1 rollout, with stand-in arms.
- Clamped − unclamped is paired by seed and start; diffusion − clamped BC is paired by start.

**Separated / not separated.** A difference is *separated* (+ or −) when its 95 % interval
excludes 0, and *not separated* otherwise. "Not separated" is never read as equality. No
p-values and no multiplicity correction.

**Primary (1).** Lift, `ood_arm_offset`: drop(diffusion) − drop(clamped BC-MLP-chunk), where
drop = `robustness_base` − OOD success. It is compared with the frozen
drop(diffusion) − drop(BC-MLP-chunk) = −40.0 pp [−54.0, −26.0], separated (−).

**Secondary (56).** The supporting and base legs below are among them.

| Task | Clamped − unclamped (per BC family and set) | Clamped drop (per BC family and OOD set) | Diffusion − clamped BC (per family and set) | drop(diffusion) − drop(clamped BC) | drop(BC) − drop(clamped BC) |
|---|---|---|---|---|---|
| lift | 9 | 6 | 9 | 6 (one is the primary) | 6 |
| can | 6 | 3 | 6 | 3 | 3 |

Total 57 = 1 primary + 56 secondary. Also reported descriptively, never compared:
- clamp activity, recomputed from the records: the fraction of executed actions, and of
  entries per dimension, that the clamp changed. The fraction of decoded actions with any
  normalized entry outside [−1, 1] is also reported;
- the §14.5 failure taxonomy of every clamped episode;
- wall time.

**Readings, fixed verbatim now.** The analysis script prints the one that matches the verdict,
followed by "Exploratory, after the freeze; the frozen numbers stand."

*Primary* (P = drop(diffusion) − drop(clamped BC-MLP-chunk), Lift arm offset):
- **separated (−):** "R1. The bound does not account for all of the frozen H4 gap: with the
  same action bound, BC-MLP-chunk still degrades more than diffusion under the arm offset on
  Lift." The supporting leg then states how much of the gap the bound does account for.
- **not separated:** "R2. With the same action bound, the drop difference between diffusion and
  BC-MLP-chunk under the arm offset on Lift is not separated: X1 cannot attribute the frozen H4
  advantage to the generative model rather than to the bound. This is not evidence that the two
  drops are equal."
- **separated (+):** "R3. With the same action bound, BC-MLP-chunk degrades less than diffusion
  under the arm offset on Lift: the bound accounts for more than the frozen H4 gap."

*Supporting leg*, reported with the primary: S = drop(BC-MLP-chunk) − drop(clamped BC-MLP-chunk),
Lift arm offset.
- **separated (+):** "The clamp reduced BC-MLP-chunk's drop under the arm offset on Lift."
- **not separated:** "No separated change in BC-MLP-chunk's drop under the arm offset on Lift."
- **separated (−):** "The clamp increased BC-MLP-chunk's drop under the arm offset on Lift."

*Base leg*, each task, for H2-c: B = clamped − unclamped BC-MLP-chunk on `robustness_base`.
- **separated (−):** "The bound is not free in distribution: it lowers BC-MLP-chunk's success on
  fresh in-distribution starts. H2-c parity stands as measured, with BC unclamped."
- **not separated:** "No separated effect of the bound on BC-MLP-chunk's in-distribution
  success."
- **separated (+):** "The bound raises BC-MLP-chunk's in-distribution success: H2-c was measured
  with BC-MLP-chunk unclamped, a disadvantage the bound removes on these starts."

The H2-c parity itself was measured on the held-out starts and is not revisited.

**Three kinds of negative result, kept apart.**
- *Engineering-incomplete:* a missing summary, fewer than 50 starts, a failed audit, a
  non-canonical platform, or `engineering_error` episodes.
- *Task failure:* the taxonomy label of a failed episode.
- *Statistical uncertainty:* a difference that is not separated.

**Outputs.** `reports/session13/x1_clamp/`:
- `results.json`: every input hash, the bootstrap and the readings;
- `tables.md`: generated;
- `taxonomy.json`.

## 7. Engineering checks (done before this commit; not results)

Checks 1–3 ran on the canonical container from the clean commit `09d1a90` (D-106,
D-107 and the X1 code), with `ROBOFORGE_EVIDENCE_SESSION=13`. They ran 06:46–06:54 UTC,
4 + 2 containers. Evidence is in `logs/session13/x1_checks/`.

1. **Reproduction with the clamp disabled.** PASS on both tasks. The wrapper with the clamp
   disabled reproduced the two hash-checked Session 7 summaries exactly: all 50 starts have
   equal `physics_sha256`, success, first-success step, step count and stop reason, and the
   `verify_record` audit passed.
   - Lift BC-MLP-chunk seed 0 on `ood_arm_offset`: 26/50 then and now.
   - Can BC-MLP seed 0 on `ood_arm_offset`: 13/50 then and now.
   - Files: `reproduce_lift.json`, `reproduce_can.json`.
2. **The clamp on diffusion.** PASS on both tasks. The seed-0 diffusion checkpoint of each task
   (Lift step 100000, Can step 65000) ran on 5 development starts (seeds 13,000,000–13,000,004,
   partition `eval_adhoc`), once through the frozen path and once with the clamp. Every episode
   was identical: `physics_sha256`, raw and executed actions. The clamp changed **0** entries,
   and every raw diffusion action lay in [−1, 1]. Files: `noop_s6_diffusion_unet_…json`,
   `noop_s7_diffusion_unet_…json`.
3. **The clamp on BC.** Descriptive; PASS, since every executed action lay inside the clamp
   bounds. Same development starts, seed-0 BC-MLP-chunk.
   - **Lift.** Most normalized entries outside [−1, 1] were on the gripper, where the clamp
     equals the controller clip. Four episodes are identical with and without the clamp. In the
     fifth, a failure in both runs (seed 13,000,004), the clamp changed dpos_y and drot_y
     entries.
   - **Can.** The entries outside [−1, 1] were on dimensions where the clamp cannot bind
     (dpos_x, dpos_y, gripper). All 5 episodes are identical.
   - **Consequence.** Counting entries outside [−1, 1] overstates what the clamp does. The
     analysis therefore also reports, recomputed from the records, the fraction of executed
     actions the clamp **changed** (§6). It also checks that every recorded executed action
     equals the clamped decode.
4. **The analysis's record checks on real records.** PASS. On the 100 clamp-disabled records
   of check 1, the record checks of `scripts/analyze_x1_clamp.py` held from the host:
   - `executed_action` (float64) equals the frozen decode of `raw_policy_action` bitwise (the
     clamped path adds one `np.clip`);
   - `records_dir` is repo-relative;
   - the host taxonomy's success counts equal the summaries' (26 and 13).
   File: `analysis_record_checks.log`.
5. **Isolation.** `isolation_before.json` was taken before any X1 container job: 1,246
   top-level entries in `runs/` and content digests of the 60 Session 7 H4 record directories.
   It is compared with a snapshot taken after the 45 evaluations.

## 8. Biases and limits, stated now

- **Post hoc design.** X1 was designed after the Session 7 robustness results and the audit's
  scan of out-of-range actions were known. That is why it is labelled exploratory.
- **Selection.** The BC checkpoints were selected unclamped (§7, validation), which can only
  favour the unclamped arm.
- **Reused starts.** The robustness starts were already used by Session 7 and, for Lift
  `robustness_base` and `ood_cube_shift`, by Level 2.
- **Narrow scope.** One bound (the training range), three seeds and two perturbations. A
  separated or not-separated outcome says nothing about other bounds or other perturbations.
- **Power, an estimate.** With 3 seeds × 50 starts, the frozen drop contrasts have 95 % intervals
  about ±14 pp wide. Only large changes can separate.
