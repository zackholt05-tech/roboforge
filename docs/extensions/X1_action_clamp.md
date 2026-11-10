# X1: evaluation-only action-clamp test (KI-052)

**Session:** 13. **Label of every result:** "exploratory, after the freeze". **Training:** none.
**Paid compute:** none.

## 1. Why

The diffusion sampler clamps its predicted clean action chunk to [−1, 1] in min–max normalized
space (`_Sampler._x0`, `clip_sample=True`, `src/roboforge/models/noise_schedule.py`), so every
diffusion action lies inside each dimension's training range. The BC heads are clipped only to
the controller limits (`ActionCodec.decode`); for example, Lift's drot_x training range is
[−0.151, 0.119] against ±0.5 rad for BC. The audit (A-6) found that BC policies execute
out-of-range motion actions on the Lift OOD sets (BC-MLP-chunk 0.53–0.54, BC-MLP 0.64–0.71 per
the audit's scan), almost all in failed episodes, while diffusion executes none.

Two frozen comparisons therefore compare the generative head *together with* this bound:
- **H4 (robustness):** under the initial arm offset on Lift, diffusion degraded least
  (diffusion − BC-MLP-chunk drop: −40.0 pp [−54.0, −26.0], separated);
- **H2-c (held-out parity):** diffusion and BC-MLP-chunk are not separated on either task.

X1 asks: **if the BC heads get the same bound, does the diffusion advantage under perturbation
remain?** D-096 named this test and deferred it to its own decision.

## 2. Design (to be fixed by a Session 13 decision and the preregistration)

- **Checkpoints:** the 9 BC checkpoints per task already selected and frozen (BC-MLP,
  BC-RNN, BC-MLP-chunk × seeds 0–2), as listed in `reports/session6/lift_test_configuration.json`
  and `reports/session7/can_test_configuration.json`, verified by SHA-256.
- **Suites:** the Session 7 robustness sets (RESEARCH_PROTOCOL §14.1–14.2,
  `configs/eval/robustness.yaml`): Lift `robustness_base`, `ood_cube_shift`, `ood_arm_offset`;
  Can `robustness_base`, `ood_arm_offset`; 50 starts each, hashes in their manifests under
  `data/eval_states/`. `robustness_base` is in-distribution and disjoint from the held-out
  starts, so it gives evidence relevant to H2-c without revisiting the test. **Total: 45
  evaluations** (Lift 9 × 3, Can 9 × 2).
- **The clamp:** after the policy produces an action in the environment's action space, clip
  each motion dimension to `[min_i, max_i]`, the per-dimension training range behind the
  diffusion policy's min–max normalization. Session 13 confirms from the code and the
  checkpoints which statistics the diffusion runs used (`data/normalization/*_ph_lowdim_v1.json`
  or the checkpoint's own copy) and that they come from the same training split as the BC runs.
  The gripper dimension is clipped the same way (Session 13 records whether that is a no-op).
- **Comparisons (paired by start, the frozen bootstrap method and seed of §14.3 unless the
  preregistration states otherwise):**
  1. within each BC family and set: clamped − unclamped (the Session 7 summary of the same
     checkpoint and set);
  2. across families, per set: diffusion − clamped BC-MLP-chunk, and diffusion − clamped BC-RNN
     and − clamped BC-MLP, compared with the frozen unclamped contrasts.
- **Reading the outcome (to be fixed verbatim in the preregistration):** if clamping BC closes
  the Lift arm-offset gap to diffusion, the frozen H4 advantage is attributable to the bound,
  not to the generative model. If the gap stays separated, the bound does not explain it. If
  clamping hurts BC on `robustness_base`, the bound is not free and H2-c parity stands as
  measured. Either way the frozen numbers stand; X1 adds an attribution.

## 3. Hard constraints found while planning

1. **No frozen file may change.** `docs/FINAL_TEST_FREEZE.md` pins the SHA-256 of nearly every
   file under `src/roboforge/` (including `eval/learned.py`, `eval/rollout.py`, the models and
   `ActionCodec`) and `scripts/eval_heldout.py`, `scripts/run_heldout.py`,
   `scripts/analyze_heldout.py`. The existing inference-variant mechanism
   (`roboforge.eval.learned.VARIANTS`) applies to diffusion checkpoints only and lives in a
   frozen file. The clamp must be a wrapper in **new** files (for example
   `src/roboforge/exploratory/clamp.py` and `scripts/run_x1_clamp.py`); Session 13 lists every
   pinned file before writing code, and `make freeze-check` passes before and after.
2. **The string `final_test`** must not appear in `src/` or `scripts/*.py`
   (`tests/research_integrity/`).
3. **`scripts/analyze_robustness.py` is never rerun** and refuses to overwrite its output. X1
   needs a new analysis script that reads the Session 7 summaries
   (`reports/session7/robustness/evals/`) only after their SHA-256 equals the value recorded in
   `reports/session7/robustness/analysis.json` (the chained-hash pattern of D-096).
4. **Output locations.** Session 7 evaluation records live in `runs/<train_run>__<suite>_step…`
   directories. Clamped evaluations must write to new, distinctly named directories and must
   refuse to write into an existing one. Session 13 checks the writer before the first rollout.
5. **Comparability.** The new clamped evaluations are compared with summaries recorded in
   Session 7 on the same image. Before any reported rollout, the wrapper with the clamp disabled
   must reproduce at least one recorded Session 7 BC robustness summary per task exactly
   (same successes, same per-episode outcomes), and the clamp applied to a diffusion checkpoint
   must change no action on a few development starts (seeds outside every reserved block).

## 4. Work plan

1. **Decisions first:** Makefile `PREFIX ?= s13`, evidence session 13 (as D-097 did for 12);
   the X1 design of §2 as a dated decision.
2. **Implement** the wrapper, the runner and the analysis script in new files, with unit
   tests (clamp arithmetic, refusal to overwrite, hash checks) and a research-integrity test
   that the runner refuses held-out suites.
3. **Engineering checks** of §3.5 on the container, recorded in `logs/session13/`.
4. **Preregistration:** commit `docs/extensions/X1_PREREGISTRATION.md` (checkpoints with
   SHA-256, suites with content hashes, clamp ranges per task, contrasts, bootstrap method and
   seed, the outcome readings of §2) before the first reported rollout.
5. **Run** the 45 evaluations once, from the preregistration commit, clean tree,
   ≤ 4 containers.
6. **Analyze** once; write `reports/session13/x1_clamp/` (results JSON, tables, the analysis
   record with every input hash).
7. **Record:** KI-052 updated with the outcome; ledger rows; STATUS and HANDOFF.

## 5. Cost (estimate)

The same 45 BC evaluations without a clamp recorded **3.12 container-hours** in Session 7
(sum of `wall_s` of the matching summaries in `reports/session7/robustness/evals/`: Lift 1.41,
Can 1.71). With ≤ 4 containers at once that is about **0.8 h of wall-clock time** for the
reported runs, plus the engineering checks. Disk: the 75 Session 7 robustness evaluation
directories use 0.79 GB (about 11 MB each), so about 0.5 GB (estimate). $0.

## 6. Acceptance criteria (Session 13 STOP)

- `make freeze-check` unchanged before and after; no frozen file, freeze or result changed.
- Engineering checks of §3.5 passed and logged.
- Preregistration committed before the first reported rollout (the commit order shows it).
- 45 summaries, each canonical, n = 50, 0 errors, `verify_record` passed, git clean.
- Analysis written once; KI-052 updated; every number in the docs comes from the analysis
  record, labelled "exploratory, after the freeze".
- Lint, full suite alone, `tests/research_integrity`, `make release-check` pass.

## 7. Out of scope for X1

Training; any held-out-suite rollout (the owner may choose a held-out revisit later, labelled
as an exploratory revisit of the test, with its own decision); changing the published paper,
README or Observatory (D-105).
