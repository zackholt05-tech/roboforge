# X3 preregistration: the Level 1 Can comparison with 6 training seeds, on fresh starts

**Label of every result: separate replication.** This file was written and committed on
2026-11-10 (Session 14), after the X3 held-out suite was generated and committed, and before
the first X3 training run. The commit order shows it. It governs X3 only, under DECISIONS
D-105, D-108, D-109, D-110 and D-111. Level 1, the RGB study, Level 2 and X1 stay frozen. No
frozen file, freeze, result, label or executed outcome changes. X3 is reported beside the
frozen Level 1 result, never instead of it, and the two are never pooled.

## 1. Question

With 3 training seeds per family, the seed component of the preregistered hierarchical
bootstrap rests on 2 degrees of freedom. Two frozen Can separations do not hold under a
seed-level t interval computed after the audit (KI-053, D-095):

| Contrast (Can, frozen, 3 seeds × 50 starts) | Difference | Bootstrap 95 % | Bootstrap verdict | Seed-level Welch 95 % (df) |
|---|---|---|---|---|
| H2 diffusion − BC-MLP | +18.7 pp | [+4.7, +32.7] | separated (+) | [−4.5, +41.8] (2.35) |
| H2 diffusion − BC-RNN | −0.7 pp | [−8.0, +6.7] | not separated | [−9.0, +7.6] (3.74) |
| H2-c diffusion − BC-MLP-chunk | −0.7 pp | [−8.7, +7.3] | not separated | [−9.0, +7.6] (3.74) |
| H3 BC-RNN − BC-MLP | +19.3 pp | [+5.3, +34.0] | separated (+) | [−2.8, +41.5] (2.59) |

The analysis recomputes every frozen row from `reports/session7/results/results.json` (pinned
below) and prints it beside the replication.

X3 asks: **do these declared Can contrasts hold with 6 seeds per family, on 100 fresh held-out
starts?**

## 2. Configurations and training (new seeds only; nothing is retuned)

Each family trains its frozen chosen Can configuration (`docs/FINAL_TEST_FREEZE.md` §3) with
seeds **3, 4 and 5** (owner, D-109).

| Family | Frozen chosen trial | Base config (X3) | SHA-256 of the file |
|---|---|---|---|
| diffusion_unet | 2, `optim.schedule=cosine` | `configs/extensions/x3/diffusion_unet_cos_can.yaml` | `26a64d295c59808221813b76ab72eb939a1d4f73163cc925649cd4f1801792f5` |
| bc_mlp | 2, `optim.schedule=cosine` | `configs/extensions/x3/bc_mlp_cos_can.yaml` | `e1b1ffa6a38fd143560d9109bd8e17a55d5355b8a36595091eb07e87c3fb47e0` |
| bc_mlp_chunk | 4, `optim.weight_decay=0.1` | `configs/extensions/x3/bc_mlp_chunk_wd1e1_can.yaml` | `deaf5ae9cf57f6ee6634a57f49b5d76e905cc0de76e5b5e82898265ec7fd9ef7` |
| bc_rnn | 2, `optim.schedule=cosine` | `configs/extensions/x3/bc_rnn_cos_can.yaml` | `10f5fe742fde396d2216c32daa46c4226a7daecdf27d2086eabbe0095d008e30` |

- Each file is the frozen base config with the chosen override written in, nothing else.
- With seeds 0–2 these files resolve to the frozen config SHA-256 and run ID of all 12 frozen
  Can runs. With seeds 3–5 they differ from those runs in the seed alone
  (`tests/unit/test_x3_configs.py`).
- **Plan:** `configs/experiments/x3_can.yaml` (SHA-256 `bdb779a3dd116f330a44d06cc044498a8bba2e4d03a7f7460b09ce30413b5e9f`), prefix `s14`, tuning
  seed 3, replicate seeds 4 and 5, `replicates_with_tuning: true`. The one-trial "tuning"
  record chooses trivially; it is not a tuning step.
- **Budget:** 100,000 steps, batch 100, checkpoints every 5,000 (K = 20), 1 torch thread, as in
  Session 7.
- **Where:** the macOS host (KI-061), through `scripts/run_suite.py` (`make x3-train`), from a
  clean tree. At most 6 training runs and 4 containers at a time (host-wide slots); no launch
  below 4 GiB free.
- **Selection:** every checkpoint runs all 50 Can validation starts on the canonical container,
  and the RESEARCH_PROTOCOL §7 rule selects one per run (`scripts/select_checkpoint.py`):
  highest validation success count, then lower mean first-success step, then the earlier
  checkpoint. **Selection uses validation starts only.**

The 12 runs, resolved from the plan before any of them trained (`make x3-plan` printed the
seed-3 runs; its record is committed as `logs/session14/x3/plan_resolved_dryrun.json`;
`tests/unit/test_x3_configs.py` pins all 12):

| Family | Seed 3 | Seed 4 | Seed 5 |
|---|---|---|---|
| diffusion_unet | `s14_diffusion_unet_cos_can_seed3_a6116493` | `s14_diffusion_unet_cos_can_seed4_de04fdae` | `s14_diffusion_unet_cos_can_seed5_9119b2ec` |
| bc_mlp | `s14_bc_mlp_cos_can_seed3_04223e1b` | `s14_bc_mlp_cos_can_seed4_7c8e5c7b` | `s14_bc_mlp_cos_can_seed5_c001f43c` |
| bc_mlp_chunk | `s14_bc_mlp_chunk_wd1e1_can_seed3_2fec9f65` | `s14_bc_mlp_chunk_wd1e1_can_seed4_26a20b25` | `s14_bc_mlp_chunk_wd1e1_can_seed5_4b41dda8` |
| bc_rnn | `s14_bc_rnn_cos_can_seed3_24bd8478` | `s14_bc_rnn_cos_can_seed4_bc40d865` | `s14_bc_rnn_cos_can_seed5_ad65ce36` |

A training run that fails is reported and not replaced. If a family then has fewer than 6
seeds, the analysis uses the seeds it has and says so; no seed is added.

## 3. The X3 held-out suite

- **`x3_heldout`, Can, 100 starts**, seeds 14,000,000 + i (i = 0…99), reserved block
  [14,000,000, 15,000,000) (`configs/extensions/x3.yaml`, SHA-256 `7a863c954ff928d3f6dfee7a8b705d01cc3443fc8ec6638903522e4e17f34d15`).
- Drawn as the frozen suites are (`roboforge.sim.snapshot.generate`). Each start restores
  exactly and does not succeed at t = 0.
- **Files:** `data/eval_states/can/x3_heldout.json` and `x3_heldout.data.jsonl.xz`, frozen format
  (`roboforge.eval_suite/1`). **Manifest content SHA-256: `9c199e19f69708872a01665cf1367d2e34741388c2f546ef96d3ffc900a2b34d`.**
- Generated on the canonical container from clean `e0b8a4e`. Regenerated in memory and
  equal bitwise in the container, and within 1e-12 with identical XML on the host
  (`logs/session14/x3/`). Committed in `81afa3b`, before any rollout on it.
- **Disjoint** from every reserved seed block of the frozen studies and from X1's development
  seeds, and by state hash from every committed Can suite (`tests/research_integrity/test_x3_scope.py`).
- **Partition label `eval_adhoc`.** The frozen vocabulary has no X3 label, and the frozen code
  files these seeds as ad hoc (D-111). Every selection here is by suite name and content hash,
  never by partition.

## 4. Evaluations (24, each once, on all 100 X3 starts)

- **Frozen checkpoints (12):** the Can entries of the committed `docs/FINAL_TEST_FREEZE.md`
  (git blob `bfdb4b77d478bdb1908b34b96e1dad70eb88147c`), 4 families × seeds 0–2, with their
  checkpoint SHA-256 (freeze §3). The X3 suite is a new suite, not a revisit of the frozen
  held-out suite. No frozen held-out start is stepped.
- **New checkpoints (12):** the §7 selection of each run of §2. Each selection record
  (`reports/session14/selection/<run_id>.json`) is committed unmodified before the first X3
  rollout. Its chosen SHA-256 must equal the training manifest's and the file's.
- **Settings, as frozen:** canonical container
  (`linux-aarch64-container-1f685638f28c`), R = 1, policy seed 0. Diffusion noise comes from
  `SeedSequence([0, sha256(state_id)[:8], 0])`. Episodes end at the first success or at 400
  steps, reset only from the stored snapshots, and the first 3 are re-simulated
  (`verify_record`).
- **No random baseline:** H1 is not replicated (every frozen H1 difference is separated by
  ≥ 76.7 pp).
- **Code.** The evaluator, orchestrator, engineering check and analysis
  (`src/roboforge/replication/x3.py` and `x3_jobs.py`, `scripts/eval_x3.py`,
  `scripts/run_x3.py`, `scripts/check_x3_engineering.py`, `scripts/analyze_x3.py`, their
  tests and supporting modules) are written after this file, while the X3 runs train, and
  outside the repository.
  - They are committed and tested before the first X3 rollout, with a dated ledger row giving
    each file's SHA-256.
  - They implement this file exactly. Anything they cannot implement as written is recorded
    there as a dated deviation before the first X3 rollout.
  - The orchestrator refuses unless this file is committed unmodified and names the suite's
    content hash, every new selection record is committed unmodified, and the tree is clean. It
    launches at most 4 containers.
- **Engineering check (not a result), before the first X3 rollout:** the evaluator in its
  check mode (validation suite only, records under `runs/x3_checks/`) must reproduce the
  committed Session 7 validation summaries of BC-MLP seed 0 at step 55,000 and diffusion seed 0
  at step 65,000 exactly: equal per-episode `physics_sha256`, success and first-success step on
  all 50 starts.
- **No reruns,** except one engineering rerun under the frozen §14.6 rule (a fault outside the
  policy before the summary is written; the incident recorded).
- **Outputs:** records `runs/x3/<eval_id>/`, summaries `reports/session14/x3/evals/`
  (schema `roboforge.x3_eval/1`), logs `logs/session14/x3/`. Nothing is overwritten.

## 5. Analysis (`make x3-analyze`, `scripts/analyze_x3.py`, once)

**Inputs, through a hash chain.**
- The 24 X3 summaries. Each must be canonical, complete (n = 100), audited, from one clean
  commit that contains this file, with the X3 suite's content hash and the expected checkpoint
  SHA-256.
- The frozen result, `reports/session7/results/results.json`, SHA-256
  `b8b097c38765c588711615917e3cfd5be6847e63083ddebe92f0f1488672408c`. Its `per_seed` values
  give the frozen seed-level intervals.

**Outcome matrices.** Per family, 6 seeds (rows in seed order 0–5) × 100 starts (columns in
the manifest's state order). The entry is 1 for success before the horizon and 0 otherwise.
Episodes labelled `engineering_error` count as failures and are reported separately.

**Primary interval.** The frozen §9 hierarchical paired bootstrap,
`roboforge.eval.statistics.hierarchical_paired_bootstrap`, unchanged:
- B = 10,000, seed 20261106, 95 % percentile intervals (numpy linear interpolation);
- draw order: states, bc_mlp, bc_rnn, bc_mlp_chunk, diffusion_unet;
- states resampled once per replicate and shared by every family (pairing by start); each
  family's seeds resampled independently.

**Seed-level interval (co-primary for robustness).** The D-095 Welch t 95 % interval on the
6 + 6 per-seed success rates (`roboforge.paper.sensitivity.welch`, Welch–Satterthwaite df). If
both families have zero seed variance it is "undefined (no seed variance)", never a zero-width
interval.

**Declared contrasts (Can).**

| ID | Contrast (A − B) |
|---|---|
| H2 | diffusion_unet − bc_mlp |
| H2 | diffusion_unet − bc_rnn |
| H2c | diffusion_unet − bc_mlp_chunk |
| H3 | bc_rnn − bc_mlp |

No p-values and no multiplicity correction; every interval is reported.

**Verdict per contrast.** An interval *separates* (+ or −) when it excludes 0.
- **robust (±):** both intervals separate, on the same side;
- **bootstrap only (±):** the bootstrap separates and the seed-level interval does not;
- **seed-level only (±):** the seed-level interval separates and the bootstrap does not;
- **not separated:** neither interval separates. This is not evidence of equality;
- **conflicting:** the two intervals separate on opposite sides. This is not expected, since
  both are centred near the same estimate; it gives no verdict.

**Relation to the frozen result**, from the frozen bootstrap verdict:
- frozen separated (s):
  - replication robust (s) → *holds, robust*;
  - bootstrap only (s) → *holds, bootstrap only*;
  - bootstrap not separated → *not reproduced*;
  - bootstrap separated in the other direction → *reversed*;
- frozen not separated:
  - replication not separated → *unchanged, not separated*;
  - any separation → *new separation*;
- a conflicting replication verdict → *no relation*.

**Secondary (declared now, no readings).**
- The censored episode length (first-success step if successful, else 400) for the 4
  contrasts, with the same bootstrap and draws.
- The 4 contrasts on the frozen seeds 0–2 alone (the same checkpoints as the frozen test, on
  the new starts) and on the new seeds 3–5 alone (new checkpoints, new starts). Each uses the
  same bootstrap with its own 3-seed matrices, and the seed-level interval. These tell the start
  draw from the seed count.
- Descriptive only: per-seed success rates; family means with their bootstrap intervals; the
  validation selections of the new runs (development numbers); the frozen failure taxonomy
  (`configs/eval/failure_taxonomy.yaml`, RESEARCH_PROTOCOL §14.5) applied to every X3 episode;
  wall time.

**Engineering incompleteness (frozen rule, `docs/FINAL_TEST_FREEZE.md` §4).** An evaluation
with no summary, fewer than 100 starts, a failed audit or a non-canonical platform is rerun
once, only for a fault outside the policy before its summary. If it is still incomplete, its
family is ENGINEERING-INCOMPLETE. Its complete evaluations are listed, it is left out of the
bootstrap (the draw order skips it), and its contrasts are reported as not computed.

**Readings, fixed verbatim now.** For each contrast the script prints the verdict reading, the
relation reading and the closing line. `{name}` is the contrast (for example "H2 diffusion −
BC-MLP"), `{sign}` is + or −, and `{method}` is "under both interval methods", "under the
bootstrap only" or "under the seed-level interval only".

*Verdict:*
- **robust:** "{name}: separated ({sign}) under both the bootstrap and the seed-level
  interval, with 6 seeds on 100 fresh starts. In this replication the difference does not rest
  on 3 seeds or on the 50 frozen starts."
- **bootstrap only:** "{name}: separated ({sign}) under the bootstrap but not under the
  seed-level interval. With 6 seeds the verdict still depends on the interval method."
- **seed-level only:** "{name}: separated ({sign}) under the seed-level interval but not under
  the bootstrap. The verdict depends on the interval method."
- **not separated:** "{name}: not separated under either interval, with 6 seeds on 100 fresh
  starts. This replication gives no evidence of a difference; that is not evidence of
  equality."
- **conflicting:** "{name}: the bootstrap and the seed-level interval separate on opposite
  sides; no verdict."

*Relation:*
- **holds, robust:** "The frozen separation holds in this replication, now under both interval
  methods."
- **holds, bootstrap only:** "The frozen separation holds in this replication, still under the
  bootstrap only."
- **not reproduced:** "The frozen bootstrap separation is not reproduced in this replication."
- **reversed:** "This replication separates in the opposite direction to the frozen result."
- **unchanged, not separated:** "As in the frozen result, not separated."
- **new separation:** "Unlike the frozen result, this replication separates ({sign}) {method}."
- **no relation:** "No relation to the frozen result can be stated."

*Closing line:* "Separate replication; the frozen result stands as reported."

A unit test keeps these strings equal to the script's constants.

**Outputs.** `reports/session14/x3/`: `results.json` (every input hash, both intervals, the
verdicts and the readings), `tables.md` (generated; the frozen and replication columns side by
side), `taxonomy.json`.

## 6. Expected precision (estimates, stated now)

From the frozen per-seed spread (SD 10.3 pp BC-MLP, 4.0 BC-RNN, 4.0 BC-MLP-chunk, 3.1
diffusion, at 50 starts), the seed-level 95 % half-widths with 6 seeds per family would be
about ±10.8 pp for H2 diffusion − BC-MLP and H3, and about ±4.6 pp for H2-c (estimates). If
the true differences were near the frozen +19 pp, both would be expected to separate under both
methods; this is an estimate, not a power calculation. With 100 starts per seed the binomial
share of the per-seed spread is halved, so these half-widths are conservative.

## 7. Biases and limits, stated now

- **Selection optimism** stays in validation (max over 20 checkpoints on the same 50 starts).
  The X3 suite is unbiased for the selected checkpoints, as the frozen held-out suite was.
- **The frozen checkpoints have already been tested once** (on the frozen suite). The X3
  starts are new, so their outcomes on X3 are not a revisit. The frozen verdicts were known
  when X3 was designed; that is why X3 exists. Nothing in X3 is tuned to them.
- **The seed-level interval treats per-seed rates as the units and ignores pairing by start.**
  It checks the seed dimension only.
- **R = 1:** KI-028's start/seed confound for diffusion remains.
- **Scope:** Can, privileged low-dimensional state, one simulator, one dataset (PH), 6 seeds.
  It covers neither Lift nor the robustness sets (H4).
