# Independent critical audit (Session 11)

**Status:** complete (audits A–D).
**Rule (Session 11 prompt, step 4):** every finding gets an engineering correction or a
transparent limitation. No frozen file was edited. No new evaluation, training or paid compute
was run for this audit; every check reads committed files or the gitignored `runs/` records,
read-only.

## Method

Four separate passes, each run by a fresh agent that read the code and the frozen files, not
the handoff's conclusions. Each agent read `docs/KNOWN_ISSUES.md` only after forming its
findings, to mark what was already documented.

| Pass | Scope |
|---|---|
| A | Simulator and controller alignment, train/rollout observation equality, split leakage, evaluation-set hygiene, privilege and like-for-like comparisons |
| B | Statistics: bootstrap and pairing recomputation, interval validity, seed sensitivity, multiplicity, censored length, saturation, dataset reuse |
| C | Paid compute reconciliation, declared vs reported analyses, null results, the timeline of freezes against the runs they govern |
| D | Every claim of the paper (`paper/`) against the original frozen results and MASTER_SPEC §11.3 |

Severity: HIGH = a reported conclusion could be wrong or misattributed; MEDIUM = a reader
could be misled without the disclosure; LOW = a record or engineering defect with no effect
on a result; INFO = checked and found correct.

**Disposition key:** *Corrected* = an engineering or reporting fix made in Session 11, with its
location. *Limitation* = disclosed in the paper (section given) and here; no data changed.

## Summary of dispositions

| ID | Sev. | Finding (short) | Disposition |
|---|---|---|---|
| A-6 | HIGH | Diffusion actions are bounded to the training range; BC heads are not | Limitation (paper §4, §6, §7, §9); KI-052; the evaluation-only test that would resolve it stays open and needs its own decision before any rollout |
| B-1 | HIGH | Can H2 and H3 separations depend on the interval method | Corrected by reporting: seed-level sensitivity analysis, not preregistered (D-095; paper §6, Tables 17–18); KI-053 |
| B-2 | HIGH | Several H4 separations depend on the method | As B-1 (paper §7, Table 18) |
| B-3 | MEDIUM | Two verdicts change with the bootstrap seed | Limitation (paper §9, "verdicts that depend on the method"); KI-053 |
| B-4 | MEDIUM | Degenerate and boundary intervals | Limitation with a generated count (paper §9); KI-053 |
| B-5 | MEDIUM | Censored length is mechanically tied to success | Corrected by reporting: exact success/speed decomposition (D-095; paper §6) |
| B-6 | MEDIUM | Unadjusted multiplicity | Limitation with counts (paper §5, §7, §9) |
| B-7 | MEDIUM | Saturation restatements (difference of drops = −H5) | Limitation (paper §8.2) |
| B-8, A B-5, C-15 | MEDIUM | The same Lift starts served three sequentially designed studies; H5's set chosen after earlier outcomes | Limitation (paper §8.2, §9) |
| A B-6 | MEDIUM | RGB/PC selection under saturation carries no information | Limitation (paper §8.1, §9) |
| C-4, C-5 | MEDIUM | Cross-study comparisons are not like-for-like; H5 compares packages | Limitation (paper §8, §8.2) |
| C-9 | MEDIUM | Declared secondary metrics never aggregated | Corrected by reporting: post-hoc table from hash-chained summaries (D-096; paper Table 4); KI-054 |
| C-14 | MEDIUM | Unequal early look at two tuning configurations before the Lift search space was committed | Limitation (paper §9) |
| C-19 | MEDIUM | Robustness, RGB and Level 2 analysis scripts were not hash-frozen and were edited after data | Limitation (paper §9) |
| C-5 (spend) | MEDIUM | A-01 launch count exceeded before its amendment | Limitation (paper §9); recorded here and in KI-057 |
| A-7 | LOW | Our success predicate can fire one step before robosuite's stale one | Limitation (paper §9) |
| A B-4 | LOW | After the Level 1 freeze the base lock is open to any caller | Existing KI-039/040/051; no change (all held-out runs are complete) |
| C-3 (A) | LOW | Point-cloud box uses the known table height | Limitation (paper §9) |
| A-9 | LOW | Can open-loop replay diverges at contacts | Existing KI-004; no change |
| B-9, B-10 | LOW | Level 1 informed Level 2's design; ablations share the primary arm's evaluations | Limitation (paper §7, §8.2) |
| B-12 | LOW | `scripts/analyze_robustness.py` had no run-once guard | Corrected: guard plus test (KI-055) |
| B-13 | LOW | `scripts/analyze_level2.py` hashed a report before checking it exists | Corrected: existence first (KI-055) |
| C-6 | LOW | One RGB run's hours omit a preempted attempt | Corrected by reporting: footnote b of the paper's cost table (KI-056) |
| C-8 | INFO | A-04's ms/step stop rule was checked by hand for seed 0's first segment | Recorded here |
| C-10 | LOW | KI-028's confound was promised but never reported | Limitation (paper §5, §9) |
| C-11 | LOW | DiT, GMM-BC and Square not run, reason not recorded | Limitation with reason (paper §9; D-096) |
| C-13 | LOW | Undeclared context rows; H4 pair set not stated in the protocol | Limitation (paper §9) |
| C-18 | LOW | One RGB run's validation summaries came from a dirty tree | Limitation (paper §9) |
| C-20(e) | LOW | D-086 prose time and E9-11 order are wrong | Corrected: annotated in DECISIONS D-086 and the ledger E9-11 |
| C-21 | LOW | The freeze calls a content hash "SHA-256" | Limitation (KI-057; the freeze is frozen) |
| all INFO | INFO | Alignment, pairing, splits, spend, timeline, frozen files | Checked and correct (below) |

---

## A. Simulator, splits, privilege

**A-1 INFO.** Controller, rate, horizon and termination match the dataset `env_args`:
OSC_POSE delta, world frame, input ±1, output ±0.05 m / ±0.5 rad, 20 Hz, horizon 400, first
success. Asserted at pipeline build (`src/roboforge/data/pipeline.py:120-132`) and at every
evaluation (`src/roboforge/sim/env_adapter.py:102-105`).

**A-2 INFO.** Training and rollout observations are the same function of the same-step state.
Processed `obs[t]` equals robomimic `obs[t]` exactly for every `relation: equal` feature;
recomputed rollout features from recorded post-step states match the recorded `obs` to
≤ 1e-15 (3 Lift + 3 Can validation episodes). Float32 casting and window indices are shared.
KI-003 and KI-015 cover the intended Can feature correction.

**A-3, A-4, A-5, A-8, A-10 INFO.** Quaternion convention shared by data and evaluation;
evaluation starts drawn from the demonstrations' start distribution (nearest-demo cube
distance: Lift median 1.8–2.2 mm, Can 8.4–8.9 mm); gripper sign and decoding correct;
demonstration and evaluation XMLs compile to the same physics; the normalized size of OOD
inputs is the same for every family.

**A-6 HIGH (attribution), the measured differences stand.** The diffusion sampler clamps the
predicted x̂₀ to [−1, 1] in min–max space (`src/roboforge/models/noise_schedule.py`, `_x0`,
`clip_sample=True`), and the final DDIM step returns it, so every diffusion action lies within
each dimension's training range. BC outputs are bounded only by the controller limits
(`src/roboforge/data/normalization.py:120-124`). Example: Lift drot_x training range
[−0.151, 0.119] vs ±0.5 rad per step for BC. Fraction of executed motion actions outside the
training range (scan of `runs/` arrays): diffusion 0.000 on every set; Lift OOD BC-MLP-chunk
0.53–0.54, BC-MLP 0.64–0.71; in failed episodes 0.61–0.74 vs 0.003–0.03 in successes. Counter-
evidence (so correlational): 0.000 in the first 20 steps of failed cube-shift episodes; BC-MLP
on Lift base is 12.5 % out of range yet 98 % successful; BC-RNN is ≤ 3.5 % out of range and
still drops to 50 % under arm offset. **Consequence:** H2-c and the H4 between-family gaps
compare the generative head *and* this bound. **Disposition:** limitation (paper §4, §6, §7,
§9; KI-052). An evaluation-only exploratory test (BC checkpoints with outputs clamped to the
training range on the robustness sets; optionally diffusion with `clip_sample=False`) would
resolve it without retraining; it is a new post-freeze evaluation, so it needs its own
recorded decision, a clean tree, `ROBOFORGE_EVIDENCE_SESSION`, and the label "exploratory,
after the freeze". It was not run; it stays open under KI-052.

**A-7 LOW.** The success predicate on current kinematics (D-024/D-026) fires one control step
before robosuite's stale predicate in 2.6–4.2 % of episodes per suite (e.g. Lift held-out
43/1100, Can 17/650). Timing only; success unchanged. Limitation (paper §9).

**A-9 LOW.** Can open-loop replay diverges at contacts (median max object error 5–6.5 cm).
Existing KI-004.

## A. Split leakage and evaluation-set hygiene

**B-1, B-2, B-3 (pass A) INFO.** Train/valid 180/20, no overlap; normalizers refit on the train
demos reproduce the committed statistics bitwise (a fit on all 200 does not); windows are
episode-bounded and causal. Validation demonstrations select nothing (only closed-loop
validation rollouts do). Suites are disjoint by state hash and seed; a census of 64,316
records finds only named-suite starts; held-out counts reconcile (Lift 1100 = 13×50 + 6×50 +
3×50; Can 650 = 13×50); every held-out manifest was created after its gate.

**B-4 (pass A) LOW.** Once `docs/FINAL_TEST_FREEZE.md` exists, `eval_states.load_suite` admits
any caller; later studies were protected by their own gates and a name-scan test that cannot
see access through the `LOCKED_SUITE` constant. Existing KI-039, KI-040, KI-051. No change: all
held-out runs are complete and no further held-out use is planned.

**B-5 (pass A) MEDIUM.** The same 50 Lift held-out starts and the robustness base and
cube-shift starts were reused by three studies designed in sequence, and H5's primary set was
chosen after the Level 1 robustness and RGB held-out outcomes were known
(RESEARCH_PROTOCOL §13 amendment of 2026-11-09; LEVEL2_GO_NO_GO). No model, checkpoint or
configuration was selected on them; placement started 4 min after the point-cloud selections
were committed. Limitation (paper §8.2, §9).

**B-6 (pass A) MEDIUM.** In the RGB and point-cloud studies 7–12 of 12 checkpoints per run tie
at 50/50 on validation and selection is decided by first-success-step gaps of 0.1–0.3 steps.
Limitation (paper §8.1, §9); same class as KI-034.

## A. Privilege and like-for-like

**C-1, C-2 (pass A) INFO.** Level 1 privileged inputs are flagged and labelled; RGB inputs
carry no privileged state (renders plus 9-D proprioception; live and restored renders bitwise
equal).

**C-3 (pass A) LOW.** The point cloud is sensor-like but uses noise-free depth, exact
calibration and a crop box whose lower edge (z = 0.805 m) removes the table at its known
height (0.8 m). Limitation (paper §9).

**C-4 (pass A) MEDIUM.** Cross-study numbers on the same starts are not like-for-like (samples
10 M vs 3.84 M, 20 vs 12 checkpoints, 4 vs 0 tuning trials, CPU fp32 vs L4 with TF32, no
augmentation vs random crops, privileged pose vs images). The paper never compares them
(paper §8 opening paragraph).

**C-5 (pass A) MEDIUM.** H5 compares packages: encoders 22,394,176 vs 357,184 parameters,
random crops vs none, world-frame coordinates. Limitation (paper §8.2).

---

## B. Statistics

Recomputations were independent of the analysis code (auditor's own scripts, scratch only).

**INFO (verified).** Pairing is correct: 650 Level 1 records, 3,750 robustness records and the
Level 2 records pair by start index exactly as documented; diffusion noise seeds follow
`roboforge.eval.learned.noise_seed`; recomputed bootstrap estimates and intervals equal the
frozen ones to ≤ 5e-14.

**B-1 HIGH.** Can H2 (diffusion − BC-MLP +18.7 [+4.7, +32.7]) and H3 (BC-RNN − BC-MLP +19.3)
are separated under the preregistered bootstrap but not under alternatives that model the seed
level differently: Welch on per-seed rates [−4.46, +41.80] (df 2.35); a variance-components
interval [−1.27, +38.60]; Bonferroni over the task's 8 comparisons [−0.7, +38.7]. Conversely,
Lift H2 (diffusion − BC-MLP) becomes separated under some alternatives. Cause: BC-MLP's Can
seeds (68, 88, 74 %) differ widely and the bootstrap resamples only 3 seeds.
**Disposition:** the frozen verdict stays the reported result; the paper adds the seed-level
t analysis labelled "not preregistered" (D-095) beside every declared difference
(Tables 17–18) and calls the Can separations fragile (§6, abstract). KI-053.

**B-2 HIGH.** Several H4 separations (e.g. Lift diffusion − BC-MLP-chunk on cube shift) lose
separation under the same alternatives. Disposition as B-1 (paper §7: 6 of 41 robustness
verdicts change, all from separated to not separated).

**B-3 MEDIUM.** Bootstrap-seed sensitivity: with 50 alternative bootstrap seeds, Lift
diffusion − BC-RNN on cube shift (tail 0.0264) is separated under 20/50 seeds, and Can
drop(diffusion) − drop(BC-MLP) under arm offset (tail 0.0237) is not separated under 25/50.
Limitation (paper §9, "verdicts that depend on the method"); KI-053.

**B-4 MEDIUM.** 21 frozen intervals (rates and differences) are degenerate or end exactly at a
boundary; of the 64 declared success differences, 10 have a bound at exactly zero or zero
width (generated count `sens.all.boundary`). With 50 binary outcomes per seed the bootstrap
distribution is discrete, so those verdicts rest on ties. Limitation (paper §9); KI-053.

**B-5 MEDIUM.** Censored length counts failures at 400 steps, so length differences restate
success. Exact decomposition ΔL = −Δp (H − T̄_B) + p_A ΔT̄ (D-095): Can diffusion − BC-MLP
−56.0 = −52.6 (success) −3.4 (speed); Lift diffusion − BC-RNN +11.0 = +9.5 + 1.5. The auditor's
speed-only comparison of diffusion and BC-RNN on the 46 Lift starts that succeeded under every
seed gives +1.5 [−0.6, +3.2] steps (scratch recomputation, not a reported result).
Corrected by reporting (paper §6).

**B-6 MEDIUM.** 73 unadjusted verdicts across the analyses; a within-analysis Bonferroni
correction flips 9. Limitation with counts (paper §5, §7, §9).

**B-7 MEDIUM.** Saturation makes some contrasts restatements: with both arms at 100 % on the
base starts, Level 2's difference of drops equals −H5. Limitation (paper §8.2).

**B-8 MEDIUM.** The Level 2 held-out comparison (both arms 150/150) could not have favoured the
point-cloud arm; it is a replication-type check, not a test. Limitation (paper §8.2, §9).

**B-9 LOW.** Level 1 results informed Level 2's design (disclosed in RESEARCH_PROTOCOL §13).
**B-10 LOW.** H4, A1, A2 and A3 share the primary arm's evaluations on the base starts, so
they are not independent. Limitation (paper §7).

**B-11 INFO.** Selection optimism (validation vs fresh base vs held-out) is reported (paper
Table 15).

**B-12 LOW, corrected.** `scripts/analyze_robustness.py` had no run-once guard; a rerun in the
same evidence session would have overwritten `analysis.json`. Fix: the script now refuses when
`analysis.json` exists, with a unit test
(`tests/unit/test_analyze_robustness.py::test_main_refuses_to_overwrite_an_existing_analysis`).
The script is not hash-frozen by any freeze; `make freeze-check` passed before and after. KI-055.

**B-13 LOW, corrected.** `scripts/analyze_level2.py` hashed an RGB held-out report before
checking it exists, so a missing file raised a traceback instead of the intended message. Fix:
existence first. Not hash-frozen; freeze-check passed before and after. KI-055.

---

## C. Spend, declared analyses, timeline

**C-1, C-2, C-3 INFO.** All 24 paid launches reconcile with Modal's billing snapshots: $10.61436
in total (A-01 $0.14825 over 6 launches, A-02 $0.03165, A-03 $8.05978 over 12, A-04 $2.37467
over 5); every app in the final month-to-date snapshot is in the ledger and vice versa; every
launch had a granted approval committed in a clean tree and passed its cap check (largest
committed + upper bound: A-03 $12.376 ≤ $15, A-04 $6.119 ≤ $9, A-01 $0.828 ≤ $2); every
failed, cut or preempted launch is billed and counted. Sessions 6, 7, 10 and 11 used no paid
compute. The cost ledger has no launch event after launch 24 (Session 9); the last billing
snapshot is from 2026-11-09 22:55 UTC, so Sessions 10 and 11's $0 rests on the ledger, not on a
later billing snapshot.

**C-4 INFO (wording).** "Billed" is Modal's usage report at list price before credits, not an
invoice. The paper's cost captions say so.

**C-5 MEDIUM.** A-01 authorized "1 L4 GPU probe"; two probe launches ran (5688db17 failed in
5 s, $0.00001; 4489c09f $0.0657) before the 00:20 EST amendment authorized a retry. The guard
checks function and cap, not launch counts (D-049). Limitation (paper §9) and here.

**C-6 LOW.** RGB diffusion seed 2: segment 2 (launch 116f287e, attempt 0) was preempted and
has no `train_wall_s`, so the 1.29 h in the RGB results omits ≈ 0.1–0.2 h; its billed $1.60
includes it. Also: "77.9 ms/step" (E8-10) is the steady-state rate; segment wall time over
steps gives 79.5–84.1 ms. Corrected by reporting: footnote b of the paper's cost table
(rule: more segments than launches). KI-056.

**C-7 INFO.** KI-049 (Level 2 seed 0 24.62 → 42.2 ms/step) is reproduced from the committed
launch record and shown beside the row it corrects.

**C-8 INFO.** A-04's 66.7 ms/step stop rule was not machine-checked for seed 0's cut first
segment (`ms_per_step: null` in the queue log); the hand-measured 41.6 ms/step (E9-07) is below
the threshold. Recorded here.

**C-9 MEDIUM, corrected by reporting.** RESEARCH_PROTOCOL §8 declares gripper sign changes and
action smoothness; every held-out summary computes them per episode, but no frozen result
aggregates them, and FINAL_TEST_FREEZE §4 dropped them without an amendment. Plan diversity
was measured only for the Session 4 pilot; point-cloud throughput (LEVEL2 freeze "also
reported") is absent. Fix: `src/roboforge/paper/secondary.py` reads the 26 held-out summaries
cited by the Level 1 results file, each verified against the SHA-256 recorded there, and
reports plain means labelled "declared secondary, aggregated after the frozen analysis"
(D-096; paper Table 4). Plan diversity and point-cloud throughput are listed as declared, not
reported (paper §9). KI-054.

**C-10 LOW.** RESEARCH_PROTOCOL §14.6 and FINAL_TEST_FREEZE §5 say KI-028's start/noise-seed
confound "is reported"; no frozen result mentions it. Limitation (paper §5, §9).

**C-11 LOW.** DiT (MASTER_SPEC requires a reason if omitted), GMM-BC and Square were not run.
Reason recorded in D-096: not run within the project's time and compute budget; the declared
hypotheses do not depend on them (paper §9).

**C-12 INFO.** Every declared comparison and evaluation is reported, nulls included (Level 1
`not_computed: []`; H4; A1–A3 all not separated; RGB not separated; H5 separated (−)). Every
evaluation summary is referenced by a selection or analysis (480/480, 540/540, 72/72, 36/36;
75/75, 26/26, 6/6, 12/12, 3/3) and every training run is accounted for. Tuning trials: 4 per
family per task, 0 for the extensions.

**C-13 LOW.** Undeclared rows: Level 2 "base success (context)"; H4 computes 4 of 6 family pairs,
fixed in code (`49043d7`) before the first robustness rollout. Limitation (paper §9).

**C-14 MEDIUM.** The E6-02 rehearsal wrote six Lift validation summaries (200 training steps:
diffusion default 0/50, lr 3e-4 6/50 and 8/50; BC-MLP default 50/50, lr 3e-4 50/50 and 48/50)
11–25 min before the Lift search space was committed (`6dce01b`); Lift diffusion later chose
lr 3e-4. lr 3e-4 was already in the rehearsal plan and the trial list is identical for every
family, but BC-RNN and BC-MLP-chunk had no early look. Limitation (paper §9).

**C-15 LOW.** Level 2 reuses the Session 7 robustness starts that helped choose H5's endpoint.
Limitation (paper §8.2, §9).

**C-16, C-17 INFO.** Every freeze and preregistration was committed before the first record it
governs (table in the audit transcript; e.g. FINAL_TEST_FREEZE `bf0a0fa` 22:42:48 UTC →
first held-out start 22:50:52). No frozen file has changed: 83 Level 1 code hashes, both
test-configuration records, 12 RGB and 12 Level 2 file hashes equal the current files;
`b5fc2ee` (Session 10) touched `shadow_render.py` after every evaluation and `d2c8add`
restored it (KI-050).

**C-18 LOW.** The 12 validation summaries of RGB BC seed 2 record `git a1aa1a8, dirty: true`;
the next commit changed 31 files, all docs, logs and reports; the frozen RGB code hashes match.
Limitation (paper §9).

**C-19 MEDIUM.** Only Level 1's `scripts/analyze_heldout.py` is hash-frozen; the robustness, RGB
and Level 2 analysis scripts are not, and each got a post-data fix (`565c6e4`, `8c728f4`,
`620c541`) touching input reading and training accounting only; pairing, resampling and
inclusion are untouched and all import the frozen bootstrap. Each script predates its first
data. Limitation (paper §9).

**C-20 LOW.** (a)–(d) amendments after related results changed no analysis (disclosed in §13
and DECISIONS). (e) Record errors: D-086's prose time "≈ 17:55 EST" is after the point-cloud
held-out run ended (17:52:26 EST; commit `08c3aa8` at 17:49:10 EST), and E9-11 says the
rehearsal followed the amendment, but the rehearsal summary records `4ebebc1`, before
`08c3aa8`. Corrected: both annotated in place, with this audit as the source.

**C-21 LOW.** FINAL_TEST_FREEZE gives `5a5d998f…` as the "SHA-256" of
`configs/eval/failure_taxonomy.yaml`; it is the content hash (canonical JSON) returned by
`taxonomy_sha256()`, while the file bytes hash to `e9aea501…`. Both are consistent; the freeze
is frozen, so the wording is recorded in KI-057.

---

## D. Claims in the paper

A fresh agent compared every claim in `paper/` with the original frozen files and
MASTER_SPEC §11.3. Every finding was corrected in the text before the final build; the
section numbers below are those of the final `paper/main.pdf`.

**INFO (verified).** All 16 Level 1 success rates, per-seed values, intervals and verdicts and
all 8 censored-length differences match `reports/session7/results/tables.md`; the length
decompositions recompute exactly; H4 rates, contrasts, verdicts and counts (24, 12), A1–A3,
RGB and Level 2 values match their frozen tables; the sensitivity counts and every quoted t
interval match; the selected-checkpoint statements, CPU and latency ratios, secondary-metric
columns, selection table, chosen trials and spend match; the point-cloud arm is always
"DP3-inspired"; D-095 and D-096 outputs are labelled everywhere and replace no frozen verdict;
no §11.3 forbidden claim appears, and the report is labelled "not peer reviewed".

| ID | Sev. | Finding | Correction in the paper |
|---|---|---|---|
| D-1 | HIGH | "Not separated" written as equality ("as robust", "matches", "degrades as little", "is matched by") against the preregistered rule | Every instance rewritten as "not separated", with the interval; abstract, §6, §7, §10 |
| D-2 | MEDIUM | "Temporal context" attribution relied on the undeclared BC-MLP-chunk − BC-MLP comparison; "history alone" ignores BC-RNN's different architecture | §6, §9, §10: "no evidence that the generative head adds to history and chunking"; the chunk − BC-MLP gap labelled undeclared and exploratory; BC-RNN's architecture difference stated |
| D-3 | MEDIUM | H2-c control described as differing "only in its deterministic head" | §4 lists head and EMA, action bound and selected tuning trial |
| D-4 | MEDIUM | Preregistered A1/A2 selection-bias disclosure missing; sampler sentence too reassuring | §7 adds the bias and the shared evaluations; DDIM-10 vs DDPM stated as not separated with both point estimates favouring DDPM |
| D-5 | MEDIUM | Out-of-range-action confound stated causally | §7 and §9 call it a correlation and give the counter-evidence |
| D-6 | MEDIUM | Bootstrap-seed flips and Bonferroni results missing; "the audit lists which" was false | §9 states both and points to B-1, B-3, B-6 for the intervals and count |
| D-7 | MEDIUM | Missing limitations: scaled-down diffusion and optimizer recipe; in-distribution Lift starts (KI-011); small tuning budget; replay fidelity (KI-004) | §9 adds "Scale of the diffusion policy", "In-distribution starts" and the tuning budget |
| D-8 | MEDIUM | Robustness failure taxonomy omitted while the text said "every failure" | New generated table `taxonomy_robustness` (Table 10) and macros; §7 cites the arm-offset counts |
| D-9 | MEDIUM | "No measured gain" contradicted the Lift arm-offset result | §6: "no measured gain in held-out success", with the exception |
| D-10 | MEDIUM | Abstract used the t analysis in one direction only; Lift H2 length separation unmentioned | Abstract mentions both directions; §6 reports the Lift length separation |
| D-11 | MEDIUM | "No amendment changed an analysis" contradicted the post-data script fixes | §5 distinguishes method amendments from input-reading fixes |
| D-12 | MEDIUM | Appendix overstated hash-freezing and source coverage (renders and chained summaries missing) | Sources table now lists the three renders and the 26 chained summaries; wording qualified |
| D-13 | LOW–MED | Scope paragraph cited the fragile Can result | §9 scope cites Lift H3 under both methods and Can H3 under the bootstrap only |
| D-14 | LOW | "Matched RGB diffusion policy" without qualification | Abstract and introduction: "frozen RGB diffusion policy (same head and training, different encoder and augmentation)" |
| D-15 | LOW | Cube-shift result missing from the H4 reading | §7 adds BC-RNN degrading least under cube shift |
| D-16 | LOW | Secondary-metric wording (computed by summaries, not the analysis; mechanism offered; clip column undefined) | §6 and the Table 4 caption corrected; no mechanism; clip column defined; diffusion's zero stated as by construction |
| D-17 | LOW | "Each under a committed approval" contradicted the A-01 launch-count finding | §6 states the exception |
| D-18 | LOW | Caption mismatches (held-out table columns, latency figure markers, taxonomy random rows, Level 2 context row, renders not labelled as illustrations) | All captions corrected; "64 declared differences" now "differences in the sensitivity tables (the declared ones and one context row)" |
| D-19 | LOW | Small imprecisions (base learning rate, success predicate, "size of that bias", untested mechanism, "a few hundred demonstrations", intro clause) | Each corrected |
| D-20 | LOW | Extension items declared but not reported (RGB per-checkpoint validation, Level 2 per-checkpoint rates); extension runs that selected their last checkpoint | §9 lists both |
| D-21 | LOW | Bibliography years from the latest arXiv revision | Years set to the first release or venue, with a venue note; `make paper-bibcheck` rerun: 31/31 resolved |
