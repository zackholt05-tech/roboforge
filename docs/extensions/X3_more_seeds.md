# X3: more training seeds (a separate replication)

**Sessions:** 1 to 2 (estimate). **Label:** "separate replication". **Paid compute:** none
planned; Level 1 trained on local CPU.

## 1. Why

Level 1 used 3 training seeds per family and task. On Can, two preregistered separations
(diffusion − BC-MLP, +18.7 pp; BC-RNN − BC-MLP, +19.3 pp) do not hold under a seed-level t
interval computed after the audit ([−4.5, +41.8] and [−2.8, +41.5]; KI-053). With 3 seeds the
seed-level interval has 2 degrees of freedom, so it is very wide. X3 asks whether those
separations, and the H2-c parity, hold with more seeds.

## 2. Design (to be fixed by a decision and the preregistration)

- **No retuning.** Each family trains with its frozen chosen configuration (the chosen trial
  in `docs/FINAL_TEST_FREEZE.md` and the Session 6/7 test configurations), the same 100k-step
  budget, checkpoints every 5k, and the same validation-success selection rule. Only the seed
  changes (seeds 3, 4, …).
- **Fresh held-out starts.** Evaluating new seeds on the already-used held-out suite would be
  a revisit of the test. X3 generates a new held-out suite per task on the container
  (`ROBOFORGE_WRITE_EVAL_STATES=1`, `scripts/generate_eval_states.py`) from a seed block outside
  every reserved block in `data/eval_states/<task>/partitions.json`, named for X3 (never
  `final_test`). Its size (50 starts, or more) is part of the preregistration.
- **All seeds on the new suite.** The frozen seed 0–2 checkpoints are also evaluated on the
  new suite (a new suite, not a revisit), so the replication has N seeds × the new starts,
  paired by start across families.
- **Task.** Can is where the fragility is; Lift saturates (KI-034). Recommendation: Can only,
  Lift optional.
- **Analysis.** The preregistration names the primary interval (the frozen hierarchical
  bootstrap, the seed-level t interval, or both with a stated rule) and its seed, the contrasts
  (H2 diffusion − BC-MLP, H2-c diffusion − BC-MLP-chunk, H3 BC-RNN − BC-MLP), and how the
  replication is reported next to the frozen result (side by side, never merged into it).
- **Seed count.** Chosen in the X3 session from the recorded per-seed spread of the frozen Can
  results and the cost below, and fixed in the preregistration before any new seed trains.

## 3. Cost per extra seed (estimate, from measured Level 1 runs)

| Task | Training, 4 families (CPU-hours, seed 1–2 values) | Validation selection (container-hours, seed 1–2 runs) | Disk (training runs + 20 validation + 1 held-out evaluation each) |
|---|---|---|---|
| Can | ≈ 5.2 (BC-MLP 0.16, BC-RNN 0.85, BC-MLP-chunk 0.18, diffusion 3.98) | ≈ 6.4 (2.0, 1.4, 1.0, 2.0) | 2.3 GB |
| Lift | ≈ 4.6 (0.17, 0.93, 0.18, 3.36) | ≈ 3.7 (1.6, 0.7, 0.7, 0.7) | 1.5 GB |

Sources: `paper/generated/tables/cost_latency.tex`; `wall_s` of the recorded validation
summaries of the seed 1–2 runs under `runs/`; `du` of those runs (2026-11-10). Training runs
single-threaded and in parallel (Session 7 used up to 8 training jobs and 4 evaluation
containers), so wall-clock time is much shorter than the sums. Plus the held-out evaluations of
every seed on the new suite.

**Disk is the binding limit:** about 20 GB free on 2026-11-10, so about 8 extra Can seeds at
most before anything else; the X3 session measures free space first and proposes what can be
pruned (never a frozen run; KI-008).

KI-061: training on the macOS host is unaffected. Training inside the CPU container would crash
unless `torch._dynamo` is imported before mujoco.

## 4. Acceptance criteria

New suite generated, checked (`make check-eval-states`-style regeneration) and committed with
its hash before any rollout on it; preregistration committed before the first new seed trains;
every new run trained from a clean tree with its configuration hash equal to the frozen one
except the seed; selection by the frozen rule on validation starts only; the new suite run once
per checkpoint; analysis once; results labelled "separate replication" and reported beside,
never instead of, the frozen ones.

## 5. Prompt template for the X3 session

```text
Implement RoboForge 3D SESSION <N> ONLY: post-release extension X3, a separate replication
with more training seeds, as specified in docs/extensions/X3_more_seeds.md, under DECISIONS
D-105 and the rules of MASTER_SPEC.md. Begin by reading MASTER_SPEC.md,
docs/extensions/README.md, docs/extensions/X3_more_seeds.md, docs/STATUS.md, docs/HANDOFF.md,
docs/DECISIONS.md, docs/KNOWN_ISSUES.md (KI-008, KI-053, KI-061), docs/EXPERIMENT_LEDGER.md and
docs/FINAL_TEST_FREEZE.md, and inspect the actual repository and test state. First ask the
owner to confirm the task(s) and the number of extra seeds, with the measured cost and disk.
<Binding boundaries of docs/extensions/README.md "Exact prompt for Session 13", with
EVIDENCE_SESSION=<N>, and: training is allowed only for the frozen configurations with new
seeds; no rollout on the frozen held-out suites.> At the STOP boundary update the records, run
the checks, commit as Zackary Holt with no AI co-author trailers, write the exact prompt for
the next session, and STOP.
```
