# Errata to `results.json` / `tables.md` (Level 2 analysis, generated once at `8c728f4`)

The analysis is never rerun or overwritten (Level 2 freeze §7). Corrections are recorded here.

1. **Training row of seed 0 (`s9_pc_diffusion_dp3_lift_seed0_8fd35613`), `training` block.**
   - The script counted each segment's steps from `start_step`/`resumed_from`, but the
     manifest key is `resumed_from_step`. It therefore counted 60,000 steps for the one
     completed segment, which resumed at 25,000.
   - Its two earlier segments (KI-047) have no `train_wall_s`:
     - the first trained 25,000 steps in 1,040 s (from `metrics.jsonl`) before the client's
       network loss stopped it;
     - the second was stopped about 20 s after it started, before any step was saved.
   - **Corrected:** the completed segment trained 35,000 steps in 1,477 s (0.41 h),
     **42.2 ms/step** (not 24.62). Seed 0's total training time, including the 1,040 s of the
     cut first segment, is about 2,517 s, **0.70 h**. The generated 0.41 h is the
     completed-segment time only, so it understates the total.
   - Billed amounts are unaffected: $0.80 for seed 0 over its three launches, $2.37 for all.
2. Nothing else is affected: every success, interval, separation, taxonomy count and latency
   in the files is as generated.

The script is fixed for the record (`scripts/analyze_level2.py`, `_training`). It now counts
completed segments only, reporting 1,477 s (0.41 h) at 42.2 ms/step for seed 0; segments without
an end record (KI-047) are left out. The fix changes no generated file.
