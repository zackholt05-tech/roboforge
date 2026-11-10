# X2: a harder or multimodal task

**Sessions:** 3 or more (estimate; Level 1 took Sessions 3–7 for two tasks). **Label:** a
separate study with its own decisions, freeze and held-out suite. **Paid compute:** none
planned; Level 1 trained on local CPU.

## 1. Why

On the frozen tasks every learned policy except one-step BC-MLP reaches 95–100 % held-out
success, so the benchmark cannot rank strong methods, and the proficient-human (PH)
demonstrations are close to unimodal. Diffusion policies are usually motivated by multimodal
action distributions. X2 moves to a setting with headroom or multimodality.

## 2. The owner's choice at the start of X2

| Option | Data (pinned Hugging Face revision `74fa0184…`, sizes from its file listing) | What it tests | New infrastructure |
|---|---|---|---|
| **A. Can, mixed-human (MH)** | `v1.5/can/mh/low_dim_v15.hdf5` (112.8 MB), `demo_v15.hdf5` (79.8 MB). Not in the manifest yet | The multimodality claim directly, on a simulator contract already validated for Can | Manifest entries (a data decision, SHA-256 from the LFS ids, re-verified on download); operator-aware split; a new held-out suite |
| **B. Square, proficient-human (PH)** | `v1.5/square/ph/low_dim_v15.hdf5` (51.1 MB), `demo_v15.hdf5` (69.0 MB). **Already in `data/manifests/robomimic_v1.5_ph.json` with SHA-256**, `download_by_default: false` | A harder precision task (nut insertion) with headroom | A full Session 0–2 style contract: replay check, observation and action conventions, success check, eval suites |
| C. Square, MH | `v1.5/square/mh/` (123.3 MB low-dim, 73.9 MB raw) | Both, at once | Everything in A and B |

**Recommendation: A (Can MH).** It isolates the variable the frozen study could not test
(demonstration multimodality) while holding the task, simulator, controller and evaluation
machinery fixed, and it needs the least new infrastructure. B is the better choice if the owner
cares more about task difficulty than about multimodality. Lift MH also exists (50.7 MB
low-dim) but Lift saturates (KI-034).

## 3. Constraints

- **Operator identity.** MASTER_SPEC §3.3: if the dataset carries demonstrator identity,
  preserve it in the split and report within- versus held-out-operator claims. The first X2
  session inspects the MH file's masks or filter keys and records what operator information it
  actually contains; nothing is assumed.
- **Fairness.** All four families train on the same demonstrations, normalization (training
  partition only) and budget; the tuning budget per family is preregistered and equal, as in
  Sessions 6–7.
- **Headroom check before the full design.** A short pilot (validation starts only, one seed,
  the frozen Level 1 configurations) confirms the task is neither saturated nor at zero before
  the search space and budget are frozen. The pilot never touches held-out starts.
- **Held-out suite.** A new suite generated on the container from a seed block outside every
  reserved block in `data/eval_states/*/partitions.json`, named for X2 (never `final_test`).
- **Frozen code.** Training and evaluation code pinned by `docs/FINAL_TEST_FREEZE.md` is reused
  as-is; any task-specific code goes into new files. `make freeze-check` before and after.
- **Disk.** Measured on the Can PH study: one seed of the four families (training runs, 20
  validation evaluations and one held-out evaluation each) used **2.3 GB**. The X2 tuning plus
  replicates will need several times that; the first X2 session measures free space and
  proposes pruning or relocation of non-frozen scratch before training (KI-008).

## 4. Phases

1. **X2-a: data and contract.** Owner's choice of option; data decision; download and verify
   SHA-256; inspect and split (operator-aware for MH); for Square, the full replay and
   simulator contract; validation and held-out suites generated and checked; headroom pilot.
2. **X2-b: tuning, training, selection.** Preregistered search space and budget; local CPU
   training; checkpoint selection on validation starts only; latency benchmark.
3. **X2-c: freeze and test.** `docs/extensions/X2_FREEZE.md` committed (the Level 1 freeze is
   the template); held-out suite run once; analysis once; report.

## 5. Cost (estimate, to be replaced by pilot measurements)

Level 1 Can PH as the reference: the tuning and replicate suite recorded **46.36
container-hours** of validation evaluations (E7-02), and one training run per family per seed
took 0.16–4.31 CPU-hours (BC-MLP to diffusion; `paper/generated/tables/cost_latency.tex`). The
Can MH low-dim file is 2.4 times the size of the PH one (112.8 vs 46.9 MB), so expect more or
longer demonstrations; the first X2 session counts them, and the pilot measures the real step
time and rollout time before any budget is fixed.

## 6. Prompt template for the first X2 session

```text
Implement RoboForge 3D SESSION <N> ONLY: post-release extension X2, phase X2-a (data and
contract), as specified in docs/extensions/X2_harder_task.md, under DECISIONS D-105 and the
rules of MASTER_SPEC.md. Begin by reading MASTER_SPEC.md, docs/extensions/README.md,
docs/extensions/X2_harder_task.md, docs/STATUS.md, docs/HANDOFF.md, docs/DECISIONS.md,
docs/KNOWN_ISSUES.md, docs/EXPERIMENT_LEDGER.md and docs/RESEARCH_PROTOCOL.md, and inspect the
actual repository and test state. First ask the owner to choose option A, B or C of §2, with
the facts. <Binding boundaries of docs/extensions/README.md "Exact prompt for Session 13",
with EVIDENCE_SESSION=<N>, and: no held-out rollout in X2-a; the headroom pilot uses
validation starts only.> At the STOP boundary update the records, run the checks, commit as
Zackary Holt with no AI co-author trailers, write the exact prompt for X2-b, and STOP.
```
