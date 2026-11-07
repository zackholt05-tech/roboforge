<!-- parsable status header -->
```yaml
CURRENT_SESSION: 5
SESSION_STATUS: PASS            # Modal integration built and tested on Modal (Gate A); costs projected; Gate C decided (local CPU, $0)
NEXT_SESSION: 6
BLOCKERS: none for Session 6    # open items carried forward: KI-001 (EGL vs OSMesa on Modal), KI-008, KI-011, KI-021, KI-023, KI-024, KI-028, KI-033
LAST_VERIFIED_COMMIT: db10a1b   # lint clean, 281 tests; later commits are docs/evidence only
PROTOCOL_COMMITS: {v1: 8b08b95, diffusion_amendment: 3aed462, main_budget_amendment: ec9626b}
EVIDENCE_COMMITS: {modal_gate_a: [2477312, a2fe706, feef27e], profiles: [a2fe706, feef27e], projection: db10a1b, pilot_equivalence: ec9626b}   # all dirty=false (exceptions recorded in LEDGER E5-03, E5-12)
EVALUATION_PLATFORM: linux-aarch64-container-1f685638f28c   # D-017, configs/eval/platform.yaml
FINAL_TEST: LOCKED              # docs/FINAL_TEST_FREEZE.md does not exist; no test state loaded
RESOURCE_SPEND: {usd: 0.148, gpu_hours: 0.09, notes: "Modal Gate A under approval A-01 ($2.00 cap, closed); billed list price; month-to-date incl. another app $5.42 of $30"}
APPROVED_LIMIT: {gate_c: "local CPU only, $0", modal: "none until a new approval row"}
UPDATED: 2026-11-07
```

# RoboForge Status

## Session 5 acceptance criteria (MASTER_SPEC Session 5)
| Criterion | Status | Evidence |
|---|---|---|
| Prerequisites re-verified at `d956292` | DONE: all PASS, `data/` clean, 257 tests, smoke macOS 33/33, container 17/17 + 16/16, image ID equal, 4 pilot runs present and hash-checked | LEDGER E5-01 |
| Modal entrypoint: pinned image from `uv.lock`, Volumes, no secrets, cost/time log, dry run by default, approved launch path | DONE: `infra/modal_app.py`. `max_containers=1`, `retries=0`, explicit timeout, at most 4 attempts per launch, refuses existing run IDs. Approval gate plus cost ledger | D-045, D-047, D-049, E5-16 |
| Resumable, interrupt-safe checkpoints; hash-validated retrieval | DONE: content-identical to uninterrupted runs, locally (9 tests, two simulations) and **on Modal**. Real-SIGINT drill → Modal restart → clean stop → resume launch → hash-verified download | D-046, E5-02, E5-03, E5-08, E5-12 |
| Modal-trained checkpoint reusable outside Modal | DONE: loaded locally under D-035/D-042; equivalence A–F PASS on the canonical container; 5-start validation 4/5, 0 errors | E5-08 |
| GPU training profile; CPU/parallel rollout profile; rendering bottleneck | DONE: L4 profile of all 4 families; local concurrency and parallel-container profiles. Physics dominates evaluation. EGL vs OSMesa on Modal **PENDING** (KI-033) | E5-09, E5-10, E5-11, E5-13 |
| Projected cost of the main suite with reserve; alternative budgets | DONE: 4 families stated explicitly. Local, Modal CPU and Modal L4 for five variants | COMPUTE_PLAN §4, E5-15 |
| Main training budget confirmed or amended before Session 6 | DONE: amended to **100k steps, checkpoints every 5k** (owner) | RESEARCH_PROTOCOL §6.3, §13 (2026-11-07) |
| Gate C asked before any full suite; no automatic retry or multi-GPU | DONE: owner chose **local CPU only, $0**. `retries=0`, attempts capped at 4, one GPU at most | D-051, E5-17, KI-031 |
| Billed cost accounted | DONE: **$0.148**, reconciled per launch | COMPUTE_PLAN §2 |
| Committed data, eval suites, `uv.lock` and dependencies unchanged | DONE | `git diff d956292 -- uv.lock pyproject.toml data/ configs/eval` empty |

## Pilot validation results (unchanged since Session 4; development numbers, one seed, Lift validation suite)
| Family | Successes at 5k / 10k / 15k / 20k / 25k / 30k steps | Selected (§7) |
|---|---|---|
| BC-MLP (Session 3) | 47 / 41 / 45 / 41 / 42 / 42 of 50 | step 5000 (47/50) |
| BC-RNN (Session 3) | 50 / 50 / 50 / 48 / 49 / 49 | step 15000 (50/50) |
| BC-MLP-chunk (Session 3) | 49 / 50 / 49 / 48 / 48 / 46 | step 10000 (50/50) |
| Diffusion U-Net (Session 4) | 49 / 50 / 50 / 49 / 49 / 49 | step 15000 (50/50) |

These are not results: they come from one seed, and selection used the same validation suite.
The four selected checkpoints still pass reload equivalence with the Session 5 code (E5-14).

## Owner decisions pending
1. Optional: set a Modal **workspace budget** (the only hard spending cap) before any future
   paid session.
2. Optional: a future approved L4 probe to settle EGL vs OSMesa (KI-001). It is not needed
   for Level 1.
3. Optional: prune the Docker build cache if disk space is needed (KI-008).
