<!-- parsable status header -->
```yaml
CURRENT_SESSION: 4
SESSION_STATUS: PASS            # protocol amended first; diffusion implemented, reference-equal, trained, reloaded, rolled out on Lift
NEXT_SESSION: 5
BLOCKERS: none for Session 5    # open items carried forward: KI-001, KI-008, KI-011, KI-021, KI-023, KI-024, KI-028
LAST_VERIFIED_COMMIT: 183b56c   # lint clean, 257 tests; later commits are docs/evidence only
PROTOCOL_COMMITS: {v1: 8b08b95, diffusion_amendment: 3aed462}   # amendment before any reported diffusion run
EVIDENCE_COMMITS: {diffusion_training: 3aed462, validation_selection_equivalence_latency_export_diagnostics: 183b56c}   # all dirty=false
EVALUATION_PLATFORM: linux-aarch64-container-1f685638f28c   # D-017, configs/eval/platform.yaml
FINAL_TEST: LOCKED              # docs/FINAL_TEST_FREEZE.md does not exist; no test state loaded
RESOURCE_SPEND: {usd: 0.00, gpu_hours: 0, notes: "local CPU only; no remote compute"}
UPDATED: 2026-11-06
```

# RoboForge Status

## Session 4 acceptance criteria (MASTER_SPEC Session 4)
| Criterion | Status | Evidence |
|---|---|---|
| Prerequisites re-verified at `71ab833` | DONE: all PASS, `data/` clean, 231 tests, smoke macOS 33/33, container 17/17 + 16/16, image ID equal, Session 3 runs present | `logs/session4/prereq/`, LEDGER E4-01 |
| Protocol TO FINALIZE items filled before the first reported diffusion run | DONE: amendment 2026-11-06 at `3aed462` (RESEARCH_PROTOCOL §6.4, §13); pilot launched from that clean commit | E4-08 |
| Observation encoder, temporal U-Net, timestep conditioning | DONE: flatten encoder (0 params) + `ConditionalUnet1D` (64, 128), kernel 5, GN 8, FiLM, 64-D sinusoidal step embedding; **1,092,551 params** | D-040, `reports/session4/param_counts.json` |
| DDPM/DDIM sampler, normalized action-chunk training, EMA | DONE: K = 100 `squaredcos_cap_v2`, ε-prediction, masked denoising MSE, DDIM-10 (η = 0, `clip_sample`), DDPM-100 alternative, EMA power 0.75 | D-040, `src/roboforge/models/` |
| Horizon logic, step-inference control loop, episode reset | DONE: T_o 2 / T_p 16 / T_a 8, replans every 8 steps; generator re-seeded per episode by `(policy_seed, state_id, repeat)`; receding-horizon checks PASS on all 300 pilot episodes | D-041, `reports/session4/diagnostics/actions.json` |
| Compare with robomimic's reference (schedule equations, shapes, seeded reproducibility, synthetic arrays, reconstruction) | DONE: **bitwise** vs diffusers 0.11.1 and robomimic `d309eae` on macOS and the container, including full DDIM-10 / DDPM-100 sampling; oracle-noise reconstruction test; licenses in `THIRD_PARTY_NOTICES.md` | `reports/session4/reference/`, E4-02 |
| Tiny-batch overfit on real Lift windows, finite gradients | DONE: sampled-chunk MSE ×0.014 (32 windows) and ×0.062 (2 demos); all losses and gradient norms finite | `reports/session4/overfit_lift.json`, E4-04 |
| Checkpoint reload equivalence: identical seeded samples and closed-loop `physics_sha256` | DONE: selected (15k) and final (30k) checkpoints pass A–F on the container (two loads, fresh process, recorded eval, Session 2 sanity, seeded samples); 30k also on macOS; BC selected checkpoints still pass A–E | `reports/session4/checkpoint_equivalence/`, E4-05, E4-12 |
| `export_denoising` with actual per-timestep tensors | DONE: 9 decisions exported from 2 canonical evaluations; every replayed chunk equals the recorded actions bitwise; normalized + denormalized `A_{t_k}`, ε̂, x̂₀, coefficients, conditioning window, noise seed; labeled proposals | `reports/session4/denoising/`, D-043, E4-14 |
| Small validation rollout; debug normalization and replanning | DONE: 6 canonical evaluations × 50 Lift starts, **0 invalid actions or errors**, 0 clipping in every dimension, raw actions within [−1, 1], replanning checks PASS | `reports/session4/validation/`, E4-10, E4-15 |
| Container inference latency (KI-023) | DONE: DDIM-10 36.3 ms per decision (native 9.9 ms), DDPM-100 649 ms (native 99.5 ms) | `reports/session4/cost_latency/`, E4-13 |
| No paid compute, no publication, committed data, eval suites, `uv.lock` and dependencies unchanged | DONE | `git diff 71ab833 -- uv.lock pyproject.toml data/ configs/eval` empty |

## Pilot validation results (development numbers, one seed, Lift validation suite)
| Family | Successes at 5k / 10k / 15k / 20k / 25k / 30k steps | Selected (§7) |
|---|---|---|
| BC-MLP (Session 3) | 47 / 41 / 45 / 41 / 42 / 42 of 50 | step 5000 (47/50) |
| BC-RNN (Session 3) | 50 / 50 / 50 / 48 / 49 / 49 | step 15000 (50/50) |
| BC-MLP-chunk (Session 3) | 49 / 50 / 49 / 48 / 48 / 46 | step 10000 (50/50) |
| Diffusion U-Net (Session 4) | 49 / 50 / 50 / 49 / 49 / 49 | step 15000 (50/50) |

These are not results. They come from one training seed on the in-distribution validation
suite, and selection used that same suite. Every diffusion failure is the same start,
`lift_validation_0029`, under the same noise seed (KI-028).

## Owner decisions pending
1. **Session 5 (Modal):** any paid smoke test needs explicit approval first (COMPUTE_PLAN
   Gate A), with a cost bound. Without it, Session 5 delivers the integration and a local
   dry-run only.
2. Optional: prune the Docker build cache if disk space is needed (KI-008).
