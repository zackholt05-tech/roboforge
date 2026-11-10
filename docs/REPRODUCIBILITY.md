# Reproducibility

## Environment
```bash
# prerequisites: uv >= 0.11 (https://docs.astral.sh/uv/), git; macOS arm64 or Linux
git clone https://github.com/zackholt05-tech/roboforge.git && cd roboforge
make setup            # == uv sync --frozen  -> .venv with Python 3.11 and exact pins from uv.lock
```
Pinned core: mujoco 3.2.7 · robosuite v1.5.1 (`51cc017`) · robomimic `d309eae` ·
torch 2.15.0 · numpy 2.4.6 · h5py 3.16.0 · diffusers 0.11.1 · transformers 4.41.2.
`egl_probe` installs on Linux only (D-005). On Linux it needs `cmake`, `build-essential` and
`libegl-dev`; for headless rendering also `libosmesa6` (`MUJOCO_GL=osmesa`) or an NVIDIA EGL
driver (`MUJOCO_GL=egl`).

## Session 0 evidence commands (historical)
| Command | What it proves | Output |
|---|---|---|
| `make download` | official files fetched; SHA-256 equals manifest (= HF LFS oid) | `data/raw/...` (gitignored) |
| `make smoke-sim` | real 3D reset/step/success/render; seeding and cross-process determinism | `reports/feasibility/smoke_sim_{lift,can}.json`, `frames/smoke_*.png` |
| `make verify-data` | HDF5 structure, low_dim ≡ raw, obs ⇔ state alignment, replay of all demos, rendered frames | `reports/feasibility/data_report_{lift,can}.json`, `frames/*` |
| `make probe-obs` | relative-pose observation correctness (KI-003) | `reports/feasibility/obs_consistency.json` |
| `make test` | unit + simulator + dataset integration tests | pytest summary |
| `make lint` | ruff | — |

Every JSON report embeds `environment_snapshot()`: Python, platform, package versions,
`MUJOCO_GL`, and git SHA plus dirty flag.

## Seeds and determinism
- Simulator resets: seed the **global** `np.random` before `env.reset()`. Restoring
  serialized states is exact (KI-002).
- Random actions in smoke tests: `np.random.default_rng(seed)`.
- Torch: `torch.manual_seed` in tests. Training determinism settings are defined in Session 3.

## Canonical evaluation platform (Sessions 2–11)
Every reported rollout ran in the Linux aarch64 container `roboforge-sim-cpu`
(image `sha256:1f685638f28c…`, Python 3.11.18, OSMesa), started through
`infra/containers/run_sim_cpu.sh` with `ROBOFORGE_EVIDENCE_SESSION=<N>` from a clean tree.
Policy inference uses one CPU thread, so a rollout is deterministic given its seeds. Native
macOS rollouts are development only (KI-013). Never run `uv sync` or `uv run` inside the image.

## Frozen results and how to check them
- Level 1 (`docs/FINAL_TEST_FREEZE.md`, `docs/LEVEL1_COMPLETE.md`), the RGB study
  (`docs/RGB_EXPERIMENT_FREEZE.md`) and Level 2 (`docs/LEVEL2_EXPERIMENT_FREEZE.md`,
  `docs/LEVEL2_GO_NO_GO.md`) are frozen; their results live in `reports/session7/results/`,
  `reports/session7/robustness/`, `reports/session8/rgb_results/` and
  `reports/session9/level2_results/` (with `ERRATA.md`, KI-049).
- `make freeze-check` compares the Level 1 freeze's code hashes with the working tree;
  `uv run --frozen python scripts/run_rgb_heldout.py gate` and
  `uv run --frozen python scripts/run_pc_heldout.py gate` print the state of the two later
  held-out gates. The frozen evaluation and analysis targets (`make eval-final`,
  `make analyze-final`, `make rgb-heldout`, `make pc-heldout`, `make pc-placement`,
  `make level2-analyze`) must never be rerun.
- The independent audit of all frozen work, with the disposition of every finding, is
  `docs/AUDIT.md`.

## The paper (Session 11)
| Command | What it does |
|---|---|
| `make paper-assets` | regenerates `paper/generated/` (tables, figures with data sidecars, `numbers.tex`, `index.json`) from hash-pinned frozen files; summaries cited inside a pinned file are read only after their recorded SHA-256 matches (D-094, D-096) |
| `make paper-check` | the committed `paper/generated/` equals a fresh build byte for byte (except `index.json`'s git field), and the paper lint passes (no typed numbers, no placeholders, every macro defined) |
| `make paper-bibcheck` | resolves every DOI and arXiv id of `paper/references.bib` and writes `paper/bibliography_check.json` (needs network; never part of `make test`) |
| `make paper` | `paper-check`, then Tectonic 0.17.0 with `--only-cached` and `SOURCE_DATE_EPOCH=$(PAPER_EPOCH)`; fails on undefined or duplicate references; prints the PDF's SHA-256 |

Tectonic fetches its TeX bundle on first use. A fresh machine runs one networked build first
(for example `cd paper && tectonic -X compile --outdir /tmp/x main.tex`); after that
`make paper` never reaches the network. Two `make paper` builds on this host and bundle give
byte-identical PDFs; byte equality is not claimed across Tectonic versions, bundles or
platforms (D-093).

## Fresh-clone reproduction (Session 12)
Session 12 cloned the committed tree into an empty directory (`git clone --no-local`, nothing
shared with the working repository) and ran, in order (DECISIONS D-098; results with times in
`docs/FINAL_RELEASE_AUDIT.md`, logs in `logs/session12/fresh_clone/`):

```bash
make setup                                   # uv sync --frozen (host only, never in the image)
make lint
uv run --frozen pytest tests/unit -m "not sim and not data and not render and not slow and not web"
uv run --frozen pytest tests/research_integrity
make freeze-check
uv run --frozen python scripts/run_rgb_heldout.py gate
uv run --frozen python scripts/run_pc_heldout.py gate
make paper-check
mkdir -p data/raw data/processed             # bind-mount targets of the container
ROBOFORGE_EVIDENCE_SESSION=12 infra/containers/run_sim_cpu.sh \
  python scripts/smoke_sim.py --task lift --out-dir reports/session12/container_smoke   # and can
make download check-data build-data
make test                                    # alone (KI-043)
make smoke-sim SMOKE_OUT=<scratch dir>       # host; the default rewrites committed evidence
make paper                                   # compare the PDF's SHA-256 with the committed one
make web-install web-data web-assets web-build web-test web-e2e
```

What a fresh clone cannot do, and why:
- **Re-run a policy.** Checkpoints and rollout records are in the gitignored `runs/` and are not
  redistributed. Tests that need them skip with the reason; reported numbers are regenerated
  from the committed summaries (`make paper-check`).
- **A first Tectonic build offline.** Tectonic downloads its TeX bundle on first use (D-093).
- **Bitwise physics on another platform.** Rollouts are reproducible on the canonical container
  (KI-013).

## Continuous integration
`.github/workflows/` (DECISIONS D-101) runs on every push to `main` on GitHub's Ubuntu x86_64
runners: lint and the unit tests that need no simulator, dataset, renderer, node or long run
(`python-tests`); the held-out lock tests, `make freeze-check`, both gates, `make paper-check`
and `make release-check` (`research-integrity`); and the site's data, meshes, build with its
budgets and vitest (`web-build`). No dataset is downloaded and nothing is published.

## Paid and remote compute
All paid compute ran on Modal under four committed approvals (A-01 to A-04): 24 launches,
$10.61 billed at list price before credits, reconciled launch by launch with Modal's billing
snapshots (`reports/compute/cost_ledger.jsonl`; `docs/AUDIT.md` C-1). The RGB and Level 2
policies trained on one NVIDIA L4 with deterministic CUDA kernels; their evaluation ran on the
canonical CPU platform. Level 1 trained on the local CPU at no cost.

## What is not reproducible, or not redistributed
- Physics is not bit-identical across platforms: macOS and Linux give the same pass/fail
  outcomes but different trajectories (KI-013). Reproduce rollouts on the platform recorded
  in the run manifest.
- EGL vs OSMesa on Modal is unverified (KI-001); every reported rollout used OSMesa.
- Trained checkpoints, rollout records and per-episode arrays live in `runs/` (gitignored) and
  are not redistributed. The committed summaries, selections and analyses regenerate every
  reported number, but re-running a policy requires retraining it from the committed
  configurations.
- The datasets are downloaded by `make download` and verified by SHA-256; they are not
  redistributed.
