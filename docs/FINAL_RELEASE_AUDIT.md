# Final release audit (Session 12)

**Release:** RoboForge v1.0.0 (MASTER_SPEC Session 12). **Author:** Zackary Holt.
**Scope:** fresh-clone reproduction, the full test matrix, release scans, README and citation,
and publication as the owner authorized (DECISIONS D-097 to D-104). No training, no evaluation,
no paid compute; no frozen file, freeze or executed outcome changed.

Every row below names the command that was run and its observed result. Logs:
`logs/session12/` (fresh clone: `logs/session12/fresh_clone/`, local paths replaced by
`<clone>`, `<scratch>` and `~`). Machine-readable reports: `reports/session12/`.

## 1. Release identifiers

| Item | Value |
|---|---|
| Tested code tree | Fresh clone (§3): `bd3bf6b`. After it, Session 12 changed release tooling and the site credits only (`src/roboforge/release/`, `scripts/release_check.py`, `readme_findings.py`, `license_inventory.py`, `website/src/sections/Methods.tsx`; nothing under the simulator, evaluation, models, training or analysis code). The STOP checks of §4 ran on the release-candidate commit recorded in `docs/STATUS.md` (`LAST_VERIFIED_COMMIT`), and CI reran the checks of D-101 on it from a fresh checkout on Linux x86_64 (§7). Later commits change documentation only |
| Report | `paper/main.pdf`, 23 pages, SHA-256 `c648e7532e7112cd8bcb3b4e0f118342d6eb97dffd94d3bd1e5448c81b50939f` (unchanged since Session 11; rebuilt byte-identical in the fresh clone) |
| Tag | `v1.0.0`, annotated, on the commit that completes the release records (§7) |
| Frozen protocols | Level 1 `bf0a0fa` (`docs/FINAL_TEST_FREEZE.md`), RGB `d9c6217`, Level 2 `b93c50c`; held-out results `97cc48a`, `3b23255`, `620c541` |
| Platform of every reported rollout | Linux aarch64 container, image `sha256:1f685638f28c…` (D-017) |

## 2. Frozen results (generated, not typed)

The block below is written by `scripts/readme_findings.py` from `paper/generated/numbers.tex`,
which `make paper-check` proves equal to a fresh build from the hash-pinned frozen files; every
verdict stated in its prose is checked against the frozen macros before it is written, and
`make release-check` fails if the block is stale.

<!-- BEGIN GENERATED: findings (scripts/readme_findings.py from paper/generated/numbers.tex; do not edit by hand) -->

**Level 1 held-out success** (exact simulator state as input, which is privileged information; 3 training seeds × 50 paired held-out starts = 150 episodes per cell, run once under the frozen protocol; 95 % hierarchical bootstrap interval, preregistered):

| Policy | Lift | Can |
|---|---|---|
| BC-MLP (one step) | **90.7 %** (136/150) [83.3, 96.7] | **76.7 %** (115/150) [62.7, 90.0] |
| BC-RNN (LSTM) | **100.0 %** (150/150) [100.0, 100.0] | **96.0 %** (144/150) [89.3, 100.0] |
| BC-MLP-chunk (16-step chunks) | **98.0 %** (147/150) [94.0, 100.0] | **96.0 %** (144/150) [88.7, 100.0] |
| Diffusion U-Net (10 DDIM steps) | **97.3 %** (146/150) [93.3, 100.0] | **95.3 %** (143/150) [89.3, 100.0] |
| Random actions (sanity) | **0.0 %** (0/50) | **0.0 %** (0/50) |

Differences below are in percentage points (pp) with the preregistered 95 % interval and verdict.

- **Diffusion − BC-MLP (H2):** Can +18.7 pp [+4.7, +32.7], separated (+). This separation is fragile: a seed-level t interval, computed after the audit and not preregistered, is [−4.5, +41.8] and includes 0 (KI-053). Lift +6.7 pp [−0.7, +14.7], not separated (seed-level interval [+1.9, +11.5], separated (+)).
- **Diffusion − BC-MLP-chunk (H2-c, same inputs, horizons and chunking; deterministic head):** Lift −0.7 pp [−5.3, +4.7], not separated; Can −0.7 pp [−8.7, +7.3], not separated. These data give no evidence that the generative model adds to observation history and action chunking. Diffusion is also not separated from BC-RNN on either task.
- **Action bound (KI-052):** the diffusion sampler keeps every action inside the training range; the BC heads are bounded only by the controller limits. This confounds H2-c and the robustness comparisons.
- **BC-RNN − BC-MLP (H3):** Lift +9.3 pp [+3.3, +16.7], separated (+); Can +19.3 pp [+5.3, +34.0], separated (+) (seed-level interval [−2.8, +41.5], fragile in the same way).
- **Cost:** diffusion took 19–26× the chunked MLP's CPU training time, and a diffusion decision (10 DDIM steps) took 22–27× as long.
- **Robustness (H4, preregistered perturbation sets, not the held-out test):** under the initial arm offset, diffusion degrades least on Lift (its drop is separated from each BC family's; minus the chunked MLP's: −40.0 pp [−54.0, −26.0], separated (−)); on Can the same contrast is +4.7 pp [−10.7, +20.7], not separated. KI-052 applies to both.
- **RGB study (Lift; two 84×84 cameras plus proprioception, ResNet-18 encoder):** visual diffusion 100.0 % (150/150) vs visual BC-chunk 99.3 % (149/150): +0.7 pp [+0.0, +4.0], not separated. Lift saturates for every policy, so it cannot rank them (KI-034).
- **Level 2 (Lift; DP3-inspired point clouds from sensor-like depth, 512 points):** under cube placements outside the demonstrations' range, the point-cloud diffusion policy reached 56.7 % vs 98.0 % for the frozen RGB diffusion policy (same head and training, different encoder and augmentation): −41.3 pp [−54.0, −28.0], separated (−). H5 is not supported; the explicit 3D input generalized worse here.
- **Paid compute:** $10.61 over 24 Modal launches (list price before credits); Level 1 trained on a local CPU.

<!-- END GENERATED: findings -->

## 3. Fresh-clone reproduction (D-098)

`git clone --no-local` of the committed tree into an empty scratch directory on the
development host (macOS arm64), then the canonical container from that clone. Commit `21fc749`,
then `d9da1dd` and `bd3bf6b` pulled into the clone after the two fixes it found. The code
changed after `bd3bf6b` is listed in §1; the release candidate's own fresh checkouts are the CI
runs of §7 (Linux x86_64).

| # | Command (in the clone) | Result | Time |
|---|---|---|---|
| 1 | `git clone --no-local`; `git status` | clean, HEAD `21fc749`, author Zackary Holt | < 1 s |
| 2 | `uv sync --frozen` | PASS; 1.9 GB `.venv`, packages from the local uv cache (a cold download is exercised by CI on Linux) | 2 s |
| 3 | `make lint` | PASS | < 1 s |
| 4 | CI unit selection `pytest tests/unit -m "not sim and not data and not render and not slow and not web"` | **FAIL then PASS:** 9 tests needed the processed dataset without skipping (KI-058, fixed `d9da1dd`); rerun 304 passed, 11 skipped | 18 s |
| 5 | `pytest tests/research_integrity` | **FAIL then PASS:** 4 gate tests met missing gitignored records (KI-059, fixed `bd3bf6b`); rerun 41 passed, 4 skipped (records not present) | 9 s |
| 6 | `make freeze-check`; both gate commands | frozen code unchanged; `"open": true, "problems": []` twice | < 1 s |
| 7 | `make paper-check` | PASS: 35 generated files equal a fresh build; paper lint clean | 3 s |
| 8 | `run_sim_cpu.sh python scripts/smoke_sim.py --task lift` / `can` (`ROBOFORGE_EVIDENCE_SESSION=12`) | PASS: 33/33 checks, `all_passed` on both tasks, clean `bd3bf6b` (`reports/session12/fresh_clone/container_linux_*`) | 25 s, 21 s |
| 9 | `make download` | PASS: 4 files from the pinned Hugging Face revision `74fa018`, every SHA-256 equal to the manifest (163 MB) | 71 s |
| 10 | `make check-data`; `make build-data` | PASS: rebuilt features equal the committed manifests; max feature − dataset difference 0.0 | 65 s; 40 s |
| 11 | `make test` (full suite, alone) | **PASS: 519 passed, 7 skipped, 0 failed** (526 collected) | 8 min 42 s |
| 12 | `make smoke-sim SMOKE_OUT=<scratch>` (host) | PASS: 33/33 (`reports/session12/fresh_clone/host_macos_*`) | |
| 13 | `make paper` | PASS: no undefined reference; **PDF SHA-256 `c648e753…` equal to the committed file** | 6 s |
| 14 | `make web-install web-data web-assets web-build web-test` | PASS: 162 data files hash-checked; 61 meshes equal the container's to 0.0 m; build within every budget (first view 3.94 of 6.50 MB, all data 18.50 of 80 MB); vitest 49/49 | 15 s |
| 15 | `make web-e2e` | PASS: 7/7 (sections, keyboard, reduced motion, phone, no WebGL, missing artifact, render loop) | 17 s |
| 16 | `make web-contract`; `pytest tests/web_contract` | PASS: three.js 0.187 conversion and forward kinematics within 1e-15 m | 1 s |
| 17 | Frozen checkpoint (Can diffusion U-Net seed 0, step 65000) from the working repository's `runs/`, loaded with the clone's code | SHA-256 `41dd66dd…` equal to the freeze; strict load as `DiffusionPolicy`, 1,092,551 parameters. No rollout | 2 s |

Skipped in the clone's full suite, each with its reason: 4 research-integrity and 2 Observatory
tests need the gitignored `runs/` records; 1 web-contract test needs `npm ci` in
`tests/web_contract` (run separately in row 16).

**Limits of this reproduction, stated:** the clone ran on the development host, so platform
independence rests on the canonical container (row 8) and on CI (§5); the Tectonic bundle cache
already existed (E11-07), so a first networked Tectonic build was not exercised; checkpoints and
rollout records are not in git and are not redistributed.

Disk of the fresh clone after all steps: 2.5 GB (`.venv` 1.9 GB, datasets 163 MB, processed
features 4.4 MB, `website/node_modules` 141 MB, `.git` 81 MB).

## 4. Test matrix and checks on the release candidate

| Check | Command | Result |
|---|---|---|
| Lint | `make lint` (ruff) | see `docs/STATUS.md` STOP row |
| Python, full suite, alone | `make test` | see `docs/STATUS.md` STOP row (526 tests: 315 unit, 54 integration, 111 simulator, 45 research-integrity, 1 web contract; markers: 100 sim, 96 data, 24 render, 3 slow, 1 web) |
| Simulator | included in `make test` (`-m sim`, 100 tests) and the two smoke runs of §3 | PASS in the clone |
| Research integrity | `pytest tests/research_integrity`; `make freeze-check`; both gates | see `docs/STATUS.md` |
| Website | `make web-build` (TypeScript `tsc --noEmit` type check and budgets), `make web-test` (vitest), `make web-e2e` (Playwright) | PASS on the release candidate: build within budgets, 49/49, 7/7 |
| Type checks | Python: none configured (no mypy or pyright in the lock; ruff only); TypeScript: `tsc` in `make web-build` | TypeScript PASS; Python type checking not configured, stated as a gap |
| Paper | `make paper-check`, `make paper` | PASS; PDF byte-identical |
| Release scans | `make release-check` | see `docs/STATUS.md` |
| Dependency vulnerabilities | `pip-audit -r <uv export, PyPI packages> --no-deps` | 72 packages audited; advisories in diffusers 0.11.1 and transformers 4.41.2 (29 IDs), pinned by robomimic, not reachable from RoboForge code (KI-060, ACCEPTED). robosuite and robomimic (git installs) cannot be audited by pip-audit (`reports/session12/pip_audit.json`) |
| | `npm audit --omit=dev`; `npm audit` | 0 vulnerabilities, runtime and dev (`reports/session12/npm_audit_*.json`) |
| Licenses | `scripts/license_inventory.py` | 76 Python distributions, 95 npm packages (`reports/session12/licenses.json`). For review: pynput and qpsolvers (LGPL-3.0, transitive, installed from PyPI, never imported, not redistributed); robosuite and robomimic declare no metadata license (MIT in their repositories). Shipped site packages: MIT, BSD-3-Clause, OFL-1.1 |
| External links | `scripts/release_check.py --network` | before publication: 17 of 20 URLs answered; the failures were the local preview URL (now excluded from the scan as not external) and the repository URL in two forms, 404 because the repository did not exist yet. Rechecked after publication (§7, `reports/session12/release_check_network.json`) |

## 5. Release scans (D-104)

| Scan | What | Result |
|---|---|---|
| History identity | author, committer and tagger of every commit and annotated tag on every ref | every commit `Zackary Holt <zackholt05@gmail.com>` for both; no other identity (`make release-check`) |
| AI trailers and bylines | `Co-authored-by:` and AI bylines in every commit and tag message | none |
| Secrets | credential patterns (AWS, GitHub, Slack, Google, OpenAI- and Anthropic-style keys, Hugging Face, Modal tokens, private-key blocks) over `git log -p --all` and the tree | none. A pattern scan, not a full secret scanner (gitleaks is not installed) |
| Private paths | absolute home paths | only the 136 files disclosed by D-100 (logs of Sessions 0–9, four evidence reports, `docs/KNOWN_ISSUES.md`, one test fixture); accepted, not scrubbed (owner decision; KI-009) |
| Tracked content | datasets, checkpoints, meshes, files over 50 MB, ignored credential files | none tracked; `.env`, key files, `.modal.toml` and `secrets/` are ignored; largest tracked file 2.2 MB |
| Dataset copies | committed data are manifests, splits, normalization statistics and evaluation start states generated by RoboForge | no third-party dataset copy |
| Placeholders | to-do, to-be-determined and fix-me markers, repository placeholders and filler text in the public documents and paper sections (patterns in `roboforge.release.scan`) | none (a clone-URL placeholder in `docs/REPRODUCIBILITY.md` was replaced by the real URL) |
| Relative links | every relative link in every tracked Markdown file | all resolve |
| Citations | `paper/bibliography_check.json` (Session 11): 31/31 entries resolved by DOI or arXiv; `CITATION.cff` names Zackary Holt as the only author, version 1.0.0, Apache-2.0 | PASS |
| Data download links and commands | `make download` in the fresh clone (§3 row 9) | PASS |

## 6. README, citation and public claims

- README follows MASTER_SPEC §11.2: the question, a preview from real rollout renders with
  its selection rule and sources (`docs/media/readme_preview.json`), one paragraph, the
  generated findings with the audit's caveats (Can H2 fragile under the seed-level check,
  H2-c not separated, the diffusion action bound KI-052, Lift saturation, H5 negative), a link
  to the paper labelled "a local build, not peer reviewed", and an Observatory link only once
  the hosted site is verified (§7).
- MASTER_SPEC §11.3: no forbidden claim (no "human-level", "real-world", "first", "proved",
  "published paper"); the point-cloud arm is always "DP3-inspired"; Level 1 is labelled as
  using exact simulator state.
- `CITATION.cff`: Zackary Holt only, version 1.0.0, Apache-2.0; external work credited in
  `THIRD_PARTY_NOTICES.md` and the paper's bibliography.

## 7. Publication

Recorded only after each step is verified; see the "Publication record" section added after
the release in `docs/STATUS.md` and `docs/HANDOFF.md`.

## 8. MASTER_SPEC §13.2 release acceptance matrix

| Requirement | Status | Evidence |
|---|---|---|
| True 3D control | PASS | MuJoCo 3.2.7 + robosuite Panda; smoke checks of robot motion, contacts and success in §3 |
| Data provenance | PASS | `data/manifests/robomimic_v1.5_ph.json`; download re-verified in §3 |
| Replay | PASS (Session 0–2) | `make verify-data`; open-loop Can replay diverges at contacts (KI-004, accepted) |
| Baselines | PASS | BC-MLP, BC-RNN, BC-MLP-chunk; checkpoint hashes in the freeze; one reloaded in §3 |
| Diffusion | PASS | conditional 1D U-Net, DDPM training, DDIM-10 sampling; recorded denoising tensors |
| Main benchmark | PASS | paired held-out starts, denominators, hierarchical bootstrap (§2) |
| Two tasks | PASS | Lift and Can |
| Honest modalities | PASS | privileged state, RGB and DP3-inspired point clouds labelled separately |
| Level 1 freeze | PASS | `bf0a0fa` before any Level 2 work |
| Level 2 | PASS (run) | calibrated depth and point clouds, matched evaluation, negative H5 |
| Compute governance | PASS | four approvals, $10.61 billed, reconciled (`docs/AUDIT.md` C-1) |
| 3D showcase | PASS | viewer verified against 18 MuJoCo frames; hosting per §7 |
| Denoising | PASS | real sampler tensors replayed bitwise |
| Research report | PASS | `paper/main.pdf`, not peer reviewed |
| GitHub authorship | PASS | §5 |
| External credit | PASS | `THIRD_PARTY_NOTICES.md`, bibliography, site credits |
| Release | see §7 | |

## 9. Outstanding limitations and open items

- **Science:** simulation only; exact object state for Level 1; three seeds and fifty held-out
  starts per task; Lift saturation (KI-034); separations that depend on the interval method
  (KI-053); the diffusion action bound as a confound (KI-052; the evaluation-only clamp test was
  not run and needs its own decision); a compact diffusion model and a small tuning budget; the
  same Lift starts reused across three sequential studies (audit B-5). Full list: paper §9 and
  `docs/AUDIT.md`.
- **Engineering, open:** KI-001 and KI-033 (EGL vs OSMesa on Modal unverified), KI-008 and
  KI-010 (local disk, image size), KI-011 (narrow Lift start distribution), KI-021, KI-023,
  KI-024, KI-028, KI-034, KI-035, KI-042; KI-060 (dependency advisories, accepted).
- **Not done, by decision:** DiT, GMM-BC and the Square task (D-096); Python static type
  checking (not configured); a cold-cache macOS install (CI performs a cold Linux install).
