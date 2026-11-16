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
| Tested code tree | Fresh clone (§3): `bd3bf6b`. After it, Session 12 changed release tooling, the site credits, the CI workflows and the test harness only (`src/roboforge/release/`, `scripts/release_check.py`, `readme_findings.py`, `license_inventory.py`, `website/src/sections/Methods.tsx`, `.github/workflows/`, `tests/conftest.py` import order for KI-061); nothing under the simulator, evaluation, models, training or analysis code. STOP checks ran on `b4bd070` and again on `ab17fb2` after the harness change (`docs/STATUS.md`); CI reran the D-101 checks from fresh checkouts on Linux aarch64 and passed on `ee1e574` (§7). Later commits change records only |
| Report | `paper/main.pdf`, 23 pages, SHA-256 `c648e7532e7112cd8bcb3b4e0f118342d6eb97dffd94d3bd1e5448c81b50939f` (unchanged since Session 11; rebuilt byte-identical in the fresh clone) |
| Tag | `v1.0.0`, annotated (tag object `82f3219`), on `7b394fa`, the commit that completed the publication record; the commit after it records the tag and the release only (§7) |
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
runs of §7 (Linux x86_64 first, then Linux aarch64).

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
| External links | `scripts/release_check.py --network` | before publication: 17 of 20 URLs answered; the failures were the local preview URL (now excluded from the scan as not external) and the repository URL in two forms, 404 because the repository did not exist yet. Rechecked after publication: 19/19 answered with HTTP < 400 (§7, `reports/session12/release_check_network.json`) |

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

Each row was verified before it was written (owner authorization D-099, D-101, D-102).

| Step | Verified state | Evidence |
|---|---|---|
| Repository | public, `https://github.com/zackholt05-tech/roboforge`; `main` pushed (first `b4bd070`, then each later commit, checked with `git ls-remote`) | `gh repo view`; audit §7 |
| CI, Linux x86_64, `b4bd070` | research-integrity pass (run 42755805851); python-tests fail, triton segfault (42755805869); web-build fail, mesh check 7.45e-9 m (42755806070) | KI-061 |
| CI, Linux aarch64, `e11ad04` | research-integrity pass (42756520065); web-build pass (42756520080); python-tests fail, same triton segfault (42756519847) | KI-061 root cause |
| CI, Linux aarch64, `ee1e574` (after the harness fix) | python-tests pass (42758938590: lint; 303 passed, 12 skipped: 9 without the processed Lift data, 2 without `runs/`, 1 needs MPS); research-integrity pass (42758938394); web-build pass (42758938356) | D-101 amendment 2 |
| Observatory | GitHub Pages from branch `gh-pages`, orphan commit `c14e041` (Zackary Holt), staged by `make pages-bundle` from clean `b4bd070`; build `built`; `https://zackholt05-tech.github.io/roboforge/` | `gh api …/pages` |
| Live site check | all 186 files listed in `deploy.json` fetched from the live URL with equal SHA-256; the site's E2E run against the live URL 7/7 | `logs/session12/pages_live_e2e.log` |
| External links after publication | 19/19 answered with HTTP < 400 | `reports/session12/release_check_network.json` |
| CI, Linux aarch64, `7b394fa` (the tagged commit) | python-tests pass (42760766015), research-integrity pass (42760766276), web-build pass (42760766117) | |
| Tag `v1.0.0` | annotated tag object `82f3219` on `7b394fa`, tagger Zackary Holt <zackholt05@gmail.com>, no trailer; pushed (`git ls-remote`) | `git cat-file -p v1.0.0` |
| GitHub release | `https://github.com/zackholt05-tech/roboforge/releases/tag/v1.0.0`, published 2026-11-10 05:24 UTC, not a draft or prerelease; notes written by hand (no generated notes, no byline); asset `main.pdf` (878,204 bytes), GitHub digest and the SHA-256 of the downloaded copy both `c648e7532e7112cd8bcb3b4e0f118342d6eb97dffd94d3bd1e5448c81b50939f`, equal to the committed `paper/main.pdf` | `gh release view`, `gh release download` |

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
| Release | PASS | §7 |

## 9. Outstanding limitations and open items

- **Science:** simulation only; exact object state for Level 1; three seeds and fifty held-out
  starts per task; Lift saturation (KI-034); separations that depend on the interval method
  (KI-053); the diffusion action bound as a confound (KI-052; the evaluation-only clamp test was
  not run and needs its own decision); a compact diffusion model and a small tuning budget; the
  same Lift starts reused across three sequential studies (audit B-5). Full list: paper §9 and
  `docs/AUDIT.md`.
- **Engineering, open:** KI-001 and KI-033 (EGL vs OSMesa on Modal unverified), KI-008 and
  KI-010 (local disk, image size), KI-011 (narrow Lift start distribution), KI-021, KI-023,
  KI-024, KI-028, KI-034, KI-035, KI-042; KI-060 (dependency advisories, accepted); KI-061
  (partial: triton crashes after Mesa's LLVM is loaded globally; fixed in the test harness, a
  recorded workaround elsewhere; Linux x86_64 meshes differ by 7.45e-9 m, so x86_64 stays
  unverified).
- **Not done, by decision:** DiT, GMM-BC and the Square task (D-096); Python static type
  checking (not configured); a cold-cache macOS install (CI performs a cold Linux install).

## 10. v1.1.0 publication (Session 19)

**Release:** RoboForge v1.1.0, the post-release follow-up studies X1, X2 and X3 (owner decision
D-131; content D-133, D-134; built and verified in Session 18, E18-01 to E18-07). **Author:**
Zackary Holt. Sections 1 to 9 above are the v1.0.0 record and stay unchanged; v1.0.0 (tag,
release, `paper/main.pdf` `c648e753…`) stays as published. No training, no evaluation, no
rollout and no paid compute; nothing in the Session 18 build was rebuilt or changed.

Each row was verified before it was written, in the D-131 order. Logs: `logs/session19/`;
reports: `reports/session19/`.

| Step | Verified state | Evidence |
|---|---|---|
| Prerequisites | Pinned hashes equal (X1, X3, X2 `results.json`, `X2_FREEZE.md`, `paper/main.pdf`, `paper/addendum/addendum.pdf`, the Session 18 Observatory `index.json`); lint, freeze-check (unchanged), addendum-check, paper-check, release-check, `tests/research_integrity` (81) and the full suite alone (720 passed) pass; site build within budgets, vitest 75, E2E 8/8; `origin/main` `abbb9c3`, `gh-pages` `c14e041`, `v1.0.0` `82f3219` on `7b394fa`, release v1.0.0 saved in full | E19-01, `logs/session19/prereq_*.log` |
| Release commit | `907c901` (`CITATION.cff` version 1.1.0, date-released 2026-11-16), after `ab6c49b` (D-135, Makefile defaults); author and committer Zackary Holt <zackholt05@gmail.com>, no trailer; `make release-check` PASS on it | `git show -s 907c901` |
| `main` | pushed as a fast-forward, `abbb9c3..907c901` (51 commits, oldest `536fa24`); `git ls-remote` `refs/heads/main` `907c901`; no commit message carries a co-author trailer or AI byline | `git ls-remote` |
| CI, Linux aarch64, `907c901` | python-tests pass (43832425639: lint; 454 passed, 13 skipped: 9 without the processed Lift data, 3 without `runs/` records, 1 needs a non-CPU device); research-integrity pass (43832425616: 77 passed, 4 skipped; freeze-check, both gates, paper-check, the new addendum-check step (15 files, lint), release-check); web-build pass (43832429173: meshes 0.0 m, every budget, vitest 75); each run's `headSha` is `907c901` | `logs/session19/ci_907c901.json` |
| Tag `v1.1.0` | annotated tag object `d805cb5` on `907c901`, tagger Zackary Holt <zackholt05@gmail.com>, message without trailer; pushed; `git ls-remote` `refs/tags/v1.1.0` `d805cb5`, `^{}` `907c901`; `refs/tags/v1.0.0` still `82f3219` on `7b394fa` | `git cat-file -p v1.1.0` |
| GitHub release | `https://github.com/zackholt05-tech/roboforge/releases/tag/v1.1.0`, created with `--verify-tag`, published 2026-11-16 08:39 UTC, not a draft or prerelease, marked Latest; notes written by hand (no generated notes, no AI byline; the body equals `logs/session19/release_notes_v1.1.0.md`); asset `addendum.pdf` (121,963 bytes), GitHub digest and the SHA-256 of the downloaded copy both `bedfb27753722604e6f31b6fc171c646975ba6b7b635bdd9884dd6bc16917fe2`, equal to the committed `paper/addendum/addendum.pdf` | `logs/session19/v1_1_0_release.json`, `gh release download` |
| Release v1.0.0 | unchanged: `gh release view v1.0.0 --json` (body, assets, dates, author) identical before and after; only the Latest label moved to v1.1.0 | `logs/session19/v1_0_0_release_{before,after}.json` |
| Observatory build | from the clean tagged commit `907c901`: `make web-data` (161 + 74 files, hash-checked), `make web-assets` (61 meshes, host vs container 0.0 m), `make web-build` (every budget; all data 29.31 of 80 MB, first view 3.95 of 6.50 MB), `make pages-bundle`: 263 files, 35.7 MB, `deploy.json` source `907c901`, dirty false, SHA-256 `b4b641e2…` | `logs/session19/pages_build.log`, `reports/session19/pages/deploy.json` |
| `gh-pages` | new commit `d3f47dd` on top of `c14e041` (parent checked before the push; author and committer Zackary Holt, no trailer), pushed as a fast-forward `c14e041..d3f47dd`; never an orphan commit or a force push, so the v1.0.0 site stays in history. Only hashed JS chunks, `index.html` and `deploy.json` changed; `data/ext/` added; every v1.0 data file unchanged | `git ls-remote` |
| Pages | build `built` for `d3f47dd` (2026-11-16 08:40:14 to 08:40:44 UTC), source branch `gh-pages`, `https://zackholt05-tech.github.io/roboforge/` | `logs/session19/pages_build_latest.json` |
| Live site check | `scripts/pages_bundle.py verify`: all 263 files listed in `deploy.json` match (262 fetched with equal SHA-256; `.nojekyll` is not served), first attempt; the live `deploy.json` equals the staged one (`b4b641e2…`). The site's E2E against the live URL (a scratch copy of `e2e/site.spec.ts` whose only change is relative `page.goto` paths under `/roboforge/`, including the follow-up studies test): 8/8 | `logs/session19/pages_verify.log`, `logs/session19/pages_live_e2e.log` |
| External links | `scripts/release_check.py --network`: 21/21 answered with HTTP < 400; offline checks PASS | `reports/session19/release_check_network.json` |
| README | the Observatory paragraph names the build commit `907c901` (tag `v1.1.0`) and says, without numbers, that the site also shows the three follow-up studies under their own labels (D-133); committed after the live checks passed | `README.md` |

**Not changed by Session 19:** every frozen file and result, the Session 18 build (addendum,
README extensions block and chart, Observatory extension export and site code),
`pyproject.toml` (0.0.1) and `website/package.json` (0.10.0) (D-099), and the repository
description.

## 11. Observatory redeploy after the visual quality pass (Session 20)

**What was published:** the Session 20 site build (owner decisions D-136, D-138): presentation
changes in `website/` only. No release, no tag, no result, no number, no clip and no export
changed; v1.0.0 and v1.1.0 (tags, releases, `paper/main.pdf`, `paper/addendum/addendum.pdf`) stay
as published. Sections 1 to 10 above stay unchanged. No training, no evaluation, no rollout and
no paid compute.

Each row was verified before it was written, in the order of `docs/HANDOFF.md` # Session 20
Handoff §5. Logs: `logs/session20/`; reports: `reports/session20/`.

| Step | Verified state | Evidence |
|---|---|---|
| Local checks before the push | On `e5cff56` plus the Session 20 records: lint, full suite alone (721 passed), freeze-check (unchanged), `tests/research_integrity` (81), addendum-check, paper-check and release-check pass; site build within every budget, vitest 79, E2E 10/10; viewer re-verified 18/18 and 12/12 | E20-03 to E20-05 |
| `main` | pushed as a fast-forward, `92c8cc4..57d8b7b` (4 commits: `88707cd`, `586ef34`, `e5cff56`, `57d8b7b`); author and committer of each Zackary Holt <zackholt05@gmail.com>; no commit message carries a co-author trailer or AI byline; `git ls-remote` `refs/heads/main` `57d8b7b` | `git ls-remote` |
| CI, Linux aarch64, `57d8b7b` | python-tests pass (43882198857: 455 passed, 13 skipped, 1 deselected); research-integrity pass (43882198783: 77 passed, 4 skipped; the release scans all PASS, `private_paths` included); web-build pass (43882198727: every budget, vitest 79); each run's `headSha` is `57d8b7b` | `logs/session20/ci_57d8b7b.json` |
| Observatory build | from the pushed commit `57d8b7b`, tracked files clean: `make web-data` (161 + 74 files, hash-checked), `make web-assets` (61 meshes, host vs container 0.0 m), `make web-build` (every budget; JavaScript 391.8 of 520 KB gzip, all data 29.31 of 80 MB, first view 3.95 of 6.50 MB), `make pages-bundle`: 263 files, 35.7 MB, `deploy.json` source `57d8b7b`, dirty false, SHA-256 `38a02eac…` | `logs/session20/pages_build.log`, `reports/session20/pages/deploy.json` |
| `gh-pages` | new commit `b4467f7` on top of `d3f47dd` (parent checked before the push; author and committer Zackary Holt, no trailer), pushed as a fast-forward `d3f47dd..b4467f7`; never an orphan commit or a force push, so the v1.0.0 and v1.1.0 sites stay in history. 264 tracked files (the 263 of the bundle and `deploy.json`), every hash equal to the manifest. Changed against `d3f47dd`: 12 hashed script and stylesheet files replaced, `index.html` and `deploy.json` modified; no file under `data/`, no mesh, font or video changed | `git ls-remote` |
| Pages | build `built` for `b4467f7` (2026-11-16 14:09:19 to 14:09:50 UTC), `https://zackholt05-tech.github.io/roboforge/` | `logs/session20/pages_build_latest.json` |
| Live site check | `scripts/pages_bundle.py verify`: the first attempt (14:09:59 UTC) stopped on one HTTP 503 from the Pages CDN for one file; the second (14:10:27 UTC) matched all 263 files listed in `deploy.json` (`.nojekyll` is not served); the live `deploy.json` equals the staged one (`38a02eac…`). The site's E2E against the live URL (a scratch copy of `e2e/site.spec.ts` whose only change is 10 relative `page.goto` paths under `/roboforge/`): 10/10, including the two Session 20 tests (caveats visible by default and provenance one click away; phone chart layout) | `logs/session20/pages_verify.log`, `logs/session20/pages_live_e2e.log` |
| README | the Observatory paragraph names the build commit `57d8b7b` (on `main`, after tag `v1.1.0`); nothing else in the README changed; committed after the live checks passed | `README.md` |

**Disclosed:** three Session 20 logs were committed at `88707cd` and `586ef34` with the absolute
home directory in them and rewritten with `~` at `57d8b7b` (E20-05). History is never rewritten
(D-100), so those earlier blobs are public in the history of `main`. The path holds the
author's name, which the repository already states, and nothing else.

**Not changed by this redeploy:** every frozen file and result, the Session 10 and Session 18
Observatory exports and their viewer verifications, `website/public/data`, `paper/`, the README
findings, setup and extensions blocks, the README preview and chart, `CITATION.cff` (1.1.0),
`website/package-lock.json`, both tags and both releases.
