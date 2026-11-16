RoboForge v1.1.0: three follow-up studies after v1.0.0 (X1, X2, X3), published as a report addendum, a generated README section and a new section of the Observatory. Author: Zackary Holt.

**Addendum:** `addendum.pdf` is attached (6 pages, a local build, not peer reviewed). SHA-256 `bedfb27753722604e6f31b6fc171c646975ba6b7b635bdd9884dd6bc16917fe2`. Every number in it is generated from the three hash-pinned extension result files; none was typed by hand.

**Unchanged:** the v1.0.0 report (`paper/main.pdf`, SHA-256 `c648e7532e7112cd8bcb3b4e0f118342d6eb97dffd94d3bd1e5448c81b50939f`), its frozen protocols and results, and the v1.0.0 release. Each follow-up study was preregistered and run once under its own rules. None of them changes a frozen number; each is shown beside the frozen results and never pooled with them.

**The three studies, each under its own label:**
- **X1, the action bound (exploratory, after the freeze).** The frozen BC checkpoints were re-run with the diffusion sampler's action bound, in the validation regime (the Session 7 robustness starts), not on the held-out test. It tested the BC side of KI-052: BC's missing bound does not explain the robustness gaps. The diffusion side, clipping inside the sampler, was not tested and stays open.
- **X3, more training seeds on Can (separate replication).** The frozen Can configurations with more training seeds, evaluated on fresh held-out starts. The Can separations that were fragile with three seeds (KI-053) hold under both interval methods. The frozen result stands as reported.
- **X2 (Can MH), mixed-quality demonstrations from several operators.** Reported beside the frozen Level 1 Can PH result, never instead of it and never pooled with it. Within-operator only (operator-stratified split). X2 episodes end at 500 steps and Level 1 Can PH episodes at 400, so the two success scales differ.

**Findings:** see the generated `extensions` block in `README.md` and the addendum, both written from the same pinned macros. As that block states: "On Can, in the frozen result, X3 and X2 alike, no primary analysis separates the diffusion policy from the deterministic MLP that sees the same history and predicts the same action chunks (H2-c)." "Not separated" is never read as equality.

**Observatory:** the site built from this tag adds the section "After v1.0.0: three follow-up studies": X1 and X2 recorded evaluation episodes, re-rendered from their recorded simulator states (rule-selected cases, labelled as such, not representative samples), and the X3 results, each under its own label. It is deployed to https://zackholt05-tech.github.io/roboforge/ after this release; the verified deployment is recorded in `docs/FINAL_RELEASE_AUDIT.md`.

**Read with the findings:** simulation only, with exact simulator state as input (privileged information). Small diffusion models, the frozen tuning budgets, a few seeds and one canonical platform. The diffusion side of KI-052 is open; KI-053 describes the interval-method dependence of the frozen Can PH separations; KI-065 notes that the X2 records carry the frozen evaluator's PH label (the site shows the Can MH label). The addendum, like the v1.0.0 report, is a local build, not peer reviewed.

**Reproduce:** `make addendum-check` regenerates every table, figure and macro of the addendum from the pinned files and compares them byte for byte; `make addendum` builds the PDF with the v1.0.0 toolchain. Study records: `docs/extensions/`, `docs/DECISIONS.md` (D-105 to D-134), `docs/EXPERIMENT_LEDGER.md`.

License: Apache-2.0. Third-party credits: `THIRD_PARTY_NOTICES.md`.
