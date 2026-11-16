# Observatory visual audit (Session 20)

Owner request (D-136, 2026-11-16): "a full visual quality improvement pass on the observatory,
making sure things are best visually represented, rendered, that there isnt too much text strewn
across the observatory, that everything makes sense". This is presentation work on `website/`
only. No result, clip, export, number or verdict changes.

Part 1 is the baseline, written and committed before any site code changed. Part 2 (the plan),
Part 3 (what changed, test changes, caveat locations after) and Part 4 (the after measurements)
are added as the session proceeds.

## Part 1. Baseline (site as published, `92c8cc4`, built from the unchanged code)

### 1.1 Method
- `make web-audit AUDIT_TAG=before` (`website/e2e/audit-s20.spec.ts`, Playwright on the installed
  Chrome against `vite preview` of the build): 162 JPEG viewport tiles, 10 MB, in
  `logs/session20/visual_audit/before/`.
  - `1440x900/`, `1280x800/`, `390x844/`: every section, scrolled top to bottom
    (`<section>_NN.jpg`);
  - `reduced_motion/` and `nowebgl/`: the first two tiles of every section at 1440×900;
  - `viewers/`: each 3D viewer during playback (hero, lab, compare, inside, follow-ups X1 and
    X2, pixels) and the denoising explorer at its first and last step;
  - `word_counts.json`: words and text blocks visible by default, from the DOM;
  - `index.json`: the tile list.
- Word counting: text nodes that are rendered and not inside a closed `<details>` (a `<summary>`
  counts). "Reading words" are body text plus small print (notes, captions, legends, errata).
  Controls, table cells and chart labels are counted separately.
- Fresh-context review: a subagent that saw only the screenshots (no code, no documents, no
  opinion of mine) answered MASTER_SPEC §8.1's 30-second test per section and listed what is
  confusing, redundant, cluttered, badly rendered or hard to read. Its record is in §1.5.

### 1.2 Baseline word counts (1440×900; `before/word_counts.json`)

| Section | Body text | Small print | Reading words | Controls | Table | Chart labels | Visible words | In closed disclosures | Text blocks | Charts | Height (px) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| hero | 55 | 16 | 71 | 4 | 0 | 0 | 75 | 0 | 3 | 0 | 828 |
| why | 58 | 113 | 171 | 5 | 0 | 28 | 204 | 0 | 4 | 2 | 951 |
| lab | 101 | 84 | 185 | 48 | 0 | 7 | 240 | 0 | 17 | 0 | 1,235 |
| compare | 54 | 139 | 193 | 25 | 0 | 21 | 239 | 0 | 10 | 1 | 1,093 |
| inside | 179 | 137 | 316 | 57 | 0 | 28 | 401 | 169 | 24 | 2 | 1,453 |
| findings | 113 | 512 | 625 | 8 | 37 | 331 | 1,001 | 302 | 25 | 10 | 3,133 |
| followups | 216 | 1,040 | 1,256 | 80 | 333 | 430 | 2,099 | 508 | 54 | 10 | 5,804 |
| pixels | 147 | 524 | 671 | 27 | 52 | 148 | 898 | 0 | 31 | 8 | 3,957 |
| methods | 700 | 103 | 803 | 0 | 173 | 0 | 976 | 0 | 32 | 0 | 3,023 |
| **page** | 1,623 | 2,668 | **4,291** | 254 | 595 | 993 | **6,133** | 979 | 200 | 33 | 21,477 |

At 390×844 the page is 36,575 px tall with 6,174 visible words. Small print (2,668 words) is
larger than the body text (1,623): most of the reading load is captions and notes.

### 1.3 Findings, ranked by visible impact

**1. Text: every chart repeats its own legend (text, charts; findings, followups, pixels).**
Each difference and estimate chart ends with the same caption of about 25 to 45 words ("Diamond:
paired difference in percentage points; bar: 95 % interval. Filled: … hollow: … which is not
evidence of equivalence"). The same "not separated" sentence then appears again as a note
beside the chart. In findings, 512 of 625 reading words are small print; in followups 1,040 of
1,256.

**2. Charts: a chart's title is below it, inside the caption (charts; all chart sections).**
The reader sees marks first and learns what they are last. No chart has a heading above it or
an axis title. Chart pairs sit side by side with nothing saying how they relate.

**3. Rendering: the 3D scene is dark on dark (rendering; every viewer).** Table, pedestal, bins,
mount and floor grid all sit within a few percent of the graphite background, so the dark robot
links, the bin walls and the shadows barely separate. The manipulated object (a 4 cm cube or a
can) is small and has the same value as the bins. Shadows are faint and there is no contact
shadow under the object. The robot mount and stand legs read as clutter under the table.

**4. Rendering: default camera framing (rendering; lab, compare, followups).** The lab's default
"Free orbit" on Can crops the top of the arm and the front of the bin. Head-to-head and the
follow-up viewers default to MuJoCo's frontview, where the robot is about a third of the canvas
height and the object a few pixels; most of each canvas is empty floor. In the X1 three-up the
cube, the thing the case is about, is about a dozen pixels wide.

**5. Coherence: the page does not say what the research found until the sixth section
(coherence; hero).** The hero states the setup but no finding. The 30-second test's fourth
question (what did the research find?) cannot be answered from the first screen.

**6. Text and layout: provenance is spelled out in full beside every viewer (text, layout; hero,
lab, compare, followups, pixels).** Each clip caption ends with an evaluation
run id of about 100 characters in 11 px type. The lab also shows a nine-row provenance table by default, next to a short
action chart, which leaves a large hole in the grid (`1440x900/lab_02.jpg`). In head-to-head and
the follow-ups the caption and "Executed actions sent to the simulator …" repeat under every
viewer (three times in X1).

**7. Density: the follow-up section is 5,804 px tall and 2,099 words (layout, text; followups).**
Three studies in sequence, each with an intro note, two charts, a seven-column table, readings,
a method note and a viewer group. The X1 "clamped − unclamped" chart is 15 rows whose message is
one count ("0 of 15 separate"), which the text already states.

**8. Charts: small defects (charts).** The H1 chart's axis labels collide ("+75 +100+125").
In the X2 and X3 contrast charts the frozen and new rows alternate with no visual pairing.
Outcome strips encode failure type by colour with no legend (hover only). Verdict tags ("sep. +",
"not sep.") are 11 px.

**9. Methods is a wall of prose and an 18-row table (text; methods).** 700 body words, the
longest section. The verification table (18 rows, all "pass", values 1.000) and the list of
frozen source paths repeat material that a one-line summary and the Findings source table carry.

**10. Accessibility and phone (accessibility).** Type below 12 px in several places: run ids
(11 px), inset video caption (10.5 px), sampler tick labels (10 px), chart keys (11 px). The nav
has no current-section state, and on a phone it scrolls sideways with no cue. On a phone the
hero stage leaves a blank band under the video (`390x844/hero_01.jpg`). Contrast, focus rings,
keyboard transport and reduced motion work (E2E 8/8) and are kept.

**11. Performance (performance).** Viewers render on demand and release off-screen canvases
(E2E: at most 3 live). Each viewer allocates a 2048² shadow map; the X1 three-up holds three.
Budgets: JS 389.1 KB gzip of 520, CSS 5.2 KB gzip of 40, first view 3.95 MB of 6.5, all data 29.31 MB of 80. Nothing here
needs fixing; the pass must stay inside these budgets.

### 1.4 Findings per section

| Section | Rendering | Layout | Text | Coherence | Charts | Accessibility |
|---|---|---|---|---|---|---|
| hero | Robot reads well; table is a floating dark slab; cube small; shadows faint | Caption box top right competes with the nav; large empty left half above the title | Lede is setup only; run id in 11 px | No finding stated (finding 5) | none | Phone: blank band under the video |
| why | n/a (2D) | Schematic and tensor strip are balanced | Recorded-figure caption is one long paragraph | Clear | Tensor strip cells have 10 px labels and no legend except the caption | 10 px labels |
| lab | Default orbit crops the arm and bin on Can; dark on dark | Hole under the action chart; provenance table dominates | Provenance open by default; two closing notes | Clear | Action chart has no time axis | Run id 11 px |
| compare | Frontview default: robot small, floor empty | Two viewers and captions align | Caption and action note repeat on both sides | Clear, but the case chips do not say which case is the representative one | Strip has no failure-colour legend | ok |
| inside | Close-up starts too close on Lift (the hand fills the frame); path colours are clear | Dense right panel | Long lede and closing note; provenance list open, with a paragraph-long "path spread" row | Proposals and recorded path are labelled and distinct (keep) | Chunk chart key is 11 px | 10 px tick labels |
| findings | n/a | Title-below-chart; callout floats mid-height | Finding 1; KI-052 is not mentioned beside the H2-c and H4 numbers | Can/Lift tabs work; robustness is set apart and labelled | Finding 2, 8 | ok |
| followups | As compare; three-up is tiny | Finding 7 | Finding 1, 6; a "Method:" note in protocol jargon | Labels are present, but the reading of each study is buried under its intro note | Finding 8 | Three-up captions wrap to five lines |
| pixels | Point cloud overlay reads well; scene dark | Blank area beside the cloud legend | Chart captions repeat; training-cost table and erratum are long | H5 headline is clear (keep) | Placement maps are good | ok |
| methods | n/a | Two long columns | Finding 9 | Fine as reference | n/a | ok |

### 1.5 Fresh-context review of the baseline screenshots

Verbatim record: `logs/session20/visual_audit/before/fresh_review.md` (32 desktop tiles, 9 viewer
captures and 20 phone tiles read; no code, document or opinion given).

**The 30-second test (yes / partly / no), baseline:**

| Section | (a) task | (b) policy | (c) success | (d) finding |
|---|---|---|---|---|
| hero | partly | partly | partly | no |
| why | no | partly | no | no |
| lab | partly | yes | yes | no |
| compare | partly | yes | yes | partly |
| inside | partly | yes | partly | no |
| findings | partly | yes | yes | yes |
| followups | partly | yes | yes | partly |
| pixels | partly | yes | yes | yes |
| methods | yes | yes | no | no |

"Only Findings and Pixels pass a heading-plus-first-screen test." The tasks are first defined in
plain words in Methods, the last section. The hero names neither the policy nor a finding.

**Points the review added to §1.3:**
- The simulator's own render (inset, phone) shows a red can or cube on a light bin; the 3D
  reconstruction shows a yellow one on a dark bin. Nothing says they are the same episode.
- "7.000000000000001 cm" is printed twice in the placement-map captions (a formatting defect).
- Two table columns are both headed "seed-level"; the scrubber's cyan success mark and the
  trail have no legend; the translucent placeholder objects in the Can bins are unexplained.
- The Why tensor strip is 44 cells of about 58×36 px; the explorer's first and last denoising
  steps look the same in 3D because the default close-up hides the cube behind the hand.
- Phone: charts shrink as whole images until their labels are 4 to 6 px; "hover" instructions
  do nothing on touch; the hero title starts below the fold.
- Follow-up callouts are written in internal codes ("R1", "H4", "H2-c").
- Capture note: `inside_03.jpg` mostly shows the next section, and the −41.3 headline is split
  across `pixels_01` and `pixels_02`. The capture procedure is kept as is for the after run so
  that the two runs are comparable.

### 1.6 Where each required caveat is shown, before

| Caveat | Location before (visible by default unless noted) |
|---|---|
| Level 1 uses exact simulator state (privileged) | findings: note beside the success chart of each task ("Level 1 policies observe privileged simulator object state …"); inside: observation-window caption; pixels: lede |
| Simulation only | every clip caption ("MuJoCo simulation"); hero lede ("simulated Franka Panda"); footer ("Simulated robot control only; no physical robot was used") |
| "Not separated" is never read as equality | findings: two notes per task and every difference-chart caption; followups: lede and every difference-chart caption; pixels: RGB difference-chart caption |
| KI-052, the action-bound confound | followups, X1 intro and note only. **Not shown in findings beside the H2-c and H4 numbers** |
| Lift saturation, KI-034 | findings, Lift tab callout; pixels, RGB note |
| H5 negative | pixels: the red headline number, its heading and sentence |
| "DP3-inspired" | pixels: Level 2 intro line; methods: credits |
| X1 "exploratory, after the freeze" (validation regime, not held-out) | followups: two chips on the X1 heading; the callout text |
| X3 "separate replication" | followups: chip on the X3 heading; intro note |
| X2 (Can MH): beside the frozen Can PH result, never pooled, within-operator only, 500- vs 400-step episodes | followups: X2 heading, chips ("held-out, once", "within-operator only"), intro note (label and the two horizons) |
| X2 Can MH display label; recorded label in Provenance (KI-065) | followups: both only inside the closed "Provenance and training data" disclosure of each X2 clip. **The display label is not visible by default** |
| The paper and the addendum are local builds, not peer reviewed | methods credits: the paper ("a local build, not peer reviewed"). followups source caption names the addendum **without "not peer reviewed"** |

## Part 2. Plan (fixed before implementing)

No section is removed, the narrative order is unchanged, the §8.1 identity is unchanged (graphite
surfaces, warm ivory type, cyan for the learned policy under focus, one error red), and no npm
dependency, font, texture, environment map or other third-party asset is added. So no owner
question is needed before implementing (Session 20 prompt, step 3). `website/package-lock.json`
does not change.

| # | Change | Addresses |
|---|---|---|
| P1 | Charts: a visible title above each chart; one shared key per chart group (marks, intervals, "hollow = not separated, which is not evidence of equivalence"); captions keep only what is specific to the chart; axis title; tick collisions fixed; paired Level 1 / follow-up rows grouped; a failure-type legend on outcome strips; on phones charts keep a readable width and scroll sideways inside their own box | 1, 2, 8, phone |
| P2 | Scene: lighter table and bin, brighter object, stronger key and rim light, soft contact shadows, calmer grid, more floor separation. Styled materials and lights only; poses, meshes, transforms and the MuJoCo camera presets are untouched | 3 |
| P3 | Cameras: a new default three-quarter framing on the workspace for the lab, head-to-head and follow-up viewers; the explorer's close-up pulled back; MuJoCo's own cameras stay one click away | 4 |
| P4 | Hero: the caption names the policy, task and outcome in words; a short "what the research found" strip under the hero, every value and verdict read from `findings.json`, with its caveats | 5, 30-second test |
| P5 | Clip captions: one line (recorded rollout, policy, task, outcome, short source); the full run id, checkpoint hash, start, record, physics hash and checks move into a "Provenance" disclosure on every clip; the action-chart note is shown once per group | 6 |
| P6 | Lab: tasks described in plain words beside the task chips; provenance in a disclosure; the hole in the grid closed; the simulator inset labelled as the same episode in the simulator's own colours | 6, review |
| P7 | Follow-ups: each study opens with its labels and its reading; the 15-row clamped − unclamped chart, the two-interval tables, the method note and the episode-length table move into disclosures (the count "0 of 15 separate" stays visible); column headers disambiguated | 7 |
| P8 | Findings: KI-052 stated beside the H2-c and H4 numbers; ablation notes and the source table in disclosures | 1, caveats |
| P9 | Why: fewer, larger sampler cells (every second DDIM step, from the data), shorter caption | review |
| P10 | Methods: verification table and source list behind a one-line summary; prose tightened | 9 |
| P11 | Type and navigation: nothing under 12 px except SVG tick labels at 11 px; current-section state in the nav; scroll cue on phones; section spacing tightened; the float artefact fixed with the existing rounding helper | 10, review |
| P12 | Caveats: the X2 (Can MH) display label visible on every X2 clip group; the addendum named as a local build, not peer reviewed | §1.6 |

Not done, with the reason:
- The success axes stay 0 to 100 %. The review suggested zooming; a truncated axis would make
  small differences look large, and the random baseline sits at 0.
- The object keeps the restyled amber instead of the simulator's red: red is the site's one
  error colour. The inset is labelled instead.
- The translucent placeholder objects in the Can bins are recorded scene geometry and stay.
- Side-by-side comparison on phones: two viewers do not fit 390 px; the stacked layout stays.


## Part 3. What changed

All changes are in `website/` (source, styles, site tests), plus the audit tooling
(`scripts/visual_audit_montage.py`, `make web-audit`, `make web-verify-s20`,
`make web-verify-ext-s20`) and one section added to the generated `docs/VISUAL_PROVENANCE.md`.
`website/package-lock.json`, the exports, `website/public/data`, `paper/`, the README and
`CITATION.cff` are unchanged. No result, clip, number or verdict changed.

Before/after pictures, one per section and per viewer:
`logs/session20/visual_audit/montage/<section>.jpg` and `montage/viewers_<name>.jpg`
(`uv run --frozen python scripts/visual_audit_montage.py`). The full tile sets are in
`logs/session20/visual_audit/before/` (162 tiles) and `after/` (155 tiles; the page is shorter).

### 3.1 By plan item

| # | Done | Where to look |
|---|---|---|
| P1 | Every chart has its title above the marks. One reading key per section (Findings, After v1.0, Cameras & point clouds). Captions keep only what is specific to the chart. Difference axes are titled and never run past ±100 points. Level 1 and follow-up rows are drawn as banded pairs. Outcome strips have a failure-type legend. | `montage/findings.jpg`, `montage/followups.jpg` |
| P2 | Lighter table and bins, self-lit amber object, stronger key and rim light, soft contact shadows, a pool of light on the floor, calmer grid. The hero's full-width dark shade became a left-only shade, which is what had made the table look black. | `montage/viewers_hero_mid.jpg`, `montage/viewers_compare_mid.jpg` |
| P3 | New default "Workspace" camera (three-quarter view of the gripper, object and goal) for the lab, head-to-head, follow-up and point-cloud viewers; "Whole arm" and MuJoCo's own cameras are one click away. The explorer's close-ups are pulled back so the cube and the candidate paths are in frame. | `montage/viewers_lab_mid.jpg`, `montage/viewers_inside_mid.jpg` |
| P4 | Hero: the caption says in words what is shown (recorded rollout, simulation, policy and seed, task with a plain description, start, checkpoint, outcome, short source). Under it, "What the research found": three results read from `findings.json` (Can H2 and H2-c, Level 2 H5) with interval, verdict, a verdict-gated sentence and the caveats. The hero is shorter, so the three numbers are on the first screen at 1440×900. | `montage/hero.jpg`, `after/1440x900/hero_01.jpg` |
| P5 | One caption line per clip. Run id, checkpoint hash, start state, record, physics hash, simulator, platform and the recompute check are in a "Provenance" disclosure on every clip. The action-chart note is shown once per group. | `montage/lab.jpg`, `montage/compare.jpg` |
| P6 | Lab: each task is described in words beside the task chips; the simulator inset is captioned as the same episode in the simulator's own colours; the 3D view is tagged "3D reconstruction of the recorded states, restyled colours". | `montage/lab.jpg` |
| P7 | Follow-ups: each study opens with its question and labels, then its reading, then the charts. The 15-row clamped − unclamped chart, the two-interval tables, the method note and the episode-length table are in disclosures; "0 of 15 separate" and the 500- vs 400-step sentence stay visible. | `montage/followups.jpg` |
| P8 | Findings: KI-052 is stated beside the Can family comparison and beside H4. Ablation notes and the source table are in disclosures. | `montage/findings.jpg` |
| P9 | Why: every second DDIM step in larger cells; shorter caption. | `montage/why.jpg` |
| P10 | Methods: the 18-row verification table, the list of approximations and the frozen sources are behind one-line summaries. Section height 3,023 → 1,461 px. | `montage/methods.jpg` |
| P11 | No text under 12 px. The navigation marks the section being read and fades at the edge on narrow screens. "7.000000000000001 cm" is fixed with a rounding helper. | `after/1440x900/pixels_01.jpg` |
| P12 | The X2 (Can MH) training label is visible under every X2 clip group, with a pointer to the recorded label in Provenance (KI-065). The addendum is named "a local build, not peer reviewed". | `after/1440x900/followups_06.jpg` |

### 3.2 Changes beyond the plan
- **Phone charts.** The plan kept charts wide and scrolled them sideways. In the screenshots that
  left most plots off screen, so row charts (success rates, differences, failure types) now have
  a phone layout with each label above its row. Outcome strips, the latency scatter and the
  action channels still scroll inside their own box.
- **X2 start label.** X2 captions said "ad hoc evaluation start". They now say "X2 held-out
  start": the `x2_heldout` suite was recorded under the evaluator's `eval_adhoc` partition
  (D-123). Provenance shows both: "can_x2_heldout_NNNN (X2 held-out start; recorded partition
  eval_adhoc, D-123)".
- **Pose check wording.** "largest difference 0.0e+0 m over 37 clips" reads "identical in all 37
  clips" when the largest difference in the data is exactly 0.
- **Calibration renders** sit in one row under the failure-type bars instead of a tall stack
  beside an empty column.
- **After the round-1 review** (`after/fresh_review_round1.md`), eight of its points were fixed:
  the "3D reconstruction" tag moved off the arm to the lower-left corner; the "success" label
  sits above the timeline, and to the left of its tick near the end; the hero was shortened;
  the reading key is shown once per section instead of once per chart group; the 100-start X2
  strip wraps onto a second line at the cell size of the other strips; "Three arms, one start"
  became "Three conditions, one start"; the "−56 cm" and "+y" labels no longer collide; the
  clipped "gripper to cube" header and the one serif disclosure were fixed.
- **After the final review** (`after/fresh_review.md`), three more: the explorer's viewer is
  shorter, so the viewer, its caption and its legend fit one screen beside the sampler charts
  (the review's capture of the first denoising step had shown no chart for that reason); the
  doubled "Erratum (KI-049). Erratum (" is one; the viewer tag's background is more opaque.
  The lab viewer is slightly shorter so that the scene and the transport fit one screen.

### 3.3 What the 3D changes did not touch, and the re-verification
- Changed under `website/src/three/`: `Viewer.tsx` (lights, shadows, floor, fog, exposure, trail
  width), `robotScene.ts` (the colour table, three material parameters and one line opacity) and
  `cameras.ts` (the viewer's own presets: "Workspace" added, "hero" and "Whole arm" retuned,
  aspect-aware distance).
- Not changed: kinematics, transforms, frame conversion, `meshes.ts`, `mujocoCameraPose`, the
  MuJoCo camera presets, `src/verify/`, `fk.test.ts`. Meshes are still extracted at build time
  and not committed (D-090).
- Because files under `src/three/` changed, both viewer verifications were run again on the final
  build with new spec files that write to a new tree. `make web-verify` and
  `make web-verify-ext` were not run; the Session 10 and Session 18 results and montages were
  not written.

| Run | Output | Frames | Result | Robot IoU (min) | Object IoU (min) | Centroid (max) |
|---|---|---|---|---|---|---|
| `make web-verify-s20` | `reports/session20/observatory/viewer_verification.json` | 18 | pass | 0.9998 | 0.9970 | 0.04 px |
| `make web-verify-ext-s20` | `reports/session20/observatory/ext/viewer_verification.json` | 12 | pass | 0.9998 | 1.0000 | 0.03 px |

D-091 thresholds, unchanged: robot IoU ≥ 0.95, object IoU ≥ 0.85, centroid ≤ 1.5 px. Every frame
record equals the Session 10 or Session 18 record for the same frame.

### 3.4 Numbers and labels
- Every number on the page still comes from the generated data. The hero strip reads the three
  differences, intervals and verdicts from `findings.json`; its sentence is chosen by the
  verdict, never typed. No arithmetic was added beyond rounding for display and counting.
- Phrases that contain a digit and were already in the page's copy ("95 %", "seed 0", "3D")
  are used in the same sense. Layout constants (sizes, positions) are not results.
- Recorded states, proposals and illustrations keep their own chips ("recorded rollout",
  "proposal", "illustrative schematic, not data", "recorded sampler tensors"). Selection labels
  ("first success", "Case study: only diffusion succeeds", "rule-selected") stay with their
  clips.

### 3.5 Test changes
No existing assertion was changed, moved or removed, and no threshold or budget was touched
(`git diff website/e2e/site.spec.ts` has no deleted line). The existing provenance assertions
(`.h2h-pair.n2 .provenance` contains "KI-065" and the X2 training label) pass unchanged against
the provenance that now sits inside the disclosure.

| File | Change | Reason |
|---|---|---|
| `website/e2e/site.spec.ts` | New test "after the visual pass: caveats visible by default, each clip's source one click away (D-136)" | Pins what the text cuts and the Provenance disclosure must not lose. It reads the text a visitor sees without opening anything and checks each caveat of §3.6 in its section; the hero strip's three values (the third must equal the −41.3 headline of the point-cloud section); that every clip caption has a Provenance disclosure with "source: run_id"; that the hero and X2 disclosures are closed by default and, once opened, contain the run id, checkpoint, start state, physics hash, the recorded evaluator label and KI-065; the X2 start label with the recorded partition in Provenance; no unrounded float in the point-cloud section; the nav's current-section mark. |
| `website/e2e/site.spec.ts` | New test "phone layout: row charts fit the screen, labels above their rows" | New behaviour (§3.2): at 390 px the row charts of Findings do not scroll sideways and keep their text alternative. |
| `website/src/lib/format.test.ts` | New file, 4 tests | `startLabel` keeps every frozen partition's wording and names the X2 held-out suite; `cm` rounds without float noise. |
| `website/e2e/verify-s20.spec.ts`, `verify-ext-s20.spec.ts` | New files | The Session 10 and Session 18 verification specs with the output directory changed to `reports/session20/observatory/` (and the directory created before the first write). Same page, checks and thresholds. |
| `website/e2e/audit-s20.spec.ts` | New file (baseline commit), not changed since | Screenshots and word counts; the same file produced the before and after records. |
| `tests/unit/test_visual_provenance.py` | 1 test added | The Session 20 section is appended and everything above it is byte-identical. |

### 3.6 Where each required caveat is shown, after
Checked by the new E2E test on the text visible without opening anything.

| Caveat | Before | After (visible by default) |
|---|---|---|
| Level 1 uses exact simulator state (privileged) | findings, inside, pixels | unchanged, plus the hero strip ("policies that read exact simulator state (privileged information)") |
| Simulation only | clip captions, hero lede, footer | unchanged, plus the hero strip ("Simulation only") |
| "Not separated" is never read as equality | findings and followups notes and chart captions; pixels RGB caption | the reading key at the top of findings, followups and the point-cloud charts ("not separated: it includes 0, which is not evidence of equivalence"); the followups lede; the pixels RGB note; the hero strip, in the sentence of the not-separated card |
| KI-052, the action-bound confound | followups X1 only | followups X1 (unchanged); **added** in findings beside the Can family comparison and beside H4; **added** in the hero strip |
| Lift saturation, KI-034 | findings Lift tab; pixels RGB note | unchanged, plus the hero strip |
| H5 negative | pixels headline | unchanged ("separated, negative"), plus the hero strip's third card |
| "DP3-inspired" | pixels Level 2 intro; methods credits | unchanged, plus the hero strip's third card |
| X1 "exploratory, after the freeze" (validation regime, not held-out) | followups X1 chips and callout | unchanged |
| X3 "separate replication" | followups X3 chip and intro | unchanged |
| X2 (Can MH): beside the frozen Can PH result, never pooled, within-operator only, 500- vs 400-step episodes | followups X2 heading, chips, intro | unchanged: one intro paragraph with all four ("reported beside the frozen Level 1 Can PH result, never instead of it, never pooled; within-operator only … success before the 500-step horizon … Level 1 episodes end at 400 steps, so the two success scales differ") |
| X2 Can MH display label; recorded label in Provenance (KI-065) | both only inside a closed disclosure | **the display label is now visible** under each X2 clip group, with "(KI-065)"; the recorded evaluator label is in each clip's Provenance |
| The paper and the addendum are local builds, not peer reviewed | methods (paper); followups named the addendum without it | methods (unchanged); followups: "a local build, not peer reviewed" **added** for the addendum |

## Part 4. After measurements

### 4.1 Words visible by default (`after/word_counts.json`, same method as §1.1)

| Section | Reading words, before | after | Visible words, before | after | Words in closed disclosures, before | after | Height px, before | after |
|---|---|---|---|---|---|---|---|---|
| hero | 71 | 208 | 75 | 213 | 0 | 61 | 828 | 1,037 |
| why | 171 | 146 | 204 | 170 | 0 | 0 | 951 | 893 |
| lab | 185 | 147 | 240 | 204 | 0 | 61 | 1,235 | 1,176 |
| compare | 193 | 174 | 239 | 228 | 0 | 121 | 1,093 | 1,097 |
| inside | 316 | 219 | 401 | 310 | 169 | 335 | 1,453 | 1,280 |
| findings | 625 | 491 | 1,001 | 850 | 302 | 366 | 3,133 | 2,994 |
| followups | 1,256 | 909 | 2,099 | 1,380 | 508 | 1,076 | 5,804 | 5,193 |
| pixels | 671 | 594 | 898 | 829 | 0 | 122 | 3,957 | 3,506 |
| methods | 803 | 583 | 976 | 607 | 0 | 415 | 3,023 | 1,461 |
| **page, 1440×900** | **4,291** | **3,471** | **6,133** | **4,791** | 979 | 2,557 | **21,477** | **18,637** |
| page, 390×844 | 4,284 | 3,371 | 6,174 | 4,743 | 979 | 2,557 | 36,575 | 36,309 |

- Reading words fell by 820 (19 %), all visible words by 1,342 (22 %), page height by 13 % on
  desktop. Without the hero, reading words fell from 4,220 to 3,263 (23 %).
- The hero grew on purpose: it now states the three headline results and their caveats, which
  the baseline review found missing.
- Small-print blocks went from 86 to 77; closed disclosures from 8 to 24. Nothing was deleted
  that carries provenance or a caveat: it is either visible or one click away.
- On phones the page is about as long as before: the row charts are taller in their phone
  layout, in exchange for being readable.

### 4.2 Checks on the final build
| Check | Result | Log |
|---|---|---|
| `make web-data`, `make web-assets` | 161 + 74 data files, 61 meshes; `website/public/data` unchanged | `logs/session20/web_data_after.log`, `web_assets_after.log` |
| `make web-build` (D-088 budgets, none loosened) | JavaScript 391.8 of 520 KB gzip; CSS 6.1 of 40 KB; fonts 309.6 KB of 1.60 MB; meshes 3.00 of 4.00 MB; all data 29.31 of 80 MB; first view 3.95 of 6.50 MB; every per-file limit passes | `web_build_after.log` |
| `make web-test` | 79 passed (75 before, 4 new) | `web_test_after.log` |
| `make web-e2e` | 10 of 10 (8 before, 2 new) | `web_e2e_after.log` |
| `make web-verify-s20`, `make web-verify-ext-s20` | pass, 18 and 12 frames (§3.3) | `web_verify_s20.log`, `web_verify_ext_s20.log` |
| Accessibility, by the E2E suite and the captures | keyboard path, reduced motion, no horizontal scroll at 390 px, no-WebGL and missing-artifact messages, at most 3 live canvases: all pass; `after/reduced_motion/`, `after/nowebgl/` | `web_e2e_after.log` |

Lighthouse was not run (optional in the Session 20 prompt).

### 4.3 The 30-second test, before and after
Fresh-context reviews, screenshots only, same ten questions. Verbatim records:
`logs/session20/visual_audit/before/fresh_review.md` (baseline),
`after/fresh_review_round1.md` (first after-capture) and `after/fresh_review.md` (after the
round-1 fixes). Each cell is before → after (final review).

| Section | (a) task | (b) policy | (c) success | (d) finding |
|---|---|---|---|---|
| hero | partly → **yes** | partly → **yes** | partly → partly | no → **yes** |
| why | no → no | partly → partly | no → no | no → no |
| lab | partly → **yes** | yes → yes | yes → yes | no → no |
| compare | partly → **yes** | yes → yes | yes → yes | partly → partly |
| inside | partly → partly | yes → yes | partly → **yes** | no → no |
| findings | partly → **yes** | yes → yes | yes → yes | yes → yes |
| followups | partly → partly | yes → yes | yes → yes | partly → **yes** |
| pixels | partly → partly | yes → yes | yes → yes | yes → yes |
| methods | yes → yes | yes → partly | no → no | no → no |

- Counts over the 36 cells: yes 15 → 22, partly 13 → 7, no 8 → 7.
- The hero now passes three of four, and the visitor first learns the result on the first
  screen ("What the research found"), not five sections in.
- "Why diffusion" is unchanged by design: it is a labelled schematic of the mechanism, not a
  result. Methods (b) moved from yes to partly with no change to that text; two reviewers read
  the same paragraph differently.
- The reviews judge "recorded versus illustration" as mostly clear in both after-reviews ("the
  badges are the page's strongest feature", round 1).

### 4.4 What the final review still finds, and what was done
Fixed after it (§3.2): the explorer viewer taller than the screen; the doubled erratum prefix.

Left as they are, with the reason:
- **The "Workspace" camera crops the upper arm.** It is the trade the baseline review asked for
  (fill the panel with the gripper and the object); "Whole arm" is the next chip.
- **Success axes run 0 to 100 %** (§2, "Not done"): a zoomed axis would exaggerate differences
  that the intervals do not separate.
- **Comparison viewers open at the first frame**, where the panels look alike. Opening on a
  later frame would be a choice of frame by eye; the start is the one thing the two episodes
  share by construction.
- **Internal codes** (H1 to H5, R1, KI and D numbers) remain in the callouts and notes. They
  are the names the paper, the addendum and the repository use, and the R1 and H2-c readings are
  copied from the pinned results. The X1 reading is now preceded by a plain sentence.
- **The object is amber in 3D and red in the simulator's render** (§2); the inset caption and
  the viewer tag say which is which.
- **Phone:** stacked comparison viewers, the long follow-ups section, the small sampler grid in
  "Why diffusion", and the hero's simulator still before the title are unchanged in structure.
- **Empty right columns** beside the ablations, under the X2 chart and beside the explorer's
  observation window.
- **The billing table, erratum and calibration renders** stay in the point-cloud section, next
  to the study they belong to.
- **Hero result sentences** sit just below the first screen at 1440×900; the three numbers, their
  labels and intervals are on it.

Candidates for a later pass, none started: a phone layout for the sampler grid; plain-word
glosses for the hypothesis codes; a selection rule for an opening frame of the comparison
viewers; filling or removing the empty right columns.

