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

