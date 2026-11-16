# Fresh-context review of the after screenshots (Session 20, step 5)

A subagent with a fresh context was given only the screenshot files under
`logs/session20/visual_audit/after/` (all 29 tiles of `1440x900/`, the 9 files of `viewers/`,
and 20 tiles of `390x844/`) and the same ten questions as the baseline review
(`../before/fresh_review.md`): MASTER_SPEC §8.1's 30-second test per section, and what is
confusing, redundant, cluttered, badly rendered or hard to read. It was told not to read code,
documents or data, and it was given no opinion or plan. Its answer follows verbatim.

This is the second after-review. The first (`fresh_review_round1.md`) led to eight fixes; this
one was run on the screenshots captured after them. Three small fixes followed this review
(docs/OBSERVATORY_VISUAL_AUDIT.md §4.4): the explorer's viewer was made shorter so that it fits
one screen, which is why `viewers/denoise_first_step.jpg` showed no chart; the doubled
"Erratum (KI-049). Erratum (" was reduced to one; the viewer tag's background was made more
opaque. The screenshots in this directory were captured once more after those three fixes.

---

REVIEW RECORD: first-time visitor, screenshots only

Scope: I read all 29 desktop tiles, all 9 viewer captures and 20 phone tiles. Not captured: the tile between findings_02 and findings_03 (start of the robustness part, lower half of the cost scatter). viewers/denoise_first_step.jpg shows no chart panel at all.

1. 30-SECOND TEST (a task / b policy / c success / d finding)
- hero (hero_01, hero_02, hero_mid): a Yes, small chip "task: Lift (grasp a cube and lift it off the table)". b Yes, chip "Diffusion U-Net, seed 0". c Partly, "outcome: success at step 43" is small cyan text among five chip rows. d Yes, three big numbers, but their sentences sit below the fold.
- why (why_01): a No, abstract gripper/obstacle/goal schematic. b Partly, "the trained policy". c No. d No, it says "not a measurement".
- lab (lab_01, lab_02, lab_mid): a Yes. b Yes. c Yes, heading "Diffusion U-Net on Can: success". d No.
- compare (compare_01, compare_02, compare_mid): a Yes. b Yes, titled panels. c Yes, "success at step 121" vs "failure, 400 steps". d Partly, one case study plus a 50-square strip.
- inside (inside_01, inside_02): a Partly, "task: Lift" appears only under the viewer. b Yes. c Yes, but under the viewer. d No.
- findings (findings_01 to 04): a Yes. b Yes. c Yes, "95.3 %, 143/150". d Yes, callout "Can separates the families".
- followups (01 to 07): a Partly, tasks named but not described. b Yes. c Yes. d Yes, but callouts are worded in codes ("R1", "H4 gap", "H2-c").
- pixels (01 to 05): a Partly. b Yes. c Yes. d Yes, "−41.3 points: Point clouds generalize worse when the cube moves".
- methods (01, 02): a Yes, first sentence. b Partly. c No. d No.

2. CONFUSING
- hero: "points" (of what?), "separated", "BC-MLP-chunk", "DP3-inspired", "KI-052/KI-034", "privileged", checkpoint hash, "source: s6_diffusion_unet_lr3e4…", "Camera drift on".
- why: 24 mini-plots with four mark types and no y-scale; "t = 90"; "sampler tensors"; "validation start" used before "validation" is defined.
- lab: "start 0000, seed 0", "MuJoCo agentview", "Δroll", "DDIM-10, 16-step chunks, 8 executed", "PickPlace._check_success". The can is gold in 3D but red in the inset.
- compare: "one clock"; "the first one, in held-out order, that fits the case"; "Scrubber ticks: diffusion's decisions". Six failure colours with near-identical orange/amber and red/pink. White outline ("start shown") collides with coloured outline (failure).
- inside: three timelines (decision slider, sampler slider, playback scrubber) with no stated relation. "Run the sampler", "16 other seeds", "cloned simulation", "Export, not a result", "t−1/now" grid, "quat". The legend wording changes between sampler steps (denoise_first_step vs denoise_last_step).
- findings: H1/H2/H2c/H3/H4, "sep. +", "Level 1", "arm offset", "base", "A1 primary − raw weights", "T_o", "DDPM-100", "An order of estimates is not a ranking", "Highest success: arm offset: BC-MLP-chunk, 79.3 %".
- followups: order runs X1, X3, X2. "clamped", "bound", "frozen", "Can MH/PH", "within-operator only", "validation regime", "D-107/D-134/KI-065".
- pixels: "Level 2", "84² cameras" vs "84-pixel", "9-D proprioception", partial-pie discs, "Segments", struck-through 24.62, a billing table.
- methods: "canonical container (linux-aarch64-container-1f685638f28c)", "index 5bc54c58d893", "IoU ≥ 0.9998".

3. REDUNDANT
- hero: "recorded rollout" pill plus "Every rollout on this page is a recorded evaluation episode". Card 2 says "not separated" twice.
- why: intro and schematic caption repeat each other; the grid returns in Inside.
- lab: success is shown four times (episode pill, heading, outcome chip, scrubber flag), the task three times, "recorded" four times.
- compare: the 50-start strip reappears in findings_02; each panel carries both "recorded rollout" and "3D reconstruction…".
- inside: "proposal … run in a clone" appears in the legend and the footnote.
- findings: the mark-legend box is identical in findings_01 and followups_01. The KI-052 confound note is in hero_02, findings_01 and findings_03. The failure legend appears twice in findings_02.
- followups: "never pooled" in 01, 04 and 05; "start is fixed by rule (D-134)" in 02 and 05; +18.7 for the fourth time.
- pixels: −41.3 again; failure legend twice in pixels_04; "Erratum (KI-049). Erratum (" doubled in pixels_05.
- methods: the viewer-check sentence repeats followups_07; the footer repeats the credits.

4. CLUTTERED OR EMPTY
- hero: five chip rows compete with the headline. The third card's title wraps, so −41.3 sits lower than the other two numbers.
- why: four-line run-on caption.
- lab: four control rows push the viewer to mid-page; the first viewport shows only the arm's shoulder (lab_01). The action chart leaves about a third of the width blank (lab_02).
- compare: three flat Δroll/pitch/yaw rows per panel; blank band under the legend (compare_02).
- inside: the right column ends early and all of inside_02 is left-only.
- findings: findings_02 packs forest plots, 650 squares, bars and a scatter into one viewport. The 0 to 100 axis piles learned policies at 95 %. The H1 chart is four near-identical rows. The right column is empty in findings_03.
- followups: blank right half in followups_05 and under the X2 chart; the X2 strip wraps to two lines per row (followups_07).
- pixels: pixels_04 mixes strip, bars, photos and a billing table; the section ends on an erratum box (pixels_05).
- methods: lower right empty (methods_01).

5. BADLY RENDERED
- hero: black pedestal and dark table merge with the background; the cube is tiny and half-hidden by the gripper; the Pause/Camera buttons sit on the table edge.
- why: dots are clipped at cell borders.
- lab: the "Workspace" camera crops the upper arm (lab_01, lab_mid); bins are flat grey; the arm base is black on black.
- compare: at t=0 both panels are identical (compare_01), so nothing differs until played.
- inside: the gripper fills the frame and the arm is cropped; the floor is a pale blue-grey unlike every other viewer; the viewer is taller than the viewport. denoise_first_step.jpg has an empty right column, so the chart is either absent or not captured.
- findings: seed circles collide with each other and the mean dot (findings_01). In the scatter, "BC-MLP-chunk" sits on a gridline and a line runs off-plot (findings_02).
- followups: three small viewers are identical at t=0 (followups_02). In followups_x1_mid thick cyan trails scribble over the arm and the cube disappears in the middle panel. The action charts are tiny.
- pixels: glaring white calibration photos with microscopic baked-in headers (pixels_04); orange cloud dots on a gold cube (pixels_mid).
- methods: the reference column starts above "Credits" (methods_02).

6. HARD TO READ
- hero: small grey chips and footnote.
- why: caption; phone grid labels are a few pixels high.
- lab: chips; flat grey lines.
- compare: small chart labels.
- inside: mini-chart titles and legend are the smallest text on the page.
- findings: grey "H1" prefixes and "sep. +".
- followups: three-up chart labels; SHA table.
- pixels: erratum box; table.
- methods: small grey licence paragraph.

7. NARRATIVE FLOW
The order is sound: result, motivation, watch, compare, mechanism, evidence, follow-ups, vision, methods. The visitor first learns the result in "What the research found" at the bottom of hero_01, but must scroll to hero_02 for the sentences. Weak points:
- The mechanism is told twice (Why, Inside) with two sections between.
- "held-out", "validation", "seed" and "start" are used from the hero onward and defined only in Methods.
- The largest effect (−41.3) is headlined in the hero but lives in section 8.
- The nav label "After v1.0" is opaque.

8. TEN CHANGES, RANKED
1. lab_01: put the whole viewer and the uncropped arm in the first viewport; move the chips below it.
2. compare_01, followups_02: open on the decisive frame, not identical t=0 frames.
3. 390 why_02: rebuild the denoising grid for phone; it is illegible.
4. findings_01: zoom the success axis and separate the seed circles.
5. hero_01, lab_01: fold checkpoint hash and source id into "Provenance".
6. followups_05, inside_02, findings_03: close the empty right columns.
7. inside_01: enlarge the mini-charts and move the legend out of the grid; verify the first-step state (denoise_first_step).
8. pixels_04, pixels_05: move calibration photos, billing table and erratum to Methods.
9. followups_03 vs findings_02: use one strip encoding; thin the trails (followups_x1_mid).
10. hero_01: align the three cards, lift their sentences above the fold, and separate the table from the background.

9. RECORDING vs ILLUSTRATION vs PROPOSAL
Mostly yes. "recorded rollout", "illustrative schematic, not data", "recorded sampler tensors", "proposal" and "3D reconstruction of the recorded states, restyled colours" are explicit. It is unclear in these places:
- inside: the "recorded rollout" pill sits under a viewer that also draws proposal paths, and the "validation, never held-out" status is in small grey text.
- Phone (hero_01, lab_02): the still is a simulator render with no caption saying so. In lab_02 the inset repeats the same picture.
- pixels_04: the calibration photos come from "a training demonstration, not of these episodes", said only in the caption.
- Status pills ("exploratory", "held-out, once") share one shape with provenance pills.
- Cyan means diffusion, success, separated, proposal and point-cloud alike.
- Nothing is labelled a prediction except "Recorded, not predicted".

10. PHONE (390 px)
Works:
- Single column, result cards (hero_02), pills that wrap.
- Dot and forest plots with labels above rows (findings_02, followups_02, followups_10).
- The opt-in "Load the 3D reconstruction".

Does not:
- The nav is cut at "Head to hea" with no scroll cue (every tile).
- The headline is below the fold, behind an image and six chip lines (hero_01).
- Orphan "Pause / Camera drift on" buttons control nothing visible (hero_02).
- The denoising grid is unreadable (why_02).
- Stills are grey t=0 frames where the can is a few red pixels (lab_02, compare_02).
- Stacked panels cannot be compared, which defeats the head-to-head (compare_02, followups_05).
- The section lengths are punishing: followups alone is 15 tiles.
