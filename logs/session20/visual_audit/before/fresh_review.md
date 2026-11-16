# Fresh-context review of the baseline screenshots (Session 20, step 2)

A subagent with a fresh context was given only the screenshot files under
`logs/session20/visual_audit/before/` (all 32 tiles of `1440x900/`, the 9 files of `viewers/`,
and 20 tiles of `390x844/`) and ten questions: MASTER_SPEC §8.1's 30-second test per section,
and what is confusing, redundant, cluttered, badly rendered or hard to read. It was told not to
read code, documents or data, and it was given no opinion or plan. Its answer follows verbatim.

---

REVIEW RECORD: first-time-visitor visual audit, "before" screenshots
Base: /Users/zackaryholt/Projects/roboforge/logs/session20/visual_audit/before/
Read: all 32 desktop tiles, all 9 viewer captures, 20 phone tiles (first two per section, followups_05, followups_10). Unprefixed names are 1440x900; "390/" is phone; "viewers/" is viewer captures.
Capture gaps: inside_03.jpg shows Findings charts, so the end of "Inside" is not captured on desktop. The "−41.3" callout falls between pixels_01 and pixels_02 and was judged from 390/pixels_02.

**HERO** (hero_01, viewers/hero_mid)
1. (a) Partly: "task: Lift" is only in small grey meta top-right, and the cube is a ~35 px tan block under the gripper. (b) Partly: only "trained checkpoint 7e487665"; "diffusion_unet" is buried in the source string. (c) Partly: "outcome: success at step 43 (2.15 s)" is small cyan text; nothing in the scene signals success. (d) No: the paragraph describes the setup and gives no result.
2. "Frozen evaluations", "proposals and schematics are labelled as such", the checkpoint hash, and "Camera drift on" (state or action?).
3. The source run-id string adds nothing to the meta line above it.
4. Not dense; the top-right third is empty.
5. The robot sits bottom-right with the table cropped by the frame edge. Table, pedestal and base are near-black on a near-black background. The gripper partly hides the cube. Arm shells are visibly faceted. No trail.
6. Source string (~10 px grey) and the meta line.

**WHY** (why_01, why_02)
1. (a) No: the schematic shows gripper, obstacle and goal, which is not Lift or Can. (b) Partly: it contrasts regression with diffusion generically. (c) No. (d) No: the caption says "not a measurement".
2. The 44-cell tensor grid has no y-axes. The caption uses "DDIM", "x_t", "replayed bitwise", "not a held-out result". The schematic argues multi-modality, and the caption then says the findings do not measure modes.
3. The pill "illustrative schematic, not data" and the caption sentence say the same thing. The grid duplicates the explorer in "Inside".
4. Grid cells are ~58×36 px; the sand "16 other samples" are invisible.
5. The schematic is fine. Grid labels are tiny and only the output column stands out.
6. "t = 90…" and Δx labels (~9 px); the caption is 13 px grey at ~110 characters per line.

**LAB** (lab_01–03, viewers/lab_mid)
1. (a) Partly: "Can" is never explained until the footnote at the bottom of lab_03 ("the can rests inside the goal bin"). (b) Yes: "Diffusion U-Net on Can: success". (c) Yes: heading plus cyan outcome. (d) No.
2. Pills like "start 0000, seed 0, first success". The inset MuJoCo render shows a red can and light-wood bin, while the main view shows a yellow can and dark bin. Translucent ghost objects sit in the right-hand bins with no explanation. "physics_sha256" and "Recompute check |Δpos| 0.0e+0 m".
3. Policy, checkpoint, task and outcome appear in the heading, meta line, source line and table. The run id is printed three times in lab_03. "Recorded" is asserted three times.
4. Five pill rows (task, policy, episode, 7 cameras, 4 overlays) plus a 9-row provenance table. There is a ~190 px void under the action plot in lab_03.
5. The default camera cuts off the robot's top, and the viewport edge cuts the can at frame 0 (lab_01). Dark-brown bins sit on black with heavy black shadows. The viewer is taller than the screen, so the scene and transport controls are never visible together. The trail is clear (viewers/lab_mid).
6. Source line, inset caption, action-plot labels.

**COMPARE** (compare_01–02, viewers/compare_mid)
1. (a) Partly: a "Can" pill only. (b) Yes: colour-barred headers "Diffusion U-Net" and "BC-MLP". (c) Yes: "success at step 121" and "failure, 400 steps". (d) Partly: the 50-square strips suggest BC-MLP fails more, but no sentence says so.
2. The intro is selection-rule procedure, not meaning. The failure outline colours have no legend ("hover"). There is an unexplained cyan marker at ~30% of the scrubber. The trail has no legend.
3. Outcome is stated in the header and again in the meta line. The action caption is repeated verbatim under both panels.
4. Moderate.
5. At the start the two panels look identical. The robot is ~80 px wide in a 610 px panel and the can ~15 px. Once placed, the dark-gold can vanishes in the dark bin (viewers/compare_mid, left). In the failure panel the can is hidden behind the gripper. The top third of each panel is empty.
6. Strip caption (~200 characters on one 13 px line); source strings.

**INSIDE** (inside_01–02, viewers/inside_mid, viewers/denoise_first_step, viewers/denoise_last_step)
1. (a) Partly: a "Lift" pill and a visible cube at frame 0. (b) Yes: the intro names the diffusion policy. (c) Partly: only the pill and a table cell say "success at step 47". (d) No.
2. The intro is jargon ("16-step × 7-dimension", "DDIM", "bitwise", "cloned simulation"). There are two look-alike sliders plus a "Run the sampler" button. The "t−1/now" heat strip. "Proposal" labels a chunk that was partly executed.
3. Three disclaimers: the "What this is" row, the "Export, not a result" paragraph, and the validation label.
4. Small multiples are ~130×45 px with an 8 px legend, and the page shows two tables' worth of text.
5. In the default "Close, from above" view the gripper fills half the frame and hides the cube in the mid-playback captures. The first and last denoise steps are almost indistinguishable in 3D, so the main payoff is invisible. The path is thin and short. inside_02 opens with ~150 px of empty floor.
6. Small-multiple titles, the "t=90…out" labels, the faint sand band.

**FINDINGS** (findings_01–04)
1. (a) Partly: Can/Lift toggles with no task description. (b) Yes. (c) Yes: rates with counts. (d) Yes: "Can separates the families… beats the one-step MLP by 18.7 points".
2. "Separates the families", "hierarchical paired bootstrap", H1/H2/H2c/H3, "sep. +", "T_o = 2 − T_o = 1", "RESEARCH_PROTOCOL §14". The failure-colour legend sits below the grid it explains. The latency scatter mixes circles and squares, uses a log axis, and starts y at 70%.
3. The forest-plot caption boilerplate ("Diamond… Filled… hollow… not evidence of equivalence") is repeated under every forest plot here and in later sections, plus a "'not separated' means…" note under each callout. The H1-vs-random chart restates the success chart. The SHA table recurs in followups_08 and Methods.
4. findings_02 packs a 13×50 grid, stacked bars and a scatter into one screen. The callout sits beside the charts, not above them.
5. In the dot plot, rings and estimate dots pile up at 95–96% while three quarters of the 0–100 axis is empty. Axis ticks collide ("−125−100", "+100+125", findings_01). Cyan means diffusion, "separated" and success. The 12 px outline colours are hard to tell apart.
6. "H1" prefixes, "sep." tags and axis ticks (9–10 px).

**FOLLOW-UPS** (followups_01–08, viewers/followups_x1_mid, viewers/followups_x2_mid)
1. (a) Partly. (b) Yes. (c) Yes: per-panel headers. (d) Partly: the callouts are written in internal codes ("R1", "H4", "H2-c") and X2's arrives only at the very end (followups_07–08).
2. The order is X1, X3, X2. Codes: KI-052, D-107, D-111, D-134, "Can MH/PH", "within-operator only", "clamped". Two columns are both headed "seed-level" (followups_05).
3. The clamped rows are identical to the unclamped rows (100.0/100.0, 54.7/54.7, 48.7/48.7). There is a 15-row chart in which 13 rows read "+0.0 pp not sep." and the other two "−0.7 pp" (followups_02). X3's forest plot and table show the same four contrasts. "Exploratory, after the freeze; the frozen numbers stand" appears twice in followups_01.
4. Eight desktop screens (15 on phone) hold ~10 charts, 3 tables and 2 viewer groups. followups_02 has a ~350 px empty left column.
5. The triple viewer has ~60 px robots and a ~10 px cube; only the trail scribble is readable (viewers/followups_x1_mid). Source strings break mid-word across three lines. The X2 strips have ~8 px squares and ~7 px labels (followups_07).
6. Action-plot labels in the triple view; strip row labels.

**PIXELS** (pixels_01–05, viewers/pixels_mid)
1. (a) Partly: "Lift" and "cube" appear in captions. (b) Yes: "Point-cloud diffusion" and "Visual diffusion". (c) Yes. (d) Yes: "−41.3 points: Point clouds generalize worse when the cube moves". This is the clearest statement on the page.
2. The nav says "Beyond pixels" and the heading "Beyond privileged state". "Level 1/2" are undefined. Here sand means visual diffusion and cyan means point cloud, reversing the page's coding, and the cyan-barred panel is the failure. The pie-filled discs. "7.000000000000001 cm" is printed twice (pixels_02).
3. The RGB study uses two charts for two near-100% numbers. There are three near-identical calibration frames.
4. pixels_03–05 leave ~600 px of empty left column beside the image stack.
5. These are the best-framed viewers on the page (viewers/pixels_mid), though the robot head is cropped. The white RGB thumbnails and light calibration images glare against the dark theme. The burned-in calibration captions are ~5 px. The scatter has no y ticks.
6. Calibration captions; the struck-through "24.62".

**METHODS** (methods_01–04)
1. (a) Yes: "Lift (pick up a cube) and Can (place a can in its bin)", the first plain definition, in the last section. (b) Yes: four families are named. (c) No. (d) No.
2. "0.0e+0 m", "canonical container", "DECISIONS D-089", hashes.
3. The frozen-sources list appears a third time. An 18+ row table reads all "pass" at ~1.000.
4. methods_02 has a ~750 px empty left column.
5. Clip names wrap to two lines.
6. "Frozen sources" and licence paragraphs (~12 px grey serif).

**7. Narrative flow**
The order is sensible: motivation, watch, compare, mechanism, results, follow-ups, harder inputs, methods. The result arrives five sections in, and the hero never states it. The Lab and Compare intros explain selection rules instead of what to look for. "After v1.0.0" and "Beyond privileged state" presuppose insider vocabulary. Only Findings and Pixels pass a heading-plus-first-screen test.

**8. Ten changes, by visible impact**
1. hero_01.jpg: centre and enlarge the arm and cube, lighten the table, and replace the meta and hash with a plain caption plus the headline finding.
2. viewers/compare_mid.jpg: raise scene contrast (lighter bin and table, brighter object, rim light) so the can never disappears.
3. compare_02.jpg and followups_03.jpg: default cameras should fill the panel with gripper and object.
4. viewers/inside_mid.jpg: pull the camera back so the cube and candidate paths are visible, and thicken the paths.
5. findings_01.jpg: put the takeaway above the charts, zoom the success axis, and fix the colliding ticks.
6. lab_03.jpg: collapse the provenance table, source strings and hashes behind one disclosure, and remove the void.
7. findings_03.jpg: use one shared forest-plot key instead of the repeated caption.
8. followups_02.jpg: delete the all-zero chart and duplicate rows, and lead each study with a plain-language result.
9. why_01.jpg: show 3–4 enlarged denoising steps instead of 44 micro-cells.
10. pixels_03.jpg and methods_02.jpg: fix the one-column voids, the float artefact and the light image stack.

**9. Recorded versus illustration**
- The desktop shows a restyled dark scene with a yellow can or cube, while the inset and phone show a MuJoCo render with a red one. A visitor cannot tell which is "the recording", or may think they are different episodes.
- "Proposal" labels a chunk that was partly executed, and the hero uses "proposals" in another sense.
- The Why schematic has equal visual weight to the recorded data beside it and implies a multi-modality benefit that the results do not test.
- Validation-regime content (Inside, "Can under perturbation", X1) looks identical to held-out content, marked only by small pills.
- The ghost objects in the bins and the unlabelled trails could be read as predictions.
- The calibration frames come from a training demonstration but sit beside the failure analysis.
- The follow-up strips use red outlines for "failure" where the earlier strips use colour for failure type.

**10. Phone**
- The nav shows three of eight links with no scroll cue (390/hero_01).
- The hero image, meta and source string all precede the H1, which starts at y≈630.
- "Pause" and "Camera drift on", plus 11 camera/overlay pills, appear before any 3D is loaded (390/hero_02, 390/lab_02).
- The lab inset duplicates the image it sits on (390/lab_02).
- Charts are shrunk as whole images, with labels at ~4–6 px and grid squares at ~5 px (390/findings_02, 390/why_02, 390/inside_02).
- "Hover" instructions are useless on touch.
- The contrast table is clipped at the right edge, and its caption is cut mid-word (390/followups_10).
- Compared viewers stack vertically, so "same start" can never be seen side by side (390/compare_02, 390/followups_05).
- Captions are larger than the charts they describe.
