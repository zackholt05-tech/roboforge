# Fresh-context review of the after screenshots, round 1 (Session 20, step 5)

A subagent with a fresh context was given only the screenshot files of the first after-capture
(all 29 tiles of `1440x900/`, the 9 files of `viewers/`, and 20 tiles of `390x844/`) and the same
ten questions as the baseline review. It was told not to read code, documents or data, and it was
given no opinion or plan. Its answer follows verbatim.

Eight of its points were then fixed (docs/OBSERVATORY_VISUAL_AUDIT.md §3.2), the screenshots in
this directory were captured again from the final build, and the same review was run once more on
them (`fresh_review.md`). The tiles this record cites are therefore the earlier capture of the
same procedure; the tile names are the same, the contents differ where a fix was made.

---

REVIEW OF THE "AFTER" SCREENSHOTS (first-time visitor, images only)

I read all 29 desktop tiles, the 9 viewer captures and 20 phone tiles. Tiles why_02, lab_03, findings_04 and pixels_05 mostly show the next section's header. denoise_first_step.jpg shows only the viewer's bottom edge and a legend, so the first denoising step is not captured.

1. 30-second test (a task, b policy, c success, d finding)
- hero (hero_01, hero_02, hero_mid): a Partly, "task: Lift" chip and a small cube, but nothing says "pick up the cube". b Yes, "Diffusion U-Net, seed 0". c Yes, cyan "outcome: success at step 43", in small corner text. d Partly, the three result cards sit below the fold.
- why (why_01): a No, abstract gripper/obstacle/goal sketch. b Partly, "trained policy's first decision". c No. d No, and it says so.
- lab (lab_01, lab_02, lab_mid): a Yes, "Can: pick up a can and put it in its bin". b Yes. c Yes, heading ends ": success". d No.
- compare (compare_01, compare_02, compare_mid): a Yes. b Yes. c Yes, cyan "success at step 121" against red "failure, 400 steps". d Partly, one case study plus a 50-square strip.
- inside (inside_01, inside_02, inside_mid): a Partly, chip only. b Yes. c Yes, but the outcome chip is under the viewer. d No.
- findings (findings_01 to 03): a Yes. b Partly, families named, no robot. c Yes, as rates. d Yes, "Can separates the families", though mid-screen in the right column.
- followups (followups_01 to 07, x1_mid, x2_mid): a Partly. b Yes. c Yes. d Partly, "The bound does not close the gap" is wrapped in R1/H4/KI-052/D-107.
- pixels (pixels_01 to 04, pixels_mid): a Partly. b Yes. c Yes. d Yes, red "-41.3 points, Point clouds generalize worse when the cube moves".
- methods (methods_01, methods_02): a Yes, the clearest task description on the page, placed last. b Yes. c No. d No.

2. Confusing
- hero: "points" (percentage points?), "separated", "privileged information", KI-052, KI-034, checkpoint hash.
- why: "sampler tensors", "t = 90 ... 10", three mark types explained only in a four-line caption.
- lab: "chosen by rule", "MuJoCo agentview", "DDIM-10, 16-step chunks, 8 executed"; the action chart has no value or time axis.
- compare: "one clock", yet the two action charts span different lengths (121 vs 400 steps) and the playhead appears only on the right (compare_mid).
- inside: "validation, never held-out" after two sections of "held-out"; "cloned simulation", "controller units", an empty dashed "t-1" row, "Export, not a result".
- findings: H1/H2/H2c/H3/H4, "A3 T_o = 2 - T_o = 1", "sep. +", "Level 1", and a sign convention that needs its own footnote (findings_03).
- followups: "Three arms, one start" above three robot arms, where "arms" means study conditions; order X1, X3, X2; "clamped", "Can MH/PH", D-107, D-134, KI-065.
- pixels: "Level 2", "84²", "9-D proprioception", H5, pie-filled discs.
- methods: container and commit hashes, "DECISIONS D-089".

3. Redundant
- hero: the "source:" string repeats the chips.
- why: "not data" and "not a measurement".
- lab: success is stated four times (episode pill, heading, chip, scrubber label).
- compare: third Task selector; the strip repeats rows of the findings grid.
- inside: recorded/proposal is explained in the legend and again in the footnote.
- findings: legend strip twice (findings_01, findings_03); KI-052 note twice; failure legend twice (findings_02).
- followups: the same legend strip three more times (followups_01, _04, _05); "never pooled" at least four times.
- pixels: the same failures as scatter, strip and bars; failure legend twice (pixels_04).
- methods: the footer repeats the credits.

4. Cluttered, dense or empty
- hero: the top-left third of hero_01 is empty; chips float in a void above the arm.
- why: 24 cells of about 110 x 58 px.
- lab: selectors, chips and source line push the viewer to y = 570, so the first screen shows only the upper arm (lab_01); the chart stops at 960 px, leaving a void (lab_02).
- compare: about 150 px empty under the strip (compare_02).
- inside: seven thumbnail charts (about 130 x 45 px), then a large empty panel beside a tall viewer (inside_mid).
- findings: nine-plus charts; plots use under half their panels; empty column beside the ablations (findings_03).
- followups: seven tiles of near-identical charts; voids in followups_02 and followups_05.
- pixels: scatters fill half of each column (pixels_02); the cost table and erratum are bookkeeping (pixels_04, pixels_05).
- methods: fine.

5. Badly rendered
- hero: dark table and base on a dark background; the cube is small and hidden by the gripper mid-playback (hero_mid).
- why: dots clipped at cell edges; cyan and sand overlap in the output column.
- lab: the default camera crops the upper arm (lab_01, lab_mid); "success" is jammed against the end marker (lab_02).
- compare: the caption "3D reconstruction of the recorded states..." lies over the arm in both viewers; arm cropped at the top (compare_01).
- inside: caption chip over the arm and a stray white sliver top-left (inside_mid); "gripper to cube" header clipped (inside_02).
- findings: diamonds at +96 sit on the axis end (findings_02).
- followups: caption over the arm in all three viewers (followups_02); the X2 strip shrinks to roughly 6 px squares and labels (followups_07).
- pixels: "-56 cm" runs into "+y" (pixels_02); "success" is struck through by ticks (pixels_03); glaring white calibration renders with illegible burned-in text (pixels_04).
- methods: the "Frozen sources" disclosure is serif, the others sans (methods_01).

6. Hard to read
- hero: small grey chips and caveat note.
- why: grid labels; very long caption lines.
- lab: grey footnote and inset caption.
- compare: orange/yellow/red outlines on 12 px squares.
- inside: chart titles and legend of roughly 9 to 10 px, with dark swatches on dark.
- findings: "sep. +" and "143/150" at about 10 px.
- followups: roughly 8 px axis labels on the three-up charts (followups_03).
- pixels: burned-in image text.
- methods: small grey licence print (methods_02).

7. Narrative flow
- The order is hero, result cards, why, lab, head to head, inside, findings, follow-ups, cameras, methods.
- The visitor first learns the result in the three cards of hero_02, one scroll below the first screen. That is early, but should be visible without scrolling.
- "Why diffusion" argues from an abstract sketch before the task has been seen.
- "Inside the policy" (mechanism, validation data) interrupts the evidence chain between head-to-head and findings.
- The plain task description is in Methods, last.
- Nav says "After v1.0", the heading "v1.0.0", the text "v1.1 addendum".

8. Ten changes, by visible impact
1. lab_01: collapse selectors and chips so the viewer is on the first screen.
2. compare_01, followups_02, lab_01: reframe the default camera to show the whole arm and move the caption off the robot.
3. hero_01: put a plain outcome line on the viewer ("picks up the cube: success in 2.15 s") and bring the result cards above the fold.
4. inside_01: enlarge the sampler charts to fill the right column.
5. hero_01: raise table/base contrast and make the cube larger.
6. findings_01: show the legend once, not five times.
7. followups_07: draw the X2 strip at the size of the other strips.
8. followups_03: readable labels on the three-up action charts.
9. followups_01 and _02: replace KI/D/H/R codes with words and rename "Three arms".
10. pixels_04 and pixels_02: frame or dim the white calibration renders and fix the "-56 cm +y" collision.

9. Recording, illustration or proposal
The badges are the page's strongest feature: "recorded rollout", "illustrative schematic, not data" (dashed), "recorded sampler tensors", "proposal". It is unclear in these places:
- inside_01: the viewer is badged "recorded rollout" while the cyan dashed path inside it is a proposal; the legend separating them is below the viewer (inside_02).
- why_01: the badge says "recorded" but the caption says "replayed ... not a held-out result".
- "3D reconstruction, restyled colours" is only explained in Methods.
- pixels_03 and pixels_04: the point cloud ("recomputed"), the RGB thumbnails ("rendered from the recorded state") and the calibration images ("not of these episodes") carry no badge.
- Outcome strips and charts carry no badge.

10. Phone (390 px)
Works:
- Single column with readable body text; chips wrap.
- Result cards stack (hero_02).
- Chart rows put labels above the marks (findings_02, followups_02, followups_10).

Does not work:
- Nav is clipped at "Head to hea" with no scroll cue.
- The hero opens on a pale simulator still and the title sits near y = 690 (hero_01); Pause/Camera drift buttons appear under a still (hero_02).
- Stills clash with the dark theme, and the object is a speck of about 10 px.
- The lab viewer comes after a full screen of controls (lab_01), and its inset repeats the same image (lab_02).
- The tensor grid is unreadably small (why_02).
- Side-by-side comparisons become a vertical stack, losing "same start" (compare_02, followups_05).
- The mini-chart grid shows six panels; Δroll is not visible (inside_02).
- "success" collides with the end marker (lab_02).
- Follow-ups alone runs 15 screens.
