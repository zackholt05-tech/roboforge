# Superseded container feasibility run (commit a20bfe3)

Kept for disclosure. Canonical container, git `a20bfe3` clean, 2026-11-09 15:08–15:17 UTC
(`logs/session9/pc_feasibility/container.log`).

Calibration verdict `pass: false`: 1 of 368,570 "interior" pixels differed from `mj_ray` by
9.7 mm (`fixed_mount0_pedestal_vis`, demo_0 t = 29, agentview, pixel (6, 35)), a silhouette inside
one mesh. Every other check passed. `depth_vs_ray` then added a planarity condition computed from
exact ray depth only, with no labels (`b16b692`). The rerun from `b16b692` replaces this run as
the evidence; its other files were overwritten by that rerun. Coverage and cost here are produced
by unchanged code.
