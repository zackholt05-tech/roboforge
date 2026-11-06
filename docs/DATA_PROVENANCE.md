# Data Provenance

Machine-readable source of truth: `data/manifests/robomimic_v1.5_ph.json`. Local copies
live under `$ROBOFORGE_DATA_ROOT/raw/robomimic_v1.5/` (gitignored, never redistributed).

## Source
- Hugging Face dataset `robomimic/robomimic_datasets`, revision
  `74fa018461f479cd9fd15b924a16103012096203`, last modified 2025-04-13. Public, ungated;
  card metadata `license: mit`. Citation in `THIRD_PARTY_NOTICES.md`.
- Registry cross-check: robomimic@`d309eae` `robomimic/__init__.py` lists the same paths
  (`v1.5/{task}/ph/{demo_v15,low_dim_v15}.hdf5`), the eval horizon (400 for Lift/Can/Square
  PH), and `image: None`.
- The upstream CLI
  `python robomimic/scripts/download_datasets.py --tasks lift can --dataset_types ph --hdf5_types low_dim raw --dry_run`
  works: `logs/session0/upstream_download_datasets_dry_run.log`. Note its default download
  dir is inside site-packages; RoboForge uses its own fetcher.

## Files (downloaded 2026-11-05; SHA-256 = HF LFS oid, re-verified locally)
| Task | Type | Size (bytes) | SHA-256 |
|---|---|---|---|
| lift | low_dim | 21,084,088 | `2067777cb8b532e9263dd09fd6448c41cc31224bb27be4a3b734010ae13eb540` |
| lift | raw | 36,699,586 | `68f2a8cfe2502de1224f0708fb82e46da07abce108c63e721d72fc1618272f37` |
| can | low_dim | 46,889,752 | `3f2eb92e0a5025d0095e866ac16cc8092d6a762abe27dec90dbaff9027282962` |
| can | raw | 64,932,974 | `86961df85af1b9c6b9d4182a3755a8a4db6d0660cd550e45cca1f7accdb6d73d` |
| square | low_dim / raw | 51,145,256 / 68,982,198 | in manifest; **not downloaded** |

## Verified structure (programmatic; `reports/feasibility/data_report_{lift,can}.json`)
- `data/demo_<i>` × 200 per task. Per-demo `actions [T,7]`, `states [T,D]`
  (Lift D=32, Can D=71), `rewards`, `dones`; low_dim only: `obs/*` and `next_obs/*`.
  Attributes: `model_file` (MJCF XML) and `num_samples`.
- `mask/{train,valid,20_percent*,50_percent*}`. train/valid = 180/20, disjoint.
- `low_dim` and `raw` carry **identical** states, actions and model XML for every demo
  (max abs diff 0.0). Masks are identical.
- `env_args`: low_dim `env_version 1.5.1`, raw `1.5.0`. Controller BASIC/OSC_POSE identical.
- Observations recomputed from `states[t]` in the pinned simulator equal stored `obs[t]`
  exactly, except Can's lagged relative blocks (KI-003).
- The raw files also contain teleoperation logs (`controller_info`, `user_info`,
  `interventions`). These are unused.

## Derived data
None yet. Session 1 creates `data/processed/` and `data/splits/` with their own hashes and
converter versions.
