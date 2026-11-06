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
  `interventions`, `policy_acting`, `user_acting`). These are unused. They contain no
  operator identity: a single channel `'0'` per demo, 0 intervention steps and 0
  policy-acting steps (Session 1 audit).

## Derived data (Session 1)
Built by `make build-data` (`scripts/build_data.py`, converter `lowdim_v1`) from the hash-verified
`low_dim` files only. Each derived artifact records its source SHA-256 and its own content hash:

| Artifact | Location | Committed | Lift SHA-256 | Can SHA-256 |
|---|---|---|---|---|
| Processed features (float32 obs/actions/rewards/dones) | `$ROBOFORGE_DATA_ROOT/processed/robomimic_v1.5_ph/<task>/lowdim_v1.hdf5` | no (gitignored) | `74e70119993830798c8b4ed87f202f64d8d2c9d573c0f376d54c4130e52d63c8` | `e6055a6f066f53b987ed27a7b5894f20ec7053fc9cdf3222b4ad79ed3834ea7b` |
| Split manifest | `data/splits/<task>_ph.json` | yes | `73097265526c03dfb7535aa9bff50aadd285cbfb554e563636d66395909c331e` | `743779b0ad9cacd5bab85a74421f6fba4ace5e980f5979a4a3534c82f042b46f` |
| Normalization stats | `data/normalization/<task>_ph_lowdim_v1.json` | yes | `253bfc7e63570bf3e8f8646a0094d1da4c5b729ba8f463b4955f01410b4812d6` | `e0437cd7b8e48403366d19f7b0e6116e2224723b8a5240b2d7ba008403b3228d` |
| Task manifest (spec §4.4) | `data/manifests/tasks/<task>_ph.json` | yes | `2ac6070d18cc53f565ec11e67e01f6b5a91de7e268df6a491590b0e45f0528ac` | `d8f85ecea43c0e616162079757598af6b06c0edd8f03a3c563c08b06dea47162` |

- Hashes are of canonical content (no timestamps or paths). `make check-data` rebuilds
  everything and requires equality. Verified bit-identical on macOS arm64 and Linux aarch64
  (KI-018); linux/x86_64 unverified.
- Statistics and features computed from the data are derived works of MIT-licensed
  datasets. The processed HDF5 files are not redistributed; they are rebuilt locally.
- Column meanings: `docs/DATA_DICTIONARY.md`. Audit: `reports/data_audit.json`.
