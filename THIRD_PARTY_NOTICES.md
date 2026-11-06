# Third-Party Notices and Attribution

RoboForge (project author: Zackary Holt; licensed under Apache-2.0, see `LICENSE` and
`NOTICE`) builds on the work of other researchers and open-source projects. This file records what is used, under which license, and how it
must be cited. RoboForge does **not** redistribute any third-party dataset, simulator
asset, mesh, or pretrained weight; datasets are fetched from their official source by
`scripts/download_data.py` and verified against SHA-256 hashes in
`data/manifests/robomimic_v1.5_ph.json`.

Status: licenses below were checked in Session 0 (2026-11-05) against the pinned
revisions. BibTeX entries were copied from each project's own README, dataset card, or
official project page (source noted). Re-verify before publication (Session 11).

## Software dependencies (installed, not vendored)

| Component | Pinned version | License | Source checked |
|---|---|---|---|
| MuJoCo (Python bindings) | `mujoco==3.2.7` | Apache-2.0 | GitHub `google-deepmind/mujoco` license field |
| robosuite | v1.5.1, commit `51cc01785bab80ffeed20da15e67d7dd4140e76a` | MIT (© 2022 Stanford Vision and Learning Lab and UT Robot Perception and Learning Lab); the LICENSE notes it includes partial MuJoCo code under Apache-2.0 | `LICENSE` in the pinned checkout |
| robomimic | commit `d309eaecc18acf4152a830a895a6984b8ac71b05` (v0.5.0 + 15) | MIT | GitHub license field / `LICENSE` |
| PyTorch, torchvision, NumPy, h5py, diffusers, transformers, etc. | see `uv.lock` | their respective licenses (BSD/MIT/Apache-2.0) | to be enumerated per package before release |

Robot and object assets (Franka Panda meshes, Lift cube, PickPlace bins/can) ship inside
the robosuite package (`robosuite/models/assets/`). Session 0 found no separate license or
notice file in that directory; they are distributed with robosuite under its MIT license.
Images in `reports/feasibility/frames/` are RoboForge renders of these assets, produced by
MuJoCo from the official demonstration states (see `reports/feasibility/STACK_REPORT.md`).

## Datasets

**robomimic v0.1 proficient-human (PH) datasets, robosuite-v1.5 repackaging**
(`Lift`, `PickPlaceCan`), Hugging Face dataset `robomimic/robomimic_datasets` at revision
`74fa018461f479cd9fd15b924a16103012096203`. Dataset card metadata: `license: mit`.
The robomimic documentation notes these repackaged datasets may not exactly reproduce the
original CoRL 2021 results (which used the `mujoco-py` `offline_study` robosuite branch).

## Required citations

robomimic (datasets and framework) — source: Hugging Face dataset card / robomimic README
```bibtex
@inproceedings{robomimic2021,
  title={What Matters in Learning from Offline Human Demonstrations for Robot Manipulation},
  author={Ajay Mandlekar and Danfei Xu and Josiah Wong and Soroush Nasiriany and Chen Wang and Rohun Kulkarni and Li Fei-Fei and Silvio Savarese and Yuke Zhu and Roberto Mart\'{i}n-Mart\'{i}n},
  booktitle={Conference on Robot Learning (CoRL)},
  year={2021}
}
```

robosuite — source: robosuite v1.5.1 README
```bibtex
@inproceedings{robosuite2020,
  title={robosuite: A Modular Simulation Framework and Benchmark for Robot Learning},
  author={Yuke Zhu and Josiah Wong and Ajay Mandlekar and Roberto Mart\'{i}n-Mart\'{i}n and Abhishek Joshi and Soroush Nasiriany and Yifeng Zhu and Kevin Lin},
  booktitle={arXiv preprint arXiv:2009.12293},
  year={2020}
}
```

MuJoCo — source: google-deepmind/mujoco README
```bibtex
@inproceedings{todorov2012mujoco,
  title={MuJoCo: A physics engine for model-based control},
  author={Todorov, Emanuel and Erez, Tom and Tassa, Yuval},
  booktitle={2012 IEEE/RSJ International Conference on Intelligent Robots and Systems},
  pages={5026--5033},
  year={2012},
  organization={IEEE},
  doi={10.1109/IROS.2012.6386109}
}
```

Diffusion Policy (algorithmic reference for the main model) — source: official project page
https://diffusion-policy.cs.columbia.edu/ (code: `real-stanford/diffusion_policy`, MIT)
```bibtex
@inproceedings{chi2023diffusionpolicy,
  title={Diffusion Policy: Visuomotor Policy Learning via Action Diffusion},
  author={Chi, Cheng and Feng, Siyuan and Du, Yilun and Xu, Zhenjia and Cousineau, Eric and Burchfiel, Benjamin and Song, Shuran},
  booktitle={Proceedings of Robotics: Science and Systems (RSS)},
  year={2023}
}
@article{chi2024diffusionpolicy,
  author={Cheng Chi and Zhenjia Xu and Siyuan Feng and Eric Cousineau and Yilun Du and Benjamin Burchfiel and Russ Tedrake and Shuran Song},
  title={Diffusion Policy: Visuomotor Policy Learning via Action Diffusion},
  journal={The International Journal of Robotics Research},
  year={2024}
}
```

3D Diffusion Policy / DP3 (reference for the optional Level 2 extension only) — source:
`YanjieZe/3D-Diffusion-Policy` README (MIT)
```bibtex
@inproceedings{Ze2024DP3,
  title={3D Diffusion Policy: Generalizable Visuomotor Policy Learning via Simple 3D Representations},
  author={Yanjie Ze and Gu Zhang and Kangning Zhang and Chenyuan Hu and Muhan Wang and Huazhe Xu},
  booktitle={Proceedings of Robotics: Science and Systems (RSS)},
  year={2024}
}
```

If any code is later adapted (not just imported) from robomimic, Diffusion Policy or DP3,
the file must carry a header naming the source, revision and license.
