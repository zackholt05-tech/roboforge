# Third-Party Notices and Attribution

RoboForge (project author: Zackary Holt; licensed under Apache-2.0, see `LICENSE` and
`NOTICE`) builds on the work of other researchers and open-source projects. This file records what is used, under which license, and how it
must be cited. The repository's `main` branch does **not** redistribute any third-party
dataset, simulator asset, mesh, or pretrained weight; datasets are fetched from their official
source by `scripts/download_data.py` and verified against SHA-256 hashes in
`data/manifests/robomimic_v1.5_ph.json`. The hosted Observatory (the `gh-pages` branch, GitHub
Pages) is the one exception, described under "Hosted Observatory" below.

Status: licenses below were checked in Session 0 (2026-11-05) against the pinned
revisions and re-verified in Session 12 (2026-11-09) before publication: every distribution
of the locked Python environment and every npm package of the site is inventoried with its
declared license in `reports/session12/licenses.json` (`scripts/license_inventory.py`). BibTeX
entries were copied from each project's own README, dataset card, or official project page
(source noted); the paper's bibliography was resolved against DOI and arXiv metadata
(`paper/bibliography_check.json`).

## Software dependencies (installed, not vendored)

| Component | Pinned version | License | Source checked |
|---|---|---|---|
| MuJoCo (Python bindings) | `mujoco==3.2.7` | Apache-2.0 | GitHub `google-deepmind/mujoco` license field |
| robosuite | v1.5.1, commit `51cc01785bab80ffeed20da15e67d7dd4140e76a` | MIT (© 2022 Stanford Vision and Learning Lab and UT Robot Perception and Learning Lab); the LICENSE notes it includes partial MuJoCo code under Apache-2.0 | `LICENSE` in the pinned checkout |
| robomimic | commit `d309eaecc18acf4152a830a895a6984b8ac71b05` (v0.5.0 + 15) | MIT | GitHub license field / `LICENSE` |
| PyTorch, torchvision, NumPy, h5py, diffusers, transformers, etc. | see `uv.lock` | their respective licenses (mostly BSD, MIT, Apache-2.0; matplotlib's PSF-style license; MPL-2.0 for two) | enumerated per package in `reports/session12/licenses.json` (76 distributions) |
| pynput, qpsolvers (transitive dependencies of robosuite and its IK stack) | see `uv.lock` | LGPL-3.0 | installed from PyPI by `uv sync`, never imported by RoboForge code and not redistributed |

robosuite and robomimic declare no license in their package metadata; both repositories are
MIT at the pinned revisions (robosuite's installed distribution carries its `LICENSE`).

## Adapted code (Session 4)

| RoboForge file | Adapted from | License | What was taken |
|---|---|---|---|
| `src/roboforge/models/unet1d.py` | robomimic `d309eae`, `robomimic/models/diffusion_policy_nets.py` (robomimic's copy of the `ConditionalUnet1D` of Diffusion Policy, Chi et al., `real-stanford/diffusion_policy`) | MIT (robomimic: Copyright (c) 2021 Stanford Vision and Learning Lab) | module structure, parameter names and forward arithmetic of the conditional 1D U-Net (sinusoidal timestep embedding, FiLM residual blocks, down/up sampling). Loading robomimic's weights reproduces its outputs bitwise (`tests/unit/test_diffusion_reference.py`) |
| `src/roboforge/models/noise_schedule.py` | `diffusers==0.11.1` `DDPMScheduler`, `DDIMScheduler`, `betas_for_alpha_bar` | Apache-2.0 (file headers: `scheduling_ddpm.py` Copyright 2022 UC Berkeley Team and The HuggingFace Team; `scheduling_ddim.py` Copyright 2022 Stanford University Team and The HuggingFace Team) | the scheduler equations and their float32 operation order, re-implemented so every coefficient can be recorded; bitwise equal to diffusers in tests |
| `src/roboforge/train/ema.py` | `diffusers==0.11.1` `training_utils.EMAModel` | Apache-2.0 (the diffusers package license; this file has no own header) | the EMA warm-up decay rule and update, re-implemented with checkpointable state; bitwise equal in tests |

diffusers and robomimic are installed from `uv.lock`; their code is not vendored. The
Diffusion Policy algorithm is credited to Chi et al. (citation below); RoboForge's
contribution is the controlled benchmark, not the algorithm.

robomimic license (applies to the adapted portions of `src/roboforge/models/unet1d.py`):

```text
MIT License

Copyright (c) 2021 Stanford Vision and Learning Lab

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Hosted Observatory (GitHub Pages, Session 12)

The Observatory site is built locally (`make web-data web-assets web-build`) and published from
the `gh-pages` branch (DECISIONS D-102). Unlike `main`, that build contains:
- the visual meshes of the Franka Panda, gripper, cube and can, extracted from the compiled
  MuJoCo model of robosuite 1.5.1 (MIT, © 2022 Stanford Vision and Learning Lab and UT Robot
  Perception and Learning Lab);
- the Archivo and Newsreader variable fonts from fontsource (SIL Open Font License 1.1);
- the bundled JavaScript of React, React DOM, scheduler, three.js, React Three Fiber, zustand,
  its-fine, suspend-react, react-use-measure, use-sync-external-store and zod (MIT), and of
  small helpers such as ieee754 (BSD-3-Clause).

The site ships `THIRD_PARTY_LICENSES.txt` with the full license text of robosuite and of every
runtime npm package (`scripts/pages_bundle.py`). React Three Fiber's npm package ships no
license file, so its upstream `LICENSE` at the same version is kept in
`website/licenses/react-three-fiber.LICENSE` with its source URL. The build tools (Vite,
TypeScript, lightningcss under MPL-2.0, Playwright, vitest) are not shipped.

## Robot and object assets

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

Any code adapted (not just imported) from robomimic, Diffusion Policy, diffusers or DP3
must carry a header naming the source, revision and license, and be listed under "Adapted
code" above.

3D Diffusion Policy (DP3): the method that inspired the Level 2 point-cloud arm; no DP3 code,
weights or data are used, and RoboForge's encoder is reimplemented from the paper
(`src/roboforge/models/point_encoder.py`) — source: arXiv:2403.03954 abstract page (comments:
RSS 2024)
```bibtex
@inproceedings{ze2024dp3,
  title={3D Diffusion Policy: Generalizable Visuomotor Policy Learning via Simple 3D Representations},
  author={Yanjie Ze and Gu Zhang and Kangning Zhang and Chenyuan Hu and Muhan Wang and Huazhe Xu},
  booktitle={Robotics: Science and Systems (RSS)},
  year={2024}
}
```
