"""Policy architectures: BC-MLP and BC-RNN (Session 3); conditional diffusion U-Net (Session 4).

Every network maps a normalized observation window ``obs_history [B, T_o, D]`` with its
pre-start padding mask ``obs_valid [B, T_o]`` to a normalized action chunk
``[B, T_p, A]``. Deterministic families do it in one ``forward``; the diffusion policy
(:mod:`~roboforge.models.diffusion`) does it by seeded iterative denoising
(``sample(..., generator)``). Trainer and rollout policy call the same functions, so training
and inference cannot diverge (RESEARCH_PROTOCOL §6). :mod:`~roboforge.models.registry` maps
family names to networks, configs and losses.

Adapted third-party code: :mod:`~roboforge.models.unet1d` (Diffusion Policy / robomimic, MIT).
The scheduler and EMA arithmetic follow diffusers 0.11.1 (Apache-2.0) and are re-implemented.
See ``THIRD_PARTY_NOTICES.md``.
"""
