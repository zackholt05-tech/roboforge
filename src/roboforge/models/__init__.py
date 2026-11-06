"""Policy architectures: BC-MLP and BC-RNN (Session 3); conditional diffusion U-Net (Session 4).

Every network maps a normalized observation window ``obs_history [B, T_o, D]`` with its
pre-start padding mask ``obs_valid [B, T_o]`` to a normalized action chunk
``[B, T_p, A]``. Trainer and rollout policy call the same ``forward``, so training and
inference cannot diverge (RESEARCH_PROTOCOL §6).
"""
