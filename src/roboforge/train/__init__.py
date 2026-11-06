"""Training for every policy family (Session 3 on).

* :mod:`~roboforge.train.config` holds the strict, hashed run configs.
* :mod:`~roboforge.train.data` holds the in-memory windows and the seeded sampler.
* :mod:`~roboforge.train.loop` is the shared trainer.
* :mod:`~roboforge.train.checkpoint` writes and reads the ``weights_only`` checkpoints.
* :mod:`~roboforge.train.run` runs one training run and writes its immutable manifest.
"""
