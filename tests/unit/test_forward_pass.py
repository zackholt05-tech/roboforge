"""One batch from the window sampler through a tiny MLP (forward only; no training).

This is a pipeline sanity check of batch collation, shapes, masked loss and finite outputs.
It is not a policy. The real-data version lives in tests/integration/test_data_pipeline.py.
Training code (backward passes, optimizers) belongs to Session 3.
"""

from __future__ import annotations

import numpy as np
import torch

from roboforge.data.normalization import Normalizer
from roboforge.data.sequence import SequenceDataset, collate


def masked_mse(pred: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Mean squared error over valid (non-padded) action steps only."""
    mask = valid.float().unsqueeze(-1)
    return (((pred - target) ** 2) * mask).sum() / (mask.sum() * target.shape[-1])


def one_batch_forward(ds: SequenceDataset, batch_size: int = 8) -> tuple[torch.Tensor, float]:
    """Forward one batch through a tiny MLP; return (prediction, masked MSE value)."""
    torch.manual_seed(0)
    b = collate([ds[i] for i in range(min(batch_size, len(ds)))])
    n, t_o, d = b["obs_history"].shape
    _, t_p, a = b["future_action"].shape
    net = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(t_o * d, 64), torch.nn.ReLU(),
                              torch.nn.Linear(64, t_p * a))
    with torch.no_grad():
        pred = net(b["obs_history"]).view(n, t_p, a)
        loss = masked_mse(pred, b["future_action"], b["valid_mask"])
    return pred, float(loss)


def _ds(lengths: list[int], t_p: int = 16) -> SequenceDataset:
    rng = np.random.default_rng(0)
    eps = {f"demo_{i}": {"obs": rng.normal(size=(n, 19)).astype(np.float32),
                         "actions": rng.uniform(-1, 1, size=(n, 7)).astype(np.float32)}
           for i, n in enumerate(lengths)}
    obs_n = Normalizer.fit([e["obs"] for e in eps.values()], "minmax")
    act_n = Normalizer.fit([e["actions"] for e in eps.values()], "minmax")
    return SequenceDataset(eps, list(eps), obs_n, act_n, 2, t_p)


def test_one_batch_forward_synthetic() -> None:
    pred, loss = one_batch_forward(_ds([12, 30]))
    assert pred.shape == (8, 16, 7)
    assert torch.isfinite(pred).all() and np.isfinite(loss)


def test_padded_actions_do_not_contribute_to_masked_loss() -> None:
    b = collate([_ds([1], t_p=4)[0]])  # single-step episode: 1 valid action, 3 padded
    assert b["valid_mask"].tolist() == [[True, False, False, False]]
    pred = torch.zeros_like(b["future_action"])
    corrupted = b["future_action"].clone()
    corrupted[:, 1:] += 100.0  # change only padded targets
    assert torch.equal(masked_mse(pred, b["future_action"], b["valid_mask"]),
                       masked_mse(pred, corrupted, b["valid_mask"]))
