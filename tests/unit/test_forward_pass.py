"""Session 0 trivial model check: one batch from the window loader through a tiny MLP.

This is a pipeline sanity test (batch collation, shapes, masked loss, finite
outputs), not a policy and not training. Gradient/backward checks belong to the
model sessions (3-4); see docs/KNOWN_ISSUES.md regarding the ml-pipeline hook.
"""

from __future__ import annotations

import numpy as np
import torch

from roboforge.data.windows import LowDimWindowDataset


def collate(ds: LowDimWindowDataset, idx: list[int]) -> dict[str, torch.Tensor]:
    """Stack window items into tensors."""
    items = [ds[i] for i in idx]
    return {k: torch.as_tensor(np.stack([it[k] for it in items]))
            for k in ("obs", "obs_valid", "action", "action_valid")}


def one_batch_forward(ds: LowDimWindowDataset, batch_size: int = 8) -> tuple[torch.Tensor, float]:
    """Forward one batch through a tiny MLP; return (prediction, masked MSE value)."""
    torch.manual_seed(0)
    b = collate(ds, list(range(min(batch_size, len(ds)))))
    n, t_o, d = b["obs"].shape
    _, t_p, a = b["action"].shape
    net = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(t_o * d, 64), torch.nn.ReLU(),
                              torch.nn.Linear(64, t_p * a))
    with torch.no_grad():
        pred = net(b["obs"]).view(n, t_p, a)
        mask = b["action_valid"].float().unsqueeze(-1)
        masked_mse = (((pred - b["action"]) ** 2) * mask).sum() / (mask.sum() * a)
    return pred, float(masked_mse)


def test_one_batch_forward_synthetic(synthetic_hdf5) -> None:
    ds = LowDimWindowDataset(synthetic_hdf5, ["demo_0", "demo_2"], ["robot0_eef_pos", "object"],
                             obs_horizon=2, pred_horizon=16)
    pred, loss = one_batch_forward(ds)
    assert pred.shape == (8, 16, 7)
    assert torch.isfinite(pred).all() and np.isfinite(loss)


def test_padded_actions_do_not_contribute_to_masked_loss(synthetic_hdf5) -> None:
    ds = LowDimWindowDataset(synthetic_hdf5, ["demo_1"], ["object"], obs_horizon=2, pred_horizon=4)
    b = collate(ds, [0])  # single-step episode: 1 valid action, 3 padded
    assert b["action_valid"].tolist() == [[True, False, False, False]]
    pred = torch.zeros_like(b["action"])
    corrupted = b["action"].clone()
    corrupted[:, 1:] += 100.0  # change only padded targets
    mask = b["action_valid"].float().unsqueeze(-1)
    loss_a = (((pred - b["action"]) ** 2) * mask).sum()
    loss_b = (((pred - corrupted) ** 2) * mask).sum()
    assert torch.equal(loss_a, loss_b)
