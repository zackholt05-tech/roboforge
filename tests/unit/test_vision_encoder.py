"""Candidate image encoders for the RGB study (``roboforge.models.vision``)."""

from __future__ import annotations

import pytest
import torch

from roboforge.models.vision import (
    RGBObservationEncoder,
    SpatialSoftmax,
    VisualCore,
    crop,
    parameter_count,
)


def test_spatial_softmax_returns_the_peak_location() -> None:
    ss = SpatialSoftmax(channels=1, height=5, width=7, num_kp=1)
    with torch.no_grad():
        ss.keypoints.weight.fill_(1.0)
        ss.keypoints.bias.zero_()
    x = torch.full((1, 1, 5, 7), -1e4)
    x[0, 0, 4, 0] = 1e4  # bottom-left: x = -1, y = +1 (rows grow downwards)
    assert torch.allclose(ss(x), torch.tensor([[-1.0, 1.0]]))


def test_eval_crop_is_the_centre_and_train_crops_are_seeded() -> None:
    img = torch.arange(10 * 10).reshape(1, 10, 10, 1).repeat(3, 1, 1, 1)
    c = crop(img, 6, None, train=False)
    assert c.shape == (3, 6, 6, 1) and c[0, 0, 0, 0] == 22
    a = crop(img, 6, torch.Generator().manual_seed(1), train=True)
    b = crop(img, 6, torch.Generator().manual_seed(1), train=True)
    assert torch.equal(a, b)
    with pytest.raises(ValueError):
        crop(img, 11, None, train=False)


@pytest.mark.parametrize("trunk", ["compact", "resnet18"])
def test_visual_core_shapes(trunk: str) -> None:
    core = VisualCore(trunk, image_size=84, crop_size=76)
    out = core(torch.randint(0, 256, (2, 84, 84, 3), dtype=torch.uint8))
    assert out.shape == (2, 64) and torch.isfinite(out).all()


def test_observation_encoder_takes_only_images_and_proprioception() -> None:
    enc = RGBObservationEncoder(["agentview", "robot0_eye_in_hand"], 9, "compact", 84, 76)
    imgs = torch.randint(0, 256, (3, 2, 2, 84, 84, 3), dtype=torch.uint8)
    out = enc(imgs, torch.zeros(3, 2, 9))
    assert out.shape == (3, 2, 2 * 64 + 9) == (3, 2, enc.out_dim)
    assert torch.equal(out[..., -9:], torch.zeros(3, 2, 9))  # proprio passes through last
    with pytest.raises(ValueError):
        enc(imgs[:, :, :1], torch.zeros(3, 2, 9))
    # separate weights per camera, as robomimic
    assert parameter_count(enc) == 2 * parameter_count(enc.cores[0])
