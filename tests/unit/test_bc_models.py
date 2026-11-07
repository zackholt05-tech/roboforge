"""BC-MLP / BC-RNN networks: shapes, exact parameter counts, gradients, history masking."""

from __future__ import annotations

import pytest
import torch

from roboforge.models.bc import BCRNN, build_model, count_parameters, model_config

MLP = {"hidden": [1024, 1024], "activation": "relu", "layer_norm": False, "dropout": 0.0}
RNN = {"cell": "lstm", "hidden": 296, "layers": 2, "dropout": 0.0}


def _window(b: int, t_o: int, n_valid: list[int] | None = None, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(b, t_o, 19, generator=g)
    valid = torch.ones(b, t_o, dtype=torch.bool)
    for i, n in enumerate(n_valid or []):
        valid[i, : t_o - n] = False
        x[i, : t_o - n] = x[i, t_o - n]  # padding repeats the first observation (D-022)
    return x, valid


@pytest.mark.parametrize(("family", "cfg", "t_o", "t_p", "params"), [
    ("bc_mlp", MLP, 1, 1, 1_077_255),
    ("bc_rnn", RNN, 10, 1, 1_080_703),
    ("bc_mlp", MLP, 2, 16, 1_204_336),
])
def test_shapes_and_protocol_parameter_counts(family, cfg, t_o, t_p, params) -> None:
    torch.manual_seed(0)
    net = build_model(family, cfg, 19, 7, t_o, t_p)
    assert count_parameters(net) == params  # RESEARCH_PROTOCOL §6.1
    x, v = _window(5, t_o)
    assert net(x, v).shape == (5, t_p, 7)


@pytest.mark.parametrize(("family", "cfg", "t_o", "t_p"), [
    ("bc_mlp", MLP, 1, 1), ("bc_rnn", RNN, 10, 1), ("bc_mlp", MLP, 2, 16),
    ("bc_rnn", {**RNN, "cell": "gru", "hidden": 32}, 4, 3)])
def test_backward_reaches_every_parameter_with_finite_gradients(family, cfg, t_o, t_p) -> None:
    torch.manual_seed(0)
    net = build_model(family, cfg, 19, 7, t_o, t_p)
    x, v = _window(8, t_o, n_valid=[1, 2, t_o] if family == "bc_rnn" else None)
    loss = ((net(x, v) - torch.randn(8, t_p, 7)) ** 2).mean()
    loss.backward()
    grads = [p.grad for p in net.parameters()]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)
    assert all(float(g.abs().sum()) > 0 for g in grads)
    norm = torch.nn.utils.get_total_norm(grads)
    assert torch.isfinite(norm) and float(norm) > 0


def test_rnn_ignores_padded_steps_and_starts_from_zero_at_the_first_valid_step() -> None:
    torch.manual_seed(0)
    net = build_model("bc_rnn", RNN, 19, 7, 10, 1).eval()
    x, v = _window(3, 10, n_valid=[1, 4, 10])
    with torch.no_grad():
        base = net(x, v)
        y = x.clone()
        y[0, :9] += 5.0  # padded positions only
        y[1, :6] -= 3.0
        assert torch.equal(net(y, v), base)
        # equal to the recurrence over the valid suffix alone, from a zero state
        for i, n in enumerate([1, 4, 10]):
            _, (h, _) = net.rnn(x[i: i + 1, 10 - n:])
            ref = net.head(h[-1]).view(1, 1, 7)
            torch.testing.assert_close(base[i: i + 1], ref, rtol=0, atol=1e-6)


def test_rnn_output_depends_on_older_valid_observations() -> None:
    torch.manual_seed(0)
    net = build_model("bc_rnn", RNN, 19, 7, 10, 1).eval()
    x, v = _window(1, 10)
    with torch.no_grad():
        y = x.clone()
        y[0, 0] += 1.0  # the oldest valid observation
        assert not torch.equal(net(x, v), net(y, v))


def test_mlp_history_uses_both_frames() -> None:
    torch.manual_seed(0)
    net = build_model("bc_mlp", MLP, 19, 7, 2, 16).eval()
    x, v = _window(1, 2)
    with torch.no_grad():
        y = x.clone()
        y[0, 0] += 1.0
        assert not torch.equal(net(x, v), net(y, v))


@pytest.mark.parametrize("bad", ["newest_invalid", "not_prefix"])
def test_rnn_rejects_malformed_masks(bad: str) -> None:
    net = BCRNN(19, 7, 4, 1, model_config("bc_rnn", RNN))
    x, v = _window(1, 4)
    v[0, -1 if bad == "newest_invalid" else 1] = False
    with pytest.raises(ValueError):
        net(x, v)


def test_model_config_is_strict() -> None:
    with pytest.raises(ValueError):
        model_config("bc_mlp", {**MLP, "width": 3})
    with pytest.raises(ValueError):
        model_config("bc_rnn", {k: v for k, v in RNN.items() if k != "layers"})
    with pytest.raises(ValueError):
        model_config("diffusion", {})


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs a non-CPU device")
def test_rnn_runs_on_a_non_cpu_device() -> None:
    """Gate A found ``encode`` building an index on the CPU for CUDA inputs (Session 5)."""
    torch.manual_seed(0)
    net = build_model("bc_rnn", RNN, 19, 7, 10, 1)
    x, valid = _window(4, 10, [10, 3, 1, 7])
    ref = net(x, valid)
    out = net.to("mps")(x.to("mps"), valid.to("mps"))
    assert out.device.type == "mps"
    torch.testing.assert_close(out.cpu(), ref, rtol=1e-4, atol=1e-5)
