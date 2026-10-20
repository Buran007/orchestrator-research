"""Torch-лосс K2.5 сверяется с эталоном `algorithm.k25_token_clip_reference`: значение, градиенты, маска, env."""

import json

import pytest

torch = pytest.importorskip("torch")

from orchestrator_research.algorithm import K25Params, k25_token_clip_reference  # noqa: E402
from orchestrator_research.training.k25_loss import (  # noqa: E402
    ENV_VAR,
    LOSS_MODE,
    from_env,
    k25_token_clip_loss,
    register,
    to_env,
)


def _token_mean(mat, mask):
    return (mat * mask).sum() / mask.sum().clamp(min=1)


def _batch(seed=0, B=4, T=16):
    g = torch.Generator().manual_seed(seed)
    old = torch.randn(B, T, generator=g, dtype=torch.float64) * 0.5 - 1.0
    delta = torch.randn(B, T, generator=g, dtype=torch.float64) * 0.3
    adv = torch.randn(B, T, generator=g, dtype=torch.float64)
    mask = (torch.rand(B, T, generator=g) < 0.7).to(torch.float64)
    mask[:, 0] = 1.0
    return old, old + delta, adv, mask


def test_matches_reference_on_random_batch():
    old, new, adv, mask = _batch()
    # В батче должны быть токены вне полосы, иначе клип не проверяется.
    delta = (new - old).flatten().tolist()
    assert any(abs(d) > 0.2 for d, m in zip(delta, mask.flatten().tolist(), strict=True) if m)

    params = K25Params()
    total, metrics = k25_token_clip_loss(old, new, adv, mask, params=params, agg=_token_mean)
    ref = k25_token_clip_reference(
        old_log_probs=old.flatten().tolist(),
        new_log_probs=new.flatten().tolist(),
        advantages=adv.flatten().tolist(),
        trainable_mask=[int(v) for v in mask.flatten().tolist()],
        params=params,
    )
    assert total.item() == pytest.approx(ref.total_loss, abs=1e-6)
    assert metrics["actor/k25_pg_loss"] == pytest.approx(ref.policy_loss, abs=1e-6)
    assert metrics["actor/k25_tau_loss"] == pytest.approx(ref.regularizer_loss, abs=1e-6)
    assert metrics["actor/k25_in_band_frac"] == pytest.approx(ref.in_band_fraction, abs=1e-6)


def test_out_of_band_tokens_get_only_tau_gradient():
    params = K25Params(alpha=-0.2, beta=0.2, tau=0.05)
    old = torch.zeros(1, 6, dtype=torch.float64)
    new = torch.tensor([[0.5, -0.5, 0.05, 0.0, 1.0, -0.1]], dtype=torch.float64, requires_grad=True)
    adv = torch.tensor([[1.0, -2.0, 0.5, 3.0, -1.0, 0.7]], dtype=torch.float64)
    mask = torch.ones(1, 6, dtype=torch.float64)

    total, _ = k25_token_clip_loss(old, new, adv, mask, params=params, agg=_token_mean)
    (grad,) = torch.autograd.grad(total, new)

    r = new.detach()
    tau_grad = 2 * params.tau * r / 6  # token-mean по всей маске ответа (N = 6)
    out_of_band = [0, 1, 4]
    assert torch.allclose(grad[0, out_of_band], tau_grad[0, out_of_band], atol=1e-12, rtol=0)

    in_band = [2, 3, 5]  # три токена в полосе: pg-член даёт -A / 3 на каждый
    expected_in = -adv[0, in_band] / 3 + tau_grad[0, in_band]
    assert torch.allclose(grad[0, in_band], expected_in, atol=1e-12, rtol=0)


def test_masked_tokens_get_zero_gradient():
    old, new, adv, mask = _batch(seed=3)
    new = new.clone().requires_grad_(True)
    total, _ = k25_token_clip_loss(old, new, adv, mask, params=K25Params(), agg=_token_mean)
    (grad,) = torch.autograd.grad(total, new)
    assert torch.all(grad[mask == 0] == 0)
    assert torch.any(grad[mask == 1] != 0)


def test_no_in_band_tokens_gives_finite_loss_equal_to_tau_term():
    params = K25Params(alpha=-0.2, beta=0.2, tau=0.1)
    old = torch.zeros(1, 3, dtype=torch.float64)
    new = torch.tensor([[1.0, -1.0, 0.5]], dtype=torch.float64, requires_grad=True)
    adv = torch.ones(1, 3, dtype=torch.float64)
    mask = torch.ones(1, 3, dtype=torch.float64)
    total, metrics = k25_token_clip_loss(old, new, adv, mask, params=params, agg=_token_mean)
    assert torch.isfinite(total)
    assert metrics["actor/k25_in_band_frac"] == 0.0
    assert total.item() == pytest.approx(0.1 * (1.0 + 1.0 + 0.25) / 3)


@pytest.mark.parametrize("environ", [{}, {ENV_VAR: "   "}])
def test_from_env_is_strict_when_unset_or_blank(environ):
    with pytest.raises(ValueError, match="not set"):
        from_env(environ)


def test_env_round_trip():
    params = K25Params(alpha=-0.1, beta=0.3, tau=0.01)
    assert from_env({ENV_VAR: to_env(params)}) == params
    assert from_env({ENV_VAR: json.dumps({"alpha": -0.1, "beta": 0.3, "tau": 0.01})}) == params


@pytest.mark.parametrize(
    "raw",
    [
        "not json",
        "[1, 2]",
        '{"gamma": 1}',
        '{"alpha": 0.2, "beta": 0.2}',
        '{"beta": -0.2}',
        '{"tau": -1}',
        '{"tau": "0.1"}',
        '{"tau": true}',
        '{"tau": NaN}',
    ],
)
def test_from_env_rejects_bad_values(raw):
    with pytest.raises(ValueError):
        from_env({ENV_VAR: raw})


def test_register_is_idempotent():
    pytest.importorskip("verl")
    from verl.trainer.ppo.core_algos import POLICY_LOSS_REGISTRY

    register()
    first = POLICY_LOSS_REGISTRY[LOSS_MODE]
    register()
    assert POLICY_LOSS_REGISTRY[LOSS_MODE] is first
