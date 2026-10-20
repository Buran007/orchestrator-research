"""Групповые преимущества и эталонный лосс K2.5 (token clip); формулы — в `algorithm`."""

import pytest

from orchestrator_research.algorithm import K25Params, group_relative_advantages, k25_token_clip_reference


def test_group_advantages_are_centred_per_group():
    values = group_relative_advantages([1.0, 0.0, 0.5, 0.5], ["a", "a", "b", "b"])
    assert values == pytest.approx([0.5, -0.5, 0.0, 0.0])


def test_identical_policies_have_zero_loss():
    result = k25_token_clip_reference(
        old_log_probs=[-1.0, -2.0],
        new_log_probs=[-1.0, -2.0],
        advantages=[1.0, -1.0],
        trainable_mask=[1, 1],
    )
    assert result.total_loss == 0.0
    assert result.in_band_fraction == 1.0


def test_out_of_band_token_only_keeps_regularizer():
    result = k25_token_clip_reference(
        old_log_probs=[0.0],
        new_log_probs=[1.0],
        advantages=[5.0],
        trainable_mask=[1],
        params=K25Params(tau=0.001),
    )
    assert result.policy_loss == 0.0
    assert result.regularizer_loss == pytest.approx(0.001)


def test_masked_observation_token_does_not_affect_loss():
    result = k25_token_clip_reference(
        old_log_probs=[-1.0, 0.0],
        new_log_probs=[-1.0, 10.0],
        advantages=[1.0, 100.0],
        trainable_mask=[1, 0],
    )
    assert result.total_loss == 0.0
    assert result.in_band_fraction == 1.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"alpha": 0.0, "beta": 0.2},
        {"alpha": -0.2, "beta": 0.0},
        {"tau": -1e-3},
        {"alpha": float("nan")},
        {"beta": True},
        {"tau": "0.1"},
    ],
)
def test_k25_params_reject_invalid_values(kwargs):
    with pytest.raises(ValueError):
        K25Params(**kwargs)
