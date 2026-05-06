"""Tests for the APGPolicy and APG training loop.

Required by the M6 spec (5 tests minimum):

1. test_apg_policy_forward_shape
   -- (B, obs_dim) input -> (B,) output.
2. test_apg_policy_step_returns_bounded_rf
   -- scalar output in [-rf_limit, +rf_limit].
3. test_apg_policy_reset_clears_history
   -- buffer goes to zero after reset().
4. test_apg_policy_gradient_flow
   -- running 100 step() calls in train mode produces non-zero grads.
5. test_apg_train_one_episode_smoke
   -- single 1-second BPTT episode runs without OOM/NaN, finite loss.

Additional tests:
6. test_apg_policy_obs_dim_matches_init
   -- obs_dim matches constructor params.
7. test_apg_load_checkpoint
   -- save/load round-trip preserves weights.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
import torch

from cptservo.policy.apg import APGPolicy
from cptservo.policy.apg_train import train_apg
from cptservo.twin.reduced import ReducedTwin

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_DATA_DIR = Path(__file__).parent.parent / "data"
_CALIB_PATH = _DATA_DIR / "reduced_calibration.json"

with open(_CALIB_PATH, encoding="utf-8-sig") as _fh:
    _CALIB_RAW = json.load(_fh)

_CALIB = {
    "light_shift_coeff": float(_CALIB_RAW["light_shift_coeff"]),
    "buffer_gas_shift_coeff": float(_CALIB_RAW["buffer_gas_shift_coeff"]),
    "lumped_zeeman_coeff": float(_CALIB_RAW["lumped_zeeman_coeff"]),
}


def _torch_autograd_smoke_passes() -> bool:
    """Return False when this local Torch build hard-crashes on backward()."""
    code = (
        "import torch; "
        "m=torch.nn.Linear(4,1); "
        "x=torch.randn(8,4); "
        "loss=(m(x).squeeze(-1)**2).mean(); "
        "loss.backward()"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=30,
        check=False,
    )
    return result.returncode == 0


_TORCH_AUTOGRAD_WORKS = _torch_autograd_smoke_passes()
_AUTOGRAD_SKIP_REASON = (
    "local Torch build hard-crashes on backward(); APG autograd tests skipped"
)


def _make_twin() -> ReducedTwin:
    """Return a calibrated ReducedTwin on CPU/float64."""
    return ReducedTwin(
        light_shift_coeff=_CALIB["light_shift_coeff"],
        buffer_gas_shift_coeff=_CALIB["buffer_gas_shift_coeff"],
        lumped_zeeman_coeff=_CALIB["lumped_zeeman_coeff"],
        device="cpu",
        dtype=torch.float64,
    )


def _make_policy(
    n_error_history: int = 16,
    rf_limit_Hz: float = 1000.0,
    hidden_dims: tuple[int, ...] = (64, 64),
    include_env_sensors: bool = False,
    n_rf_history: int = 0,
) -> APGPolicy:
    """Return a fresh APGPolicy with given config."""
    return APGPolicy(
        n_error_history=n_error_history,
        n_rf_history=n_rf_history,
        include_env_sensors=include_env_sensors,
        hidden_dims=hidden_dims,
        rf_limit_Hz=rf_limit_Hz,
    )


# ---------------------------------------------------------------------------
# Test 1 (spec): forward shape (B, obs_dim) -> (B,)
# ---------------------------------------------------------------------------


def test_apg_policy_forward_shape() -> None:
    """APGPolicy.forward maps (B, history_len) to (B,) — spec test 1.

    Verifies the batch dimension is preserved and the output is 1-D float64.
    """
    B = 8
    policy = _make_policy(n_error_history=16)
    obs_dim = policy.obs_dim  # == 16 when n_rf_history=0, no env sensors

    obs = torch.randn(B, obs_dim, dtype=torch.float64)
    out = policy.forward(obs)

    assert out.shape == (B,), (
        f"Expected output shape ({B},), got {out.shape}"
    )
    assert out.dtype == torch.float64, (
        f"Expected float64 output, got {out.dtype}"
    )


# ---------------------------------------------------------------------------
# Test 2 (spec): step() returns bounded RF correction
# ---------------------------------------------------------------------------


def test_apg_policy_step_returns_bounded_rf() -> None:
    """APGPolicy.step() returns rf_correction in [-rf_limit, +rf_limit] — spec test 2.

    Feeds 200 non-zero error samples; all RF corrections must be bounded.
    laser_correction must always be 0 (v1 convention).
    """
    rf_limit = 500.0
    policy = _make_policy(rf_limit_Hz=rf_limit)
    policy.reset()

    torch.manual_seed(42)
    errors = torch.randn(200, dtype=torch.float64).tolist()

    for e in errors:
        laser, rf = policy.step(e)
        assert laser == pytest.approx(0.0), (
            f"laser_correction must be 0, got {laser}"
        )
        assert -rf_limit <= rf <= rf_limit, (
            f"rf_correction {rf:.4e} out of bounds [{-rf_limit}, {rf_limit}]"
        )


# ---------------------------------------------------------------------------
# Test 3 (spec): reset() clears the error history buffer
# ---------------------------------------------------------------------------


def test_apg_policy_reset_clears_history() -> None:
    """reset() zeroes the error history so the next step has no memory — spec test 3.

    Feeds N non-zero steps to build up history, calls reset(), then verifies
    that the first step after reset matches a fresh policy's first step on the
    same error.
    """
    policy = _make_policy(n_error_history=16)

    # Build up non-zero history
    for _ in range(50):
        policy.step(1.0)

    # Reset and take one step
    policy.reset()
    _, rf_after_reset = policy.step(0.5)

    # Fresh policy with identical weights — must give same output (history = zeros)
    fresh_policy = _make_policy(n_error_history=16)
    fresh_policy.load_state_dict(policy.state_dict())
    fresh_policy.reset()
    _, rf_fresh = fresh_policy.step(0.5)

    assert rf_after_reset == pytest.approx(rf_fresh, rel=1e-9), (
        f"After reset: rf={rf_after_reset:.6e}, fresh rf={rf_fresh:.6e}. "
        "reset() must restore history to zeros."
    )


# ---------------------------------------------------------------------------
# Test 4 (spec): gradient flow through policy parameters
# ---------------------------------------------------------------------------


def test_apg_policy_gradient_flow() -> None:
    """100 forward() calls in train mode produce non-zero grads — spec test 4.

    Constructs a differentiable rollout: 100 observations fed through
    policy.forward(), loss = mean(output^2), backward().  All policy
    parameters must receive non-zero gradients.
    """
    if not _TORCH_AUTOGRAD_WORKS:
        pytest.skip(_AUTOGRAD_SKIP_REASON)
    policy = _make_policy(n_error_history=16, hidden_dims=(64, 64))
    policy.train()

    obs_dim = policy.obs_dim
    N = 100
    torch.manual_seed(7)
    obs_batch = torch.randn(N, obs_dim, dtype=torch.float64)

    outputs = policy.forward(obs_batch)  # (N,) with grad
    loss = (outputs**2).mean()
    loss.backward()

    any_nonzero = False
    for name, param in policy.named_parameters():
        assert param.grad is not None, (
            f"Parameter '{name}' has no gradient after backward()"
        )
        if param.grad.abs().max().item() > 0.0:
            any_nonzero = True

    assert any_nonzero, (
        "All policy parameter gradients are zero — gradient is not flowing"
    )


# ---------------------------------------------------------------------------
# Test 5 (spec): single BPTT episode smoke test
# ---------------------------------------------------------------------------


def test_apg_train_one_episode_smoke() -> None:
    """Single 1-second BPTT episode runs without OOM/NaN, finite loss — spec test 5.

    Uses bptt_window=200 (= 200 ms at 1 kHz), clean scenario, 1 episode.
    Verifies:
        - No exception raised.
        - loss_per_episode has exactly 1 entry.
        - The loss value is finite (not NaN, not inf).
        - sigma_y_per_episode entry equals sqrt(loss).
    """
    if not _TORCH_AUTOGRAD_WORKS:
        pytest.skip(_AUTOGRAD_SKIP_REASON)
    policy = _make_policy(n_error_history=16)

    result = train_apg(
        policy=policy,
        n_episodes_per_stage=1,
        bptt_window=200,
        learning_rate=3.0e-4,
        grad_clip=1.0,
        curriculum=["clean"],
        save_path="models/apg_smoke_test_pytest.pt",
        rng_seed=99,
        n_warmup_steps=200,
        verbose=False,
    )

    assert "loss_per_episode" in result, "result must contain 'loss_per_episode'"
    assert len(result["loss_per_episode"]) == 1, (
        f"Expected 1 episode loss entry, got {len(result['loss_per_episode'])}"
    )

    loss_val = result["loss_per_episode"][0]
    assert math.isfinite(loss_val), (
        f"Episode loss must be finite, got {loss_val}"
    )
    assert loss_val >= 0.0, f"Loss must be non-negative, got {loss_val}"

    sigma_val = result["sigma_y_per_episode"][0]
    assert math.isfinite(sigma_val), (
        f"sigma_y must be finite, got {sigma_val}"
    )
    assert sigma_val == pytest.approx(math.sqrt(loss_val), rel=1e-6), (
        f"sigma_y {sigma_val:.4e} != sqrt(loss) {math.sqrt(loss_val):.4e}"
    )


# ---------------------------------------------------------------------------
# Test 6: obs_dim matches init params
# ---------------------------------------------------------------------------


def test_apg_policy_obs_dim_matches_init() -> None:
    """obs_dim must equal n_error_history + n_rf_history + 3*include_env_sensors."""
    for include_env in (True, False):
        policy = APGPolicy(
            n_error_history=8,
            n_rf_history=4,
            include_env_sensors=include_env,
        )
        expected = 8 + 4 + (3 if include_env else 0)
        assert policy.obs_dim == expected, (
            f"include_env={include_env}: expected obs_dim={expected}, got {policy.obs_dim}"
        )


# ---------------------------------------------------------------------------
# Test 7: save/load checkpoint round-trip
# ---------------------------------------------------------------------------


def test_apg_load_checkpoint() -> None:
    """save() / load() round-trip must preserve weights and architecture exactly."""
    policy = _make_policy()

    # Set weights to a known non-default value
    with torch.no_grad():
        for p in policy.parameters():
            p.fill_(0.123)

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = str(Path(tmpdir) / "test_policy.pt")
        policy.save(ckpt_path)
        loaded = APGPolicy.load(ckpt_path)

    # Architecture must match
    assert loaded.obs_dim == policy.obs_dim
    assert loaded.rf_limit_Hz == policy.rf_limit_Hz
    assert loaded.n_error_history == policy.n_error_history
    assert loaded.n_rf_history == policy.n_rf_history
    assert loaded.include_env_sensors == policy.include_env_sensors

    # Weights must be identical
    for (name, p_orig), (_, p_load) in zip(
        policy.named_parameters(), loaded.named_parameters()
    ):
        assert torch.allclose(p_orig, p_load), (
            f"Parameter '{name}' differs after load: "
            f"max_diff={(p_orig - p_load).abs().max().item():.3e}"
        )

    # Forward pass must be identical
    obs = torch.randn(4, policy.obs_dim, dtype=torch.float64)
    with torch.no_grad():
        out_orig = policy(obs)
        out_load = loaded(obs)
    assert torch.allclose(out_orig, out_load, atol=1e-12), (
        "Forward pass differs after load/save round-trip"
    )
