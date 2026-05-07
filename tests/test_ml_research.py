"""Tests for ML autoresearch controller plumbing."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cptservo.evaluation.batched_runner import run_batched_loop
from cptservo.policy.cpt_env import CPTServoEnv
from cptservo.policy.ml_research import (
    CfCDirectConfig,
    CfCDirectController,
    CfCFeedforwardConfig,
    CfCFeedforwardController,
    MLControllerConfig,
    MLServoController,
    PhysicsResidualConfig,
    PhysicsResidualController,
    read_rhlqr_reference,
)
from cptservo.twin.disturbance import Disturbance
from cptservo.twin.reduced import ReducedTwin


class _ZeroController:
    """Stateless controller for deterministic runner comparisons."""

    def reset(self) -> None:
        """No state to clear."""

    def step(self, error: float, env: dict[str, float] | None = None) -> tuple[float, float]:
        """Ignore feedback and hold RF correction at zero."""
        return error, 0.0


def test_research_env_features_and_scaled_reward() -> None:
    """Research env exposes richer obs and nonzero finite scaled rewards."""
    env = CPTServoEnv(
        disturbance_recipe_name="thermal_ramp",
        rollout_duration_s=0.05,
        disc_noise_amp_ci=0.0,
        lo_noise_scale=0.0,
        include_env_derivatives=True,
        include_running_features=True,
        include_lqr_hint=True,
        reward_mode="window_var",
        reward_scale=1.0e20,
        rng_seed=123,
    )
    obs, _ = env.reset(seed=123)
    assert obs.shape == (22,)
    rewards = []
    for _ in range(4):
        _, reward, terminated, _, info = env.step(np.array([0.0], dtype=np.float32))
        rewards.append(reward)
        assert np.isfinite(reward)
        assert "lqr_hint_Hz" in info
        if terminated:
            break
    assert any(abs(r) > 0.0 for r in rewards)


def test_ml_controller_residual_action_clipping() -> None:
    """Residual controller output remains inside RF actuator bounds."""
    controller = MLServoController(
        MLControllerConfig(mode="residual_lqr", residual_limit_Hz=250.0)
    )
    for param in controller.parameters():
        param.data.fill_(10.0)
    _, rf = controller.step(
        1.0e-3,
        {"T_K": 334.0, "B_uT": 51.0, "I_norm": 1.02},
    )
    assert abs(rf) <= controller.config.rf_limit_Hz


def test_physics_residual_uses_env_deviations() -> None:
    """Linear residual adds only deviations from nominal env sensors."""
    controller = PhysicsResidualController(
        PhysicsResidualConfig(k_T_Hz_per_K=0.05, k_B_Hz_per_uT=0.0, k_I_Hz_per_norm=0.0)
    )
    _, rf_nom = controller.step(
        0.0,
        {"T_K": 333.15, "B_uT": 50.0, "I_norm": 1.0},
    )
    controller.reset()
    _, rf_hot = controller.step(
        0.0,
        {"T_K": 334.15, "B_uT": 50.0, "I_norm": 1.0},
    )
    assert rf_hot - rf_nom == pytest.approx(0.05)


def test_physics_residual_multisensor_terms() -> None:
    """T/B/I coefficients all contribute with the configured signs."""
    cfg = PhysicsResidualConfig(
        k_T_Hz_per_K=0.05,
        k_B_Hz_per_uT=-0.1,
        k_I_Hz_per_norm=2.0,
        intensity_tau_s=0.001,
    )
    controller = PhysicsResidualController(cfg)
    _, rf = controller.step(
        0.0,
        {"T_K": 334.15, "B_uT": 52.0, "I_norm": 1.1},
    )
    expected = 0.05 * 1.0 - 0.1 * 2.0 + 2.0 * 0.1
    assert rf == pytest.approx(expected)


def test_physics_residual_derivative_terms_and_reset() -> None:
    """Derivative terms are reset with controller state."""
    cfg = PhysicsResidualConfig(k_dT_Hz_per_K_s=1.0e-4)
    controller = PhysicsResidualController(cfg)
    _, rf_step = controller.step(
        0.0,
        {"T_K": 333.16, "B_uT": 50.0, "I_norm": 1.0},
    )
    assert rf_step == pytest.approx(1.0e-3)
    controller.reset()
    _, rf_after_reset = controller.step(
        0.0,
        {"T_K": 333.15, "B_uT": 50.0, "I_norm": 1.0},
    )
    assert rf_after_reset == pytest.approx(0.0)


def test_physics_residual_clips_residual_before_final_rf() -> None:
    """Residual is independently bounded before final actuator clipping."""
    cfg = PhysicsResidualConfig(
        k_T_Hz_per_K=100.0,
        residual_limit_Hz=0.25,
    )
    controller = PhysicsResidualController(cfg)
    _, rf = controller.step(
        0.0,
        {"T_K": 334.15, "B_uT": 50.0, "I_norm": 1.0},
    )
    assert rf == pytest.approx(0.25)


def test_physics_residual_rejects_bad_sensor_values() -> None:
    """Missing and non-finite sensors fall back to last good values."""
    cfg = PhysicsResidualConfig(k_T_Hz_per_K=1.0)
    controller = PhysicsResidualController(cfg)
    _, rf_good = controller.step(
        0.0,
        {"T_K": 333.16, "B_uT": 50.0, "I_norm": 1.0},
    )
    controller.reset()
    _, rf_bad = controller.step(
        0.0,
        {"T_K": float("nan"), "B_uT": 50.0, "I_norm": 1.0},
    )
    assert rf_good == pytest.approx(0.01)
    assert rf_bad == pytest.approx(0.0)


def test_cfc_zeroed_residual_matches_rhlqr() -> None:
    """A zeroed CfC with no base residual is exactly the DLQR inner loop."""
    cfg = CfCFeedforwardConfig(base_residual=PhysicsResidualConfig())
    cfc = CfCFeedforwardController(cfg)
    lqr = cfc._lqr.__class__.from_recipe()
    errors = [0.0, 1.0e-4, -2.0e-4, 3.0e-4]
    for error in errors:
        _, rf_lqr = lqr.step(error)
        _, rf_cfc = cfc.step(
            error,
            {"T_K": 333.2, "B_uT": 50.1, "I_norm": 1.01},
        )
        assert rf_cfc == pytest.approx(rf_lqr)


def test_cfc_safe_init_matches_linear_residual_sequence() -> None:
    """Default CfC safe init preserves the known linear residual controller."""
    base_cfg = PhysicsResidualConfig(k_T_Hz_per_K=0.05)
    linear = PhysicsResidualController(base_cfg)
    cfc = CfCFeedforwardController(
        CfCFeedforwardConfig(base_residual=base_cfg)
    )
    envs = [
        {"T_K": 333.15, "B_uT": 50.0, "I_norm": 1.0},
        {"T_K": 333.20, "B_uT": 50.1, "I_norm": 1.01},
        {"T_K": 333.25, "B_uT": 49.9, "I_norm": 0.99},
    ]
    errors = [0.0, 1.0e-4, -1.5e-4]
    for error, env in zip(errors, envs):
        _, rf_linear = linear.step(error, env)
        _, rf_cfc = cfc.step(error, env)
        assert rf_cfc == pytest.approx(rf_linear)


def test_cfc_reset_clears_recurrent_state() -> None:
    """CfC reset clears recurrent state and reproduces the same first action."""
    cfc = CfCFeedforwardController()
    cfc.weights["W_out"][:] = 1.0
    _, rf_first = cfc.step(
        1.0e-4,
        {"T_K": 334.0, "B_uT": 50.0, "I_norm": 1.0},
    )
    assert np.max(np.abs(cfc._h)) > 0.0
    cfc.reset()
    assert np.max(np.abs(cfc._h)) == pytest.approx(0.0)
    _, rf_repeat = cfc.step(
        1.0e-4,
        {"T_K": 334.0, "B_uT": 50.0, "I_norm": 1.0},
    )
    assert rf_repeat == pytest.approx(rf_first)


def test_cfc_save_load_roundtrip(tmp_path: Path) -> None:
    """CfC checkpoints preserve config and weights."""
    cfg = CfCFeedforwardConfig(hidden_size=4)
    cfc = CfCFeedforwardController(cfg)
    cfc.weights["W_out"][:] = np.arange(4, dtype=np.float64)
    path = tmp_path / "cfc.json"
    cfc.save(path)
    loaded = CfCFeedforwardController.load(path)
    assert loaded.config.hidden_size == 4
    assert np.allclose(loaded.weights["W_out"], cfc.weights["W_out"])


def test_cfc_direct_save_load_and_clipping(tmp_path: Path) -> None:
    """Standalone direct CfC persists weights and clips full RF action."""
    direct = CfCDirectController(CfCDirectConfig(hidden_size=4, rf_limit_Hz=0.25))
    direct.weights["W_feat_out"][:] = 100.0
    _, rf = direct.step(
        0.0,
        {"T_K": 334.15, "B_uT": 50.0, "I_norm": 1.0},
    )
    assert rf == pytest.approx(0.25)
    path = tmp_path / "direct.json"
    direct.save(path)
    loaded = CfCDirectController.load(path)
    assert loaded.config.hidden_size == 4
    assert loaded.config.rf_limit_Hz == pytest.approx(0.25)
    assert np.allclose(loaded.weights["W_feat_out"], direct.weights["W_feat_out"])


def test_cfc_direct_zero_output_does_not_use_rhlqr() -> None:
    """Zero direct CfC emits zero RF even for nonzero error."""
    direct = CfCDirectController()
    _, rf = direct.step(
        1.0e-3,
        {"T_K": 333.15, "B_uT": 50.0, "I_norm": 1.0},
    )
    assert rf == pytest.approx(0.0)


def test_cfc_rejects_bad_sensors_and_clips_residual() -> None:
    """CfC inherits sensor fallback and bounded residual behavior."""
    cfg = CfCFeedforwardConfig(
        base_residual=PhysicsResidualConfig(
            k_T_Hz_per_K=10.0,
            residual_limit_Hz=0.1,
        )
    )
    cfc = CfCFeedforwardController(cfg)
    _, rf_bad = cfc.step(
        0.0,
        {"T_K": float("nan"), "B_uT": 50.0, "I_norm": 1.0},
    )
    cfc.reset()
    _, rf_hot = cfc.step(
        0.0,
        {"T_K": 334.15, "B_uT": 50.0, "I_norm": 1.0},
    )
    assert rf_bad == pytest.approx(0.0)
    assert rf_hot == pytest.approx(0.1)


def test_cfc_linear_regime_matches_rhlqr_runner() -> None:
    """Zeroed CfC does not perturb DLQR in a clean linear-regime rollout."""
    trace = Disturbance.from_recipe("clean").generate(
        duration_s=0.02,
        sample_rate_Hz=10_000.0,
        seed=12,
    )
    kwargs = {
        "disturbance_traces": [trace],
        "duration_s": 0.02,
        "physics_rate_Hz": 10_000.0,
        "decimation_rate_Hz": 1_000.0,
        "n_warmup_steps": 10,
        "rng_seed": 12,
        "disc_noise_amp_ci": 0.0,
        "lo_noise_scale": 0.0,
    }
    lqr_res = run_batched_loop(
        twin=ReducedTwin(device="cpu"),
        controller=CfCFeedforwardController()._lqr.__class__.from_recipe(),
        **kwargs,
    )
    cfc_res = run_batched_loop(
        twin=ReducedTwin(device="cpu"),
        controller=CfCFeedforwardController(
            CfCFeedforwardConfig(base_residual=PhysicsResidualConfig())
        ),
        **kwargs,
    )
    assert np.allclose(cfc_res["y"], lqr_res["y"])


def test_batched_runner_passes_env_to_ml_controller() -> None:
    """Canonical evaluator can call ML controllers with env sensor dicts."""
    controller = MLServoController(MLControllerConfig(mode="standalone"))
    trace = Disturbance.from_recipe("clean").generate(
        duration_s=0.02,
        sample_rate_Hz=10_000.0,
        seed=5,
    )
    twin = ReducedTwin(device="cpu")
    res = run_batched_loop(
        twin=twin,
        controller=controller,
        disturbance_traces=[trace],
        duration_s=0.02,
        physics_rate_Hz=10_000.0,
        decimation_rate_Hz=1_000.0,
        n_warmup_steps=10,
        rng_seed=5,
        disc_noise_amp_ci=0.0,
        lo_noise_scale=0.0,
    )
    assert res["y"].shape == (1, 20)
    assert res["noise_injection_point"] == "rf_actual_pre_step+disc_noise_pre_controller"


def test_shared_noise_paired_runner_matches_canonical_single_run() -> None:
    """Shared-noise batches preserve the exact canonical B=1 RNG stream."""
    trace = Disturbance.from_recipe("clean").generate(
        duration_s=0.015,
        sample_rate_Hz=10_000.0,
        seed=17,
    )
    kwargs = {
        "duration_s": 0.015,
        "physics_rate_Hz": 10_000.0,
        "decimation_rate_Hz": 1_000.0,
        "n_warmup_steps": 10,
        "rng_seed": 99,
        "disc_noise_amp_ci": 7.0e-4,
        "lo_noise_scale": 1.0,
    }
    single = run_batched_loop(
        twin=ReducedTwin(device="cpu"),
        controller=_ZeroController(),
        disturbance_traces=[trace],
        **kwargs,
    )
    paired = run_batched_loop(
        twin=ReducedTwin(device="cpu"),
        controller=_ZeroController(),
        disturbance_traces=[trace, trace],
        shared_noise_across_batch=True,
        **kwargs,
    )
    assert np.allclose(paired["y"][0], single["y"][0])
    assert np.allclose(paired["y"][1], single["y"][0])
    assert np.allclose(paired["error_signal"][0], single["error_signal"][0])
    assert np.allclose(paired["error_signal"][1], single["error_signal"][0])


def test_read_current_rhlqr_reference() -> None:
    """ML gates read the current M5 reference instead of a stale constant."""
    project_root = Path(__file__).resolve().parents[1]
    value = read_rhlqr_reference(project_root)
    assert value == 5.0736994150999564e-12
