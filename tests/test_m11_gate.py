"""Tests for the M11 direct-CfC promotion gate helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.cfc_improvement_campaign import score_candidate
from scripts.run_m11_gate import SensorTransformController, summarize_controller


class _RecorderController:
    """Small controller that records transformed env inputs."""

    def __init__(self) -> None:
        self.envs: list[dict[str, float]] = []
        self.reset_count = 0

    def reset(self) -> None:
        self.reset_count += 1

    def step(self, error: float, env: dict[str, float] | None = None) -> tuple[float, float]:
        self.envs.append(dict(env or {}))
        return 0.0, error


def test_sensor_transform_applies_bias_and_lag() -> None:
    """M11 sensor wrapper perturbs controller sensors, not physics traces."""
    inner = _RecorderController()
    wrapped = SensorTransformController(
        inner,
        bias={"T_K": 1.0, "B_uT": -2.0, "I_norm": 0.1},
        lag_alpha=0.5,
    )
    wrapped.reset()
    assert inner.reset_count == 1

    wrapped.step(0.0, {"T_K": 333.0, "B_uT": 50.0, "I_norm": 1.0})
    wrapped.step(0.0, {"T_K": 335.0, "B_uT": 54.0, "I_norm": 1.4})

    assert inner.envs[0] == {
        "T_K": pytest.approx(334.0),
        "B_uT": pytest.approx(48.0),
        "I_norm": pytest.approx(1.1),
    }
    assert inner.envs[1] == {
        "T_K": pytest.approx(335.0),
        "B_uT": pytest.approx(50.0),
        "I_norm": pytest.approx(1.3),
    }


def test_m11_summary_requires_m5_improvement_and_all_robust_wins() -> None:
    """Promotion requires nominal improvement plus 5/5 robust ties/wins."""
    probes = {
        "m5_thermal_ramp": {
            "controllers": {
                "candidate": {
                    "over_rhlqr_10s": 0.96,
                    "ties_or_wins_10s": True,
                    "metrics": {"rf_abs_max_Hz": 50.0},
                }
            }
        }
    }
    for key in [
        "ood_3x_thermal_slope",
        "high_disc_noise_3x",
        "low_disc_noise_third",
        "reality_gap_5pct",
        "worst_case_2x_stacked",
    ]:
        probes[key] = {
            "controllers": {
                "candidate": {
                    "over_rhlqr_10s": 1.004,
                    "ties_or_wins_10s": True,
                    "metrics": {"rf_abs_max_Hz": 60.0},
                }
            }
        }

    summary = summarize_controller("candidate", Path("candidate.json"), probes)
    assert summary["gate_pass"] is True
    assert summary["promotion_decision"] == "promote"

    probes["ood_3x_thermal_slope"]["controllers"]["candidate"]["ties_or_wins_10s"] = False
    summary = summarize_controller("candidate", Path("candidate.json"), probes)
    assert summary["gate_pass"] is False
    assert summary["promotion_decision"] == "do_not_promote"


def test_cfc_campaign_score_penalizes_robustness_miss() -> None:
    """Campaign ranking heavily penalizes candidates that miss robust probes."""
    def candidate(ratio: float, tau1: float = 1.0, rf_abs: float = 50.0) -> dict:
        return {
            "ml_over_rhlqr_10s": ratio,
            "ml_over_rhlqr_1s": tau1,
            "ml": {"rf_abs_max_Hz": rf_abs},
        }

    base = {
        "m5_thermal_ramp": {
            "candidates": {
                "robust": candidate(0.968),
                "fragile": candidate(0.95),
            }
        }
    }
    for key in [
        "ood_3x_thermal_slope",
        "high_disc_noise_3x",
        "low_disc_noise_third",
        "reality_gap_5pct",
        "worst_case_2x_stacked",
    ]:
        base[key] = {
            "candidates": {
                "robust": candidate(1.004),
                "fragile": candidate(1.02),
            }
        }

    assert score_candidate(base, "robust") < score_candidate(base, "fragile")
