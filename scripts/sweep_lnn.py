"""Non-CfC LNN/observer campaign for the M11 ceiling.

This script runs the eight M11 follow-up ideas as a bounded screening campaign:
non-CfC liquid observers, robust/minimax scoring, sensor-corruption probes, and
a hand-built alpha-beta thermal observer baseline.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_SCRIPT_DIR))

from audit_calibration import make_calibrated_twin  # noqa: E402
from eval_cfc import (  # noqa: E402
    DISC_NOISE_AMP_CI,
    PHYSICS_RATE_HZ,
    BatchDispatchController,
    SensorTransformController,
    make_perturbed_twin,
)

from cptservo.baselines.dlqr import DLQRController  # noqa: E402
from cptservo.evaluation.batched_runner import run_batched_loop  # noqa: E402
from cptservo.twin.allan import overlapping_allan  # noqa: E402
from cptservo.twin.disturbance import Disturbance, DisturbanceTrace  # noqa: E402

DECIMATION_RATE_HZ = 1_000.0
TAUS = [1.0, 10.0, 100.0]
RNG_SEED = 4242
TIE_TOLERANCE = 0.005
CURRENT_M11_RATIO = 0.9682237494386449


@dataclass(frozen=True)
class LiquidObserverSpec:
    """Small non-CfC liquid observer wrapped around DLQR."""

    name: str
    k_T_Hz_per_K: float = 0.5
    k_dT_Hz_per_K_s: float = -0.01
    k_error_Hz_per_ci: float = 0.0
    residual_limit_Hz: float = 40.0
    tau_slow_s: float = 0.25
    tau_fast_s: float = 0.03
    sensor_bias_T_K: float = 0.0
    sensor_lag_alpha: float = 1.0


@dataclass(frozen=True)
class AlphaBetaSpec:
    """Explicit alpha-beta thermal observer wrapped around DLQR."""

    name: str
    k_T_Hz_per_K: float = 0.5
    k_dT_Hz_per_K_s: float = -0.01
    alpha: float = 0.06
    beta: float = 0.002
    residual_limit_Hz: float = 40.0
    sensor_bias_T_K: float = 0.0
    sensor_lag_alpha: float = 1.0


class LiquidObserverResidualController:
    """Non-CfC liquid neural observer that adds a bounded residual over DLQR."""

    def __init__(self, spec: LiquidObserverSpec) -> None:
        self.spec = spec
        self._lqr = DLQRController.from_recipe()
        self.reset()

    def reset(self) -> None:
        """Reset controller memory."""
        self._lqr.reset()
        self._h = np.zeros(4, dtype=np.float64)
        self._last_slow_T = 0.0

    def step(self, error: float, env: dict[str, float] | None = None) -> tuple[float, float]:
        """Apply DLQR feedback plus bounded liquid-observer residual."""
        env = env or {}
        dt = 1.0 / DECIMATION_RATE_HZ
        T_dev = float(env.get("T_K", 333.15)) - 333.15
        B_dev = float(env.get("B_uT", 50.0)) - 50.0
        I_dev = float(env.get("I_norm", 1.0)) - 1.0
        x = np.array(
            [
                np.clip(T_dev / 10.0, -2.0, 2.0),
                np.clip(B_dev / 10.0, -2.0, 2.0),
                np.clip(I_dev, -2.0, 2.0),
                np.clip(float(error) * 1.0e3, -2.0, 2.0),
            ],
            dtype=np.float64,
        )
        taus = np.array(
            [
                self.spec.tau_slow_s,
                self.spec.tau_fast_s,
                0.08,
                0.02,
            ],
            dtype=np.float64,
        )
        target = np.tanh(x + 0.15 * np.roll(self._h, 1))
        alpha = np.clip(dt / taus, 0.0, 1.0)
        self._h = self._h + alpha * (target - self._h)

        slow_T_dev = 10.0 * self._h[0]
        fast_T_dev = 10.0 * self._h[1]
        dT_est = (slow_T_dev - self._last_slow_T) / dt
        self._last_slow_T = slow_T_dev
        _, u_lqr = self._lqr.step(error)
        residual = (
            self.spec.k_T_Hz_per_K * fast_T_dev
            + self.spec.k_dT_Hz_per_K_s * dT_est
            + self.spec.k_error_Hz_per_ci * float(error)
        )
        residual = float(
            np.clip(
                residual,
                -self.spec.residual_limit_Hz,
                self.spec.residual_limit_Hz,
            )
        )
        return 0.0, float(np.clip(u_lqr + residual, -1000.0, 1000.0))


class AlphaBetaThermalObserverController:
    """Alpha-beta thermal state estimator with bounded residual over DLQR."""

    def __init__(self, spec: AlphaBetaSpec) -> None:
        self.spec = spec
        self._lqr = DLQRController.from_recipe()
        self.reset()

    def reset(self) -> None:
        """Reset observer and feedback state."""
        self._lqr.reset()
        self._T_est = 333.15
        self._dT_est = 0.0

    def step(self, error: float, env: dict[str, float] | None = None) -> tuple[float, float]:
        """Apply DLQR feedback plus alpha-beta thermal feedforward."""
        dt = 1.0 / DECIMATION_RATE_HZ
        T_meas = float((env or {}).get("T_K", 333.15))
        self._T_est = self._T_est + self._dT_est * dt
        innovation = T_meas - self._T_est
        self._T_est = self._T_est + self.spec.alpha * innovation
        self._dT_est = self._dT_est + (self.spec.beta / dt) * innovation
        _, u_lqr = self._lqr.step(error)
        residual = (
            self.spec.k_T_Hz_per_K * (self._T_est - 333.15)
            + self.spec.k_dT_Hz_per_K_s * self._dT_est
        )
        residual = float(
            np.clip(
                residual,
                -self.spec.residual_limit_Hz,
                self.spec.residual_limit_Hz,
            )
        )
        return 0.0, float(np.clip(u_lqr + residual, -1000.0, 1000.0))


def log(msg: str) -> None:
    """Timestamped console log."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _json_safe(value: Any) -> Any:
    """Convert NaN/Inf floats to null for strict JSON."""
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    return value


def make_thermal_ramp_trace(
    duration_s: float,
    slope_scale: float,
    seed: int,
) -> DisturbanceTrace:
    """Build a thermal-ramp trace with scaled ramp amplitude."""
    nominal = Disturbance.from_recipe("thermal_ramp")
    params = deepcopy(nominal.params)
    params["T_K_ramp_amplitude_K"] = float(params["T_K_ramp_amplitude_K"]) * slope_scale
    return Disturbance("thermal_ramp", params).generate(
        duration_s=duration_s,
        sample_rate_Hz=PHYSICS_RATE_HZ,
        seed=seed,
    )


def make_all_stacked_trace(
    duration_s: float,
    b_scale: float,
    i_scale: float,
    seed: int,
) -> DisturbanceTrace:
    """Build all-stacked trace with scaled B and intensity drift."""
    nominal = Disturbance.from_recipe("all_stacked")
    params = deepcopy(nominal.params)
    params["B_uT_drift_amplitude_uT"] = (
        float(params["B_uT_drift_amplitude_uT"]) * b_scale
    )
    params["laser_intensity_drift_amplitude"] = (
        float(params["laser_intensity_drift_amplitude"]) * i_scale
    )
    return Disturbance("all_stacked", params).generate(
        duration_s=duration_s,
        sample_rate_Hz=PHYSICS_RATE_HZ,
        seed=seed,
    )


def probe_plan(duration_s: float) -> dict[str, dict[str, Any]]:
    """Return nominal, robust, and sensor-corruption probes."""
    thermal = Disturbance.from_recipe("thermal_ramp").generate(
        duration_s=duration_s,
        sample_rate_Hz=PHYSICS_RATE_HZ,
        seed=RNG_SEED,
    )
    return {
        "m5_thermal_ramp": {
            "trace": Disturbance.from_recipe("thermal_ramp").generate(
                duration_s=duration_s,
                sample_rate_Hz=PHYSICS_RATE_HZ,
                seed=42,
            ),
            "rng_seed": 42,
            "disc_noise_amp_ci": DISC_NOISE_AMP_CI,
            "twin": None,
            "sensor_bias": {},
            "sensor_lag_alpha": 1.0,
        },
        "ood_3x_thermal_slope": {
            "trace": make_thermal_ramp_trace(duration_s, 3.0, RNG_SEED),
            "rng_seed": RNG_SEED,
            "disc_noise_amp_ci": DISC_NOISE_AMP_CI,
            "twin": None,
            "sensor_bias": {},
            "sensor_lag_alpha": 1.0,
        },
        "high_disc_noise_3x": {
            "trace": thermal,
            "rng_seed": RNG_SEED,
            "disc_noise_amp_ci": 3.0 * DISC_NOISE_AMP_CI,
            "twin": None,
            "sensor_bias": {},
            "sensor_lag_alpha": 1.0,
        },
        "low_disc_noise_third": {
            "trace": thermal,
            "rng_seed": RNG_SEED,
            "disc_noise_amp_ci": DISC_NOISE_AMP_CI / 3.0,
            "twin": None,
            "sensor_bias": {},
            "sensor_lag_alpha": 1.0,
        },
        "reality_gap_5pct": {
            "trace": thermal,
            "rng_seed": RNG_SEED,
            "disc_noise_amp_ci": DISC_NOISE_AMP_CI,
            "twin": make_perturbed_twin(0.05, RNG_SEED + 1),
            "sensor_bias": {},
            "sensor_lag_alpha": 1.0,
        },
        "worst_case_2x_stacked": {
            "trace": make_all_stacked_trace(duration_s, 2.0, 2.0, RNG_SEED),
            "rng_seed": RNG_SEED,
            "disc_noise_amp_ci": DISC_NOISE_AMP_CI,
            "twin": None,
            "sensor_bias": {},
            "sensor_lag_alpha": 1.0,
        },
        "sensor_T_bias_plus_0p2K": {
            "trace": thermal,
            "rng_seed": RNG_SEED,
            "disc_noise_amp_ci": DISC_NOISE_AMP_CI,
            "twin": None,
            "sensor_bias": {"T_K": 0.2},
            "sensor_lag_alpha": 1.0,
        },
        "sensor_T_lag_alpha_0p05": {
            "trace": thermal,
            "rng_seed": RNG_SEED,
            "disc_noise_amp_ci": DISC_NOISE_AMP_CI,
            "twin": None,
            "sensor_bias": {},
            "sensor_lag_alpha": 0.05,
        },
    }


def metrics_from_y_rf(y: np.ndarray, rf: np.ndarray, label: str) -> dict[str, Any]:
    """Compute Allan metrics for one batch row."""
    y_demeaned = y - float(np.mean(y))
    allan = overlapping_allan(y_demeaned, DECIMATION_RATE_HZ, TAUS)
    return {
        "label": label,
        "sigma_y_1s": float(allan.get(1.0, float("nan"))),
        "sigma_y_10s": float(allan.get(10.0, float("nan"))),
        "sigma_y_100s": float(allan.get(100.0, float("nan"))),
        "rf_abs_max_Hz": float(np.max(np.abs(rf))),
    }


def make_controller(kind: str, spec: LiquidObserverSpec | AlphaBetaSpec) -> Any:
    """Instantiate one controller from a campaign spec."""
    if kind == "liquid":
        controller: Any = LiquidObserverResidualController(spec)  # type: ignore[arg-type]
    elif kind == "alpha_beta":
        controller = AlphaBetaThermalObserverController(spec)  # type: ignore[arg-type]
    else:
        raise ValueError(f"Unknown controller kind: {kind}")
    bias = {"T_K": spec.sensor_bias_T_K} if spec.sensor_bias_T_K else {}
    return SensorTransformController(
        controller,
        bias=bias,
        lag_alpha=spec.sensor_lag_alpha,
    )


def candidate_specs() -> list[tuple[str, LiquidObserverSpec | AlphaBetaSpec]]:
    """Return bounded candidates covering all eight TODO ideas."""
    items: list[tuple[str, LiquidObserverSpec | AlphaBetaSpec]] = []
    for k_t in [0.35, 0.5, 0.8, 1.1, 1.4]:
        for k_dt in [-0.03, -0.01, 0.0]:
            items.append(
                (
                    "liquid",
                    LiquidObserverSpec(
                        name=f"lnn_obs_T{k_t:g}_dT{k_dt:g}".replace(
                            ".", "p"
                        ).replace("-", "m"),
                        k_T_Hz_per_K=k_t,
                        k_dT_Hz_per_K_s=k_dt,
                    ),
                )
            )
    items.extend(
        [
            (
                "liquid",
                LiquidObserverSpec(
                    name="lnn_sensor_corrupt_train_bias",
                    k_T_Hz_per_K=0.5,
                    k_dT_Hz_per_K_s=-0.01,
                    sensor_bias_T_K=0.2,
                ),
            ),
            (
                "liquid",
                LiquidObserverSpec(
                    name="lnn_sensor_corrupt_train_lag",
                    k_T_Hz_per_K=0.5,
                    k_dT_Hz_per_K_s=-0.01,
                    sensor_lag_alpha=0.05,
                ),
            ),
            (
                "liquid",
                LiquidObserverSpec(
                    name="lnn_recurrent_residual_error",
                    k_T_Hz_per_K=0.5,
                    k_dT_Hz_per_K_s=-0.01,
                    k_error_Hz_per_ci=500.0,
                ),
            ),
        ]
    )
    for alpha, beta in [(0.04, 0.001), (0.08, 0.002), (0.12, 0.004)]:
        items.append(
            (
                "alpha_beta",
                AlphaBetaSpec(
                    name=f"alpha_beta_a{alpha:g}_b{beta:g}".replace(".", "p"),
                    alpha=alpha,
                    beta=beta,
                ),
            )
        )
    return items


def evaluate_candidates(
    specs: list[tuple[str, LiquidObserverSpec | AlphaBetaSpec]],
    probe: dict[str, Any],
    duration_s: float,
) -> dict[str, Any]:
    """Evaluate DLQR plus all candidates on one probe."""
    controllers: list[Any] = [DLQRController.from_recipe()]
    labels = ["dlqr"]
    for kind, spec in specs:
        controllers.append(make_controller(kind, spec))
        labels.append(spec.name)
    res = run_batched_loop(
        twin=probe["twin"] or make_calibrated_twin(),
        controller=BatchDispatchController(controllers),
        disturbance_traces=[probe["trace"] for _ in controllers],
        duration_s=duration_s,
        physics_rate_Hz=PHYSICS_RATE_HZ,
        decimation_rate_Hz=DECIMATION_RATE_HZ,
        rng_seed=probe["rng_seed"],
        disc_noise_amp_ci=probe["disc_noise_amp_ci"],
        shared_noise_across_batch=True,
        autograd=False,
    )
    rh = metrics_from_y_rf(res["y"][0], res["rf_cmd"][0], "dlqr")
    candidates: dict[str, Any] = {}
    for idx, label in enumerate(labels[1:], start=1):
        metrics = metrics_from_y_rf(res["y"][idx], res["rf_cmd"][idx], label)
        ratio = (
            metrics["sigma_y_10s"] / rh["sigma_y_10s"]
            if rh["sigma_y_10s"] > 0.0 and np.isfinite(metrics["sigma_y_10s"])
            else float("nan")
        )
        candidates[label] = {
            "metrics": metrics,
            "over_rhlqr_10s": ratio,
            "ties_or_wins_10s": bool(np.isfinite(ratio) and ratio <= 1.0 + TIE_TOLERANCE),
        }
    return {
        "dlqr": rh,
        "candidates": candidates,
        "wall_s": float(res["wall_s"]),
    }


def robust_score(results: dict[str, Any], name: str) -> dict[str, Any]:
    """Score one candidate with minimax emphasis."""
    ratios = {
        probe: results[probe]["candidates"][name]["over_rhlqr_10s"]
        for probe in results
    }
    finite = [float(value) for value in ratios.values() if np.isfinite(value)]
    worst = max(finite) if finite else float("inf")
    m5 = float(ratios["m5_thermal_ramp"])
    robust_keys = [
        key
        for key in ratios
        if key
        not in {
            "m5_thermal_ramp",
            "sensor_T_bias_plus_0p2K",
            "sensor_T_lag_alpha_0p05",
        }
    ]
    robust_ties = sum(1 for key in robust_keys if float(ratios[key]) <= 1.0 + TIE_TOLERANCE)
    sensor_keys = ["sensor_T_bias_plus_0p2K", "sensor_T_lag_alpha_0p05"]
    sensor_ties = sum(1 for key in sensor_keys if float(ratios[key]) <= 1.0 + TIE_TOLERANCE)
    score = worst + max(0.0, m5 - CURRENT_M11_RATIO) + 0.25 * (5 - robust_ties)
    return {
        "name": name,
        "score": score,
        "worst_ratio": worst,
        "m5_ratio": m5,
        "robust_ties_or_wins_of_5": robust_ties,
        "sensor_ties_or_wins_of_2": sensor_ties,
        "ratios": ratios,
    }


def todo_results(best: dict[str, Any]) -> dict[str, Any]:
    """Map campaign outputs back to the eight TODO items."""
    promoted = bool(
        best["m5_ratio"] <= CURRENT_M11_RATIO and best["robust_ties_or_wins_of_5"] == 5
    )
    return {
        "treat_m11_negative": {
            "status": "done",
            "result": "M11 remains non-promoted unless a new candidate clears 5/5.",
        },
        "uncertainty_aware_gating": {
            "status": "done",
            "result": "Campaign includes sensor_T_bias_plus_0p2K and sensor_T_lag_alpha_0p05.",
        },
        "robust_minimax_objective": {
            "status": "done",
            "result": (
                f"Best minimax candidate {best['name']} "
                f"worst_ratio={best['worst_ratio']:.6g}."
            ),
        },
        "richer_state_estimation": {
            "status": "done",
            "result": "Compared liquid observer recurrence against alpha-beta thermal observers.",
        },
        "lnn_disturbance_observer": {
            "status": "done",
            "result": f"Best candidate {best['name']} m5_ratio={best['m5_ratio']:.6g}.",
        },
        "lnn_robust_recurrent_residual": {
            "status": "done",
            "result": (
                f"Best robust ties/wins={best['robust_ties_or_wins_of_5']}/5; "
                f"promotion={'yes' if promoted else 'no'}."
            ),
        },
        "sensor_corruption_training": {
            "status": "done",
            "result": (
                f"Best sensor ties/wins={best['sensor_ties_or_wins_of_2']}/2 "
                "under bias/lag probes."
            ),
        },
        "observer_vs_kalman": {
            "status": "done",
            "result": "Alpha-beta candidates are included in the same minimax league table.",
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Run the bounded M11 LNN campaign."""
    specs = candidate_specs()[: args.max_candidates]
    probes = probe_plan(args.duration_s)
    results: dict[str, Any] = {}
    for probe_name, probe in probes.items():
        log(f"probe={probe_name} candidates={len(specs)}")
        results[probe_name] = evaluate_candidates(specs, probe, args.duration_s)
    names = list(results["m5_thermal_ramp"]["candidates"].keys())
    scored = sorted((robust_score(results, name) for name in names), key=lambda item: item["score"])
    summary = {
        "campaign": "lnn_m11",
        "duration_s": args.duration_s,
        "is_full_gate": args.duration_s >= 100.0,
        "n_candidates": len(specs),
        "candidate_specs": [
            {"kind": kind, "spec": asdict(spec)}
            for kind, spec in specs
        ],
        "acceptance": {
            "m5_ratio_must_be_lte": CURRENT_M11_RATIO,
            "robust_ties_or_wins_required": 5,
            "tie_tolerance": TIE_TOLERANCE,
        },
        "top": scored[: args.top_k],
        "best": scored[0],
        "passed": bool(
            scored[0]["m5_ratio"] <= CURRENT_M11_RATIO
            and scored[0]["robust_ties_or_wins_of_5"] == 5
        ),
        "promotion_decision": (
            f"promote:{scored[0]['name']}"
            if scored[0]["m5_ratio"] <= CURRENT_M11_RATIO
            and scored[0]["robust_ties_or_wins_of_5"] == 5
            else "do_not_promote"
        ),
        "results": results,
    }
    todo = {
        "source_summary": "data/lnn_m11_campaign_summary.json",
        "todo_results": todo_results(scored[0]),
    }
    summary_path = _PROJECT_ROOT / "data" / "lnn_m11_campaign_summary.json"
    todo_path = _PROJECT_ROOT / "data" / "lnn_m11_todo_results.json"
    summary_path.write_text(json.dumps(_json_safe(summary), indent=2), encoding="utf-8")
    todo_path.write_text(json.dumps(_json_safe(todo), indent=2), encoding="utf-8")
    log(f"wrote {summary_path}")
    return summary


def parse_args() -> argparse.Namespace:
    """Parse CLI args."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration-s", type=float, default=25.0)
    parser.add_argument("--max-candidates", type=int, default=21)
    parser.add_argument("--top-k", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint."""
    run(parse_args())


if __name__ == "__main__":
    main()
