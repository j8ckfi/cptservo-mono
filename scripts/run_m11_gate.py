"""M11 direct-CfC promotion gate.

This gate compares RH-LQR, the current promoted direct-CfC checkpoint, and one
or more candidate direct-CfC checkpoints through the canonical batched runner
with shared noise.  It never overwrites historical ML-M8 artifacts.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_SCRIPT_DIR))

from run_m3_m4_gates import make_calibrated_twin  # noqa: E402

from cptservo.baselines.rh_lqr import RHLQRController  # noqa: E402
from cptservo.evaluation.batched_runner import run_batched_loop  # noqa: E402
from cptservo.policy.ml_research import CfCDirectController  # noqa: E402
from cptservo.twin.allan import overlapping_allan  # noqa: E402
from cptservo.twin.disturbance import Disturbance, DisturbanceTrace  # noqa: E402
from cptservo.twin.reduced import ReducedTwin  # noqa: E402

PHYSICS_RATE_HZ = 10_000.0
DECIMATION_RATE_HZ = 1_000.0
DISC_NOISE_AMP_CI = 7.0e-4
RNG_SEED = 4242
TAUS = [1.0, 10.0, 100.0]
TIE_TOLERANCE = 0.005
RF_LIMIT_HZ = 1000.0
CURRENT_PROMOTED_M5_RATIO = 0.9682237494386449
DEFAULT_BASELINE_CHECKPOINT = "data/cfc_direct_T1p5_kp1p5_ki1p0.json"


class BatchDispatchController:
    """Route batched runner callbacks to one controller per batch row."""

    def __init__(self, controllers: list[Any]) -> None:
        self.controllers = controllers
        self._call_idx = 0

    def reset(self) -> None:
        """Reset callback routing and all child controllers."""
        self._call_idx = 0
        for controller in self.controllers:
            if hasattr(controller, "reset"):
                controller.reset()

    def step(self, error: float, env: dict[str, float] | None = None) -> tuple[float, float]:
        """Dispatch this step to the controller matching the batch row."""
        idx = self._call_idx % len(self.controllers)
        self._call_idx += 1
        controller = self.controllers[idx]
        try:
            return controller.step(error, env)
        except TypeError:
            return controller.step(error)


class SensorTransformController:
    """Apply gate-only sensor bias and lag before calling a controller."""

    def __init__(
        self,
        controller: Any,
        bias: dict[str, float] | None = None,
        lag_alpha: float = 1.0,
    ) -> None:
        self.controller = controller
        self.bias = bias or {}
        self.lag_alpha = float(np.clip(lag_alpha, 0.0, 1.0))
        self._lagged_env: dict[str, float] | None = None

    def reset(self) -> None:
        """Reset wrapper and wrapped controller state."""
        self._lagged_env = None
        if hasattr(self.controller, "reset"):
            self.controller.reset()

    def transform_env(self, env: dict[str, float] | None) -> dict[str, float]:
        """Return the biased and lagged sensor environment."""
        raw = dict(env or {})
        biased = {
            "T_K": float(raw.get("T_K", 333.15)) + float(self.bias.get("T_K", 0.0)),
            "B_uT": float(raw.get("B_uT", 50.0)) + float(self.bias.get("B_uT", 0.0)),
            "I_norm": float(raw.get("I_norm", 1.0)) + float(self.bias.get("I_norm", 0.0)),
        }
        if self._lagged_env is None:
            self._lagged_env = dict(biased)
        else:
            for key, value in biased.items():
                self._lagged_env[key] = (
                    self.lag_alpha * value
                    + (1.0 - self.lag_alpha) * self._lagged_env[key]
                )
        return dict(self._lagged_env)

    def step(self, error: float, env: dict[str, float] | None = None) -> tuple[float, float]:
        """Controller protocol with transformed env sensors."""
        return self.controller.step(error, self.transform_env(env))


def log(msg: str) -> None:
    """Timestamped console log."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


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
    """Build an all-stacked trace with scaled B and intensity drift."""
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


def make_perturbed_twin(frac: float, rng_seed: int) -> ReducedTwin:
    """Return a calibrated twin with selected scalar parameters perturbed."""
    twin = make_calibrated_twin()
    rng = np.random.default_rng(rng_seed)
    for attr in (
        "light_shift_coeff",
        "buffer_gas_shift_coeff",
        "lumped_zeeman_coeff",
        "temperature_coeff_Hz_per_K",
    ):
        if hasattr(twin, attr):
            old = float(getattr(twin, attr))
            setattr(twin, attr, old * (1.0 + rng.uniform(-frac, frac)))
    return twin


def _probe_plan(duration_s: float) -> dict[str, dict[str, Any]]:
    """Build all M11 probe definitions."""
    thermal_4242 = Disturbance.from_recipe("thermal_ramp").generate(
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
            "disc_noise_amp_ci": DISC_NOISE_AMP_CI,
            "rng_seed": 42,
            "twin": None,
            "tie_tolerance": 0.0,
        },
        "ood_3x_thermal_slope": {
            "trace": make_thermal_ramp_trace(duration_s, 3.0, RNG_SEED),
            "disc_noise_amp_ci": DISC_NOISE_AMP_CI,
            "rng_seed": RNG_SEED,
            "twin": None,
            "tie_tolerance": TIE_TOLERANCE,
        },
        "high_disc_noise_3x": {
            "trace": thermal_4242,
            "disc_noise_amp_ci": 3.0 * DISC_NOISE_AMP_CI,
            "rng_seed": RNG_SEED,
            "twin": None,
            "tie_tolerance": TIE_TOLERANCE,
        },
        "low_disc_noise_third": {
            "trace": thermal_4242,
            "disc_noise_amp_ci": DISC_NOISE_AMP_CI / 3.0,
            "rng_seed": RNG_SEED,
            "twin": None,
            "tie_tolerance": TIE_TOLERANCE,
        },
        "reality_gap_5pct": {
            "trace": thermal_4242,
            "disc_noise_amp_ci": DISC_NOISE_AMP_CI,
            "rng_seed": RNG_SEED,
            "twin": make_perturbed_twin(0.05, RNG_SEED + 1),
            "tie_tolerance": TIE_TOLERANCE,
        },
        "worst_case_2x_stacked": {
            "trace": make_all_stacked_trace(duration_s, 2.0, 2.0, RNG_SEED),
            "disc_noise_amp_ci": DISC_NOISE_AMP_CI,
            "rng_seed": RNG_SEED,
            "twin": None,
            "tie_tolerance": TIE_TOLERANCE,
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


def parse_candidate_arg(raw: str) -> tuple[str, Path]:
    """Parse LABEL=PATH or PATH candidate syntax."""
    if "=" in raw:
        label, path = raw.split("=", 1)
        return label.strip(), Path(path.strip())
    path = Path(raw.strip())
    return path.stem, path


def resolve_project_path(path: str | Path) -> Path:
    """Resolve repo-relative paths used in saved artifacts and CLI args."""
    p = Path(path)
    if not p.is_absolute():
        p = _PROJECT_ROOT / p
    return p


def default_baseline_checkpoint() -> Path:
    """Return the current promoted direct-CfC checkpoint path."""
    summary = _PROJECT_ROOT / "data" / "cfc_improvement_final_summary.json"
    if summary.exists():
        data = json.loads(summary.read_text(encoding="utf-8-sig"))
        checkpoint = data.get("promoted_checkpoint")
        if checkpoint:
            return resolve_project_path(checkpoint)
    return resolve_project_path(DEFAULT_BASELINE_CHECKPOINT)


def controller_from_checkpoint(
    checkpoint: Path,
    sensor_bias: dict[str, float],
    sensor_lag_alpha: float,
) -> SensorTransformController:
    """Load a direct-CfC checkpoint wrapped with gate-only sensor transforms."""
    return SensorTransformController(
        CfCDirectController.load(checkpoint),
        bias=sensor_bias,
        lag_alpha=sensor_lag_alpha,
    )


def evaluate_probe(
    labels: list[str],
    checkpoints: list[Path],
    probe: dict[str, Any],
    duration_s: float,
    sensor_bias: dict[str, float],
    sensor_lag_alpha: float,
) -> dict[str, Any]:
    """Evaluate all controllers for one probe as a paired shared-noise batch."""
    controllers: list[Any] = [RHLQRController.from_recipe()]
    controllers.extend(
        controller_from_checkpoint(path, sensor_bias, sensor_lag_alpha)
        for path in checkpoints
    )
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
    rh = metrics_from_y_rf(res["y"][0], res["rf_cmd"][0], "rh_lqr")
    controller_results: dict[str, Any] = {}
    for idx, label in enumerate(labels, start=1):
        metrics = metrics_from_y_rf(res["y"][idx], res["rf_cmd"][idx], label)
        ratio = (
            metrics["sigma_y_10s"] / rh["sigma_y_10s"]
            if rh["sigma_y_10s"] > 0.0 and np.isfinite(metrics["sigma_y_10s"])
            else float("nan")
        )
        controller_results[label] = {
            "metrics": metrics,
            "over_rhlqr_10s": ratio,
            "ties_or_wins_10s": bool(
                np.isfinite(ratio) and ratio <= 1.0 + probe["tie_tolerance"]
            ),
        }
    return {
        "rh_lqr": rh,
        "controllers": controller_results,
        "disc_noise_amp_ci": probe["disc_noise_amp_ci"],
        "rng_seed": probe["rng_seed"],
        "tie_tolerance": probe["tie_tolerance"],
        "wall_s": float(res["wall_s"]),
    }


def summarize_controller(
    label: str,
    checkpoint: Path,
    probe_results: dict[str, Any],
) -> dict[str, Any]:
    """Create the public M11 summary for one controller."""
    probe_ratios = {
        key: value["controllers"][label]["over_rhlqr_10s"]
        for key, value in probe_results.items()
    }
    robust_keys = [key for key in probe_ratios if key != "m5_thermal_ramp"]
    robust_ties = sum(
        1 for key in robust_keys if probe_results[key]["controllers"][label]["ties_or_wins_10s"]
    )
    rf_abs_max = max(
        float(value["controllers"][label]["metrics"]["rf_abs_max_Hz"])
        for value in probe_results.values()
    )
    m5_ratio = float(probe_ratios["m5_thermal_ramp"])
    gate_pass = bool(
        m5_ratio <= CURRENT_PROMOTED_M5_RATIO
        and robust_ties == 5
        and rf_abs_max < 0.95 * RF_LIMIT_HZ
    )
    return {
        "milestone": "M11",
        "controller": label,
        "checkpoint": str(checkpoint),
        "m5_ratio": m5_ratio,
        "probe_ratios": probe_ratios,
        "robust_ties_or_wins": robust_ties,
        "rf_abs_max_Hz": rf_abs_max,
        "gate_pass": gate_pass,
        "promotion_decision": "promote" if gate_pass else "do_not_promote",
    }


def run_gate(args: argparse.Namespace) -> dict[str, Any]:
    """Run the M11 gate and write non-overwriting artifacts."""
    duration_s = float(args.duration_s)
    baseline_checkpoint = resolve_project_path(args.baseline_checkpoint) if (
        args.baseline_checkpoint
    ) else default_baseline_checkpoint()
    candidate_items = [parse_candidate_arg(item) for item in args.candidate_checkpoint]
    labels = ["current_promoted"]
    checkpoints = [baseline_checkpoint]
    for label, path in candidate_items:
        labels.append(label)
        checkpoints.append(resolve_project_path(path))
    sensor_bias = {
        "T_K": args.sensor_bias_T_K,
        "B_uT": args.sensor_bias_B_uT,
        "I_norm": args.sensor_bias_I_norm,
    }
    probes: dict[str, Any] = {}
    for name, probe in _probe_plan(duration_s).items():
        log(f"probe={name} controllers={len(labels)}")
        probes[name] = evaluate_probe(
            labels,
            checkpoints,
            probe,
            duration_s,
            sensor_bias,
            args.sensor_lag_alpha,
        )

    summaries = {
        label: summarize_controller(label, path, probes)
        for label, path in zip(labels, checkpoints)
    }
    candidate_labels = [label for label in labels if label != "current_promoted"]
    promoted_candidates = [
        label for label in candidate_labels if summaries[label]["gate_pass"]
    ]
    best_label = min(
        labels,
        key=lambda item: (
            summaries[item]["m5_ratio"],
            -summaries[item]["robust_ties_or_wins"],
        ),
    )
    gate_pass = bool(promoted_candidates)
    gate_doc = {
        "milestone": "M11",
        "duration_s": duration_s,
        "is_full_gate": duration_s >= 100.0,
        "baseline_checkpoint": str(baseline_checkpoint),
        "candidate_checkpoints": {
            label: str(path) for label, path in zip(labels, checkpoints)
        },
        "acceptance": {
            "m5_ratio_must_be_lte": CURRENT_PROMOTED_M5_RATIO,
            "robust_ties_or_wins_required": 5,
            "tie_tolerance": TIE_TOLERANCE,
            "rf_abs_max_must_be_lt_Hz": 0.95 * RF_LIMIT_HZ,
        },
        "sensor_transform": {
            "bias": sensor_bias,
            "lag_alpha": args.sensor_lag_alpha,
        },
        "summaries": summaries,
        "best_label": best_label,
        "gate_pass": gate_pass,
        "promotion_decision": (
            f"promote:{promoted_candidates[0]}"
            if promoted_candidates
            else "do_not_promote"
        ),
        "results": probes,
    }
    out_path = _PROJECT_ROOT / "data" / "gate_M11.json"
    out_path.write_text(json.dumps(_json_safe(gate_doc), indent=2), encoding="utf-8")
    for label in labels:
        label_slug = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in label)
        label_doc = dict(gate_doc)
        label_doc["public_summary"] = summaries[label]
        label_path = _PROJECT_ROOT / "data" / f"gate_M11_{label_slug}.json"
        label_path.write_text(json.dumps(_json_safe(label_doc), indent=2), encoding="utf-8")
    log(f"wrote {out_path}")
    return gate_doc


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


def parse_args() -> argparse.Namespace:
    """Parse CLI args."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration-s", type=float, default=100.0)
    parser.add_argument("--baseline-checkpoint", default="")
    parser.add_argument(
        "--candidate-checkpoint",
        action="append",
        default=[],
        help="Candidate checkpoint as LABEL=PATH or PATH. May be repeated.",
    )
    parser.add_argument("--sensor-bias-T-K", type=float, default=0.0)
    parser.add_argument("--sensor-bias-B-uT", type=float, default=0.0)
    parser.add_argument("--sensor-bias-I-norm", type=float, default=0.0)
    parser.add_argument(
        "--sensor-lag-alpha",
        type=float,
        default=1.0,
        help="1.0 disables lag; lower values apply first-order lag to controller sensors.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint."""
    run_gate(parse_args())


if __name__ == "__main__":
    main()
