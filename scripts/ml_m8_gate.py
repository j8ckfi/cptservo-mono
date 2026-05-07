"""ML residual vs RH-LQR robustness gate.

This is the M8-style follow-up for the narrow M5 ML residual win.  It compares
the promoted ``PhysicsResidualController`` directly against RH-LQR on a compact
set of perturbation probes and writes ``data/gate_ML_M8.json``.
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
from cptservo.policy.ml_research import (  # noqa: E402
    CfCDirectController,
    CfCFeedforwardConfig,
    CfCFeedforwardController,
    PhysicsResidualConfig,
    PhysicsResidualController,
)
from cptservo.twin.allan import overlapping_allan  # noqa: E402
from cptservo.twin.disturbance import Disturbance, DisturbanceTrace  # noqa: E402
from cptservo.twin.reduced import ReducedTwin  # noqa: E402

PHYSICS_RATE_HZ = 10_000.0
DECIMATION_RATE_HZ = 1_000.0
DISC_NOISE_AMP_CI = 7.0e-4
RNG_SEED = 4242
TAUS = [1.0, 10.0, 100.0]


class _BatchDispatchController:
    """Dispatch batched runner callbacks to per-trajectory controllers."""

    def __init__(self, controllers: list[Any]) -> None:
        self.controllers = controllers
        self._call_idx = 0

    def reset(self) -> None:
        """Reset all child controllers and callback index."""
        self._call_idx = 0
        for controller in self.controllers:
            if hasattr(controller, "reset"):
                controller.reset()

    def step(self, error: float, env: dict[str, float] | None = None) -> tuple[float, float]:
        """Route each batch callback to the matching child controller."""
        idx = self._call_idx % len(self.controllers)
        self._call_idx += 1
        controller = self.controllers[idx]
        try:
            return controller.step(error, env)
        except TypeError:
            return controller.step(error)


def log(msg: str) -> None:
    """Timestamped console log."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def make_thermal_ramp_trace(
    duration_s: float,
    slope_scale: float,
    seed: int,
) -> DisturbanceTrace:
    """Build a thermal ramp trace with scaled ramp amplitude."""
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
    """Build all_stacked trace with scaled B and intensity drift."""
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


def evaluate_pair(
    ml_controller: Any,
    trace: DisturbanceTrace,
    duration_s: float,
    disc_noise_amp_ci: float,
    rng_seed: int,
    twin: ReducedTwin | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run RH-LQR and ML as a paired two-element batch."""
    controller = _BatchDispatchController(
        [RHLQRController.from_recipe(), ml_controller]
    )
    res = run_batched_loop(
        twin=twin or make_calibrated_twin(),
        controller=controller,
        disturbance_traces=[trace, trace],
        duration_s=duration_s,
        physics_rate_Hz=PHYSICS_RATE_HZ,
        decimation_rate_Hz=DECIMATION_RATE_HZ,
        rng_seed=rng_seed,
        disc_noise_amp_ci=disc_noise_amp_ci,
        shared_noise_across_batch=True,
        autograd=False,
    )

    def metrics(label: str, batch_idx: int) -> dict[str, Any]:
        y = res["y"][batch_idx] - float(np.mean(res["y"][batch_idx]))
        allan = overlapping_allan(y, DECIMATION_RATE_HZ, TAUS)
        return {
            "label": label,
            "sigma_y_1s": float(allan.get(1.0, float("nan"))),
            "sigma_y_10s": float(allan.get(10.0, float("nan"))),
            "sigma_y_100s": float(allan.get(100.0, float("nan"))),
            "rf_abs_max_Hz": float(np.max(np.abs(res["rf_cmd"][batch_idx]))),
            "wall_s": float(res["wall_s"]),
        }

    return metrics("rh_lqr", 0), metrics("ml_controller", 1)


def run_probe(
    name: str,
    trace: DisturbanceTrace,
    duration_s: float,
    controller_factory: Any,
    disc_noise_amp_ci: float = DISC_NOISE_AMP_CI,
    rng_seed: int = RNG_SEED,
    tie_tolerance: float = 0.0,
    twin: ReducedTwin | None = None,
) -> dict[str, Any]:
    """Compare RH-LQR and ML residual on one probe."""
    log(f"probe={name}")
    lqr_metrics, ml_metrics = evaluate_pair(
        controller_factory(),
        trace,
        duration_s,
        disc_noise_amp_ci,
        rng_seed,
        twin,
    )
    lqr_10 = lqr_metrics["sigma_y_10s"]
    ml_10 = ml_metrics["sigma_y_10s"]
    ratio = ml_10 / lqr_10 if np.isfinite(ml_10) and lqr_10 > 0.0 else float("nan")
    return {
        "probe": name,
        "rh_lqr": lqr_metrics,
        "ml": ml_metrics,
        "ml_over_rhlqr_10s": ratio,
        "ml_ties_or_wins_10s": bool(np.isfinite(ratio) and ratio <= 1.0 + tie_tolerance),
        "disc_noise_amp_ci": disc_noise_amp_ci,
        "rng_seed": rng_seed,
        "tie_tolerance": tie_tolerance,
    }


def run_gate(args: argparse.Namespace) -> dict[str, Any]:
    """Run the ML M8 gate and write the JSON artifact."""
    duration_s = float(args.duration_s)
    residual_config = PhysicsResidualConfig(
        k_T_Hz_per_K=args.k_T,
        k_B_Hz_per_uT=args.k_B,
        k_I_Hz_per_norm=args.k_I,
        k_dT_Hz_per_K_s=args.k_dT,
        k_dB_Hz_per_uT_s=args.k_dB,
        k_dI_Hz_per_norm_s=args.k_dI,
        residual_limit_Hz=args.residual_limit_Hz,
    )
    if args.controller == "cfc":
        cfc_config = CfCFeedforwardConfig(
            hidden_size=args.hidden_size,
            base_residual=residual_config,
            cfc_residual_limit_Hz=args.cfc_residual_limit_Hz,
        )

        def controller_factory() -> CfCFeedforwardController:
            if args.checkpoint:
                return CfCFeedforwardController.load(args.checkpoint)
            return CfCFeedforwardController(cfc_config)

        controller_config: dict[str, Any] = {
            "type": "cfc_feedforward",
            "hidden_size": cfc_config.hidden_size,
            "base_residual": residual_config.__dict__,
            "cfc_residual_limit_Hz": cfc_config.cfc_residual_limit_Hz,
            "checkpoint": args.checkpoint,
        }
    elif args.controller == "cfc_direct":

        def controller_factory() -> CfCDirectController:
            if not args.checkpoint:
                raise ValueError("--checkpoint is required for --controller cfc_direct")
            return CfCDirectController.load(args.checkpoint)

        controller_config = {
            "type": "cfc_direct",
            "checkpoint": args.checkpoint,
        }
    else:

        def controller_factory() -> PhysicsResidualController:
            return PhysicsResidualController(residual_config)

        controller_config = {
            "type": "physics_residual_linear",
            "base_residual": residual_config.__dict__,
        }

    probes: dict[str, Any] = {}
    probes["m5_thermal_ramp"] = run_probe(
        "m5_thermal_ramp",
        Disturbance.from_recipe("thermal_ramp").generate(
            duration_s=duration_s,
            sample_rate_Hz=PHYSICS_RATE_HZ,
            seed=42,
        ),
        duration_s,
        controller_factory,
        rng_seed=42,
    )
    probes["ood_3x_thermal_slope"] = run_probe(
        "ood_3x_thermal_slope",
        make_thermal_ramp_trace(duration_s, 3.0, RNG_SEED),
        duration_s,
        controller_factory,
        tie_tolerance=args.tie_tolerance,
    )
    probes["high_disc_noise_3x"] = run_probe(
        "high_disc_noise_3x",
        Disturbance.from_recipe("thermal_ramp").generate(
            duration_s=duration_s,
            sample_rate_Hz=PHYSICS_RATE_HZ,
            seed=RNG_SEED,
        ),
        duration_s,
        controller_factory,
        disc_noise_amp_ci=3.0 * DISC_NOISE_AMP_CI,
        tie_tolerance=args.tie_tolerance,
    )
    probes["low_disc_noise_third"] = run_probe(
        "low_disc_noise_third",
        Disturbance.from_recipe("thermal_ramp").generate(
            duration_s=duration_s,
            sample_rate_Hz=PHYSICS_RATE_HZ,
            seed=RNG_SEED,
        ),
        duration_s,
        controller_factory,
        disc_noise_amp_ci=DISC_NOISE_AMP_CI / 3.0,
        tie_tolerance=args.tie_tolerance,
    )
    probes["reality_gap_5pct"] = run_probe(
        "reality_gap_5pct",
        Disturbance.from_recipe("thermal_ramp").generate(
            duration_s=duration_s,
            sample_rate_Hz=PHYSICS_RATE_HZ,
            seed=RNG_SEED,
        ),
        duration_s,
        controller_factory,
        tie_tolerance=args.tie_tolerance,
        twin=make_perturbed_twin(0.05, RNG_SEED + 1),
    )
    probes["worst_case_2x_stacked"] = run_probe(
        "worst_case_2x_stacked",
        make_all_stacked_trace(duration_s, 2.0, 2.0, RNG_SEED),
        duration_s,
        controller_factory,
        tie_tolerance=args.tie_tolerance,
    )

    robustness_keys = [key for key in probes if key != "m5_thermal_ramp"]
    n_robust_ties_or_wins = sum(
        1 for key in robustness_keys if probes[key]["ml_ties_or_wins_10s"]
    )
    m5_pass = bool(probes["m5_thermal_ramp"]["ml_ties_or_wins_10s"])
    gate_pass = bool(m5_pass and n_robust_ties_or_wins >= 3)
    gate_doc = {
        "milestone": "ML_M8",
        "duration_s": duration_s,
        "is_full_gate": duration_s >= 100.0,
        "controller": controller_config["type"],
        "config": controller_config,
        "robustness_tie_tolerance": args.tie_tolerance,
        "results": probes,
        "m5_ml_ties_or_wins": m5_pass,
        "n_robust_ties_or_wins_of_5": n_robust_ties_or_wins,
        "gate_pass": gate_pass,
        "gate_note": (
            "Full production-readiness claim requires duration_s >= 100 and "
            "paired review of all probe metrics."
        ),
    }
    out_path = _PROJECT_ROOT / "data" / "gate_ML_M8.json"
    out_path.write_text(json.dumps(_json_safe(gate_doc), indent=2), encoding="utf-8")
    log(f"wrote {out_path}")
    return gate_doc


def _json_safe(value: Any) -> Any:
    """Convert NaN/Inf floats to null for strict JSON."""
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    return value


def parse_args() -> argparse.Namespace:
    """Parse CLI args."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--controller",
        choices=["linear", "cfc", "cfc_direct"],
        default="cfc",
    )
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--duration-s", type=float, default=25.0)
    parser.add_argument("--hidden-size", type=int, default=8)
    parser.add_argument("--k-T", type=float, default=0.05)
    parser.add_argument("--k-B", type=float, default=0.0)
    parser.add_argument("--k-I", type=float, default=0.0)
    parser.add_argument("--k-dT", type=float, default=0.0)
    parser.add_argument("--k-dB", type=float, default=0.0)
    parser.add_argument("--k-dI", type=float, default=0.0)
    parser.add_argument("--residual-limit-Hz", type=float, default=5.0)
    parser.add_argument("--cfc-residual-limit-Hz", type=float, default=1.0)
    parser.add_argument("--tie-tolerance", type=float, default=0.005)
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint."""
    run_gate(parse_args())


if __name__ == "__main__":
    main()
