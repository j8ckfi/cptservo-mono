"""Multi-method campaign to improve the direct CfC controller.

The campaign is intentionally local-first.  It screens compact direct-CfC
controllers through the canonical batched runner, saves the best checkpoint,
and screens candidates against the same probe family used by the M11 gate.
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

from run_m3_m4_gates import make_calibrated_twin  # noqa: E402

from cptservo.baselines.rh_lqr import RHLQRController  # noqa: E402
from cptservo.evaluation.batched_runner import run_batched_loop  # noqa: E402
from cptservo.policy.ml_research import (  # noqa: E402
    CfCDirectConfig,
    CfCDirectController,
    PhysicsResidualConfig,
)
from cptservo.twin.allan import overlapping_allan  # noqa: E402
from cptservo.twin.disturbance import Disturbance, DisturbanceTrace  # noqa: E402
from cptservo.twin.reduced import ReducedTwin  # noqa: E402

PHYSICS_RATE_HZ = 10_000.0
DECIMATION_RATE_HZ = 1_000.0
DISC_NOISE_AMP_CI = 7.0e-4
RNG_SEED = 42
TAUS = [1.0, 10.0, 100.0]
CURRENT_PROMOTED_M5_RATIO = 0.9682237494386449
CURRENT_PROMOTED_CHECKPOINT = "data/cfc_direct_T1p5_kp1p5_ki1p0.json"
TIE_TOLERANCE = 0.005
RF_LIMIT_HZ = 1000.0


@dataclass(frozen=True)
class StructuredSpec:
    """Direct-CfC feature readout coefficients in physical units."""

    name: str
    k_T: float = 0.5
    k_B: float = 0.0
    k_I: float = 0.0
    k_dT: float = 0.0
    k_dB: float = 0.0
    k_dI: float = 0.0
    kp_scale: float = 1.0
    ki_scale: float = 1.0
    kd_error: float = 0.0
    k_mean_error: float = 0.0
    k_std_error: float = 0.0
    k_last_rf: float = 0.0
    k_delta_rf: float = 0.0
    bias_Hz: float = 0.0


class BatchDispatchController:
    """Route batched runner callbacks to one controller per batch row."""

    def __init__(self, controllers: list[Any]) -> None:
        self.controllers = controllers
        self._call_idx = 0

    def reset(self) -> None:
        self._call_idx = 0
        for controller in self.controllers:
            if hasattr(controller, "reset"):
                controller.reset()

    def step(self, error: float, env: dict[str, float] | None = None) -> tuple[float, float]:
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


def _json_safe(value: Any) -> Any:
    """Convert NaN/Inf floats to null for strict JSON."""
    if isinstance(value, float):
        return value if np.isfinite(value) else None
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
    """Build a thermal-ramp trace with scaled thermal amplitude."""
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
    """Build an all-stacked trace with scaled B and intensity drifts."""
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


def apply_structured_readout(controller: CfCDirectController, spec: StructuredSpec) -> None:
    """Install physical coefficients into the direct-CfC feature skip readout."""
    lqr_gain = RHLQRController.from_recipe().K
    kp = float(lqr_gain[0, 0]) * spec.kp_scale
    ki = float(lqr_gain[0, 1]) * spec.ki_scale
    for value in controller.weights.values():
        value[:] = 0.0
    w = controller.weights["W_feat_out"]
    w[0] = spec.k_T * 10.0
    w[1] = spec.k_B * 10.0
    w[2] = spec.k_I
    w[3] = spec.k_dT * 10.0
    w[4] = spec.k_dB * 10.0
    w[5] = spec.k_dI
    w[6] = kp / 1.0e3
    w[7] = spec.kd_error / 1.0e3
    w[8] = ki / 1.0e6
    w[9] = spec.k_mean_error / 1.0e3
    w[10] = spec.k_std_error / 1.0e3
    w[11] = spec.k_last_rf * controller.config.rf_limit_Hz
    w[12] = spec.k_delta_rf * 10.0
    w[13] = spec.bias_Hz


def make_direct_controller(spec: StructuredSpec, hidden_size: int = 8) -> CfCDirectController:
    """Create one structured direct-CfC controller candidate."""
    controller = CfCDirectController(
        CfCDirectConfig(
            hidden_size=hidden_size,
            seed=1729,
            sensor_config=PhysicsResidualConfig(),
            output_feature_skip=True,
        )
    )
    apply_structured_readout(controller, spec)
    return controller


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


def evaluate_specs(
    specs: list[StructuredSpec],
    trace: DisturbanceTrace,
    duration_s: float,
    rng_seed: int,
    disc_noise_amp_ci: float = DISC_NOISE_AMP_CI,
    twin: ReducedTwin | None = None,
) -> dict[str, Any]:
    """Evaluate RH-LQR and candidate specs as one paired batch."""
    controllers: list[Any] = [RHLQRController.from_recipe()]
    controllers.extend(make_direct_controller(spec) for spec in specs)
    controller = BatchDispatchController(controllers)
    res = run_batched_loop(
        twin=twin or make_calibrated_twin(),
        controller=controller,
        disturbance_traces=[trace for _ in controllers],
        duration_s=duration_s,
        physics_rate_Hz=PHYSICS_RATE_HZ,
        decimation_rate_Hz=DECIMATION_RATE_HZ,
        rng_seed=rng_seed,
        disc_noise_amp_ci=disc_noise_amp_ci,
        shared_noise_across_batch=True,
        autograd=False,
    )
    rh = metrics_from_y_rf(res["y"][0], res["rf_cmd"][0], "rh_lqr")
    candidates: dict[str, Any] = {}
    for idx, spec in enumerate(specs, start=1):
        ml = metrics_from_y_rf(res["y"][idx], res["rf_cmd"][idx], spec.name)
        ratio = (
            ml["sigma_y_10s"] / rh["sigma_y_10s"]
            if rh["sigma_y_10s"] > 0.0 and np.isfinite(ml["sigma_y_10s"])
            else float("nan")
        )
        tau1_ratio = (
            ml["sigma_y_1s"] / rh["sigma_y_1s"]
            if rh["sigma_y_1s"] > 0.0 and np.isfinite(ml["sigma_y_1s"])
            else float("nan")
        )
        candidates[spec.name] = {
            "spec": asdict(spec),
            "ml": ml,
            "ml_over_rhlqr_10s": ratio,
            "ml_over_rhlqr_1s": tau1_ratio,
        }
    return {
        "rh_lqr": rh,
        "candidates": candidates,
        "wall_s": float(res["wall_s"]),
    }


def generate_structured_specs(max_specs: int) -> list[StructuredSpec]:
    """Generate deterministic structured candidate specs."""
    specs: list[StructuredSpec] = [
        StructuredSpec(
            name="current_promoted_T1p5_kp1p5_ki1p0_dTm0p01",
            k_T=1.5,
            k_dT=-0.01,
            kp_scale=1.5,
            ki_scale=1.0,
        ),
        StructuredSpec(name="baseline_direct_T0p5", k_T=0.5),
        StructuredSpec(name="direct_T0p5_dTm0p01", k_T=0.5, k_dT=-0.01),
    ]
    for k_t in [1.2, 1.3, 1.4, 1.5, 1.6]:
        for kp_scale in [1.35, 1.45, 1.5, 1.55]:
            specs.append(
                StructuredSpec(
                    name=(
                        f"curated_T{k_t:g}_dTm0p01_kp{kp_scale:g}_ki1"
                    ).replace(".", "p"),
                    k_T=k_t,
                    k_dT=-0.01,
                    kp_scale=kp_scale,
                    ki_scale=1.0,
                )
            )
    for k_t in [1.3, 1.4, 1.5]:
        for k_dt in [-0.04, -0.02, -0.005, 0.0]:
            specs.append(
                StructuredSpec(
                    name=f"curated_T{k_t:g}_dT{k_dt:g}_kp1p5".replace(
                        ".", "p"
                    ).replace("-", "m"),
                    k_T=k_t,
                    k_dT=k_dt,
                    kp_scale=1.5,
                    ki_scale=1.0,
                )
            )
    for k_t in [1.05, 1.15, 1.25, 1.35, 1.45, 1.5, 1.55, 1.65, 1.8]:
        for k_dt in [-0.06, -0.03, -0.02, -0.01, -0.005, 0.0]:
            for kp_scale in [1.25, 1.35, 1.45, 1.5, 1.55, 1.65]:
                for ki_scale in [0.85, 1.0, 1.15]:
                    specs.append(
                        StructuredSpec(
                            name=(
                                f"m11_T{k_t:g}_dT{k_dt:g}_kp{kp_scale:g}_ki{ki_scale:g}"
                            )
                            .replace(".", "p")
                            .replace("-", "m"),
                            k_T=k_t,
                            k_dT=k_dt,
                            kp_scale=kp_scale,
                            ki_scale=ki_scale,
                        )
                    )
    for kd in [-2000.0, -750.0, -250.0, 250.0, 750.0, 2000.0]:
        for k_last in [-0.02, -0.005, 0.005, 0.02]:
            specs.append(
                StructuredSpec(
                    name=(
                        f"m11_memory_kd{kd:g}_last{k_last:g}"
                    ).replace(".", "p").replace("-", "m"),
                    k_T=1.45,
                    k_dT=-0.01,
                    kp_scale=1.5,
                    ki_scale=1.0,
                    kd_error=kd,
                    k_last_rf=k_last,
                )
            )
    for k_t in [0.25, 0.35, 0.45, 0.5, 0.55, 0.65, 0.8, 1.0]:
        for k_dt in [-0.04, -0.02, -0.01, -0.005, 0.0, 0.005]:
            specs.append(
                StructuredSpec(
                    name=f"grid_T{k_t:g}_dT{k_dt:g}".replace(".", "p").replace("-", "m"),
                    k_T=k_t,
                    k_dT=k_dt,
                )
            )
    for kp_scale in [0.96, 0.98, 1.0, 1.02, 1.04]:
        for ki_scale in [0.85, 1.0, 1.15]:
            specs.append(
                StructuredSpec(
                    name=(
                        f"fb_T0p5_dTm0p01_kp{kp_scale:g}_ki{ki_scale:g}"
                    ).replace(".", "p"),
                    k_T=0.5,
                    k_dT=-0.01,
                    kp_scale=kp_scale,
                    ki_scale=ki_scale,
                )
            )
    for k_b in [-0.05, -0.02, 0.02, 0.05]:
        for k_i in [-1.0, -0.25, 0.5, 1.5]:
            specs.append(
                StructuredSpec(
                    name=f"sensor_B{k_b:g}_I{k_i:g}".replace(".", "p").replace("-", "m"),
                    k_T=0.5,
                    k_dT=-0.01,
                    k_B=k_b,
                    k_I=k_i,
                )
            )
    for kd in [-2000.0, -1000.0, -250.0, 250.0, 1000.0, 2000.0]:
        specs.append(
            StructuredSpec(
                name=f"derr_{kd:g}".replace(".", "p").replace("-", "m"),
                k_T=0.5,
                k_dT=-0.01,
                kd_error=kd,
            )
        )
    for k_last in [-0.03, -0.01, 0.01, 0.03]:
        specs.append(
            StructuredSpec(
                name=f"action_memory_{k_last:g}".replace(".", "p").replace("-", "m"),
                k_T=0.5,
                k_dT=-0.01,
                k_last_rf=k_last,
            )
        )
    seen: set[str] = set()
    unique_specs: list[StructuredSpec] = []
    for spec in specs:
        if spec.name not in seen:
            seen.add(spec.name)
            unique_specs.append(spec)
    return unique_specs[:max_specs]


def score_candidate(probe_results: dict[str, Any], candidate_name: str) -> float:
    """Composite short-screen score. Lower is better."""
    m5 = probe_results["m5_thermal_ramp"]["candidates"][candidate_name]
    m5_ratio = _finite_ratio(m5["ml_over_rhlqr_10s"])
    tau1_ratio = _finite_ratio(m5["ml_over_rhlqr_1s"], default=1.0)
    rf_abs_max = _finite_ratio(m5["ml"]["rf_abs_max_Hz"], default=RF_LIMIT_HZ)
    robust_keys = [key for key in probe_results if key != "m5_thermal_ramp"]
    robust_ratios = [
        _finite_ratio(probe_results[key]["candidates"][candidate_name]["ml_over_rhlqr_10s"])
        for key in robust_keys
    ]
    robust_misses = sum(1 for ratio in robust_ratios if ratio > 1.0 + TIE_TOLERANCE)
    worst_excess = max(0.0, max(robust_ratios) - (1.0 + TIE_TOLERANCE))
    m5_regression = max(0.0, m5_ratio - CURRENT_PROMOTED_M5_RATIO)
    tau1_penalty = max(0.0, tau1_ratio - 1.02)
    headroom_penalty = max(0.0, rf_abs_max / RF_LIMIT_HZ - 0.2)
    return (
        m5_ratio
        + 3.0 * m5_regression
        + 0.9 * robust_misses
        + 80.0 * worst_excess
        + 0.25 * tau1_penalty
        + 0.05 * headroom_penalty
    )


def _finite_ratio(value: Any, default: float = float("inf")) -> float:
    """Return a finite float for ranking; non-finite values sort last."""
    result = float(value)
    return result if np.isfinite(result) else default


def _probe_plan(duration_s: float) -> dict[str, dict[str, Any]]:
    """Build the M11-style screen probe definitions."""
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
        },
        "ood_3x_thermal_slope": {
            "trace": make_thermal_ramp_trace(duration_s, 3.0, 4242),
            "rng_seed": 4242,
            "disc_noise_amp_ci": DISC_NOISE_AMP_CI,
            "twin": None,
        },
        "high_disc_noise_3x": {
            "trace": Disturbance.from_recipe("thermal_ramp").generate(
                duration_s=duration_s,
                sample_rate_Hz=PHYSICS_RATE_HZ,
                seed=4242,
            ),
            "rng_seed": 4242,
            "disc_noise_amp_ci": 3.0 * DISC_NOISE_AMP_CI,
            "twin": None,
        },
        "low_disc_noise_third": {
            "trace": Disturbance.from_recipe("thermal_ramp").generate(
                duration_s=duration_s,
                sample_rate_Hz=PHYSICS_RATE_HZ,
                seed=4242,
            ),
            "rng_seed": 4242,
            "disc_noise_amp_ci": DISC_NOISE_AMP_CI / 3.0,
            "twin": None,
        },
        "reality_gap_5pct": {
            "trace": Disturbance.from_recipe("thermal_ramp").generate(
                duration_s=duration_s,
                sample_rate_Hz=PHYSICS_RATE_HZ,
                seed=4242,
            ),
            "rng_seed": 4242,
            "disc_noise_amp_ci": DISC_NOISE_AMP_CI,
            "twin": make_perturbed_twin(0.05, 4243),
        },
        "worst_case_2x_stacked": {
            "trace": make_all_stacked_trace(duration_s, 2.0, 2.0, 4242),
            "rng_seed": 4242,
            "disc_noise_amp_ci": DISC_NOISE_AMP_CI,
            "twin": None,
        },
    }


def run_structured_screen(args: argparse.Namespace, run_dir: Path) -> dict[str, Any]:
    """Run the structured direct-CfC screening campaign."""
    all_specs = generate_structured_specs(args.max_specs)
    chunks = [
        all_specs[start : start + args.batch_size]
        for start in range(0, len(all_specs), args.batch_size)
    ]
    combined: dict[str, Any] = {
        key: {"candidates": {}}
        for key in _probe_plan(args.screen_duration_s)
    }
    for i, chunk in enumerate(chunks, start=1):
        log(f"screen chunk {i}/{len(chunks)} candidates={len(chunk)}")
        for key, probe in _probe_plan(args.screen_duration_s).items():
            log(f"screen chunk {i}/{len(chunks)} probe={key}")
            result = evaluate_specs(
                chunk,
                probe["trace"],
                args.screen_duration_s,
                probe["rng_seed"],
                disc_noise_amp_ci=probe["disc_noise_amp_ci"],
                twin=probe["twin"],
            )
            combined[key]["rh_lqr"] = result["rh_lqr"]
            combined[key]["wall_s"] = combined[key].get("wall_s", 0.0) + result["wall_s"]
            combined[key]["candidates"].update(result["candidates"])

    scored = []
    for name in combined["m5_thermal_ramp"]["candidates"]:
        scored.append((score_candidate(combined, name), name))
    scored.sort()
    top = []
    for score, name in scored[: args.top_k]:
        probe_ratios = {
            key: combined[key]["candidates"][name]["ml_over_rhlqr_10s"]
            for key in combined
        }
        robust_keys = [key for key in probe_ratios if key != "m5_thermal_ramp"]
        item = {
            "name": name,
            "score": score,
            "m5_thermal_ramp": combined["m5_thermal_ramp"]["candidates"][name],
            "probe_ratios": probe_ratios,
            "robust_ties_or_wins_of_5": sum(
                1 for key in robust_keys if probe_ratios[key] <= 1.0 + TIE_TOLERANCE
            ),
        }
        top.append(item)
        log(
            f"top candidate {name}: score={score:.6g} "
            f"m5={item['probe_ratios']['m5_thermal_ramp']:.9g} "
            f"robust={item['robust_ties_or_wins_of_5']}/5"
        )

    winner_spec = StructuredSpec(**top[0]["m5_thermal_ramp"]["spec"])
    winner = make_direct_controller(winner_spec)
    ckpt_path = run_dir / "cfc_improved_structured.json"
    winner.save(ckpt_path)
    summary = {
        "method": "structured_direct_feature_search",
        "checkpoint": str(ckpt_path),
        "winner": asdict(winner_spec),
        "screen_duration_s": args.screen_duration_s,
        "n_candidates": len(all_specs),
        "tie_tolerance": TIE_TOLERANCE,
        "current_promoted_m5_ratio": CURRENT_PROMOTED_M5_RATIO,
        "current_promoted_checkpoint": CURRENT_PROMOTED_CHECKPOINT,
        "top": top,
        "probe_results": combined,
    }
    (run_dir / "structured_screen.json").write_text(
        json.dumps(_json_safe(summary), indent=2),
        encoding="utf-8",
    )
    return summary


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Run the CfC improvement campaign."""
    run_dir = _PROJECT_ROOT / "data" / "ml_research" / time.strftime(
        "cfc_improvement_%Y%m%d_%H%M%S"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    structured = run_structured_screen(args, run_dir)
    summary = {
        "campaign": "cfc_improvement",
        "run_dir": str(run_dir),
        "structured_screen": structured,
        "current_promoted_m5_ratio_100s": CURRENT_PROMOTED_M5_RATIO,
        "current_promoted_checkpoint": CURRENT_PROMOTED_CHECKPOINT,
        "todo": "Promote the saved winner through scripts/run_m11_gate.py full gate.",
    }
    out_path = _PROJECT_ROOT / "data" / "cfc_improvement_summary.json"
    out_path.write_text(json.dumps(_json_safe(summary), indent=2), encoding="utf-8")
    log(f"wrote {out_path}")
    return summary


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--screen-duration-s", type=float, default=25.0)
    parser.add_argument("--max-specs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--top-k", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint."""
    run(parse_args())


if __name__ == "__main__":
    main()
