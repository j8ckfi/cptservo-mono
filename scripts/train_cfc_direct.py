"""Local non-autograd training for a standalone CfC direct-action controller.

The direct controller does not call DLQR at runtime.  It is distilled from a
teacher trajectory into a tiny closed-form recurrent policy:

    u_rf = CfC(error, sensors, recurrent_state)

This script stays local-safe: it uses ridge regression on a fixed recurrent
feature map and optional DAgger-style replay, avoiding Torch backward().
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_SCRIPT_DIR))

from audit_calibration import make_calibrated_twin  # noqa: E402

from cptservo.baselines.dlqr import DLQRController  # noqa: E402
from cptservo.evaluation.batched_runner import run_batched_loop  # noqa: E402
from cptservo.policy.ml_research import (  # noqa: E402
    CfCDirectConfig,
    CfCDirectController,
    CfCFeedforwardController,
    PhysicsResidualConfig,
)
from cptservo.twin.allan import overlapping_allan  # noqa: E402
from cptservo.twin.disturbance import Disturbance, DisturbanceTrace  # noqa: E402

PHYSICS_RATE_HZ = 10_000.0
DECIMATION_RATE_HZ = 1_000.0
DISC_NOISE_AMP_CI = 7.0e-4
RNG_SEED = 42
TAUS = [1.0, 10.0, 100.0]


def log(msg: str) -> None:
    """Timestamped console log."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _env_at(trace: DisturbanceTrace, k: int) -> dict[str, float]:
    """Return decimated environment sample for controller step k."""
    decimation = int(round(PHYSICS_RATE_HZ / DECIMATION_RATE_HZ))
    idx = min((k + 1) * decimation - 1, len(trace.T_K) - 1)
    return {
        "T_K": float(trace.T_K[idx]),
        "B_uT": float(trace.B_uT[idx]),
        "I_norm": float(trace.laser_intensity_norm[idx]),
    }


def _make_trace(name: str, duration_s: float, seed: int) -> DisturbanceTrace:
    """Create a named disturbance trace."""
    return Disturbance.from_recipe(name).generate(
        duration_s=duration_s,
        sample_rate_Hz=PHYSICS_RATE_HZ,
        seed=seed,
    )


def _default_teacher_checkpoint() -> str:
    """Load the current promoted hybrid CfC checkpoint path."""
    summary_path = _PROJECT_ROOT / "data" / "cfc_train_summary.json"
    if not summary_path.exists():
        return ""
    data = json.loads(summary_path.read_text(encoding="utf-8-sig"))
    return str(data.get("checkpoint", ""))


def make_teacher(checkpoint: str) -> Any:
    """Build the teacher controller used only for distillation labels."""
    if checkpoint:
        return CfCFeedforwardController.load(checkpoint)
    return DLQRController.from_recipe()


def evaluate(controller: Any, duration_s: float, seed: int = RNG_SEED) -> dict[str, float]:
    """Evaluate a controller on the canonical thermal ramp."""
    trace = _make_trace("thermal_ramp", duration_s, seed)
    res = run_batched_loop(
        twin=make_calibrated_twin(),
        controller=controller,
        disturbance_traces=[trace],
        duration_s=duration_s,
        physics_rate_Hz=PHYSICS_RATE_HZ,
        decimation_rate_Hz=DECIMATION_RATE_HZ,
        rng_seed=seed,
        disc_noise_amp_ci=DISC_NOISE_AMP_CI,
        autograd=False,
    )
    y = res["y"][0] - float(np.mean(res["y"][0]))
    allan = overlapping_allan(y, DECIMATION_RATE_HZ, TAUS)
    return {
        "sigma_y_1s": float(allan.get(1.0, float("nan"))),
        "sigma_y_10s": float(allan.get(10.0, float("nan"))),
        "sigma_y_100s": float(allan.get(100.0, float("nan"))),
        "rf_abs_max_Hz": float(np.max(np.abs(res["rf_cmd"][0]))),
        "wall_s": float(res["wall_s"]),
    }


def run_controller(
    controller: Any,
    trace: DisturbanceTrace,
    duration_s: float,
    rng_seed: int,
) -> dict[str, np.ndarray]:
    """Run one controller and return the fields needed for distillation."""
    res = run_batched_loop(
        twin=make_calibrated_twin(),
        controller=controller,
        disturbance_traces=[trace],
        duration_s=duration_s,
        physics_rate_Hz=PHYSICS_RATE_HZ,
        decimation_rate_Hz=DECIMATION_RATE_HZ,
        rng_seed=rng_seed,
        disc_noise_amp_ci=DISC_NOISE_AMP_CI,
        autograd=False,
    )
    return {
        "error": np.asarray(res["error_signal"][0], dtype=np.float64),
        "rf_cmd": np.asarray(res["rf_cmd"][0], dtype=np.float64),
    }


def teacher_targets_from_errors(
    teacher: Any,
    trace: DisturbanceTrace,
    errors: np.ndarray,
) -> np.ndarray:
    """Replay a teacher controller over an externally generated error path."""
    if hasattr(teacher, "reset"):
        teacher.reset()
    targets = np.zeros_like(errors, dtype=np.float64)
    for k, error in enumerate(errors):
        _, rf = teacher.step(float(error), _env_at(trace, k))
        targets[k] = rf
    return targets


def fit_readout(
    controller: CfCDirectController,
    sequences: list[tuple[DisturbanceTrace, np.ndarray, np.ndarray]],
    ridge: float,
) -> dict[str, float]:
    """Fit direct CfC linear readout by ridge regression."""
    rows: list[np.ndarray] = []
    targets: list[float] = []
    for trace, errors, rf_targets in sequences:
        controller.reset()
        for k, (error, target) in enumerate(zip(errors, rf_targets)):
            rows.append(
                controller.teacher_forced_design_row(
                    float(error),
                    _env_at(trace, k),
                    float(target),
                )
            )
            targets.append(float(target))

    design = np.asarray(rows, dtype=np.float64)
    target_arr = np.asarray(targets, dtype=np.float64)
    lhs = design.T @ design + ridge * np.eye(design.shape[1])
    rhs = design.T @ target_arr
    coeff = np.linalg.solve(lhs, rhs)

    h = controller.config.hidden_size
    x = controller.feature_dim
    controller.weights["W_out"] = coeff[:h]
    controller.weights["W_feat_out"] = coeff[h : h + x]
    controller.weights["b_out"] = np.array([coeff[-1]], dtype=np.float64)
    pred = design @ coeff
    return {
        "fit_samples": float(len(target_arr)),
        "fit_rmse_Hz": float(np.sqrt(np.mean((pred - target_arr) ** 2))),
        "target_abs_max_Hz": float(np.max(np.abs(target_arr))) if len(target_arr) else 0.0,
        "readout_abs_max": float(np.max(np.abs(coeff[:-1]))) if len(coeff) > 1 else 0.0,
    }


def initialize_structured_direct(
    controller: CfCDirectController,
    k_t: float,
    kp: float | None,
    ki: float | None,
    kd_error: float,
) -> dict[str, float]:
    """Initialize direct CfC readout as sensor feedforward plus feedback."""
    lqr_gain = DLQRController.from_recipe().K
    kp_eff = float(kp) if kp is not None else float(lqr_gain[0, 0])
    ki_eff = float(ki) if ki is not None else float(lqr_gain[0, 1])
    for key in controller.weights:
        controller.weights[key][:] = 0.0
    weights = controller.weights["W_feat_out"]
    weights[0] = float(k_t) * 10.0
    weights[6] = kp_eff / 1.0e3
    weights[7] = float(kd_error) / 1.0e3
    weights[8] = ki_eff / 1.0e6
    return {
        "k_T_Hz_per_K": float(k_t),
        "kp_ci_Hz_per_ci": kp_eff,
        "ki_ci_Hz_per_ci_s": ki_eff,
        "kd_error_Hz_per_ci_delta": float(kd_error),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Train and evaluate a standalone direct CfC checkpoint."""
    teacher_checkpoint = args.teacher_checkpoint or _default_teacher_checkpoint()
    scenarios = [item.strip() for item in args.fit_scenarios.split(",") if item.strip()]
    run_dir = _PROJECT_ROOT / "data" / "ml_research" / time.strftime("cfc_direct_%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    sensor_cfg = PhysicsResidualConfig(
        intensity_tau_s=args.intensity_tau_s,
        rf_limit_Hz=args.rf_limit_Hz,
    )
    controller = CfCDirectController(
        CfCDirectConfig(
            hidden_size=args.hidden_size,
            seed=args.seed,
            sensor_config=sensor_cfg,
            rf_limit_Hz=args.rf_limit_Hz,
            control_dt_s=1.0 / DECIMATION_RATE_HZ,
            output_feature_skip=not args.no_feature_skip,
        )
    )

    fit_metrics: dict[str, float] = {}
    if args.structured_direct:
        fit_metrics = initialize_structured_direct(
            controller,
            k_t=args.structured_k_T,
            kp=args.structured_kp,
            ki=args.structured_ki,
            kd_error=args.structured_kd_error,
        )
        log(
            "structured direct initialized "
            f"k_T={fit_metrics['k_T_Hz_per_K']:.6g} "
            f"kp={fit_metrics['kp_ci_Hz_per_ci']:.6g} "
            f"ki={fit_metrics['ki_ci_Hz_per_ci_s']:.6g}"
        )
    else:
        sequences: list[tuple[DisturbanceTrace, np.ndarray, np.ndarray]] = []
        for idx, scenario in enumerate(scenarios):
            trace = _make_trace(scenario, args.fit_duration_s, RNG_SEED + idx)
            teacher = make_teacher(teacher_checkpoint)
            log(f"teacher rollout scenario={scenario}")
            rollout = run_controller(teacher, trace, args.fit_duration_s, RNG_SEED + idx)
            sequences.append((trace, rollout["error"], rollout["rf_cmd"]))

        for iteration in range(args.iterations + 1):
            fit_metrics = fit_readout(controller, sequences, args.ridge)
            log(
                f"fit iteration={iteration} samples={fit_metrics['fit_samples']:.0f} "
                f"rmse_Hz={fit_metrics['fit_rmse_Hz']:.6g}"
            )
            if iteration >= args.iterations:
                break
            for idx, scenario in enumerate(scenarios):
                trace = _make_trace(scenario, args.fit_duration_s, RNG_SEED + idx)
                direct_rollout = run_controller(
                    controller,
                    trace,
                    args.fit_duration_s,
                    RNG_SEED + 100 + iteration * 17 + idx,
                )
                targets = teacher_targets_from_errors(
                    make_teacher(teacher_checkpoint),
                    trace,
                    direct_rollout["error"],
                )
                sequences.append((trace, direct_rollout["error"], targets))

    ckpt_path = run_dir / "cfc_direct.json"
    controller.save(ckpt_path)

    metrics: dict[str, Any] = {
        "controller": "cfc_direct",
        "mode": "structured_direct_search_init"
        if args.structured_direct
        else "local_ridge_distillation",
        "teacher_checkpoint": teacher_checkpoint,
        "checkpoint": str(ckpt_path),
        "fit_scenarios": scenarios,
        "fit_duration_s": args.fit_duration_s,
        "iterations": args.iterations,
        "ridge": args.ridge,
        "config": {
            "hidden_size": controller.config.hidden_size,
            "seed": controller.config.seed,
            "rf_limit_Hz": controller.config.rf_limit_Hz,
            "output_feature_skip": controller.config.output_feature_skip,
        },
        "fit_metrics": fit_metrics,
        "training_note": (
            "Standalone direct CfC checkpoint does not call DLQR at runtime. "
            "It is either distilled from a teacher trajectory or initialized "
            "from closed-loop coefficient search."
        ),
    }
    if args.eval_duration_s > 0.0:
        metrics["eval_duration_s"] = args.eval_duration_s
        metrics["dlqr"] = evaluate(DLQRController.from_recipe(), args.eval_duration_s)
        if teacher_checkpoint:
            metrics["teacher"] = evaluate(
                CfCFeedforwardController.load(teacher_checkpoint),
                args.eval_duration_s,
            )
        metrics["cfc_direct"] = evaluate(
            CfCDirectController.load(ckpt_path),
            args.eval_duration_s,
        )

    (run_dir / "metrics.json").write_text(
        json.dumps(_json_safe(metrics), indent=2),
        encoding="utf-8",
    )
    (_PROJECT_ROOT / "data" / "cfc_direct_summary.json").write_text(
        json.dumps(_json_safe(metrics), indent=2),
        encoding="utf-8",
    )
    log(f"wrote {ckpt_path}")
    return metrics


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
    parser.add_argument("--teacher-checkpoint", default="")
    parser.add_argument("--fit-scenarios", default="thermal_ramp")
    parser.add_argument("--fit-duration-s", type=float, default=4.0)
    parser.add_argument("--eval-duration-s", type=float, default=25.0)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--hidden-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--ridge", type=float, default=1.0e-3)
    parser.add_argument("--rf-limit-Hz", type=float, default=1000.0)
    parser.add_argument("--intensity-tau-s", type=float, default=0.01)
    parser.add_argument("--no-feature-skip", action="store_true")
    parser.add_argument("--structured-direct", action="store_true")
    parser.add_argument("--structured-k-T", type=float, default=4.0)
    parser.add_argument("--structured-kp", type=float, default=None)
    parser.add_argument("--structured-ki", type=float, default=None)
    parser.add_argument("--structured-kd-error", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint."""
    run(parse_args())


if __name__ == "__main__":
    main()
