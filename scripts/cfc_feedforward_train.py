"""Local-safe trainer/initializer for the CfC feedforward residual.

Local Torch autograd is known to hard-crash in this workspace, so this script
does not call ``backward()``.  It creates a tiny CfC checkpoint initialized from
the known M5-passing linear residual and optionally evaluates it through the
canonical batched runner.
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

from run_m3_m4_gates import make_calibrated_twin  # noqa: E402

from cptservo.baselines.dlqr import DLQRController  # noqa: E402
from cptservo.evaluation.batched_runner import run_batched_loop  # noqa: E402
from cptservo.policy.ml_research import (  # noqa: E402
    CfCFeedforwardConfig,
    CfCFeedforwardController,
    PhysicsResidualConfig,
)
from cptservo.twin.allan import overlapping_allan  # noqa: E402
from cptservo.twin.disturbance import Disturbance  # noqa: E402

PHYSICS_RATE_HZ = 10_000.0
DECIMATION_RATE_HZ = 1_000.0
DISC_NOISE_AMP_CI = 7.0e-4
RNG_SEED = 42


def log(msg: str) -> None:
    """Timestamped console log."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def evaluate(controller: Any, duration_s: float) -> dict[str, float]:
    """Evaluate a controller on thermal_ramp for smoke or gate checks."""
    trace = Disturbance.from_recipe("thermal_ramp").generate(
        duration_s=duration_s,
        sample_rate_Hz=PHYSICS_RATE_HZ,
        seed=RNG_SEED,
    )
    res = run_batched_loop(
        twin=make_calibrated_twin(),
        controller=controller,
        disturbance_traces=[trace],
        duration_s=duration_s,
        physics_rate_Hz=PHYSICS_RATE_HZ,
        decimation_rate_Hz=DECIMATION_RATE_HZ,
        rng_seed=RNG_SEED,
        disc_noise_amp_ci=DISC_NOISE_AMP_CI,
        autograd=False,
    )
    y = res["y"][0] - float(np.mean(res["y"][0]))
    allan = overlapping_allan(y, DECIMATION_RATE_HZ, [1.0, 10.0, 100.0])
    return {
        "sigma_y_1s": float(allan.get(1.0, float("nan"))),
        "sigma_y_10s": float(allan.get(10.0, float("nan"))),
        "sigma_y_100s": float(allan.get(100.0, float("nan"))),
        "rf_abs_max_Hz": float(np.max(np.abs(res["rf_cmd"][0]))),
        "wall_s": float(res["wall_s"]),
    }


def _load_physical_targets() -> dict[str, float]:
    """Load calibrated physical feedforward targets from reduced calibration."""
    calib = json.loads((_PROJECT_ROOT / "data" / "reduced_calibration.json").read_text())
    return {
        "k_B_target_Hz_per_uT": -float(calib["lumped_zeeman_coeff"]),
        "k_I_target_Hz_per_norm": -float(calib["light_shift_coeff"]),
    }


def fit_cfc_readout(
    controller: CfCFeedforwardController,
    duration_s: float,
    ridge: float,
    target_b: float,
    target_i: float,
) -> dict[str, Any]:
    """Fit only the CfC readout using deterministic physical targets."""
    hidden_rows: list[np.ndarray] = []
    targets: list[float] = []
    scenarios = [
        ("thermal_ramp", 0.0, 0.0),
        ("b_field_drift", target_b, 0.0),
        ("laser_intensity_drift", 0.0, target_i),
        ("all_stacked", target_b, target_i),
    ]
    decimation = int(round(PHYSICS_RATE_HZ / DECIMATION_RATE_HZ))

    for idx, (scenario, k_b, k_i) in enumerate(scenarios):
        trace = Disturbance.from_recipe(scenario).generate(
            duration_s=duration_s,
            sample_rate_Hz=PHYSICS_RATE_HZ,
            seed=RNG_SEED + idx,
        )
        controller.reset()
        for k in range(int(round(duration_s * DECIMATION_RATE_HZ))):
            last_idx = min((k + 1) * decimation - 1, len(trace.T_K) - 1)
            env = {
                "T_K": float(trace.T_K[last_idx]),
                "B_uT": float(trace.B_uT[last_idx]),
                "I_norm": float(trace.laser_intensity_norm[last_idx]),
            }
            features, base_residual = controller._features(0.0, env, 0.0)
            _ = controller._cfc_step(features)
            desired = (
                controller.config.base_residual.k_T_Hz_per_K
                * (env["T_K"] - controller.config.base_residual.T_nom_K)
                + k_b * (env["B_uT"] - controller.config.base_residual.B_nom_uT)
                + k_i * (controller._base._I_hat - controller.config.base_residual.I_nom)
            )
            target = desired - base_residual
            target = float(
                np.clip(
                    target,
                    -controller.config.cfc_residual_limit_Hz,
                    controller.config.cfc_residual_limit_Hz,
                )
            )
            hidden_rows.append(controller._h.copy())
            targets.append(target)

    design = np.column_stack(
        [np.asarray(hidden_rows, dtype=np.float64), np.ones(len(hidden_rows))]
    )
    target_arr = np.asarray(targets, dtype=np.float64)
    lhs = design.T @ design + ridge * np.eye(design.shape[1])
    rhs = design.T @ target_arr
    coeff = np.linalg.solve(lhs, rhs)
    controller.weights["W_out"] = coeff[:-1]
    controller.weights["b_out"] = np.array([coeff[-1]], dtype=np.float64)
    pred = design @ coeff
    return {
        "fit_samples": int(len(target_arr)),
        "fit_rmse_Hz": float(np.sqrt(np.mean((pred - target_arr) ** 2))),
        "target_abs_max_Hz": float(np.max(np.abs(target_arr))) if len(target_arr) else 0.0,
        "readout_abs_max": float(np.max(np.abs(coeff[:-1]))) if len(coeff) > 1 else 0.0,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Create a local-safe CfC checkpoint and metrics artifact."""
    run_dir = _PROJECT_ROOT / "data" / "ml_research" / time.strftime("cfc_%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    residual_cfg = PhysicsResidualConfig(
        k_T_Hz_per_K=args.k_T,
        k_B_Hz_per_uT=args.k_B,
        k_I_Hz_per_norm=args.k_I,
        k_dT_Hz_per_K_s=args.k_dT,
        k_dB_Hz_per_uT_s=args.k_dB,
        k_dI_Hz_per_norm_s=args.k_dI,
        residual_limit_Hz=args.residual_limit_Hz,
    )
    cfc_cfg = CfCFeedforwardConfig(
        hidden_size=args.hidden_size,
        seed=args.seed,
        base_residual=residual_cfg,
        cfc_residual_limit_Hz=args.cfc_residual_limit_Hz,
    )
    controller = CfCFeedforwardController(cfc_cfg)
    fit_metrics: dict[str, Any] | None = None
    if args.fit_readout:
        targets = _load_physical_targets()
        fit_metrics = fit_cfc_readout(
            controller=controller,
            duration_s=args.fit_duration_s,
            ridge=args.ridge,
            target_b=args.target_k_B
            if args.target_k_B is not None
            else targets["k_B_target_Hz_per_uT"],
            target_i=args.target_k_I
            if args.target_k_I is not None
            else targets["k_I_target_Hz_per_norm"],
        )
    ckpt_path = run_dir / "cfc_feedforward.json"
    controller.save(ckpt_path)

    metrics: dict[str, Any] = {
        "controller": "cfc_feedforward",
        "mode": "local_ridge_readout_fit" if args.fit_readout else "local_safe_initialization",
        "checkpoint": str(ckpt_path),
        "config": {
            "hidden_size": cfc_cfg.hidden_size,
            "seed": cfc_cfg.seed,
            "base_residual": residual_cfg.__dict__,
            "cfc_residual_limit_Hz": cfc_cfg.cfc_residual_limit_Hz,
        },
        "training_note": (
            "Local path initializes from the known M5-passing linear residual. "
            "Dynamic CfC output is trained with deterministic ridge readout "
            "fitting when --fit-readout is enabled."
        ),
    }
    if fit_metrics is not None:
        metrics["fit_metrics"] = fit_metrics
    if args.eval_duration_s > 0.0:
        metrics["eval_duration_s"] = args.eval_duration_s
        metrics["dlqr"] = evaluate(DLQRController.from_recipe(), args.eval_duration_s)
        metrics["cfc"] = evaluate(CfCFeedforwardController.load(ckpt_path), args.eval_duration_s)
    (run_dir / "metrics.json").write_text(
        json.dumps(_json_safe(metrics), indent=2),
        encoding="utf-8",
    )
    (_PROJECT_ROOT / "data" / "cfc_train_summary.json").write_text(
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
    parser.add_argument("--hidden-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--k-T", type=float, default=0.05)
    parser.add_argument("--k-B", type=float, default=0.0)
    parser.add_argument("--k-I", type=float, default=0.0)
    parser.add_argument("--k-dT", type=float, default=0.0)
    parser.add_argument("--k-dB", type=float, default=0.0)
    parser.add_argument("--k-dI", type=float, default=0.0)
    parser.add_argument("--residual-limit-Hz", type=float, default=5.0)
    parser.add_argument("--cfc-residual-limit-Hz", type=float, default=1.0)
    parser.add_argument("--fit-readout", action="store_true")
    parser.add_argument("--fit-duration-s", type=float, default=2.0)
    parser.add_argument("--ridge", type=float, default=1.0e-3)
    parser.add_argument("--target-k-B", type=float, default=None)
    parser.add_argument("--target-k-I", type=float, default=None)
    parser.add_argument("--eval-duration-s", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint."""
    run(parse_args())


if __name__ == "__main__":
    main()
