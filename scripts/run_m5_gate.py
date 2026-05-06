"""Run the M5 RH-LQR/DLQR-vs-PI benchmark.

This script uses the same ``run_batched_loop`` evaluation harness as the M8
adversarial battery. The historical controller name is RH-LQR, but the
implementation is a steady-state two-state DLQR gain.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(_PROJECT_ROOT / "scripts"))

from run_m3_m4_gates import log, make_calibrated_twin  # noqa: E402

from cptservo.baselines.pi import PIController  # noqa: E402
from cptservo.baselines.rh_lqr import RHLQRController  # noqa: E402
from cptservo.evaluation.batched_runner import run_batched_loop  # noqa: E402
from cptservo.twin.allan import overlapping_allan  # noqa: E402
from cptservo.twin.disturbance import Disturbance  # noqa: E402

DURATION_S = 100.0
PHYSICS_RATE_HZ = 10_000.0
DECIMATION_RATE_HZ = 1_000.0
DISC_NOISE_AMP_CI = 7.0e-4
RNG_SEED = 42
SCENARIO = "thermal_ramp"
TAUS = [1.0, 10.0]


def _run_controller(controller: Any) -> dict[str, Any]:
    trace = Disturbance.from_recipe(SCENARIO).generate(
        duration_s=DURATION_S,
        sample_rate_Hz=PHYSICS_RATE_HZ,
        seed=RNG_SEED,
    )
    twin = make_calibrated_twin()
    res = run_batched_loop(
        twin=twin,
        controller=controller,
        disturbance_traces=[trace],
        duration_s=DURATION_S,
        physics_rate_Hz=PHYSICS_RATE_HZ,
        decimation_rate_Hz=DECIMATION_RATE_HZ,
        rng_seed=RNG_SEED,
        disc_noise_amp_ci=DISC_NOISE_AMP_CI,
        autograd=False,
    )
    y = res["y"][0] - float(np.mean(res["y"][0]))
    allan = overlapping_allan(y, DECIMATION_RATE_HZ, TAUS)
    return {
        "sigma_y_1s": float(allan.get(1.0, float("nan"))),
        "sigma_y_10s": float(allan.get(10.0, float("nan"))),
        "wall_s": float(res["wall_s"]),
        "noise_injection_point": res["noise_injection_point"],
    }


def run_m5_gate() -> dict[str, Any]:
    """Run PI and RH-LQR/DLQR on the same thermal-ramp trace."""
    log("M5: RH-LQR/DLQR baseline gate")
    log(
        f"  scenario={SCENARIO}, duration={DURATION_S}s, "
        f"disc_noise_amp_ci={DISC_NOISE_AMP_CI:.1e}, seed={RNG_SEED}"
    )

    pi = PIController.from_recipe()
    lqr = RHLQRController.from_recipe()
    log(f"  PI gains: kp={pi.kp}, ki={pi.ki}, dt={pi.control_dt_s}")
    log(f"  LQR gains: K={lqr.K}, Q={lqr.Q}, R={lqr.R}, dt={lqr.control_dt_s}")

    log("  PI closed-loop ...")
    pi_metrics = _run_controller(pi)
    log(
        f"    sigma_y(10s)={pi_metrics['sigma_y_10s']:.3e}, "
        f"wall={pi_metrics['wall_s']:.1f}s"
    )

    log("  RH-LQR/DLQR closed-loop ...")
    lqr_metrics = _run_controller(lqr)
    log(
        f"    sigma_y(10s)={lqr_metrics['sigma_y_10s']:.3e}, "
        f"wall={lqr_metrics['wall_s']:.1f}s"
    )

    pi_10 = pi_metrics["sigma_y_10s"]
    lqr_10 = lqr_metrics["sigma_y_10s"]
    speedup = pi_10 / lqr_10 if np.isfinite(pi_10) and lqr_10 > 0.0 else float("nan")
    lqr_wins = bool(np.isfinite(speedup) and speedup > 1.0)
    gate_pass = lqr_wins

    gate: dict[str, Any] = {
        "milestone": "M5",
        "controller_label": "RH-LQR/DLQR",
        "controller_note": (
            "Historical RH-LQR name retained; implementation is steady-state "
            "two-state DLQR, not iterative online MPC."
        ),
        "Q_diag": list(lqr.Q),
        "R": lqr.R,
        "control_dt_s": lqr.control_dt_s,
        "rf_limit_Hz": lqr.rf_limit_Hz,
        "lqr_gain_K": lqr.K[0].tolist(),
        "scenario": SCENARIO,
        "duration_s": DURATION_S,
        "rng_seed": RNG_SEED,
        "disc_noise_amp_ci": DISC_NOISE_AMP_CI,
        "noise_injection_point": pi_metrics["noise_injection_point"],
        "pi_sigma_y_1s_thermal": pi_metrics["sigma_y_1s"],
        "rhlqr_sigma_y_1s_thermal": lqr_metrics["sigma_y_1s"],
        "pi_sigma_y_10s_thermal": pi_10,
        "rhlqr_sigma_y_10s_thermal": lqr_10,
        "thermal_win_factor_pi_over_lqr": speedup,
        "rhlqr_wins_on_thermal_at_tau10s": lqr_wins,
        "pi_wall_s": pi_metrics["wall_s"],
        "lqr_wall_s": lqr_metrics["wall_s"],
        "tests_passed": None,
        "ruff_clean": None,
        "gate_pass": gate_pass,
    }

    log(
        f"  M5 tau=10s: PI={pi_10:.3e}, LQR={lqr_10:.3e}, "
        f"speedup={speedup:.3f}, gate_pass={gate_pass}"
    )
    return gate


def main() -> None:
    t_start = time.perf_counter()
    data_dir = _PROJECT_ROOT / "data"
    data_dir.mkdir(exist_ok=True)

    gate = run_m5_gate()
    out_path = data_dir / "gate_M5.json"
    out_path.write_text(json.dumps(gate, indent=2), encoding="utf-8")
    log(f"Wrote {out_path}")
    log(f"Total wall time: {time.perf_counter() - t_start:.1f}s")
    log(f"GATE {'PASS' if gate['gate_pass'] else 'FAIL'}")


if __name__ == "__main__":
    main()
