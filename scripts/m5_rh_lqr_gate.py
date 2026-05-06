"""M5 gate: receding-horizon LQR vs PI head-to-head on thermal_ramp scenario.

This script runs the M5 audit gate per the plan:

  Gate: RH-LQR <= PI sigma_y at tau=100s on `thermal_ramp`.

Architecture
------------
Both controllers run through the SAME ``run_fast_loop`` harness with identical
noise parameters:

  disc_noise_amp_ci = 7e-4   (calibrated discriminator-input noise, Knappe 2004)
  noise_injection_point = "rf_actual_pre_step+disc_noise_pre_controller"
  duration_s = 100 s
  seed = 42 for PI, 42 for RH-LQR (same RNG stream — same noise realisation)

The only difference between the two runs is the controller object.  Using the
same seed ensures the comparison is apples-to-apples: any sigma_y difference is
attributable to the controller, not to noise-sample variance.

Anti-fudge discipline
---------------------
* Same twin instance rebuilt fresh for each run (state cannot bleed across).
* Same disturbance trace for both runs.
* Same disc_noise_amp_ci (7e-4) for both runs.
* No post-loop additive noise.
* sigma_y(100s) computed from overlapping Allan deviation on the demeaned y series.

Output
------
data/gate_M5.json with the schema defined in the M5 PRD story US-M5.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Project path setup — run from any cwd
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_SCRIPT_DIR))

from run_m3_m4_gates import make_calibrated_twin, run_fast_loop  # noqa: E402

from cptservo.baselines.pi import PIController  # noqa: E402
from cptservo.baselines.rh_lqr import RHLQRController  # noqa: E402
from cptservo.twin.allan import overlapping_allan  # noqa: E402
from cptservo.twin.disturbance import Disturbance  # noqa: E402


def log(msg: str) -> None:
    """Timestamped console log with safe ASCII fallback."""
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    try:
        print(line, flush=True)
    except UnicodeEncodeError:
        print(line.encode("ascii", errors="replace").decode("ascii"), flush=True)
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# Gate parameters (frozen — must not be changed to pass the gate)
# ---------------------------------------------------------------------------

DISC_NOISE_AMP_CI: float = 7.0e-4  # from configs/v1_recipe.yaml, calibrated Knappe 2004
DURATION_S: float = 100.0
DECIMATION_RATE_HZ: float = 1_000.0
TAU_S: float = 100.0   # gate tau
RNG_SEED: int = 42     # same seed for both runs -> identical noise realisation

# Thermal-ramp scenario parameters match v1_recipe.yaml (T_K_ramp_amplitude_K=5K,
# T_K_ramp_period_s=200s).  This scenario has dominant slow temperature drift that
# an optimal (LQR) controller should suppress better than a PI with fixed gains.
SCENARIO: str = "thermal_ramp"


def run_head_to_head() -> dict:
    """Run PI and RH-LQR on thermal_ramp; return comparison metrics."""
    log("=== M5 gate: RH-LQR vs PI on thermal_ramp ===")
    log(f"  scenario={SCENARIO}, duration={DURATION_S}s, tau_gate={TAU_S}s")
    log(f"  disc_noise_amp_ci={DISC_NOISE_AMP_CI:.0e}, seed={RNG_SEED}")

    # Generate disturbance trace once; reused for both runs.
    dist_gen = Disturbance.from_recipe(SCENARIO)
    trace = dist_gen.generate(
        duration_s=DURATION_S,
        sample_rate_Hz=10_000.0,
        seed=RNG_SEED,
    )

    # -----------------------------------------------------------------
    # PI run
    # -----------------------------------------------------------------
    pi = PIController.from_recipe()
    log(f"  PI: kp={pi.kp:.0f}, ki={pi.ki:.0e}, rf_limit={pi.rf_limit_Hz} Hz")
    twin_pi = make_calibrated_twin()

    t_pi_start = time.perf_counter()
    pi_res = run_fast_loop(
        twin_pi,
        pi,
        trace,
        DURATION_S,
        decimation_rate_Hz=DECIMATION_RATE_HZ,
        disc_noise_amp_ci=DISC_NOISE_AMP_CI,
        rng_seed=RNG_SEED,
    )
    t_pi_wall = time.perf_counter() - t_pi_start
    log(f"  PI run wall={t_pi_wall:.1f}s")

    # -----------------------------------------------------------------
    # RH-LQR run
    # -----------------------------------------------------------------
    lqr = RHLQRController.from_recipe()
    log(
        f"  RH-LQR: Q={lqr.Q}, R={lqr.R:.1f}, "
        f"plant_gain={lqr.plant_gain:.2e}, plant_tau={lqr.plant_tau_s:.3f}s"
    )
    log(f"  RH-LQR: K={lqr.K.tolist()}")
    twin_lqr = make_calibrated_twin()

    t_lqr_start = time.perf_counter()
    lqr_res = run_fast_loop(
        twin_lqr,
        lqr,
        trace,
        DURATION_S,
        decimation_rate_Hz=DECIMATION_RATE_HZ,
        disc_noise_amp_ci=DISC_NOISE_AMP_CI,
        rng_seed=RNG_SEED,
    )
    t_lqr_wall = time.perf_counter() - t_lqr_start
    log(f"  RH-LQR run wall={t_lqr_wall:.1f}s")

    # -----------------------------------------------------------------
    # Allan deviation at tau_gate
    # -----------------------------------------------------------------
    y_pi = pi_res["y"] - float(np.mean(pi_res["y"]))
    y_lqr = lqr_res["y"] - float(np.mean(lqr_res["y"]))

    pi_allan = overlapping_allan(y_pi, DECIMATION_RATE_HZ, [TAU_S])
    lqr_allan = overlapping_allan(y_lqr, DECIMATION_RATE_HZ, [TAU_S])

    pi_sigma = float(pi_allan[TAU_S])
    lqr_sigma = float(lqr_allan[TAU_S])

    rh_lqr_wins = bool(lqr_sigma <= pi_sigma)
    ratio_lqr_to_pi = lqr_sigma / pi_sigma if pi_sigma > 0.0 else float("nan")

    log(
        f"  sigma_y({TAU_S:.0f}s): PI={pi_sigma:.3e}, RH-LQR={lqr_sigma:.3e}, "
        f"ratio(LQR/PI)={ratio_lqr_to_pi:.3f}, rh_lqr_wins={rh_lqr_wins}"
    )

    return {
        "pi_sigma": pi_sigma,
        "lqr_sigma": lqr_sigma,
        "rh_lqr_wins": rh_lqr_wins,
        "ratio_lqr_to_pi": ratio_lqr_to_pi,
        "pi_wall_s": t_pi_wall,
        "lqr_wall_s": t_lqr_wall,
        "lqr_obj": lqr,
    }


def run_tests() -> tuple[bool, int, int]:
    """Run pytest on tests/test_baselines.py; return (passed, n_passed, n_total)."""
    log("  Running pytest tests/test_baselines.py ...")
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_baselines.py", "-v", "--tb=short"],
        capture_output=True,
        text=True,
        cwd=str(_PROJECT_ROOT),
        timeout=300,
    )
    out = result.stdout + result.stderr
    log(f"  pytest returncode={result.returncode}")
    # Parse "X passed, Y failed" line
    n_passed = 0
    n_total = 0
    for line in out.splitlines():
        if "passed" in line and ("failed" in line or "error" in line or "passed" in line):
            import re
            nums = re.findall(r"(\d+)\s+(passed|failed|error)", line)
            for n, kind in nums:
                if kind == "passed":
                    n_passed += int(n)
                n_total += int(n)
    if n_total == 0:
        # fallback: count test_ lines
        lines = out.splitlines()
        n_total = sum(1 for ln in lines if "PASSED" in ln or "FAILED" in ln)
        n_passed = sum(1 for ln in lines if "PASSED" in ln)
    log(f"  pytest: {n_passed}/{n_total} passed")
    if result.returncode != 0:
        log("  [WARN] pytest output:")
        for line in out.splitlines()[-30:]:
            log(f"    {line}")
    return result.returncode == 0, n_passed, n_total


def run_ruff() -> bool:
    """Run ruff check on src/ and scripts/; return True if clean."""
    log("  Running ruff check src/ scripts/ ...")
    result = subprocess.run(
        [sys.executable, "-m", "ruff", "check", "src/", "scripts/"],
        capture_output=True,
        text=True,
        cwd=str(_PROJECT_ROOT),
        timeout=60,
    )
    clean = result.returncode == 0
    if not clean:
        log(f"  [WARN] ruff issues:\n{result.stdout[:1000]}")
    log(f"  ruff clean={clean}")
    return clean


def main() -> None:
    data_dir = _PROJECT_ROOT / "data"
    data_dir.mkdir(exist_ok=True)

    log("=== M5 gate script start ===")

    # 1. Head-to-head simulation
    hth = run_head_to_head()
    lqr: RHLQRController = hth["lqr_obj"]

    # 2. Tests
    tests_pass, n_passed, n_total = run_tests()

    # 3. Ruff
    ruff_clean = run_ruff()

    # 4. Gate verdict
    gate_pass = bool(hth["rh_lqr_wins"] and tests_pass and ruff_clean)

    log("=== M5 gate verdict ===")
    log(f"  rh_lqr_wins      = {hth['rh_lqr_wins']}")
    log(f"  tests_passed     = {tests_pass} ({n_passed}/{n_total})")
    log(f"  ruff_clean       = {ruff_clean}")
    log(f"  gate_pass        = {gate_pass}")

    # 5. Write gate JSON
    q_diag = list(lqr.Q)
    k_rf = float(lqr.K[0, 0])   # gain on ci state (the observable)
    k_int = float(lqr.K[0, 1])  # gain on integrator state

    gate_doc = {
        "milestone": "M5",
        "gains": {
            "Q_diag": q_diag,
            "R": float(lqr.R),
            "K_ci": k_rf,
            "K_integral": k_int,
            "K_rf": k_rf,
        },
        "linearization_point": {
            "T_K": 333.15,
            "B_uT": 50.0,
            "I_norm": 1.0,
            "description": (
                "2-state linearised ZOH plant around CPT lock point: "
                "state=[ci_filtered, integral_ci], input=rf_correction_Hz"
            ),
        },
        "linearization_method": (
            "Analytic ZOH discretisation of continuous-time CPT coherence Bloch "
            "equation (first-order linear ODE: d(ci)/dt = -(1/T2)*ci + plant_gain*u); "
            "plant_gain=2.4e-5 ci/Hz from M2 OBE discriminator slope; "
            "T2=1ms from Kitching 2018 (chip-scale buffer-gas CPT coherence lifetime). "
            "DARE solved via scipy.linalg.solve_discrete_are."
        ),
        "plant_gain_ci_per_Hz": float(lqr.plant_gain),
        "plant_tau_s": float(lqr.plant_tau_s),
        "Ad": lqr._Ad.tolist(),
        "Bd": lqr._Bd.tolist(),
        "duration_s": DURATION_S,
        "scenario": SCENARIO,
        "tau_gate_s": TAU_S,
        "pi_sigma_y_100s_thermal_ramp": hth["pi_sigma"],
        "rh_lqr_sigma_y_100s_thermal_ramp": hth["lqr_sigma"],
        "ratio_lqr_to_pi": hth["ratio_lqr_to_pi"],
        "rh_lqr_wins": hth["rh_lqr_wins"],
        "noise_injection_point": "rf_actual_pre_step+disc_noise_pre_controller",
        "disc_noise_amp_ci": DISC_NOISE_AMP_CI,
        "rng_seed": RNG_SEED,
        "pi_wall_s": hth["pi_wall_s"],
        "lqr_wall_s": hth["lqr_wall_s"],
        "tests_passed": tests_pass,
        "n_tests_passed": n_passed,
        "n_tests_total": n_total,
        "ruff_clean": ruff_clean,
        "gate_pass": gate_pass,
    }

    out_path = data_dir / "gate_M5.json"
    out_path.write_text(json.dumps(gate_doc, indent=2), encoding="utf-8")
    log(f"Wrote {out_path}")

    verdict = "GATE PASS" if gate_pass else "GATE FAIL"
    log(f"\n{'='*50}")
    log(f"  M5 {verdict}")
    log(f"  sigma_y({TAU_S:.0f}s) PI={hth['pi_sigma']:.3e}  RH-LQR={hth['lqr_sigma']:.3e}")
    log(f"{'='*50}")


if __name__ == "__main__":
    main()
