"""M6 gate: APG policy vs PI head-to-head on all_stacked scenario.

Gate criterion: APG sigma_y(tau=100s) < 0.8 * PI sigma_y(tau=100s) on
``all_stacked``.

Architecture
------------
Both controllers run through the same ``run_fast_loop`` harness with identical
noise parameters (same seed, same noise realisation).  The only difference is
the controller object.

Anti-fudge discipline
---------------------
* Same twin, same disturbance trace, same disc_noise_amp_ci for both runs.
* No post-loop additive noise.
* sigma_y computed from overlapping Allan deviation on the demeaned y series.
* Gate criterion hard-coded; not adjusted post-hoc.

Output
------
data/gate_M6.json
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Project path setup
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_SCRIPT_DIR))

from run_m3_m4_gates import log, make_calibrated_twin, run_fast_loop  # noqa: E402

from cptservo.baselines.pi import PIController  # noqa: E402
from cptservo.policy.apg import APGPolicy  # noqa: E402
from cptservo.twin.allan import overlapping_allan  # noqa: E402
from cptservo.twin.disturbance import Disturbance  # noqa: E402

# ---------------------------------------------------------------------------
# Gate parameters (frozen)
# ---------------------------------------------------------------------------

DISC_NOISE_AMP_CI: float = 7.0e-4
DURATION_S: float = 1000.0
DECIMATION_RATE_HZ: float = 1_000.0
EVAL_TAUS: list[float] = [1.0, 10.0, 100.0]
RNG_SEED: int = 42
SCENARIO: str = "all_stacked"

# Gate: APG sigma_y(100s) must be < 80% of PI sigma_y(100s)
GATE_TAU: float = 100.0
GATE_RATIO_THRESHOLD: float = 0.8


def run_head_to_head(apg_path: Path | None = None) -> dict:
    """Run PI and APG on all_stacked; return comparison metrics.

    Args:
        apg_path: Path to APGPolicy checkpoint.  Defaults to
            ``models/apg_best.pt``.

    Returns:
        Dict with sigma_y values and head-to-head metrics.
    """
    if apg_path is None:
        apg_path = _PROJECT_ROOT / "models" / "apg_best.pt"

    log(f"=== M6 gate: APG vs PI on {SCENARIO} ===")
    log(f"  scenario={SCENARIO}, duration={DURATION_S}s")
    log(f"  disc_noise_amp_ci={DISC_NOISE_AMP_CI:.0e}, seed={RNG_SEED}")
    log(f"  gate: APG sigma_y({GATE_TAU:.0f}s) < {GATE_RATIO_THRESHOLD} * PI")
    log(f"  APG checkpoint: {apg_path}")

    if not apg_path.exists():
        raise FileNotFoundError(
            f"APG checkpoint not found: {apg_path}\n"
            "Run scripts/m6_apg_train.py first."
        )

    # Load APG policy
    policy = APGPolicy.load(str(apg_path))
    policy.eval()
    log(f"  APG policy: obs_dim={policy.obs_dim}, n_params={policy.n_params()}")

    # Generate disturbance trace once; reused for both runs
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
    log(f"  PI: kp={pi.kp:.0f}, ki={pi.ki:.0e}")
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
    # APG run
    # -----------------------------------------------------------------
    twin_apg = make_calibrated_twin()
    policy.reset()

    t_apg_start = time.perf_counter()
    apg_res = run_fast_loop(
        twin_apg,
        policy,
        trace,
        DURATION_S,
        decimation_rate_Hz=DECIMATION_RATE_HZ,
        disc_noise_amp_ci=DISC_NOISE_AMP_CI,
        rng_seed=RNG_SEED,
    )
    t_apg_wall = time.perf_counter() - t_apg_start
    log(f"  APG run wall={t_apg_wall:.1f}s")

    # -----------------------------------------------------------------
    # Allan deviations at all eval taus
    # -----------------------------------------------------------------
    y_pi = pi_res["y"] - float(np.mean(pi_res["y"]))
    y_apg = apg_res["y"] - float(np.mean(apg_res["y"]))

    pi_allan = overlapping_allan(y_pi, DECIMATION_RATE_HZ, EVAL_TAUS)
    apg_allan = overlapping_allan(y_apg, DECIMATION_RATE_HZ, EVAL_TAUS)

    pi_1s = float(pi_allan.get(1.0, float("nan")))
    pi_10s = float(pi_allan.get(10.0, float("nan")))
    pi_100s = float(pi_allan.get(100.0, float("nan")))

    apg_1s = float(apg_allan.get(1.0, float("nan")))
    apg_10s = float(apg_allan.get(10.0, float("nan")))
    apg_100s = float(apg_allan.get(100.0, float("nan")))

    ratio_at_100s = apg_100s / pi_100s if pi_100s > 0.0 else float("nan")
    speedup_at_100s = pi_100s / apg_100s if apg_100s > 0.0 else float("nan")

    gate_pass_sim = bool(ratio_at_100s < GATE_RATIO_THRESHOLD)

    log(f"  sigma_y(1s):   PI={pi_1s:.3e}  APG={apg_1s:.3e}")
    log(f"  sigma_y(10s):  PI={pi_10s:.3e}  APG={apg_10s:.3e}")
    log(f"  sigma_y(100s): PI={pi_100s:.3e}  APG={apg_100s:.3e}")
    log(f"  ratio APG/PI at 100s: {ratio_at_100s:.3f}  (gate<{GATE_RATIO_THRESHOLD})")
    log(f"  speedup at 100s: {speedup_at_100s:.2f}x")
    log(f"  gate_pass_sim = {gate_pass_sim}")

    return {
        "pi_sigma_y_1s": pi_1s,
        "apg_sigma_y_1s": apg_1s,
        "pi_sigma_y_10s": pi_10s,
        "apg_sigma_y_10s": apg_10s,
        "pi_sigma_y_100s": pi_100s,
        "apg_sigma_y_100s": apg_100s,
        "ratio_apg_to_pi_100s": ratio_at_100s,
        "apg_speedup_at_100s": speedup_at_100s,
        "gate_pass_sim": gate_pass_sim,
        "pi_wall_s": t_pi_wall,
        "apg_wall_s": t_apg_wall,
        "policy": policy,
    }


def run_tests() -> tuple[bool, int, int]:
    """Run pytest on tests/test_policy.py; return (passed, n_passed, n_total)."""
    log("  Running pytest tests/test_policy.py ...")
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_policy.py", "-v", "--tb=short"],
        capture_output=True,
        text=True,
        cwd=str(_PROJECT_ROOT),
        timeout=300,
    )
    out = result.stdout + result.stderr
    log(f"  pytest returncode={result.returncode}")
    n_passed = 0
    n_total = 0
    import re

    for line in out.splitlines():
        if "passed" in line:
            nums = re.findall(r"(\d+)\s+(passed|failed|error)", line)
            for n, kind in nums:
                if kind == "passed":
                    n_passed += int(n)
                n_total += int(n)
    if n_total == 0:
        lines = out.splitlines()
        n_total = sum(1 for ln in lines if "PASSED" in ln or "FAILED" in ln)
        n_passed = sum(1 for ln in lines if "PASSED" in ln)
    log(f"  pytest: {n_passed}/{n_total} passed")
    if result.returncode != 0:
        log("  [WARN] pytest output (last 30 lines):")
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
        log(f"  [WARN] ruff issues:\n{result.stdout[:2000]}")
    log(f"  ruff clean={clean}")
    return clean


def load_training_metrics() -> dict:
    """Load training metrics from data/m6_training_metrics.json if present."""
    metrics_path = _PROJECT_ROOT / "data" / "m6_training_metrics.json"
    if metrics_path.exists():
        return json.loads(metrics_path.read_text(encoding="utf-8-sig"))
    return {}


def main() -> None:
    data_dir = _PROJECT_ROOT / "data"
    data_dir.mkdir(exist_ok=True)

    log("=== M6 gate script start ===")

    # 1. Head-to-head simulation
    hth = run_head_to_head()
    policy: APGPolicy = hth["policy"]

    # 2. Tests
    tests_pass, n_passed, n_total = run_tests()

    # 3. Ruff
    ruff_clean = run_ruff()

    # 4. Load training metrics
    train_metrics = load_training_metrics()

    # 5. Gate verdict
    gate_pass = bool(hth["gate_pass_sim"] and tests_pass and ruff_clean)

    log("\n=== M6 gate verdict ===")
    log(f"  apg_beats_pi_20pct = {hth['gate_pass_sim']}")
    log(f"  tests_passed       = {tests_pass} ({n_passed}/{n_total})")
    log(f"  ruff_clean         = {ruff_clean}")
    log(f"  gate_pass          = {gate_pass}")

    # 6. Write gate JSON
    gate_doc = {
        "milestone": "M6",
        "policy_type": "APG_MLP",
        "n_params": policy.n_params(),
        "obs_dim": policy.obs_dim,
        "hidden_dims": [64, 64, 32],
        "rf_limit_Hz": policy.rf_limit_Hz,
        "n_error_history": policy.n_error_history,
        "n_rf_history": policy.n_rf_history,
        "include_env_sensors": policy.include_env_sensors,
        "training_metrics": {
            "n_epochs_total": int(train_metrics.get("total_epochs", 0)),
            "final_loss": float(train_metrics.get("final_loss", float("nan"))),
            "wall_time_s": float(train_metrics.get("wall_time_s", 0.0)),
            "checkpoints_saved": int(train_metrics.get("checkpoints_saved", 0)),
        },
        "head_to_head": {
            "scenario": SCENARIO,
            "duration_s": DURATION_S,
            "pi_sigma_y_1s": hth["pi_sigma_y_1s"],
            "apg_sigma_y_1s": hth["apg_sigma_y_1s"],
            "pi_sigma_y_10s": hth["pi_sigma_y_10s"],
            "apg_sigma_y_10s": hth["apg_sigma_y_10s"],
            "pi_sigma_y_100s": hth["pi_sigma_y_100s"],
            "apg_sigma_y_100s": hth["apg_sigma_y_100s"],
            "ratio_apg_to_pi_100s": hth["ratio_apg_to_pi_100s"],
            "apg_speedup_at_100s": hth["apg_speedup_at_100s"],
        },
        "noise_injection_point": "rf_actual_pre_step+disc_noise_pre_controller",
        "disc_noise_amp_ci": DISC_NOISE_AMP_CI,
        "rng_seed": RNG_SEED,
        "gate_criterion": (
            f"APG sigma_y({GATE_TAU:.0f}s) < {GATE_RATIO_THRESHOLD} * PI sigma_y({GATE_TAU:.0f}s)"
            f" on {SCENARIO}"
        ),
        "gate_criterion_met": hth["gate_pass_sim"],
        "tests_passed": tests_pass,
        "n_tests_passed": n_passed,
        "n_tests_total": n_total,
        "ruff_clean": ruff_clean,
        "gate_pass": gate_pass,
        "pi_wall_s": hth["pi_wall_s"],
        "apg_wall_s": hth["apg_wall_s"],
    }

    out_path = data_dir / "gate_M6.json"
    out_path.write_text(json.dumps(gate_doc, indent=2), encoding="utf-8")
    log(f"\nWrote {out_path}")

    verdict = "GATE PASS" if gate_pass else "GATE FAIL"
    log(f"\n{'='*50}")
    log(f"  M6 {verdict}")
    log(f"  sigma_y(100s): PI={hth['pi_sigma_y_100s']:.3e}  APG={hth['apg_sigma_y_100s']:.3e}")
    log(
        f"  ratio(APG/PI)={hth['ratio_apg_to_pi_100s']:.3f}"
        f"  speedup={hth['apg_speedup_at_100s']:.2f}x"
    )
    log(f"{'='*50}")


if __name__ == "__main__":
    main()
