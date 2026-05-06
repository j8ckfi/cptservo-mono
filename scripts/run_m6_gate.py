"""M6 gate: train APG policy, then compare PI vs RH-LQR vs APG on all_stacked.

Gate criterion (from spec):
    APG sigma_y(tau=100s) on all_stacked <= 0.8 * min(PI sigma_y, LQR sigma_y)
    i.e., APG beats BOTH baselines by >=20% at tau=100s.

Steps:
    1. Train APG policy via train_apg() with default curriculum.
    2. Run 100 s of all_stacked through PI, RH-LQR, and APG.
    3. Compute sigma_y at tau in {1, 10, 100} s for each.
    4. Evaluate gate criterion and write data/gate_M6.json.

Anti-fudge discipline:
    - Same twin, same disturbance trace, same disc_noise_amp_ci for all runs.
    - No post-loop additive noise.
    - sigma_y computed from overlapping Allan deviation on demeaned y series.
    - Gate criterion hard-coded; not adjusted post-hoc.
    - If APG does not beat both baselines, GATE FAIL is reported honestly.

Usage:
    cd WIP/CPTServo
    python scripts/run_m6_gate.py

Output:
    data/gate_M6.json
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Project path setup
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_SCRIPT_DIR))
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from run_m3_m4_gates import log, make_calibrated_twin, run_fast_loop  # noqa: E402

from cptservo.baselines.pi import PIController  # noqa: E402
from cptservo.baselines.rh_lqr import RHLQRController  # noqa: E402
from cptservo.policy.apg import APGPolicy  # noqa: E402
from cptservo.policy.apg_train import train_apg  # noqa: E402
from cptservo.twin.allan import overlapping_allan  # noqa: E402
from cptservo.twin.disturbance import Disturbance  # noqa: E402

# ---------------------------------------------------------------------------
# Gate parameters (frozen per spec)
# ---------------------------------------------------------------------------

DISC_NOISE_AMP_CI: float = 7.0e-4       # M3-calibrated discriminator noise
DURATION_S: float = 100.0                # 100 s evaluation run
DECIMATION_RATE_HZ: float = 1_000.0
EVAL_TAUS: list[float] = [1.0, 10.0, 100.0]
RNG_SEED: int = 42
SCENARIO: str = "all_stacked"

# Gate: APG sigma_y(100s) must be <= 80% of min(PI, LQR) sigma_y(100s)
GATE_TAU: float = 100.0
GATE_RATIO_THRESHOLD: float = 0.8

# Training parameters (conservative: 200 episodes per stage, bptt_window=200)
N_EPISODES_PER_STAGE: int = 200
BPTT_WINDOW: int = 200
LEARNING_RATE: float = 3.0e-4
CURRICULUM: list[str] = [
    "clean",
    "thermal_ramp",
    "b_field_drift",
    "laser_intensity_drift",
    "all_stacked",
]


# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------


def train_policy(models_dir: Path) -> tuple[APGPolicy, dict[str, Any]]:
    """Train the APG policy with default curriculum and return (policy, metrics).

    Args:
        models_dir: Directory for saving checkpoints.

    Returns:
        Tuple of (trained APGPolicy, training metrics dict).
    """
    models_dir.mkdir(parents=True, exist_ok=True)
    save_path = models_dir / "apg_best.pt"

    policy = APGPolicy(
        n_error_history=16,
        n_rf_history=0,
        include_env_sensors=False,
        hidden_dims=(64, 64),
        rf_limit_Hz=1000.0,
    )
    log(f"  Policy: obs_dim={policy.obs_dim}, n_params={policy.n_params()}")

    log("  Starting curriculum training ...")
    t0 = time.perf_counter()
    metrics = train_apg(
        policy=policy,
        n_episodes_per_stage=N_EPISODES_PER_STAGE,
        bptt_window=BPTT_WINDOW,
        learning_rate=LEARNING_RATE,
        grad_clip=1.0,
        curriculum=CURRICULUM,
        save_path=str(save_path),
        rng_seed=42,
        n_warmup_steps=2_000,
        verbose=True,
    )
    wall_s = time.perf_counter() - t0
    metrics["total_wall_s"] = wall_s

    log(f"  Training done: final_loss={metrics['final_loss']:.4e}, wall={wall_s:.1f}s")
    log(f"  Saved best model: {save_path}")

    # Load the best checkpoint (may differ from final episode if training diverged)
    if save_path.exists():
        policy = APGPolicy.load(str(save_path))
        log(f"  Loaded best checkpoint from {save_path}")

    return policy, metrics


# ---------------------------------------------------------------------------
# Evaluation: run all three controllers on the same trace
# ---------------------------------------------------------------------------


def run_evaluation(policy: APGPolicy) -> dict[str, Any]:
    """Run PI, RH-LQR, and APG on all_stacked and compute sigma_y.

    Args:
        policy: Trained APGPolicy in eval mode.

    Returns:
        Dict with sigma_y values for all three controllers and head-to-head metrics.
    """
    log(f"\n=== Evaluation: all three controllers on {SCENARIO} ({DURATION_S}s) ===")
    log(f"  disc_noise_amp_ci={DISC_NOISE_AMP_CI:.0e}, seed={RNG_SEED}")

    # Generate disturbance trace once -- reused for all three runs (identical conditions)
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
    t0 = time.perf_counter()
    pi_res = run_fast_loop(
        twin_pi,
        pi,
        trace,
        DURATION_S,
        decimation_rate_Hz=DECIMATION_RATE_HZ,
        disc_noise_amp_ci=DISC_NOISE_AMP_CI,
        rng_seed=RNG_SEED,
    )
    pi_wall = time.perf_counter() - t0
    log(f"  PI run wall={pi_wall:.1f}s")

    # -----------------------------------------------------------------
    # RH-LQR run
    # -----------------------------------------------------------------
    lqr = RHLQRController.from_recipe()
    log(f"  LQR: K={lqr.K[0].tolist()}, Q={lqr.Q}, R={lqr.R}")
    twin_lqr = make_calibrated_twin()
    t0 = time.perf_counter()
    lqr_res = run_fast_loop(
        twin_lqr,
        lqr,
        trace,
        DURATION_S,
        decimation_rate_Hz=DECIMATION_RATE_HZ,
        disc_noise_amp_ci=DISC_NOISE_AMP_CI,
        rng_seed=RNG_SEED,
    )
    lqr_wall = time.perf_counter() - t0
    log(f"  LQR run wall={lqr_wall:.1f}s")

    # -----------------------------------------------------------------
    # APG run
    # -----------------------------------------------------------------
    policy.eval()
    policy.reset()
    twin_apg = make_calibrated_twin()
    t0 = time.perf_counter()
    apg_res = run_fast_loop(
        twin_apg,
        policy,
        trace,
        DURATION_S,
        decimation_rate_Hz=DECIMATION_RATE_HZ,
        disc_noise_amp_ci=DISC_NOISE_AMP_CI,
        rng_seed=RNG_SEED,
    )
    apg_wall = time.perf_counter() - t0
    log(f"  APG run wall={apg_wall:.1f}s")

    # -----------------------------------------------------------------
    # Allan deviations
    # -----------------------------------------------------------------
    y_pi = pi_res["y"] - float(np.mean(pi_res["y"]))
    y_lqr = lqr_res["y"] - float(np.mean(lqr_res["y"]))
    y_apg = apg_res["y"] - float(np.mean(apg_res["y"]))

    pi_allan = overlapping_allan(y_pi, DECIMATION_RATE_HZ, EVAL_TAUS)
    lqr_allan = overlapping_allan(y_lqr, DECIMATION_RATE_HZ, EVAL_TAUS)
    apg_allan = overlapping_allan(y_apg, DECIMATION_RATE_HZ, EVAL_TAUS)

    pi_1s = float(pi_allan.get(1.0, float("nan")))
    pi_10s = float(pi_allan.get(10.0, float("nan")))
    pi_100s = float(pi_allan.get(100.0, float("nan")))

    lqr_1s = float(lqr_allan.get(1.0, float("nan")))
    lqr_10s = float(lqr_allan.get(10.0, float("nan")))
    lqr_100s = float(lqr_allan.get(100.0, float("nan")))

    apg_1s = float(apg_allan.get(1.0, float("nan")))
    apg_10s = float(apg_allan.get(10.0, float("nan")))
    apg_100s = float(apg_allan.get(100.0, float("nan")))

    log(f"\n  sigma_y(1s):   PI={pi_1s:.3e}  LQR={lqr_1s:.3e}  APG={apg_1s:.3e}")
    log(f"  sigma_y(10s):  PI={pi_10s:.3e}  LQR={lqr_10s:.3e}  APG={apg_10s:.3e}")
    log(f"  sigma_y(100s): PI={pi_100s:.3e}  LQR={lqr_100s:.3e}  APG={apg_100s:.3e}")

    # Gate evaluation: APG <= 0.8 * min(PI, LQR) at tau=100s
    best_baseline_100s = min(
        v for v in [pi_100s, lqr_100s] if np.isfinite(v)
    ) if any(np.isfinite(v) for v in [pi_100s, lqr_100s]) else float("nan")

    if np.isfinite(apg_100s) and np.isfinite(best_baseline_100s) and best_baseline_100s > 0.0:
        ratio_apg_to_best = apg_100s / best_baseline_100s
        gate_criterion_met = bool(ratio_apg_to_best <= GATE_RATIO_THRESHOLD)
    else:
        ratio_apg_to_best = float("nan")
        gate_criterion_met = False

    ratio_apg_to_pi = (
        apg_100s / pi_100s if (np.isfinite(apg_100s) and pi_100s > 0) else float("nan")
    )
    ratio_apg_to_lqr = (
        apg_100s / lqr_100s if (np.isfinite(apg_100s) and lqr_100s > 0) else float("nan")
    )

    log(f"\n  best_baseline_sigma_y(100s) = {best_baseline_100s:.3e}")
    log(f"  ratio APG/best_baseline(100s) = {ratio_apg_to_best:.3f}")
    log(f"  gate threshold = {GATE_RATIO_THRESHOLD}")
    log(f"  gate_criterion_met = {gate_criterion_met}")

    return {
        "pi_sigma_y_1s": pi_1s,
        "pi_sigma_y_10s": pi_10s,
        "pi_sigma_y_100s": pi_100s,
        "lqr_sigma_y_1s": lqr_1s,
        "lqr_sigma_y_10s": lqr_10s,
        "lqr_sigma_y_100s": lqr_100s,
        "apg_sigma_y_1s": apg_1s,
        "apg_sigma_y_10s": apg_10s,
        "apg_sigma_y_100s": apg_100s,
        "best_baseline_sigma_y_100s": best_baseline_100s,
        "ratio_apg_to_pi_100s": ratio_apg_to_pi,
        "ratio_apg_to_lqr_100s": ratio_apg_to_lqr,
        "ratio_apg_to_best_baseline_100s": ratio_apg_to_best,
        "gate_criterion_met": gate_criterion_met,
        "pi_wall_s": pi_wall,
        "lqr_wall_s": lqr_wall,
        "apg_wall_s": apg_wall,
        "policy_n_params": policy.n_params(),
        "policy_obs_dim": policy.obs_dim,
    }


# ---------------------------------------------------------------------------
# Tests and ruff
# ---------------------------------------------------------------------------


def run_tests() -> tuple[bool, int, int]:
    """Run pytest on tests/test_policy.py.

    Returns:
        Tuple of (all_passed, n_passed, n_total).
    """
    import re

    log("  Running pytest tests/test_policy.py ...")
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_policy.py", "-v", "--tb=short"],
        capture_output=True,
        text=True,
        cwd=str(_PROJECT_ROOT),
        timeout=600,
    )
    out = result.stdout + result.stderr
    n_passed = 0
    n_total = 0
    for line in out.splitlines():
        if "passed" in line or "failed" in line or "error" in line:
            nums = re.findall(r"(\d+)\s+(passed|failed|error)", line)
            for n, kind in nums:
                n_total += int(n)
                if kind == "passed":
                    n_passed += int(n)
    if n_total == 0:
        lines = out.splitlines()
        n_total = sum(1 for ln in lines if "PASSED" in ln or "FAILED" in ln)
        n_passed = sum(1 for ln in lines if "PASSED" in ln)
    log(f"  pytest returncode={result.returncode}, {n_passed}/{n_total} passed")
    if result.returncode != 0:
        for line in out.splitlines()[-30:]:
            log(f"    {line}")
    return result.returncode == 0, n_passed, n_total


def run_ruff() -> bool:
    """Run ruff check on src/ and scripts/.

    Returns:
        True if ruff is clean.
    """
    log("  Running ruff check src/ scripts/ tests/ ...")
    result = subprocess.run(
        [sys.executable, "-m", "ruff", "check", "src/", "scripts/", "tests/"],
        capture_output=True,
        text=True,
        cwd=str(_PROJECT_ROOT),
        timeout=60,
    )
    clean = result.returncode == 0
    if not clean:
        log(f"  ruff issues:\n{result.stdout[:3000]}")
    log(f"  ruff clean={clean}")
    return clean


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    t_global_start = time.perf_counter()
    data_dir = _PROJECT_ROOT / "data"
    models_dir = _PROJECT_ROOT / "models"
    data_dir.mkdir(exist_ok=True)
    models_dir.mkdir(exist_ok=True)

    log("=== M6 gate script start ===")

    # 1. Train APG policy
    log("\n--- Step 1: Train APG policy ---")
    policy, train_metrics = train_policy(models_dir)

    # Save training metrics
    train_metrics_path = data_dir / "m6_training_metrics.json"

    # Prepare serializable metrics (stage_final_loss is a dict, loss_per_episode is a list)
    serializable_metrics = {
        k: v for k, v in train_metrics.items()
        if k not in ("loss_per_episode", "sigma_y_per_episode")
    }
    serializable_metrics["n_loss_episodes"] = len(train_metrics.get("loss_per_episode", []))
    serializable_metrics["loss_per_episode"] = train_metrics.get("loss_per_episode", [])
    train_metrics_path.write_text(json.dumps(serializable_metrics, indent=2), encoding="utf-8")
    log(f"  Saved training metrics: {train_metrics_path}")

    # 2. Evaluate all three controllers
    log("\n--- Step 2: Evaluate PI, RH-LQR, APG on all_stacked ---")
    eval_results = run_evaluation(policy)

    # 3. Run tests
    log("\n--- Step 3: pytest ---")
    tests_pass, n_passed, n_total = run_tests()

    # 4. Ruff
    log("\n--- Step 4: ruff ---")
    ruff_clean = run_ruff()

    # 5. Overall gate verdict
    gate_pass = bool(eval_results["gate_criterion_met"] and tests_pass and ruff_clean)

    total_wall = time.perf_counter() - t_global_start

    log("\n=== M6 gate verdict ===")
    log(f"  apg_beats_both_baselines_20pct = {eval_results['gate_criterion_met']}")
    log(f"  tests_passed       = {tests_pass} ({n_passed}/{n_total})")
    log(f"  ruff_clean         = {ruff_clean}")
    log(f"  gate_pass          = {gate_pass}")
    log(f"  total_wall_s       = {total_wall:.1f}s ({total_wall / 60:.1f} min)")

    # 6. Write gate JSON
    gate_doc: dict[str, Any] = {
        "milestone": "M6",
        "policy_type": "APG_MLP",
        "n_params": policy.n_params(),
        "obs_dim": policy.obs_dim,
        "hidden_dims": [64, 64],
        "rf_limit_Hz": policy.rf_limit_Hz,
        "n_error_history": policy.n_error_history,
        "n_rf_history": policy.n_rf_history,
        "include_env_sensors": policy.include_env_sensors,
        "curriculum": CURRICULUM,
        "n_episodes_per_stage": N_EPISODES_PER_STAGE,
        "bptt_window": BPTT_WINDOW,
        "training": {
            "final_loss": float(train_metrics.get("final_loss", float("nan"))),
            "best_loss": float(train_metrics.get("best_loss", float("nan"))),
            "total_wall_s": float(train_metrics.get("total_wall_s", 0.0)),
            "n_episodes_total": int(train_metrics.get("n_episodes_total", 0)),
            "stage_final_loss": train_metrics.get("stage_final_loss", {}),
        },
        "head_to_head": {
            "scenario": SCENARIO,
            "duration_s": DURATION_S,
            "pi_sigma_y_1s": eval_results["pi_sigma_y_1s"],
            "pi_sigma_y_10s": eval_results["pi_sigma_y_10s"],
            "pi_sigma_y_100s": eval_results["pi_sigma_y_100s"],
            "lqr_sigma_y_1s": eval_results["lqr_sigma_y_1s"],
            "lqr_sigma_y_10s": eval_results["lqr_sigma_y_10s"],
            "lqr_sigma_y_100s": eval_results["lqr_sigma_y_100s"],
            "apg_sigma_y_1s": eval_results["apg_sigma_y_1s"],
            "apg_sigma_y_10s": eval_results["apg_sigma_y_10s"],
            "apg_sigma_y_100s": eval_results["apg_sigma_y_100s"],
            "best_baseline_sigma_y_100s": eval_results["best_baseline_sigma_y_100s"],
            "ratio_apg_to_pi_100s": eval_results["ratio_apg_to_pi_100s"],
            "ratio_apg_to_lqr_100s": eval_results["ratio_apg_to_lqr_100s"],
            "ratio_apg_to_best_baseline_100s": eval_results["ratio_apg_to_best_baseline_100s"],
        },
        "noise_injection_point": "rf_actual_pre_step+disc_noise_pre_controller",
        "disc_noise_amp_ci": DISC_NOISE_AMP_CI,
        "lo_noise_scale": 1.0,
        "rng_seed": RNG_SEED,
        "gate_criterion": (
            f"APG sigma_y({GATE_TAU:.0f}s) <= {GATE_RATIO_THRESHOLD} "
            f"* min(PI, LQR) sigma_y({GATE_TAU:.0f}s) on {SCENARIO}"
        ),
        "gate_criterion_met": eval_results["gate_criterion_met"],
        "tests_passed": tests_pass,
        "n_tests_passed": n_passed,
        "n_tests_total": n_total,
        "ruff_clean": ruff_clean,
        "gate_pass": gate_pass,
        "total_wall_s": total_wall,
    }

    out_path = data_dir / "gate_M6.json"
    out_path.write_text(json.dumps(gate_doc, indent=2), encoding="utf-8")
    log(f"\nWrote {out_path}")

    verdict = "GATE PASS" if gate_pass else "GATE FAIL"
    log(f"\n{'=' * 56}")
    log(f"  M6 {verdict}")
    log(f"  sigma_y(100s): PI={eval_results['pi_sigma_y_100s']:.3e}")
    log(f"  sigma_y(100s): LQR={eval_results['lqr_sigma_y_100s']:.3e}")
    log(f"  sigma_y(100s): APG={eval_results['apg_sigma_y_100s']:.3e}")
    log(f"  ratio APG/best_baseline = {eval_results['ratio_apg_to_best_baseline_100s']:.3f}")
    log(f"  (gate: ratio <= {GATE_RATIO_THRESHOLD})")
    log(f"{'=' * 56}")


if __name__ == "__main__":
    main()
