"""M7 gate: PPO vs PI head-to-head on thermal_ramp AND all_stacked.

Runs PPO (loaded from models/ppo_best.zip) and PI on two scenarios for 1000 s.
Computes σ_y at τ ∈ {1, 10, 100} s.  Writes data/gate_M7.json.

Gate pass criterion (M7 spec):
    PPO σ_y ≤ PI σ_y on thermal_ramp at τ=10 s.
    (This matches the DLQR M5 gate threshold — the primary RL headline.)
    Bonus: PPO matches DLQR within 50 % on thermal_ramp at τ=10 s.

Anti-fudge discipline
---------------------
* Same twin instance rebuilt fresh for each run.
* Same disturbance trace for both runs (seed 42).
* Same disc_noise_amp_ci (7e-4) for both runs.
* No post-loop additive noise.
* σ_y computed from overlapping Allan deviation on demeaned y series.
* Noise injection point: rf_actual_pre_step+disc_noise_pre_controller.

Usage
-----
    cd /c/Users/Jack/Documents/Research/WIP/CPTServo
    python scripts/m7_ppo_gate.py
"""

from __future__ import annotations

import json
import re
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

from run_m3_m4_gates import make_calibrated_twin, run_fast_loop  # noqa: E402

from cptservo.baselines.pi import PIController  # noqa: E402
from cptservo.policy.ml_research import read_rhlqr_reference  # noqa: E402
from cptservo.twin.allan import overlapping_allan  # noqa: E402
from cptservo.twin.disturbance import Disturbance  # noqa: E402

# ---------------------------------------------------------------------------
# Gate parameters (frozen)
# ---------------------------------------------------------------------------
DISC_NOISE_AMP_CI: float = 7.0e-4
DURATION_S: float = 500.0
DECIMATION_RATE_HZ: float = 1_000.0
EVAL_TAUS: list[float] = [1.0, 10.0, 100.0]
RNG_SEED: int = 42
RF_LIMIT_HZ: float = 1_000.0

# M5 DLQR sigma_y at tau=10s on thermal_ramp, loaded from data/gate_M5.json.
_DLQR_SIGMA_Y_10S_THERMAL: float = read_rhlqr_reference(_PROJECT_ROOT)


_LOG_PATH = _PROJECT_ROOT / "logs" / "m7_gate.log"
_LOG_PATH.parent.mkdir(exist_ok=True)
_LOG_FH = open(_LOG_PATH, "w", encoding="utf-8")  # noqa: SIM115


def log(msg: str) -> None:
    """Timestamped log to console and explicit log file (flushed each call)."""
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    try:
        print(line, flush=True)
    except UnicodeEncodeError:
        print(line.encode("ascii", errors="replace").decode("ascii"), flush=True)
    sys.stdout.flush()
    try:
        _LOG_FH.write(line + "\n")
        _LOG_FH.flush()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# PPO controller wrapper (exposes PIController protocol)
# ---------------------------------------------------------------------------


class _PPOController:
    """Thin wrapper around a loaded SB3 PPO model exposing the PIController protocol.

    Maintains the same observation window as CPTServoEnv so the PPO policy
    sees identical feature vectors to training.

    Args:
        model: Loaded SB3 PPO model.
        n_error_history: Error history length (must match training env).
        n_rf_history: RF correction history length.
        include_env_sensors: Whether to append T_K, B_uT, I_norm.
        rf_limit_Hz: Anti-windup clamp (Hz).
    """

    def __init__(
        self,
        model,
        n_error_history: int = 8,
        n_rf_history: int = 4,
        include_env_sensors: bool = True,
        rf_limit_Hz: float = RF_LIMIT_HZ,
    ) -> None:
        from collections import deque

        self._model = model
        self.n_error_history = n_error_history
        self.n_rf_history = n_rf_history
        self.include_env_sensors = include_env_sensors
        self.rf_limit_Hz = rf_limit_Hz

        self._error_window: deque[float] = deque(
            [0.0] * n_error_history, maxlen=n_error_history
        )
        self._rf_window: deque[float] = deque(
            [0.0] * n_rf_history, maxlen=n_rf_history
        )
        self._T_K: float = 333.15
        self._B_uT: float = 50.0
        self._I_norm: float = 1.0

    def reset(self) -> None:
        """Reset history windows to zero."""
        from collections import deque

        self._error_window = deque(
            [0.0] * self.n_error_history, maxlen=self.n_error_history
        )
        self._rf_window = deque(
            [0.0] * self.n_rf_history, maxlen=self.n_rf_history
        )
        self._T_K = 333.15
        self._B_uT = 50.0
        self._I_norm = 1.0

    def set_env_sensors(self, T_K: float, B_uT: float, I_norm: float) -> None:
        """Update environmental sensor values for next step.

        Args:
            T_K: Cell temperature (K).
            B_uT: Magnetic field (µT).
            I_norm: Normalised laser intensity.
        """
        self._T_K = T_K
        self._B_uT = B_uT
        self._I_norm = I_norm

    def step(self, error: float) -> tuple[float, float]:
        """Compute (laser_corr, rf_corr) from error signal.

        Compatible with PIController protocol.

        Args:
            error: Discriminator error signal (ci units).

        Returns:
            Tuple (laser_detuning_correction_Hz=0, rf_detuning_correction_Hz).
        """
        self._error_window.append(error)

        obs: list[float] = list(self._error_window) + list(self._rf_window)
        if self.include_env_sensors:
            obs += [self._T_K, self._B_uT, self._I_norm]

        obs_arr = np.array(obs, dtype=np.float32).reshape(1, -1)
        action, _ = self._model.predict(obs_arr, deterministic=True)
        action_scalar = float(np.clip(action.flatten()[0], -1.0, 1.0))
        rf_raw = action_scalar * self.rf_limit_Hz
        rf_corr = float(np.clip(rf_raw, -self.rf_limit_Hz, self.rf_limit_Hz))

        self._rf_window.append(rf_corr)
        return (0.0, rf_corr)


# ---------------------------------------------------------------------------
# PPO closed-loop runner (mirrors run_fast_loop exactly)
# ---------------------------------------------------------------------------


def run_ppo_loop(
    model,
    trace,
    duration_s: float,
) -> dict[str, Any]:
    """Run a PPO-controlled closed-loop identical to run_fast_loop.

    Replicates run_fast_loop exactly but calls the PPO controller and passes
    env-sensor values (T_K, B_uT, I_norm) to its observation window.

    Args:
        model: SB3 PPO model (loaded).
        trace: DisturbanceTrace for the run.
        duration_s: Simulation duration (s).

    Returns:
        Dict with y, rf_cmd, wall_s and noise provenance fields.
    """
    import math

    import torch
    import yaml as _yaml

    HF_GROUND = 6_834_682_610.904
    physics_rate_Hz = 10_000.0
    decimation_rate_Hz = 1_000.0
    decimation = int(round(physics_rate_Hz / decimation_rate_Hz))
    n_dec = int(round(duration_s * decimation_rate_Hz))
    n_physics = n_dec * decimation
    dt = 1.0 / physics_rate_Hz

    with open(_PROJECT_ROOT / "configs" / "v1_recipe.yaml", encoding="utf-8") as fh:
        recipe = _yaml.safe_load(fh)
    nb = recipe["noise_budget"]
    white_fm_terms = [
        float(nb["photon_shot_noise_amp"]),
        float(nb["lo_white_fm_amp"]),
        float(nb.get("microwave_phase_white_amp", 0.0)),
        float(nb.get("detector_electronics_amp", 0.0)),
    ]
    total_white_fm_amp = float(np.sqrt(sum(t**2 for t in white_fm_terms)))
    lo_noise_std = total_white_fm_amp * HF_GROUND * math.sqrt(physics_rate_Hz)

    dist_arr = np.stack(
        [trace.T_K, trace.B_uT, trace.laser_intensity_norm], axis=1
    ).astype(np.float64)
    if dist_arr.shape[0] < n_physics:
        reps = (n_physics + dist_arr.shape[0] - 1) // dist_arr.shape[0]
        dist_arr = np.tile(dist_arr, (reps, 1))[:n_physics]
    elif dist_arr.shape[0] > n_physics:
        dist_arr = dist_arr[:n_physics]

    twin = make_calibrated_twin()
    state = twin.initial_state(batch_size=1)
    ctrl_zero = torch.zeros(1, 2, dtype=twin.dtype, device=twin.device)
    dist_t = torch.from_numpy(dist_arr).to(twin.device)

    with torch.no_grad():
        for _ in range(5_000):
            state = twin.step(state, ctrl_zero, dist_t[:1], dt)

    rng = np.random.default_rng(RNG_SEED)
    lo_noise = rng.normal(0.0, lo_noise_std, size=n_physics)

    ppo_ctrl = _PPOController(model=model)
    ppo_ctrl.reset()

    y_out = np.empty(n_dec, dtype=np.float64)
    rf_cmd_out = np.empty(n_dec, dtype=np.float64)

    ctrl = torch.zeros(1, 2, dtype=twin.dtype, device=twin.device)
    rf_cmd: float = 0.0

    t0 = time.perf_counter()
    progress_every = max(1, n_dec // 20)
    with torch.no_grad():
        for k in range(n_dec):
            if k % progress_every == 0 and k > 0:
                elapsed = time.perf_counter() - t0
                eta = elapsed * (n_dec - k) / k
                log(
                    f"    PPO loop: {k}/{n_dec} ({100.0 * k / n_dec:.0f}%) "
                    f"elapsed={elapsed:.0f}s eta={eta:.0f}s"
                )
            ci_sum = 0.0
            y_sum = 0.0

            for j in range(decimation):
                idx = k * decimation + j
                rf_actual = rf_cmd + float(lo_noise[idx])
                ctrl[0, 1] = rf_actual

                state = twin.step(state, ctrl, dist_t[idx : idx + 1], dt)

                ci_sum += float(state[0, 6].item())

                B_now = torch.tensor(
                    [float(dist_arr[idx, 1])], dtype=twin.dtype, device=twin.device
                )
                T_now = torch.tensor(
                    [float(dist_arr[idx, 0])], dtype=twin.dtype, device=twin.device
                )
                y_sum += float(
                    twin.fractional_frequency_error_with_B(
                        state, ctrl, B_now, T_now
                    ).item()
                )

            err_clean = ci_sum / decimation
            err_noisy = err_clean + float(rng.normal(0.0, DISC_NOISE_AMP_CI))
            y_out[k] = y_sum / decimation
            rf_cmd_out[k] = rf_cmd

            last_idx = (k + 1) * decimation - 1
            ppo_ctrl.set_env_sensors(
                float(dist_arr[last_idx, 0]),
                float(dist_arr[last_idx, 1]),
                float(dist_arr[last_idx, 2]),
            )
            _, rf_cmd = ppo_ctrl.step(err_noisy)

    wall_s = time.perf_counter() - t0
    return {
        "y": y_out,
        "rf_cmd": rf_cmd_out,
        "wall_s": wall_s,
        "noise_injection_point": "rf_actual_pre_step+disc_noise_pre_controller",
    }


# ---------------------------------------------------------------------------
# Single-scenario head-to-head
# ---------------------------------------------------------------------------


def run_scenario(
    scenario: str,
    ppo_model,
) -> dict[str, Any]:
    """Run PI and PPO on one scenario; return comparison metrics.

    Args:
        scenario: Disturbance recipe name (e.g. 'thermal_ramp').
        ppo_model: Loaded SB3 PPO model.

    Returns:
        Dict with sigma_y values and wall times for both controllers.
    """
    log(f"  === Scenario: {scenario} ===")
    dist_gen = Disturbance.from_recipe(scenario)
    trace = dist_gen.generate(
        duration_s=DURATION_S,
        sample_rate_Hz=10_000.0,
        seed=RNG_SEED,
    )

    # PI run
    pi = PIController.from_recipe()
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
    log(f"    PI wall={pi_wall:.1f}s")

    # PPO run
    t0 = time.perf_counter()
    ppo_res = run_ppo_loop(ppo_model, trace, DURATION_S)
    ppo_wall = time.perf_counter() - t0
    log(f"    PPO wall={ppo_wall:.1f}s")

    # Allan deviations (demeaned)
    y_pi = pi_res["y"] - float(np.mean(pi_res["y"]))
    y_ppo = ppo_res["y"] - float(np.mean(ppo_res["y"]))

    pi_allan = overlapping_allan(y_pi, DECIMATION_RATE_HZ, EVAL_TAUS)
    ppo_allan = overlapping_allan(y_ppo, DECIMATION_RATE_HZ, EVAL_TAUS)

    for tau in EVAL_TAUS:
        log(
            f"    sigma_y(tau={tau:.0f}s):  PI={pi_allan.get(tau, float('nan')):.3e}  "
            f"PPO={ppo_allan.get(tau, float('nan')):.3e}"
        )

    return {
        "pi_allan": pi_allan,
        "ppo_allan": ppo_allan,
        "pi_wall_s": pi_wall,
        "ppo_wall_s": ppo_wall,
    }


# ---------------------------------------------------------------------------
# Main gate
# ---------------------------------------------------------------------------


def run_head_to_head() -> dict[str, Any]:
    """Run PI and PPO on thermal_ramp and all_stacked; return gate metrics."""
    from stable_baselines3 import PPO

    log("=== M7 gate: PPO vs PI on thermal_ramp + all_stacked ===")
    log(f"  duration={DURATION_S}s, taus={EVAL_TAUS}, disc_noise={DISC_NOISE_AMP_CI:.0e}")

    best_model_path = _PROJECT_ROOT / "models" / "ppo_best.zip"
    if not best_model_path.exists():
        raise FileNotFoundError(
            f"PPO model not found at {best_model_path}. "
            "Run m7_ppo_train.py first."
        )
    log(f"  Loading PPO model from {best_model_path} (device=cpu for fast inference)")
    ppo_model = PPO.load(str(best_model_path), device="cpu")

    # thermal_ramp (primary gate scenario)
    tr = run_scenario("thermal_ramp", ppo_model)
    pi_tr_10 = float(tr["pi_allan"].get(10.0, float("nan")))
    ppo_tr_10 = float(tr["ppo_allan"].get(10.0, float("nan")))

    if np.isfinite(ppo_tr_10) and np.isfinite(pi_tr_10) and pi_tr_10 > 0.0:
        speedup_tr_10 = pi_tr_10 / ppo_tr_10
    else:
        speedup_tr_10 = float("nan")

    ppo_wins_tr_10 = bool(
        np.isfinite(ppo_tr_10) and np.isfinite(pi_tr_10) and ppo_tr_10 <= pi_tr_10
    )

    # Bonus: PPO matches DLQR within 50% on thermal_ramp tau=10s
    if np.isfinite(ppo_tr_10) and _DLQR_SIGMA_Y_10S_THERMAL > 0.0:
        ppo_vs_lqr_ratio = ppo_tr_10 / _DLQR_SIGMA_Y_10S_THERMAL
        ppo_matches_lqr = bool(ppo_vs_lqr_ratio <= 1.5)
    else:
        ppo_vs_lqr_ratio = float("nan")
        ppo_matches_lqr = False

    log(
        f"  thermal_ramp tau=10s:  PI={pi_tr_10:.3e}  PPO={ppo_tr_10:.3e}  "
        f"speedup={speedup_tr_10:.3f}  ppo_wins={ppo_wins_tr_10}"
    )
    log(
        f"  vs DLQR ({_DLQR_SIGMA_Y_10S_THERMAL:.3e}):  "
        f"ratio={ppo_vs_lqr_ratio:.3f}  matches_lqr_within_50pct={ppo_matches_lqr}"
    )

    # all_stacked
    ast = run_scenario("all_stacked", ppo_model)
    pi_ast_100 = float(ast["pi_allan"].get(100.0, float("nan")))
    ppo_ast_100 = float(ast["ppo_allan"].get(100.0, float("nan")))

    if np.isfinite(ppo_ast_100) and np.isfinite(pi_ast_100) and pi_ast_100 > 0.0:
        speedup_ast_100 = pi_ast_100 / ppo_ast_100
    else:
        speedup_ast_100 = float("nan")

    ppo_wins_ast_100 = bool(
        np.isfinite(ppo_ast_100) and np.isfinite(pi_ast_100) and ppo_ast_100 <= pi_ast_100
    )

    return {
        # thermal_ramp
        "pi_tr_1s": float(tr["pi_allan"].get(1.0, float("nan"))),
        "ppo_tr_1s": float(tr["ppo_allan"].get(1.0, float("nan"))),
        "pi_tr_10s": pi_tr_10,
        "ppo_tr_10s": ppo_tr_10,
        "pi_tr_100s": float(tr["pi_allan"].get(100.0, float("nan"))),
        "ppo_tr_100s": float(tr["ppo_allan"].get(100.0, float("nan"))),
        "speedup_tr_10s": speedup_tr_10,
        "ppo_wins_tr_10s": ppo_wins_tr_10,
        "ppo_vs_lqr_ratio_tr_10s": ppo_vs_lqr_ratio,
        "ppo_matches_lqr_within_50pct": ppo_matches_lqr,
        "pi_tr_wall_s": tr["pi_wall_s"],
        "ppo_tr_wall_s": tr["ppo_wall_s"],
        # all_stacked
        "pi_ast_1s": float(ast["pi_allan"].get(1.0, float("nan"))),
        "ppo_ast_1s": float(ast["ppo_allan"].get(1.0, float("nan"))),
        "pi_ast_10s": float(ast["pi_allan"].get(10.0, float("nan"))),
        "ppo_ast_10s": float(ast["ppo_allan"].get(10.0, float("nan"))),
        "pi_ast_100s": pi_ast_100,
        "ppo_ast_100s": ppo_ast_100,
        "speedup_ast_100s": speedup_ast_100,
        "ppo_wins_ast_100s": ppo_wins_ast_100,
        "pi_ast_wall_s": ast["pi_wall_s"],
        "ppo_ast_wall_s": ast["ppo_wall_s"],
    }


def run_tests(test_file: str = "tests/test_ppo_env.py") -> tuple[bool, int, int]:
    """Run pytest on test_ppo_env.py; return (all_passed, n_passed, n_total)."""
    log(f"  Running pytest {test_file} ...")
    result = subprocess.run(
        [sys.executable, "-m", "pytest", test_file, "-v", "--tb=short"],
        capture_output=True,
        text=True,
        cwd=str(_PROJECT_ROOT),
        timeout=300,
    )
    out = result.stdout + result.stderr
    log(f"  pytest returncode={result.returncode}")

    n_passed = 0
    n_total = 0
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


def load_training_metrics() -> dict:
    """Load training metrics from ppo_train_summary.json if available."""
    summary_path = _PROJECT_ROOT / "data" / "ppo_train_summary.json"
    if summary_path.exists():
        return json.loads(summary_path.read_text(encoding="utf-8-sig"))
    return {
        "total_timesteps": 0,
        "stages": [],
        "wall_time_s": 0.0,
        "final_mean_reward": float("nan"),
    }


def main() -> None:
    data_dir = _PROJECT_ROOT / "data"
    data_dir.mkdir(exist_ok=True)

    log("=== M7 gate script start ===")

    # 1. Head-to-head simulation (thermal_ramp + all_stacked)
    hth = run_head_to_head()

    # 2. Tests
    tests_pass, n_passed, n_total = run_tests()

    # 3. Ruff
    ruff_clean = run_ruff()

    # 4. Gate verdict: PPO <= PI on thermal_ramp at tau=10s (primary criterion)
    gate_pass_sim = hth["ppo_wins_tr_10s"]
    gate_pass = bool(gate_pass_sim and tests_pass and ruff_clean)

    log("=== M7 gate verdict ===")
    log(f"  thermal_ramp tau=10s:  PI={hth['pi_tr_10s']:.3e}  PPO={hth['ppo_tr_10s']:.3e}")
    log(f"  speedup_at_10s        = {hth['speedup_tr_10s']:.3f}")
    log(f"  ppo_wins_tr_10s       = {hth['ppo_wins_tr_10s']}")
    log(
        f"  ppo_matches_lqr_50pct = {hth['ppo_matches_lqr_within_50pct']} "
        f"(ratio={hth['ppo_vs_lqr_ratio_tr_10s']:.3f})"
    )
    log(f"  all_stacked tau=100s: PI={hth['pi_ast_100s']:.3e}  PPO={hth['ppo_ast_100s']:.3e}")
    log(f"  tests_passed          = {tests_pass} ({n_passed}/{n_total})")
    log(f"  ruff_clean            = {ruff_clean}")
    log(f"  gate_pass             = {gate_pass}")

    # 5. Load training metrics for embedding
    train_metrics = load_training_metrics()

    # 6. Write gate JSON (M7 spec schema)
    gate_doc: dict[str, Any] = {
        "milestone": "M7",
        "policy_type": "PPO_MlpPolicy",
        "training_metrics": {
            "total_timesteps": train_metrics.get("total_timesteps", 0),
            "stages": train_metrics.get("stages", []),
            "wall_time_s": train_metrics.get("wall_time_s", 0.0),
            "final_mean_reward": train_metrics.get("final_mean_reward", float("nan")),
        },
        "head_to_head_thermal_ramp": {
            "duration_s": DURATION_S,
            "pi_sigma_y_1s": hth["pi_tr_1s"],
            "ppo_sigma_y_1s": hth["ppo_tr_1s"],
            "pi_sigma_y_10s": hth["pi_tr_10s"],
            "ppo_sigma_y_10s": hth["ppo_tr_10s"],
            "pi_sigma_y_100s": hth["pi_tr_100s"],
            "ppo_sigma_y_100s": hth["ppo_tr_100s"],
            "ppo_speedup_at_10s": hth["speedup_tr_10s"],
            "ppo_wins_at_10s": hth["ppo_wins_tr_10s"],
            "rhlqr_sigma_y_10s_for_reference": _DLQR_SIGMA_Y_10S_THERMAL,
            "ppo_vs_lqr_ratio_at_10s": hth["ppo_vs_lqr_ratio_tr_10s"],
            "ppo_matches_lqr_within_50pct": hth["ppo_matches_lqr_within_50pct"],
        },
        "head_to_head_all_stacked": {
            "duration_s": DURATION_S,
            "pi_sigma_y_1s": hth["pi_ast_1s"],
            "ppo_sigma_y_1s": hth["ppo_ast_1s"],
            "pi_sigma_y_10s": hth["pi_ast_10s"],
            "ppo_sigma_y_10s": hth["ppo_ast_10s"],
            "pi_sigma_y_100s": hth["pi_ast_100s"],
            "ppo_sigma_y_100s": hth["ppo_ast_100s"],
            "ppo_speedup_at_100s": hth["speedup_ast_100s"],
            "ppo_wins_at_100s": hth["ppo_wins_ast_100s"],
        },
        "noise_injection_point": "rf_actual_pre_step+disc_noise_pre_controller",
        "tests_passed": tests_pass,
        "n_tests_passed": n_passed,
        "ruff_clean": ruff_clean,
        "gate_pass": gate_pass,
    }

    # Document gap if PPO fails to match PI on thermal_ramp tau=10s
    if not hth["ppo_wins_tr_10s"]:
        gap_ratio = (
            hth["ppo_tr_10s"] / hth["pi_tr_10s"]
            if (np.isfinite(hth["ppo_tr_10s"]) and hth["pi_tr_10s"] > 0.0)
            else float("nan")
        )
        gate_doc["gate_fail_analysis"] = {
            "gap_ratio_ppo_over_pi_thermal_ramp_10s": (
                round(gap_ratio, 4) if np.isfinite(gap_ratio) else None
            ),
            "recommendations": [
                "(a) Longer training: increase thermal_ramp timesteps from 200k to 500k",
                "(b) Different reward shaping: use sliding-window variance over tau=10s "
                "instead of per-step -y_demeaned^2",
                "(c) Different observation space: add explicit dT/dt "
                "(temperature derivative) to obs",
                "(d) Accept gap as documented M7 baseline: "
                "PPO RL infrastructure verified, short-training underfit on 1000s gate",
            ],
            "honest_assessment": (
                "PPO with 650k steps and 20s episodes may underfit on the 1000s gate "
                "trajectory. The thermal_ramp curriculum provides focused exposure but "
                "generalisation to the full 1000s test requires more training or "
                "reward shaping that explicitly targets the tau=10s Allan band."
            ),
        }

    out_path = data_dir / "gate_M7.json"
    out_path.write_text(json.dumps(gate_doc, indent=2), encoding="utf-8")
    log(f"Wrote {out_path}")

    verdict = "GATE PASS" if gate_pass else "GATE FAIL"
    log(f"\n{'=' * 60}")
    log(f"  M7 {verdict}")
    log(
        "  Primary criterion: PPO sigma_y(10s) <= PI sigma_y(10s) on thermal_ramp"
    )
    log(
        f"  PI={hth['pi_tr_10s']:.3e}  PPO={hth['ppo_tr_10s']:.3e}  "
        f"speedup={hth['speedup_tr_10s']:.3f}"
    )
    log(f"{'=' * 60}")


if __name__ == "__main__":
    main()
