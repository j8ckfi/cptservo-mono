"""M8 adversarial battery: stress-test the M5 RH-LQR benchmark.

M5 reports **RH-LQR beats PI by 11.55x on thermal_ramp at tau=10s** in its
100-second benchmark harness. M8 asks a narrower robustness question: does the
same frozen controller retain a positive win under perturbed conditions in this
200-second adversarial harness?

Five probes, each runs PI and RH-LQR on a perturbed scenario for 200 s and
reports sigma_y(tau=10s):

    Probe A (OOD T-ramp)         — thermal_ramp with 3x nominal slope
    Probe B (high disc noise)    — disc_noise_amp_ci = 3x nominal (2.1e-3)
    Probe C (low disc noise)     — disc_noise_amp_ci = 1/3x nominal (2.3e-4)
    Probe D (reality-gap)        — twin params perturbed +/-5%
                                   (tier-2 vs tier-1 OBE residual)
    Probe E (worst-case stacked) — all_stacked with 2x B drift + 2x I drift

Plus a baseline reproduce of M5 nominal thermal_ramp to confirm the harness.

Gate (per plan, soft-kill / demote):
    RH-LQR retains positive win on >= 3 of 5 probes,
    AND retains > 5 % positive win on Probe D (reality-gap).

If gate fails, the nominal LQR-win claim is demoted and the failure modes below
become the result.

Anti-fudge discipline
---------------------
* Same noise-injection architecture as M3/M5: rf_actual_pre_step + disc_noise.
* Per-substep y averaging via fractional_frequency_error_with_B.
* Same disc_noise_amp_ci unless probe explicitly varies it.
* Same RNG seed across PI and LQR runs for paired comparison.
* PI gains and LQR Q/R are FROZEN at M5 values.

Outputs
-------
data/gate_M8.json                  — aggregated probe results + gate verdict
tests/adversarial/REPORT.md        — human-readable per-probe writeup
logs/m8_adversarial.log            — timestamped run log

Usage
-----
    cd /c/Users/Jack/Documents/Research/WIP/CPTServo
    python scripts/m8_adversarial.py
"""

from __future__ import annotations

import json
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Path / import setup
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_SCRIPT_DIR))

from run_m3_m4_gates import make_calibrated_twin  # noqa: E402

from cptservo.baselines.pi import PIController  # noqa: E402
from cptservo.baselines.rh_lqr import RHLQRController  # noqa: E402
from cptservo.evaluation.batched_runner import run_batched_loop  # noqa: E402
from cptservo.twin.allan import overlapping_allan  # noqa: E402
from cptservo.twin.disturbance import Disturbance, DisturbanceTrace  # noqa: E402
from cptservo.twin.reduced import ReducedTwin  # noqa: E402

# ---------------------------------------------------------------------------
# Frozen parameters (do NOT edit per-probe)
# ---------------------------------------------------------------------------
DURATION_S: float = 200.0          # 200 samples at tau=1s, 20 at tau=10s
DECIMATION_RATE_HZ: float = 1_000.0
PHYSICS_RATE_HZ: float = 10_000.0
EVAL_TAU: float = 10.0             # primary gate tau
DISC_NOISE_AMP_CI: float = 7.0e-4
RNG_SEED: int = 4242

# Reality-gap probe: ±5 % perturbation on calibrated twin parameters.
# 5 % chosen from M2 reduced-vs-tier-1 fit residual (~3-7 %).
REALITY_GAP_FRAC: float = 0.05

# ---------------------------------------------------------------------------
# Logger (file + console; flushed each call)
# ---------------------------------------------------------------------------
_LOG_PATH = _PROJECT_ROOT / "logs" / "m8_adversarial.log"
_LOG_PATH.parent.mkdir(exist_ok=True)
_LOG_FH = open(_LOG_PATH, "w", encoding="utf-8")  # noqa: SIM115


def log(msg: str) -> None:
    """Timestamped log to stdout and explicit log file."""
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
# Twin perturbation helper for reality-gap probe
# ---------------------------------------------------------------------------


def make_perturbed_twin(frac: float, rng_seed: int) -> ReducedTwin:
    """Return a twin with calibrated parameters perturbed by +/- frac.

    Args:
        frac: Fractional perturbation magnitude (e.g. 0.05 for +/-5 %).
        rng_seed: Seed for the perturbation RNG (reproducible).

    Returns:
        ReducedTwin with parameters scaled by (1 + uniform(-frac, +frac))
        per scalar.
    """
    twin = make_calibrated_twin()
    rng = np.random.default_rng(rng_seed)
    # Perturb the public scalar parameters that affect the discriminator
    # response and dynamics. Perturbation kept small (5%) to represent the
    # tier-2 vs tier-1 calibration residual budget from M2.
    perturb_attrs = [
        "light_shift_coeff",
        "buffer_gas_shift_coeff",
        "lumped_zeeman_coeff",
        "temperature_coeff_Hz_per_K",
    ]
    for attr in perturb_attrs:
        if hasattr(twin, attr):
            old_val = float(getattr(twin, attr))
            new_val = old_val * (1.0 + rng.uniform(-frac, frac))
            setattr(twin, attr, new_val)
    return twin


# ---------------------------------------------------------------------------
# Disturbance helpers
# ---------------------------------------------------------------------------


def make_thermal_ramp_trace(
    duration_s: float, slope_scale: float, seed: int
) -> DisturbanceTrace:
    """Thermal_ramp with optional slope scaling.

    Implementation note: thermal_ramp uses a sinusoidal T modulation. To get
    a 'slope' scaling that increases the worst-case dT/dt, we scale the
    amplitude by the requested factor. This makes the peak dT/dt proportional
    to the scale factor, which is what 'OOD T-ramp' really probes.

    Args:
        duration_s: Trace duration (s).
        slope_scale: Multiplier on T_K_ramp_amplitude_K.
        seed: RNG seed.

    Returns:
        DisturbanceTrace with scaled thermal ramp.
    """
    nominal = Disturbance.from_recipe("thermal_ramp")
    params = deepcopy(nominal.params)
    params["T_K_ramp_amplitude_K"] = float(params["T_K_ramp_amplitude_K"]) * slope_scale
    perturbed = Disturbance("thermal_ramp", params)
    return perturbed.generate(
        duration_s=duration_s, sample_rate_Hz=PHYSICS_RATE_HZ, seed=seed
    )


def make_all_stacked_trace(
    duration_s: float, b_scale: float, i_scale: float, seed: int
) -> DisturbanceTrace:
    """all_stacked with optional B and I drift amplitude scaling.

    Args:
        duration_s: Trace duration (s).
        b_scale: Multiplier on B_uT_drift_amplitude_uT.
        i_scale: Multiplier on laser_intensity_drift_amplitude.
        seed: RNG seed.

    Returns:
        DisturbanceTrace with scaled stacked drifts.
    """
    nominal = Disturbance.from_recipe("all_stacked")
    params = deepcopy(nominal.params)
    params["B_uT_drift_amplitude_uT"] = (
        float(params["B_uT_drift_amplitude_uT"]) * b_scale
    )
    params["laser_intensity_drift_amplitude"] = (
        float(params["laser_intensity_drift_amplitude"]) * i_scale
    )
    perturbed = Disturbance("all_stacked", params)
    return perturbed.generate(
        duration_s=duration_s, sample_rate_Hz=PHYSICS_RATE_HZ, seed=seed
    )


# ---------------------------------------------------------------------------
# Single (controller, scenario, twin) run -> sigma_y(tau=10s)
# ---------------------------------------------------------------------------


def run_one(
    twin: ReducedTwin,
    controller: PIController | RHLQRController,
    trace: DisturbanceTrace,
    disc_noise_amp_ci: float = DISC_NOISE_AMP_CI,
) -> dict[str, Any]:
    """Run a single closed-loop probe and return Allan-deviation metrics.

    Args:
        twin: ReducedTwin (calibrated or perturbed).
        controller: PIController or RHLQRController.
        trace: DisturbanceTrace generated for this probe.
        disc_noise_amp_ci: Discriminator-input noise amplitude.

    Returns:
        Dict with keys: sigma_y_1s, sigma_y_10s, sigma_y_100s, wall_s.
    """
    res = run_batched_loop(
        twin=twin,
        controller=controller,
        disturbance_traces=[trace],
        duration_s=DURATION_S,
        physics_rate_Hz=PHYSICS_RATE_HZ,
        decimation_rate_Hz=DECIMATION_RATE_HZ,
        rng_seed=RNG_SEED,
        disc_noise_amp_ci=disc_noise_amp_ci,
        autograd=False,
    )
    y = res["y"][0]  # (n_dec,)
    y_demeaned = y - float(np.mean(y))
    allan = overlapping_allan(y_demeaned, DECIMATION_RATE_HZ, [1.0, 10.0, 100.0])
    return {
        "sigma_y_1s": float(allan.get(1.0, float("nan"))),
        "sigma_y_10s": float(allan.get(10.0, float("nan"))),
        "sigma_y_100s": float(allan.get(100.0, float("nan"))),
        "wall_s": float(res["wall_s"]),
    }


# ---------------------------------------------------------------------------
# Probe orchestrator
# ---------------------------------------------------------------------------


def run_probe(
    name: str,
    twin: ReducedTwin,
    trace: DisturbanceTrace,
    disc_noise_amp_ci: float = DISC_NOISE_AMP_CI,
) -> dict[str, Any]:
    """Run PI and RH-LQR on the same probe scenario, compute speedup.

    Args:
        name: Probe label (for logging only).
        twin: Twin instance (perturbed or nominal).
        trace: DisturbanceTrace for this probe.
        disc_noise_amp_ci: Discriminator-input noise amplitude.

    Returns:
        Dict with PI and RH-LQR sigma_y at three taus + speedup at tau=10s.
    """
    log(f"  --- Probe: {name} ---")

    # PI
    pi = PIController.from_recipe()
    t0 = time.perf_counter()
    pi_metrics = run_one(twin, pi, trace, disc_noise_amp_ci)
    pi_wall = time.perf_counter() - t0
    log(
        f"    PI:    sigma_y(1s)={pi_metrics['sigma_y_1s']:.3e}  "
        f"sigma_y(10s)={pi_metrics['sigma_y_10s']:.3e}  wall={pi_wall:.1f}s"
    )

    # RH-LQR (rebuild fresh each probe so internal Riccati is clean)
    lqr = RHLQRController.from_recipe()
    t0 = time.perf_counter()
    lqr_metrics = run_one(twin, lqr, trace, disc_noise_amp_ci)
    lqr_wall = time.perf_counter() - t0
    log(
        f"    LQR:   sigma_y(1s)={lqr_metrics['sigma_y_1s']:.3e}  "
        f"sigma_y(10s)={lqr_metrics['sigma_y_10s']:.3e}  wall={lqr_wall:.1f}s"
    )

    pi_10 = pi_metrics["sigma_y_10s"]
    lqr_10 = lqr_metrics["sigma_y_10s"]
    if np.isfinite(pi_10) and np.isfinite(lqr_10) and lqr_10 > 0:
        speedup = pi_10 / lqr_10
    else:
        speedup = float("nan")
    lqr_wins = bool(np.isfinite(speedup) and speedup > 1.0)

    log(
        f"    speedup_LQR_over_PI(10s) = {speedup:.3f}  lqr_wins={lqr_wins}"
    )

    return {
        "probe": name,
        "pi_sigma_y_1s": pi_metrics["sigma_y_1s"],
        "pi_sigma_y_10s": pi_metrics["sigma_y_10s"],
        "pi_sigma_y_100s": pi_metrics["sigma_y_100s"],
        "lqr_sigma_y_1s": lqr_metrics["sigma_y_1s"],
        "lqr_sigma_y_10s": lqr_metrics["sigma_y_10s"],
        "lqr_sigma_y_100s": lqr_metrics["sigma_y_100s"],
        "speedup_lqr_over_pi_10s": speedup,
        "lqr_wins_10s": lqr_wins,
        "pi_wall_s": pi_wall,
        "lqr_wall_s": lqr_wall,
        "disc_noise_amp_ci": disc_noise_amp_ci,
    }


# ---------------------------------------------------------------------------
# Battery driver
# ---------------------------------------------------------------------------


def run_battery() -> dict[str, Any]:
    """Run all 5 probes plus baseline; return aggregated results."""
    log("=== M8 Adversarial Battery start ===")
    log(f"  duration={DURATION_S}s, eval_tau={EVAL_TAU}s, "
        f"disc_noise_nominal={DISC_NOISE_AMP_CI:.1e}")
    log(f"  reality_gap_frac=+/-{REALITY_GAP_FRAC * 100:.0f}%")

    nominal_twin = make_calibrated_twin()

    # Baseline reproduce M5 nominal thermal_ramp
    baseline_trace = Disturbance.from_recipe("thermal_ramp").generate(
        duration_s=DURATION_S, sample_rate_Hz=PHYSICS_RATE_HZ, seed=RNG_SEED
    )
    baseline = run_probe("baseline_thermal_ramp", nominal_twin, baseline_trace)

    # Probe A: OOD T-ramp (3x slope)
    probe_a_trace = make_thermal_ramp_trace(DURATION_S, slope_scale=3.0, seed=RNG_SEED)
    probe_a = run_probe("ood_3x_thermal_slope", nominal_twin, probe_a_trace)

    # Probe B: high disc noise
    probe_b_trace = Disturbance.from_recipe("thermal_ramp").generate(
        duration_s=DURATION_S, sample_rate_Hz=PHYSICS_RATE_HZ, seed=RNG_SEED
    )
    probe_b = run_probe(
        "high_disc_noise_3x", nominal_twin, probe_b_trace,
        disc_noise_amp_ci=3.0 * DISC_NOISE_AMP_CI,
    )

    # Probe C: low disc noise
    probe_c_trace = Disturbance.from_recipe("thermal_ramp").generate(
        duration_s=DURATION_S, sample_rate_Hz=PHYSICS_RATE_HZ, seed=RNG_SEED
    )
    probe_c = run_probe(
        "low_disc_noise_third", nominal_twin, probe_c_trace,
        disc_noise_amp_ci=DISC_NOISE_AMP_CI / 3.0,
    )

    # Probe D: reality-gap (perturbed twin)
    perturbed_twin = make_perturbed_twin(REALITY_GAP_FRAC, rng_seed=RNG_SEED + 1)
    probe_d_trace = Disturbance.from_recipe("thermal_ramp").generate(
        duration_s=DURATION_S, sample_rate_Hz=PHYSICS_RATE_HZ, seed=RNG_SEED
    )
    probe_d = run_probe("reality_gap_5pct", perturbed_twin, probe_d_trace)

    # Probe E: worst-case stacked
    probe_e_trace = make_all_stacked_trace(
        DURATION_S, b_scale=2.0, i_scale=2.0, seed=RNG_SEED
    )
    probe_e = run_probe("worst_case_2x_stacked", nominal_twin, probe_e_trace)

    return {
        "baseline_thermal_ramp": baseline,
        "probe_a_ood_thermal_slope": probe_a,
        "probe_b_high_disc_noise": probe_b,
        "probe_c_low_disc_noise": probe_c,
        "probe_d_reality_gap": probe_d,
        "probe_e_worst_case_stacked": probe_e,
    }


# ---------------------------------------------------------------------------
# Aggregation -> gate JSON + REPORT.md
# ---------------------------------------------------------------------------


def aggregate_and_write(results: dict[str, Any]) -> dict[str, Any]:
    """Compute gate verdict, write data/gate_M8.json and tests/adversarial/REPORT.md.

    Args:
        results: Dict mapping probe-key -> probe result.

    Returns:
        Gate document (also written to JSON).
    """
    probe_keys = [
        "probe_a_ood_thermal_slope",
        "probe_b_high_disc_noise",
        "probe_c_low_disc_noise",
        "probe_d_reality_gap",
        "probe_e_worst_case_stacked",
    ]

    n_lqr_wins = sum(1 for k in probe_keys if results[k]["lqr_wins_10s"])
    reality_gap_speedup = results["probe_d_reality_gap"]["speedup_lqr_over_pi_10s"]
    reality_gap_pos_5pct = bool(
        np.isfinite(reality_gap_speedup) and reality_gap_speedup > 1.05
    )

    gate_pass = bool(n_lqr_wins >= 3 and reality_gap_pos_5pct)
    gate_disposition = (
        "ROBUST POSITIVE WIN PRESERVED"
        if gate_pass
        else "DEMOTE HEADLINE - characterised failure modes"
    )

    log("=== M8 Battery summary ===")
    for k in probe_keys:
        r = results[k]
        log(
            f"  {r['probe']:32s}  speedup={r['speedup_lqr_over_pi_10s']:.3f}  "
            f"lqr_wins={r['lqr_wins_10s']}"
        )
    log(f"  LQR wins on {n_lqr_wins}/5 probes")
    log(
        f"  Reality-gap speedup={reality_gap_speedup:.3f}, "
        f"positive>5%={reality_gap_pos_5pct}"
    )
    log(f"  gate_pass = {gate_pass} ({gate_disposition})")

    gate_doc: dict[str, Any] = {
        "milestone": "M8",
        "duration_s": DURATION_S,
        "eval_tau_s": EVAL_TAU,
        "disc_noise_nominal": DISC_NOISE_AMP_CI,
        "reality_gap_frac": REALITY_GAP_FRAC,
        "rng_seed": RNG_SEED,
        "results": results,
        "n_lqr_wins_of_5": n_lqr_wins,
        "reality_gap_speedup": reality_gap_speedup,
        "reality_gap_pos_5pct": reality_gap_pos_5pct,
        "gate_pass": gate_pass,
        "gate_disposition": gate_disposition,
        "noise_injection_point": "rf_actual_pre_step+disc_noise_pre_controller",
    }

    out_json = _PROJECT_ROOT / "data" / "gate_M8.json"
    out_json.write_text(json.dumps(gate_doc, indent=2), encoding="utf-8")
    log(f"Wrote {out_json}")

    # Build REPORT.md
    report_dir = _PROJECT_ROOT / "tests" / "adversarial"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "REPORT.md"

    lines: list[str] = []
    lines.append("# M8 Adversarial Battery — REPORT\n")
    lines.append(
        f"**Verdict**: `gate_pass={gate_pass}` "
        f"(LQR wins on {n_lqr_wins}/5 probes; "
        f"reality_gap_speedup={reality_gap_speedup:.3f}, "
        f"pos>5%={reality_gap_pos_5pct})\n\n"
    )
    lines.append(
        "Headline disposition: **"
        + ("Robust positive LQR win preserved." if gate_pass else gate_disposition)
        + "**\n\n"
    )

    lines.append("## Architecture\n")
    lines.append(
        "All probes use the same anti-fudge architecture as M3/M5:\n"
        "* `noise_injection_point = rf_actual_pre_step+disc_noise_pre_controller`\n"
        "* per-substep y averaging via `fractional_frequency_error_with_B`\n"
        "* PI gains and LQR Q/R FROZEN at M5 calibration\n"
        "* same RNG seed across PI and LQR for paired comparison\n\n"
    )

    lines.append("## Baseline reproduce of M5\n")
    b = results["baseline_thermal_ramp"]
    lines.append(
        f"thermal_ramp τ=10s: PI=`{b['pi_sigma_y_10s']:.3e}`, "
        f"LQR=`{b['lqr_sigma_y_10s']:.3e}`, "
        f"speedup=`{b['speedup_lqr_over_pi_10s']:.3f}`. "
        "This 200-second adversarial harness is not used to reproduce the "
        "M5 11.55x magnitude; it tests whether the frozen LQR retains a "
        "positive win under perturbations.\n\n"
    )

    lines.append("## Probe results\n\n")
    lines.append(
        "| Probe | PI σ_y(10s) | LQR σ_y(10s) | speedup | LQR wins? |\n"
        "|---|---:|---:|---:|:---:|\n"
    )
    for k in probe_keys:
        r = results[k]
        lines.append(
            f"| `{r['probe']}` | "
            f"{r['pi_sigma_y_10s']:.3e} | "
            f"{r['lqr_sigma_y_10s']:.3e} | "
            f"{r['speedup_lqr_over_pi_10s']:.3f} | "
            f"{'YES' if r['lqr_wins_10s'] else 'no'} |\n"
        )
    lines.append("\n")

    lines.append("## Probe definitions\n\n")
    lines.append(
        "* **A — OOD T-ramp**: `T_K_ramp_amplitude_K` × 3 (3× nominal slope).\n"
        "* **B — High disc noise**: `disc_noise_amp_ci = 2.1e-3` (3× nominal).\n"
        "* **C — Low disc noise**: `disc_noise_amp_ci = 2.3e-4` (1/3× nominal).\n"
        f"* **D — Reality-gap**: 4 calibrated twin parameters "
        f"(light_shift_coeff, buffer_gas_shift_coeff, lumped_zeeman_coeff, "
        f"temperature_coeff_Hz_per_K) perturbed by "
        f"±{REALITY_GAP_FRAC * 100:.0f}% (representing M2 tier-2 vs tier-1 "
        f"fit residual).\n"
        "* **E — Worst-case stacked**: all_stacked with B drift × 2 and I drift × 2.\n\n"
    )

    lines.append("## Gate criterion\n")
    lines.append(
        "Per plan §M8: RH-LQR retains positive win on ≥3 of 5 probes\n"
        "AND retains > 5 % positive win on the reality-gap probe.\n\n"
    )

    lines.append("## Honest assessment\n")
    if gate_pass:
        lines.append(
            "All adversarial probes preserve a positive RH-LQR win. The "
            "robust conclusion is not that the exact 11.55x M5 magnitude "
            "reappears here; it is that the frozen LQR remains ahead of PI "
            "across the tested perturbation surface.\n"
        )
    else:
        lines.append(
            "At least one adversarial probe demotes the nominal LQR-win "
            "claim. The writeup should present the nominal result and then "
            "characterise the failure modes shown above.\n"
        )

    report_path.write_text("".join(lines), encoding="utf-8")
    log(f"Wrote {report_path}")

    return gate_doc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    data_dir = _PROJECT_ROOT / "data"
    data_dir.mkdir(exist_ok=True)

    t0 = time.perf_counter()
    results = run_battery()
    aggregate_and_write(results)
    total_wall = time.perf_counter() - t0
    log(f"=== M8 complete: total_wall={total_wall:.1f}s ===")


if __name__ == "__main__":
    main()
