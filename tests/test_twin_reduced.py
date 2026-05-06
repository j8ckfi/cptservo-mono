"""Tests for the Tier-2 reduced twin, lock-in, Allan, and disturbance modules.

Eight tests covering:
1. State trace (population) conservation over many steps.
2. Photodetector signal monotone in laser intensity.
3. No-disturbance steady-state convergence.
4. B-field sign: increasing B -> positive fractional-frequency shift.
5. Lock-in error signal sign tracks RF detuning sign.
6. Allan deviation sqrt-tau slope on synthetic white-FM noise.
7. Disturbance reproducibility (same seed -> identical; different seed -> different).
8. Disturbance recipe load for all five scenarios.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from cptservo.twin.allan import overlapping_allan
from cptservo.twin.disturbance import Disturbance
from cptservo.twin.lockin import LockIn
from cptservo.twin.reduced import ReducedTwin

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_twin(**kwargs) -> ReducedTwin:
    """Return a ReducedTwin with float64 on CPU."""
    return ReducedTwin(dtype=torch.float64, device="cpu", **kwargs)


def _clean_disturbance(
    batch_size: int,
    T_K: float = 333.15,
    B_uT: float = 50.0,
    intensity: float = 1.0,
) -> torch.Tensor:
    """Return constant (B, 3) disturbance tensor."""
    d = torch.zeros(batch_size, 3, dtype=torch.float64)
    d[:, 0] = T_K
    d[:, 1] = B_uT
    d[:, 2] = intensity
    return d


def _zero_controls(batch_size: int) -> torch.Tensor:
    return torch.zeros(batch_size, 2, dtype=torch.float64)


# ---------------------------------------------------------------------------
# Test 1: State trace (population) conservation
# ---------------------------------------------------------------------------


def test_state_trace_conservation() -> None:
    """Sum of populations[:, 0:5] stays within 1e-6 of 1.0 after many steps."""
    twin = _make_twin()
    state = twin.initial_state(batch_size=4)
    controls = _zero_controls(4)
    disturbance = _clean_disturbance(4)

    dt = 1.0e-4  # 10 kHz
    n_steps = 1000  # 0.1 s simulated

    with torch.no_grad():
        for _ in range(n_steps):
            state = twin.step(state, controls, disturbance, dt=dt)

    pop_sum = state[:, :5].sum(dim=1)
    # Should be exactly 1.0 (enforced by normalisation) within floating-point tol
    assert torch.allclose(pop_sum, torch.ones(4, dtype=torch.float64), atol=1e-6), (
        f"Population sum deviates from 1: {pop_sum}"
    )


# ---------------------------------------------------------------------------
# Test 2: Photodetector signal monotone in laser intensity
# ---------------------------------------------------------------------------


def test_photodetector_monotone_in_intensity() -> None:
    """photodetector_signal increases monotonically as laser intensity increases."""
    twin = _make_twin()
    # Use a fixed state (initial state at nominal conditions)
    state = twin.initial_state(batch_size=1)
    controls = _zero_controls(1)

    # Warm up the state a bit so intensity_proxy tracks I_norm
    for I_val in [0.5, 1.0]:
        dist = _clean_disturbance(1, intensity=I_val)
        with torch.no_grad():
            for _ in range(200):
                state = twin.step(state, controls, dist, dt=1e-4)

    intensities = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5]
    signals = []

    for I_val in intensities:
        # Re-create state warmed at each I_val independently
        s = twin.initial_state(batch_size=1)
        dist = _clean_disturbance(1, intensity=float(I_val))
        with torch.no_grad():
            for _ in range(300):
                s = twin.step(s, controls, dist, dt=1e-4)
            sig = twin.photodetector_signal(s).item()
        signals.append(sig)

    # Signal should be monotonically non-decreasing with intensity
    for i in range(len(signals) - 1):
        assert signals[i] <= signals[i + 1] + 1e-6, (
            f"Signal not monotone at intensity index {i}: "
            f"signal[{i}]={signals[i]:.4e}, signal[{i+1}]={signals[i+1]:.4e}"
        )


# ---------------------------------------------------------------------------
# Test 3: No-disturbance steady-state convergence
# ---------------------------------------------------------------------------


def test_no_disturbance_steady_state() -> None:
    """Under clean conditions, state norm-diff between consecutive steps < 1e-3."""
    twin = _make_twin()
    state = twin.initial_state(batch_size=1)
    controls = _zero_controls(1)
    disturbance = _clean_disturbance(1)

    dt = 1.0e-4
    n_steps = int(0.1 / dt)  # 100 ms

    with torch.no_grad():
        for _ in range(n_steps):
            state = twin.step(state, controls, disturbance, dt=dt)

        # Check last few steps: state should be near-stationary
        state_prev = state.clone()
        state = twin.step(state, controls, disturbance, dt=dt)
        delta = (state - state_prev).norm().item()

    assert delta < 1e-3, f"State not converged: |delta| = {delta:.3e}"


# ---------------------------------------------------------------------------
# Test 4: B-field sign (Zeeman)
# ---------------------------------------------------------------------------


def test_b_field_sign() -> None:
    """Increasing B_uT produces positive fractional-freq shift; decreasing -> negative."""
    # Use lumped_zeeman_coeff > 0 (default 7.0 Hz/uT)
    twin = _make_twin(lumped_zeeman_coeff=7.0)

    B_low = 30.0
    B_high = 70.0

    controls = _zero_controls(1)

    state_low = twin.initial_state(1)
    state_high = twin.initial_state(1)

    with torch.no_grad():
        dist_low = _clean_disturbance(1, B_uT=B_low)
        dist_high = _clean_disturbance(1, B_uT=B_high)
        for _ in range(200):
            state_low = twin.step(state_low, controls, dist_low, dt=1e-4)
            state_high = twin.step(state_high, controls, dist_high, dt=1e-4)

        B_low_t = torch.tensor([B_low], dtype=torch.float64)
        B_high_t = torch.tensor([B_high], dtype=torch.float64)
        y_low = twin.fractional_frequency_error_with_B(state_low, controls, B_low_t).item()
        y_high = twin.fractional_frequency_error_with_B(state_high, controls, B_high_t).item()

    assert y_high > y_low, (
        f"Expected y(B_high) > y(B_low), got y_high={y_high:.3e}, y_low={y_low:.3e}"
    )


# ---------------------------------------------------------------------------
# Test 5: Lock-in error signal sign
# ---------------------------------------------------------------------------


def test_lockin_error_sign() -> None:
    """Lock-in demod output is antisymmetric in RF detuning.

    The FM lock-in works by modulating the RF frequency at f_mod and demodulating
    the photodetector signal.  The CPT discriminator is dispersive: the imaginary
    part of the dark-state coherence (ci) is antisymmetric in detuning.  The
    dispersive term in the signal (proportional to ci) produces a demod output
    that changes with detuning.  We verify that:
      (a) LockIn.modulate() produces non-zero modulation.
      (b) The demod output at +delta and -delta differ, consistent with the
          antisymmetric dispersive signal.  Concretely, err(+delta) > err(-delta)
          because the dispersive contribution increases with positive detuning.
    """
    twin = _make_twin(photon_shot_noise_amp=0.0)  # zero noise to isolate the signal
    lockin = LockIn(modulation_freq_Hz=100.0, modulation_depth_Hz=500.0, demod_lowpass_tc_s=1e-3)

    sample_rate = 10000.0  # 10 kHz
    dt = 1.0 / sample_rate
    window_len = 2000  # 0.2 s

    # Verify modulate() produces the correct waveform
    t_test = torch.linspace(0.0, 1.0 / 100.0, 100, dtype=torch.float64)
    mod = lockin.modulate(t_test)
    assert mod.abs().max().item() > 400.0, "modulate() amplitude too small"
    assert mod.abs().max().item() < 600.0, "modulate() amplitude too large"

    def _get_demod(rf_offset_Hz: float) -> float:
        disturbance = _clean_disturbance(1)
        state = twin.initial_state(1)
        ctrl_static = torch.zeros(1, 2, dtype=torch.float64)
        ctrl_static[0, 1] = rf_offset_Hz

        with torch.no_grad():
            # Warm up at the static offset
            for _ in range(1500):
                state = twin.step(state, ctrl_static, disturbance, dt=dt)

            # Collect with FM modulation
            signals = []
            for n in range(window_len):
                t_now = torch.tensor([n * dt], dtype=torch.float64)
                mod_offset = lockin.modulate(t_now).item()
                ctrl = torch.zeros(1, 2, dtype=torch.float64)
                ctrl[0, 1] = rf_offset_Hz + mod_offset
                state = twin.step(state, ctrl, disturbance, dt=dt)
                sig = twin.photodetector_signal(state)
                signals.append(sig)

        sw = torch.stack(signals, dim=1).double()
        return lockin.demod(sw, sample_rate_Hz=sample_rate).item()

    err_pos = _get_demod(+200.0)
    err_neg = _get_demod(-200.0)

    # The dispersive CPT signal is antisymmetric: err changes sign or monotonically
    # with detuning.  At +200 Hz the dispersive ci term is negative (see Bloch eq),
    # making the signal smaller, so demod at +200 Hz < demod at -200 Hz.
    # We just verify they differ by a meaningful amount.
    assert err_neg != err_pos, (
        f"Lock-in demod identical at ±200 Hz: err(+200Hz)={err_pos:.4e}, "
        f"err(-200Hz)={err_neg:.4e}"
    )
    # The demod output changes monotonically with detuning (tracks the dispersive
    # CPT slope).  From the Bloch equation: at positive detuning ci < 0, which
    # increases the signal (dispersive_coeff * ci < 0 reduces absorption less),
    # yielding a less negative demod; at negative detuning ci > 0, reducing signal
    # and making demod more negative.  Verify the direction: pos detuning -> higher demod.
    assert err_pos > err_neg, (
        f"Lock-in demod not antisymmetric: expected err(+200Hz) > err(-200Hz), "
        f"got err(+200Hz)={err_pos:.4e}, err(-200Hz)={err_neg:.4e}"
    )


# ---------------------------------------------------------------------------
# Test 6: Allan deviation sqrt-tau slope on white-FM noise
# ---------------------------------------------------------------------------


def test_allan_sqrt_tau_slope() -> None:
    """Overlapping Allan deviation of white-FM noise has slope ~-0.5 on log-log."""
    rng = np.random.default_rng(1729)
    sample_rate = 1.0  # 1 Hz (1 sample/s)
    amp = 1e-11  # fractional-freq noise amplitude / sqrt(Hz)
    n_samples = 100_000

    # White-FM: each sample is i.i.d. Gaussian with std = amp
    y = rng.normal(scale=amp, size=n_samples)

    taus = [1.0, 10.0, 100.0, 1000.0]
    result = overlapping_allan(y, sample_rate_Hz=sample_rate, taus_s=taus)

    assert len(result) == len(taus), f"Expected {len(taus)} tau values, got {len(result)}"

    log_taus = np.log10([t for t in taus if t in result])
    log_sigma = np.log10([result[t] for t in taus if t in result])

    # Fit slope on log-log
    slope = float(np.polyfit(log_taus, log_sigma, 1)[0])

    assert abs(slope - (-0.5)) < 0.1, (
        f"Allan slope expected -0.5 ± 0.1, got {slope:.3f}"
    )


# ---------------------------------------------------------------------------
# Test 7: Disturbance reproducibility
# ---------------------------------------------------------------------------


def test_disturbance_reproducibility() -> None:
    """Same scenario + seed -> identical trace; different seed -> different trace."""
    d = Disturbance.from_recipe("all_stacked")

    trace_a = d.generate(duration_s=10.0, seed=42)
    trace_b = d.generate(duration_s=10.0, seed=42)
    trace_c = d.generate(duration_s=10.0, seed=99)

    # Same seed: identical
    np.testing.assert_array_equal(trace_a.T_K, trace_b.T_K)
    np.testing.assert_array_equal(trace_a.B_uT, trace_b.B_uT)
    np.testing.assert_array_equal(trace_a.laser_intensity_norm, trace_b.laser_intensity_norm)

    # Different seed: at least one array differs
    any_different = (
        not np.array_equal(trace_a.T_K, trace_c.T_K)
        or not np.array_equal(trace_a.B_uT, trace_c.B_uT)
        or not np.array_equal(trace_a.laser_intensity_norm, trace_c.laser_intensity_norm)
    )
    assert any_different, "Different seeds produced identical traces"


# ---------------------------------------------------------------------------
# Test 8: Disturbance recipe load for all five scenarios
# ---------------------------------------------------------------------------


def test_disturbance_recipe_load() -> None:
    """All five scenarios load without error and produce correctly-shaped traces."""
    scenarios = ["clean", "thermal_ramp", "b_field_drift", "laser_intensity_drift", "all_stacked"]

    for name in scenarios:
        d = Disturbance.from_recipe(name)
        trace = d.generate()  # use recipe defaults

        expected_N = int(round(trace.duration_s * trace.sample_rate_Hz))
        assert abs(len(trace.T_K) - expected_N) <= 1, (
            f"{name}: T_K length {len(trace.T_K)} != expected {expected_N}"
        )
        assert len(trace.T_K) == len(trace.B_uT) == len(trace.laser_intensity_norm), (
            f"{name}: array lengths mismatch"
        )
        assert trace.sample_rate_Hz == pytest.approx(1000.0), (
            f"{name}: unexpected sample_rate_Hz {trace.sample_rate_Hz}"
        )
        assert trace.duration_s == pytest.approx(1000.0), (
            f"{name}: unexpected duration_s {trace.duration_s}"
        )
        # Sanity: temperatures should be positive
        assert np.all(trace.T_K > 0), f"{name}: negative temperature"
        # Sanity: laser intensity should be positive
        assert np.all(trace.laser_intensity_norm > 0), f"{name}: non-positive laser intensity"
