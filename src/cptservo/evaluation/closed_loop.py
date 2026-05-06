"""Closed-loop simulation harness for CPT-clock servo evaluation.

Runs a physics twin forward in time with a controller acting on an error signal.
Two error-signal modes are supported:

``use_direct_ci=True`` (default, fast)
    The imaginary part of the CPT dark-state coherence ``ci = Im(rho_12)`` is
    used directly as the error signal.  This is physically equivalent to the
    output of an ideal FM lock-in demodulator with infinite settling time and
    zero carrier phase error.  It is zero at two-photon resonance, positive
    when the RF frequency is below resonance (delta_bloch < 0), and negative
    when above resonance.  This mode skips the FM modulation and IIR lowpass,
    making it ~10× faster than the full lock-in path.

``use_direct_ci=False`` (high-fidelity)
    The photodetector signal is FM-modulated and demodulated through the
    ``LockIn`` IIR demodulator.  A per-run baseline is estimated from the
    warm-up period and subtracted so the error is zero-centred at resonance.
    This path is slower but includes the modulation noise and demodulator
    dynamics.

Sign convention (both modes)
    ``error > 0`` means RF frequency is below the CPT resonance (need to
    increase rf_corr).  The PI controller receives this error and its positive
    gains produce a positive rf_corr increment, closing the loop correctly.

References
----------
Kitching, J. (2018). Chip-scale atomic devices.
    *Applied Physics Reviews*, 5, 031302.
Vanier, J. & Mandache, C. (2007). The passive optically pumped Rb frequency
    standard: the laser approach. *Applied Physics B*, 87, 565-593.
"""

from __future__ import annotations

import time
from typing import Protocol

import numpy as np
import torch
from torch import Tensor

from cptservo.twin.disturbance import DisturbanceTrace
from cptservo.twin.lockin import LockIn
from cptservo.twin.reduced import ReducedTwin

# ---------------------------------------------------------------------------
# Controller protocol (structural typing — any object with .step() qualifies)
# ---------------------------------------------------------------------------


class ControllerProtocol(Protocol):
    """Minimal interface required of any controller passed to run_closed_loop."""

    def step(self, error: float) -> tuple[float, float]:
        """Given lock-in error, return (laser_Hz_correction, rf_Hz_correction)."""
        ...

    def reset(self) -> None:
        """Reset internal state (called once at simulation start)."""
        ...


# ---------------------------------------------------------------------------
# Disturbance helper
# ---------------------------------------------------------------------------


def _build_trace_index_array(
    n_physics_total: int,
    trace_len: int,
    trace_sample_rate_Hz: float,
    physics_rate_Hz: float,
) -> np.ndarray:
    """Return (n_physics_total,) array of trace indices (nearest-neighbour)."""
    idx = np.arange(n_physics_total, dtype=np.int64)
    if abs(trace_sample_rate_Hz - physics_rate_Hz) > 1.0:
        scale = trace_sample_rate_Hz / physics_rate_Hz
        idx = (idx * scale).astype(np.int64)
    return np.minimum(idx, trace_len - 1)


# ---------------------------------------------------------------------------
# Main closed-loop harness
# ---------------------------------------------------------------------------


def run_closed_loop(
    twin: ReducedTwin,
    controller: ControllerProtocol,
    disturbance_trace: DisturbanceTrace,
    duration_s: float,
    physics_rate_Hz: float = 10_000.0,
    lockin: LockIn | None = None,
    decimation_rate_Hz: float = 1_000.0,
    use_direct_ci: bool = True,
    n_warmup_steps: int = 50_000,
) -> dict:
    """Run a closed-loop simulation with the controller acting on the error signal.

    The simulation runs at ``physics_rate_Hz``.  At each decimation step, the
    error signal is computed (from ``ci`` or from the FM lock-in demodulator),
    the controller issues an RF correction, and the fractional-frequency error
    is sampled.

    Args:
        twin: ReducedTwin instance (already initialised with fitted params).
        controller: Any object implementing ``step(error) -> (laser_Hz, rf_Hz)``
            and ``reset()``.
        disturbance_trace: Pre-generated disturbance time series.  Sample rate
            does not need to match ``physics_rate_Hz``; nearest-neighbour
            mapping is used.
        duration_s: Simulation duration in seconds.
        physics_rate_Hz: Physics integration rate in Hz.
        lockin: LockIn demodulator (used only when ``use_direct_ci=False``).
            Defaults to ``f_mod=100 Hz, depth=500 Hz, tc=10 ms``.
        decimation_rate_Hz: Rate at which the controller is updated and the
            fractional-frequency series is recorded.
        use_direct_ci: If ``True`` (default), use ``ci = Im(rho_12)`` directly
            as the error signal (fast, ideal lock-in equivalent).  If ``False``,
            use the FM lock-in demodulator path (slower but more realistic).
        n_warmup_steps: Number of physics steps to run before the main loop to
            warm the state to approximate steady state.  The disturbance during
            warm-up uses the first sample of the trace.

    Returns:
        Dict with keys:

        - ``"y"`` (np.ndarray, shape (N_dec,)): Fractional-frequency series at
          decimation_rate_Hz.  Includes the DC systematic offset (buffer-gas,
          light-shift, Zeeman-at-B_nom).  Use ``y - mean(y)`` for Allan
          deviation when comparing servo noise floors.
        - ``"controls"`` (np.ndarray, shape (N_dec, 2)):  [laser_Hz, rf_Hz].
        - ``"error_signal"`` (np.ndarray, shape (N_dec,)): Error at each
          decimation step (ci or lock-in demod).
        - ``"wall_s"`` (float): Wall-clock time for the simulation.
        - ``"physics_rate_Hz"``, ``"decimation_rate_Hz"``, ``"duration_s"``.

    Raises:
        ValueError: If ``decimation_rate_Hz > physics_rate_Hz`` or if
            ``physics_rate_Hz`` is not an integer multiple of
            ``decimation_rate_Hz``.
    """
    if decimation_rate_Hz > physics_rate_Hz:
        raise ValueError(
            f"decimation_rate_Hz ({decimation_rate_Hz}) must be <= "
            f"physics_rate_Hz ({physics_rate_Hz})"
        )
    decimation_factor = int(round(physics_rate_Hz / decimation_rate_Hz))
    if abs(decimation_factor - physics_rate_Hz / decimation_rate_Hz) > 1e-6:
        raise ValueError(
            f"physics_rate_Hz ({physics_rate_Hz}) must be an integer multiple "
            f"of decimation_rate_Hz ({decimation_rate_Hz})"
        )

    if not use_direct_ci and lockin is None:
        lockin = LockIn(
            modulation_freq_Hz=100.0,
            modulation_depth_Hz=500.0,
            demod_lowpass_tc_s=10.0e-3,
        )

    dt_physics = 1.0 / physics_rate_Hz
    n_dec = int(round(duration_s * decimation_rate_Hz))
    n_physics_total = n_dec * decimation_factor

    trace_len = len(disturbance_trace.T_K)
    trace_idx = _build_trace_index_array(
        n_physics_total, trace_len,
        disturbance_trace.sample_rate_Hz, physics_rate_Hz,
    )

    # Pre-allocate outputs
    y_out = np.empty(n_dec, dtype=np.float64)
    ctrl_out = np.empty((n_dec, 2), dtype=np.float64)
    err_out = np.empty(n_dec, dtype=np.float64)

    controller.reset()
    state: Tensor = twin.initial_state(batch_size=1)
    dtype = twin.dtype
    device = twin.device

    # ---------------------------------------------------------------------------
    # Warm up at first disturbance sample
    # ---------------------------------------------------------------------------
    dist_warmup = torch.tensor(
        [[
            disturbance_trace.T_K[0],
            disturbance_trace.B_uT[0],
            disturbance_trace.laser_intensity_norm[0],
        ]],
        dtype=dtype, device=device,
    )
    ctrl_zero = torch.zeros(1, 2, dtype=dtype, device=device)
    with torch.no_grad():
        for _ in range(n_warmup_steps):
            state = twin.step(state, ctrl_zero, dist_warmup, dt_physics)

    # ---------------------------------------------------------------------------
    # Optional: estimate lock-in baseline (for non-direct-ci mode)
    # ---------------------------------------------------------------------------
    lockin_baseline: float = 0.0
    if not use_direct_ci and lockin is not None:
        # Run N_baseline decimation windows to get a stable demod at zero control
        n_baseline = max(50, int(5.0 * lockin.f_mod * decimation_factor / physics_rate_Hz))
        baseline_acc = 0.0
        n_baseline_actual = min(n_baseline, 200)
        sig_buf = np.empty(decimation_factor, dtype=np.float64)
        with torch.no_grad():
            for k in range(n_baseline_actual):
                for j in range(decimation_factor):
                    t_s = k * decimation_factor * dt_physics + j * dt_physics
                    t_t = torch.tensor([t_s], dtype=dtype, device=device)
                    mod = lockin.modulate(t_t).item()
                    c = torch.zeros(1, 2, dtype=dtype, device=device)
                    c[0, 1] = mod
                    state = twin.step(state, c, dist_warmup, dt_physics)
                    sig_buf[j] = twin.photodetector_signal(state).item()
                sw = torch.tensor(sig_buf[np.newaxis, :], dtype=dtype, device=device)
                baseline_acc += lockin.demod(sw, sample_rate_Hz=physics_rate_Hz).item()
        lockin_baseline = baseline_acc / n_baseline_actual

    # ---------------------------------------------------------------------------
    # Main simulation loop
    # ---------------------------------------------------------------------------
    laser_corr: float = 0.0
    rf_corr: float = 0.0

    if not use_direct_ci:
        sig_buf = np.empty(decimation_factor, dtype=np.float64)

    t0 = time.perf_counter()

    with torch.no_grad():
        for k in range(n_dec):
            # ---- Physics sub-steps ----
            ci_sum: float = 0.0
            for j in range(decimation_factor):
                phys_idx = k * decimation_factor + j
                tr_idx = int(trace_idx[phys_idx])
                dist_t = torch.tensor(
                    [[
                        disturbance_trace.T_K[tr_idx],
                        disturbance_trace.B_uT[tr_idx],
                        disturbance_trace.laser_intensity_norm[tr_idx],
                    ]],
                    dtype=dtype, device=device,
                )

                if use_direct_ci:
                    c = torch.zeros(1, 2, dtype=dtype, device=device)
                    c[0, 1] = rf_corr
                    state = twin.step(state, c, dist_t, dt_physics)
                    ci_sum += state[0, 6].item()
                else:
                    t_s = phys_idx * dt_physics
                    t_t = torch.tensor([t_s], dtype=dtype, device=device)
                    mod = lockin.modulate(t_t).item()  # type: ignore[union-attr]
                    c = torch.zeros(1, 2, dtype=dtype, device=device)
                    c[0, 1] = rf_corr + mod
                    state = twin.step(state, c, dist_t, dt_physics)
                    sig_buf[j] = twin.photodetector_signal(state).item()  # type: ignore[name-defined]

            # ---- Error signal ----
            if use_direct_ci:
                # ci > 0 → RF below resonance → need positive correction.
                # Average ci over the decimation window.
                error = ci_sum / decimation_factor
            else:
                sw = torch.tensor(
                    sig_buf[np.newaxis, :], dtype=dtype, device=device  # type: ignore[name-defined]
                )
                raw = lockin.demod(sw, sample_rate_Hz=physics_rate_Hz).item()  # type: ignore[union-attr]
                # Negate: raw increases with positive delta → need negative feedback.
                error = -(raw - lockin_baseline)

            # ---- Controller update ----
            laser_corr, rf_corr = controller.step(error)

            # ---- Record fractional-frequency error (uses actual rf_corr) ----
            tr_now = int(trace_idx[k * decimation_factor])
            B_now = torch.tensor(
                [disturbance_trace.B_uT[tr_now]], dtype=dtype, device=device
            )
            T_now = torch.tensor(
                [disturbance_trace.T_K[tr_now]], dtype=dtype, device=device
            )
            ctrl_now = torch.zeros(1, 2, dtype=dtype, device=device)
            ctrl_now[0, 0] = laser_corr
            ctrl_now[0, 1] = rf_corr
            y_val = twin.fractional_frequency_error_with_B(
                state, ctrl_now, B_now, T_now
            ).item()

            y_out[k] = y_val
            ctrl_out[k, 0] = laser_corr
            ctrl_out[k, 1] = rf_corr
            err_out[k] = error

    wall_s = time.perf_counter() - t0

    return {
        "y": y_out,
        "controls": ctrl_out,
        "error_signal": err_out,
        "wall_s": wall_s,
        "physics_rate_Hz": physics_rate_Hz,
        "decimation_rate_Hz": decimation_rate_Hz,
        "duration_s": duration_s,
    }


# ---------------------------------------------------------------------------
# Open-loop reference runner
# ---------------------------------------------------------------------------


def run_open_loop(
    twin: ReducedTwin,
    disturbance_trace: DisturbanceTrace,
    duration_s: float,
    physics_rate_Hz: float = 10_000.0,
    decimation_rate_Hz: float = 1_000.0,
    n_warmup_steps: int = 50_000,
) -> dict:
    """Run an open-loop simulation (zero controls) for baseline comparison.

    Args:
        twin: ReducedTwin instance.
        disturbance_trace: Pre-generated disturbance time series.
        duration_s: Simulation duration in seconds.
        physics_rate_Hz: Physics integration rate.
        decimation_rate_Hz: Output sample rate.
        n_warmup_steps: Number of physics steps for warm-up.

    Returns:
        Dict with ``"y"``, ``"wall_s"``, ``"physics_rate_Hz"``,
        ``"decimation_rate_Hz"``, ``"duration_s"`` keys.
    """
    decimation_factor = int(round(physics_rate_Hz / decimation_rate_Hz))
    dt_physics = 1.0 / physics_rate_Hz
    n_dec = int(round(duration_s * decimation_rate_Hz))
    n_physics_total = n_dec * decimation_factor

    trace_len = len(disturbance_trace.T_K)
    trace_idx = _build_trace_index_array(
        n_physics_total, trace_len,
        disturbance_trace.sample_rate_Hz, physics_rate_Hz,
    )

    y_out = np.empty(n_dec, dtype=np.float64)
    state: Tensor = twin.initial_state(batch_size=1)
    dtype = twin.dtype
    device = twin.device
    ctrl_zero = torch.zeros(1, 2, dtype=dtype, device=device)

    dist_warmup = torch.tensor(
        [[
            disturbance_trace.T_K[0],
            disturbance_trace.B_uT[0],
            disturbance_trace.laser_intensity_norm[0],
        ]],
        dtype=dtype, device=device,
    )
    with torch.no_grad():
        for _ in range(n_warmup_steps):
            state = twin.step(state, ctrl_zero, dist_warmup, dt_physics)

    t0 = time.perf_counter()

    with torch.no_grad():
        for k in range(n_dec):
            for j in range(decimation_factor):
                phys_idx = k * decimation_factor + j
                tr_idx = int(trace_idx[phys_idx])
                dist_t = torch.tensor(
                    [[
                        disturbance_trace.T_K[tr_idx],
                        disturbance_trace.B_uT[tr_idx],
                        disturbance_trace.laser_intensity_norm[tr_idx],
                    ]],
                    dtype=dtype, device=device,
                )
                state = twin.step(state, ctrl_zero, dist_t, dt_physics)

            tr_now = int(trace_idx[k * decimation_factor])
            B_now = torch.tensor(
                [disturbance_trace.B_uT[tr_now]], dtype=dtype, device=device
            )
            T_now = torch.tensor(
                [disturbance_trace.T_K[tr_now]], dtype=dtype, device=device
            )
            y_out[k] = twin.fractional_frequency_error_with_B(
                state, ctrl_zero, B_now, T_now
            ).item()

    wall_s = time.perf_counter() - t0

    return {
        "y": y_out,
        "wall_s": wall_s,
        "physics_rate_Hz": physics_rate_Hz,
        "decimation_rate_Hz": decimation_rate_Hz,
        "duration_s": duration_s,
    }
