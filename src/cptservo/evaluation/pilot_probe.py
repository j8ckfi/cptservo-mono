"""FFT analysis helpers for the M3 pilot-probe gate.

The pilot probe injects a deterministic 1 Hz, 10 Hz-amplitude sinusoid on
``rf_actual`` (in-loop, before ``twin.step``). The four required measurements:

- **Open-loop pilot SNR** in the error signal: power at 1 Hz vs local median
  noise floor. Pass: > 15 dB.
- **Closed-loop / open-loop y pilot amplitude ratio**: < 0.3.
- **Cancellation phase**: ``angle(FFT(rf_cmd)/FFT(pilot_Hz))`` should be 180 ±
  45 deg.
- **Controller sensitivity**: nominal-PI vs weak-PI suppression difference >
  6 dB.

Per Codex GPT-5.5 round 2 brainstorm: bin-aligned 1 Hz on N=100000 samples
(100s × 1 kHz) lands exactly on bin 100, so a rectangular window is exact.
"""

from __future__ import annotations

import numpy as np


def pilot_amplitude(
    x: np.ndarray,
    sample_rate_Hz: float,
    pilot_freq_Hz: float,
) -> float:
    """Single-sided amplitude of x at the pilot frequency, via rfft.

    Args:
        x: Time-domain signal of length N.
        sample_rate_Hz: Sample rate in Hz.
        pilot_freq_Hz: Frequency to extract (assumed bin-aligned).

    Returns:
        Single-sided amplitude in the same units as x.
    """
    x_demean = x - float(np.mean(x))
    X = np.fft.rfft(x_demean)
    bin_idx = int(round(pilot_freq_Hz * len(x) / sample_rate_Hz))
    if bin_idx <= 0 or bin_idx >= len(X):
        return float("nan")
    return 2.0 * float(np.abs(X[bin_idx])) / len(x)


def pilot_snr_db(
    x: np.ndarray,
    sample_rate_Hz: float,
    pilot_freq_Hz: float,
    floor_band_offset: int = 5,
    floor_band_width: int = 45,
) -> float:
    """Pilot SNR in dB, against a local-median noise floor.

    Excludes bins within ``floor_band_offset`` of the pilot bin to avoid
    inflation by leakage skirts. Estimates the floor as the median |X|^2 over
    [-band-offset, -offset] and [+offset, +band] adjacent to the pilot bin.
    """
    x_demean = x - float(np.mean(x))
    X = np.fft.rfft(x_demean)
    bin_idx = int(round(pilot_freq_Hz * len(x) / sample_rate_Hz))
    if bin_idx <= 0 or bin_idx >= len(X):
        return float("nan")

    pilot_power = float(np.abs(X[bin_idx]) ** 2)
    lo = max(0, bin_idx - floor_band_offset - floor_band_width)
    hi = min(len(X), bin_idx + floor_band_offset + floor_band_width + 1)
    mask = np.ones(len(X), dtype=bool)
    mask[max(0, bin_idx - floor_band_offset) : min(len(X), bin_idx + floor_band_offset + 1)] = False
    floor_power_arr = np.abs(X[lo:hi]) ** 2 * mask[lo:hi]
    floor_power_arr = floor_power_arr[floor_power_arr > 0]
    if floor_power_arr.size == 0:
        return float("nan")
    floor_power = float(np.median(floor_power_arr))
    if floor_power <= 0.0 or pilot_power <= 0.0:
        return float("nan")
    return 10.0 * float(np.log10(pilot_power / floor_power))


def cancellation_phase_deg(
    rf_cmd: np.ndarray,
    pilot: np.ndarray,
    sample_rate_Hz: float,
    pilot_freq_Hz: float,
) -> float:
    """Phase of rf_cmd relative to the pilot at the pilot frequency.

    For a well-behaved closed loop, rf_cmd should be ~180 deg out of phase
    with the pilot (controller commands the opposite of the disturbance to
    cancel it). A phase of 180 ± 45 deg is the gate threshold.

    Returns:
        Phase in degrees, wrapped to (-180, 180].
    """
    rfc = rf_cmd - float(np.mean(rf_cmd))
    plt = pilot - float(np.mean(pilot))
    Xc = np.fft.rfft(rfc)
    Xp = np.fft.rfft(plt)
    bin_idx = int(round(pilot_freq_Hz * len(rf_cmd) / sample_rate_Hz))
    if bin_idx <= 0 or bin_idx >= len(Xc):
        return float("nan")
    if abs(Xp[bin_idx]) == 0.0:
        return float("nan")
    ratio = Xc[bin_idx] / Xp[bin_idx]
    deg = float(np.degrees(np.angle(ratio)))
    # Wrap to (-180, 180]
    while deg <= -180.0:
        deg += 360.0
    while deg > 180.0:
        deg -= 360.0
    return deg


def overlapping_allan_slope(
    sigma_y_table: dict[float, float],
    taus: list[float],
) -> float:
    """Log-log slope of sigma_y(tau) over the given tau set.

    Pass for white-FM scaling: slope in [-0.7, -0.3] (ideal -0.5).

    Args:
        sigma_y_table: Mapping tau -> sigma_y.
        taus: List of tau values to fit (must all be present in table).

    Returns:
        Slope of log10(sigma_y) vs log10(tau).
    """
    log_tau = np.log10([float(t) for t in taus])
    log_sig = np.log10([float(sigma_y_table[t]) for t in taus])
    slope, _ = np.polyfit(log_tau, log_sig, 1)
    return float(slope)
