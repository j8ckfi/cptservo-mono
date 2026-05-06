"""Overlapping Allan deviation estimator for fractional-frequency series.

Implements the standard overlapping Allan deviation formula using the
phase-data second-difference approach (IEEE Std 1139-2008; Riley NIST SP 1065).

For a fractional-frequency series y[0..M-1] with basic interval tau0, the
phase sequence (in units of tau0) is:

    phi[0] = 0
    phi[i] = sum_{k=0}^{i-1} y[k]

The overlapping Allan variance at averaging time tau = m*tau0 is:

    sigma_y^2(tau) = 1 / (2 * (M - 2m + 1) * tau^2)
                     * sum_{i=0}^{M-2m} (phi[i+2m] - 2*phi[i+m] + phi[i])^2

This is the frequency-data overlapping estimator (Riley Eq 15), which correctly
gives sigma_y(tau) ~ tau^{-1/2} for white-FM noise.

References
----------
IEEE Std 1139-2008. (2008). *IEEE Standard Definitions of Physical Quantities
    for Fundamental Frequency and Time Metrology - Random Instabilities*.
Riley, W. (2008). *Handbook of Frequency Stability Analysis*. NIST SP 1065.
"""

from __future__ import annotations

import math

import numpy as np

# ---------------------------------------------------------------------------
# Overlapping Allan deviation
# ---------------------------------------------------------------------------


def overlapping_allan(
    y: np.ndarray,
    sample_rate_Hz: float,
    taus_s: list[float],
) -> dict[float, float]:
    """Compute overlapping Allan deviation for requested averaging times.

    Uses the phase-data second-difference overlapping estimator.  For a
    fractional-frequency series y of length M and basic interval
    tau0 = 1/sample_rate_Hz, at averaging factor m = round(tau / tau0):

        sigma_y^2(tau) = 1 / (2 * n_terms * tau^2)
                         * sum_{i=0}^{n_terms-1}
                           (phi[i+2m] - 2*phi[i+m] + phi[i])^2

    where phi[i] = tau0 * sum_{k=0}^{i-1} y[k] and n_terms = M - 2m + 1.

    Args:
        y: (M,) array of fractional-frequency samples.
        sample_rate_Hz: Sample rate of y in Hz (tau0 = 1/sample_rate_Hz).
        taus_s: List of averaging times in seconds to evaluate.

    Returns:
        Dict mapping each requested tau_s to its sigma_y(tau).  Taus that
        require more averaging than the series length are omitted.
    """
    y = np.asarray(y, dtype=np.float64)
    M = len(y)
    tau0 = 1.0 / sample_rate_Hz

    # Build cumulative phase: phi[i] = tau0 * sum(y[0:i]),  length M+1
    phi = np.empty(M + 1, dtype=np.float64)
    phi[0] = 0.0
    np.cumsum(y, out=phi[1:])
    phi[1:] *= tau0

    result: dict[float, float] = {}

    for tau in taus_s:
        m = int(round(tau / tau0))
        if m < 1:
            m = 1
        n_terms = M - 2 * m + 1
        if n_terms < 1:
            # Not enough data for this tau
            continue

        # Second differences of the phase sequence
        # d[i] = phi[i+2m] - 2*phi[i+m] + phi[i],  i = 0..n_terms-1
        d = phi[2 * m : 2 * m + n_terms] - 2.0 * phi[m : m + n_terms] + phi[0:n_terms]

        avar = np.dot(d, d) / (2.0 * n_terms * tau**2)
        result[float(tau)] = math.sqrt(avar)

    return result


# ---------------------------------------------------------------------------
# Theoretical white-FM floor
# ---------------------------------------------------------------------------


def white_fm_floor(
    photon_shot_noise_amp: float,
    sample_rate_Hz: float,
    tau_s: float,
) -> float:
    """Theoretical sigma_y(tau) floor for a white-FM noise process.

    For white frequency modulation (flat S_y(f) = h_0), the Allan deviation is:

        sigma_y(tau) = sqrt(h_0 / (2 * tau))

    The ``photon_shot_noise_amp`` parameter is the one-sided amplitude of the
    white-FM noise in fractional-frequency / sqrt(Hz), so:

        h_0 = photon_shot_noise_amp^2

    and the theoretical floor is:

        sigma_y(tau) = photon_shot_noise_amp / sqrt(2 * tau)

    Args:
        photon_shot_noise_amp: White-FM noise amplitude (fractional-freq /
            sqrt(Hz)).
        sample_rate_Hz: Sample rate (Hz); included for API symmetry but the
            white-FM floor does not depend on sample rate.
        tau_s: Averaging time (s).

    Returns:
        Theoretical sigma_y(tau_s).
    """
    _ = sample_rate_Hz  # unused; kept for API symmetry
    return photon_shot_noise_amp / math.sqrt(2.0 * tau_s)
