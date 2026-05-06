"""FM modulation and lock-in demodulation for CPT clock readout.

Implements the standard FM lock-in technique used in chip-scale CPT clocks: the
RF (or laser) frequency is sinusoidally modulated at f_mod; the photodetector
signal is then demodulated at f_mod to produce an error signal proportional to
the slope of the CPT resonance.  At resonance the error is zero; off-resonance
the sign gives the direction of the correction.

References
----------
Kitching, J. (2018). Chip-scale atomic devices.
    *Applied Physics Reviews*, 5, 031302.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor


class LockIn:
    """FM modulation and lock-in demodulator.

    Parameters configure the modulation frequency, depth, and the lowpass
    time-constant of the single-pole IIR demodulator.

    Typical usage::

        lockin = LockIn(modulation_freq_Hz=100.0,
                        modulation_depth_Hz=1000.0,
                        demod_lowpass_tc_s=1e-3)
        t = torch.linspace(0, 0.01, 100)
        mod = lockin.modulate(t)          # frequency offsets
        err = lockin.demod(signal, 10000.0)  # error signal from 100-sample window
    """

    # ---------------------------------------------------------------------------

    def __init__(
        self,
        modulation_freq_Hz: float = 100.0,
        modulation_depth_Hz: float = 1000.0,
        demod_lowpass_tc_s: float = 1.0e-3,
    ) -> None:
        """Initialise LockIn.

        Args:
            modulation_freq_Hz: Modulation (dither) frequency in Hz.
            modulation_depth_Hz: Peak frequency deviation of the FM dither (Hz).
            demod_lowpass_tc_s: Time constant of the single-pole IIR lowpass
                filter applied after mixing (seconds).
        """
        self.f_mod = modulation_freq_Hz
        self.depth = modulation_depth_Hz
        self.tc = demod_lowpass_tc_s

    # ---------------------------------------------------------------------------

    def modulate(self, t: Tensor) -> Tensor:
        """Return instantaneous FM modulation offset at times t.

        Args:
            t: Time array (any shape), in seconds.

        Returns:
            Tensor of same shape as t; values in Hz.
        """
        return self.depth * torch.sin(2.0 * math.pi * self.f_mod * t)

    # ---------------------------------------------------------------------------

    def demod(
        self,
        signal_window: Tensor,
        sample_rate_Hz: float,
        carrier_phase_offset: float = 0.0,
    ) -> Tensor:
        """Demodulate a photodetector signal window.

        Multiplies by the reference sin(2*pi*f_mod*t + phase) then applies a
        single-pole IIR lowpass filter with time-constant self.tc.  Returns the
        final (steady-state) in-phase amplitude, which is proportional to the
        lock-loop error signal.

        Args:
            signal_window: (B, N) real-valued photodetector time series.  N is
                the number of samples in the window.
            sample_rate_Hz: Sample rate of the signal in Hz.
            carrier_phase_offset: Phase offset of the demodulation carrier
                (radians).  Use 0.0 for standard in-phase demod.

        Returns:
            (B,) demodulated error signal (in-phase amplitude after lowpass).
        """
        B, N = signal_window.shape
        dt = 1.0 / sample_rate_Hz

        # Build time vector for this window (relative to start of window = t=0)
        t = torch.arange(N, dtype=signal_window.dtype, device=signal_window.device) * dt

        # Reference carrier: sin(2*pi*f_mod*t + phase)
        carrier = torch.sin(
            2.0 * math.pi * self.f_mod * t + carrier_phase_offset
        )  # (N,)

        # Multiply signal by carrier — same for every batch element
        mixed = signal_window * carrier.unsqueeze(0)  # (B, N)

        # Single-pole IIR lowpass: alpha = dt / (tc + dt)
        alpha = dt / (self.tc + dt)

        # Run IIR filter across the N dimension
        # y[n] = alpha * x[n] + (1-alpha) * y[n-1]
        # We iterate; N is typically small (100–1000 samples).
        y = torch.zeros(B, dtype=signal_window.dtype, device=signal_window.device)
        for n in range(N):
            y = alpha * mixed[:, n] + (1.0 - alpha) * y

        return y
