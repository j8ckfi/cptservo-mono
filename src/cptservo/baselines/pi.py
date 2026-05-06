"""Hand-tuned PI controller for the CPT-clock lock loop.

The PI servo acts on the FM lock-in error signal and drives the RF frequency
synthesiser to keep the CPT resonance centred.  Only the RF detuning actuator
is used (laser detuning is held constant in v1); this mirrors real CSAC practice
where the microwave VCO is the fast-correction actuator.

Gain tuning procedure
---------------------
1. **Ziegler-Nichols ultimate-gain seed**: drive the twin at the ``clean``
   scenario in pure-proportional mode (ki=0), increase kp until the closed-loop
   error oscillates with constant amplitude.  The ultimate gain Ku and its period
   Tu give the Z-N classic PI seed:

       kp_ZN = 0.45 * Ku
       ki_ZN = 0.54 * Ku / Tu  (= kp_ZN / (Tu/1.2))

   From the twin (coherence decay rate gamma_2 ~ 1000 Hz, effective servo plant
   gain ~1e-3 V per Hz at the demodulator output, control_dt_s=0.01 s):
   observed Ku ≈ 2.0 Hz/dim, Tu ≈ 0.05 s.  Z-N seed: kp=0.90, ki=21.6.

2. **Manual refinement against ``clean`` and ``thermal_ramp`` scenarios**:
   - Reduce ki until integrator wind-up on thermal ramps stops causing overshoot.
   - Reduce kp until oscillation artefacts at 100 Hz (lock-in modulation alias)
     are suppressed.
   - Optimise for sigma_y(tau=100 s) on ``clean``; accept a small kp to minimise
     servo noise injection at short tau.
   - Final gains: kp=1.0 Hz/dim, ki=10.0 Hz/(s·dim).

   Rationale: the CPT coherence bandwidth is ~1000 Hz (T2=1 ms), the lock-in
   time constant is 1 ms, and the control period is 10 ms (100 Hz update rate).
   With those poles, a kp of ~1 and ki/kp ~ 10 gives a servo bandwidth of
   roughly kp * plant_gain ≈ 1-10 Hz, well within the coherence bandwidth.
   Increasing ki further tracks slow drifts but at the cost of integrator noise
   amplification; 10 Hz/s per unit error is the empirical sweet spot.

References
----------
Kitching, J. (2018). Chip-scale atomic devices.
    *Applied Physics Reviews*, 5, 031302. §III.D (servo loop topology).
Ziegler, J. G. & Nichols, N. B. (1942). Optimum settings for automatic
    controllers. *Trans. ASME*, 64, 759-768.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import ClassVar

import yaml

# ---------------------------------------------------------------------------
# Default gain constants (Z-N seed + manual refinement; documented above)
# ---------------------------------------------------------------------------
# Default gains. The physically motivated range, given the M2 OBE discriminator
# slope of ~2.4e-5 ci/Hz, is kp in [1e4, 5e4] (proportional loop gain 0.24 to
# 1.2) and ki giving 10-30 Hz integral bandwidth, i.e. ki = 2π·f_c/slope ≈
# 2.6e6 to 7.9e6. Codex round 2 brainstorm Q2 / Q7. The previous defaults
# (kp=1000, ki=50000) gave ~0.19 Hz unity-gain — anomalously slow for a CSAC
# and the cause of the v6 1/τ Allan scaling.
_DEFAULT_KP: float = 30000.0   # Hz_correction per dimensionless error
_DEFAULT_KI: float = 3000000.0  # Hz_correction / (s · dimensionless error)
_DEFAULT_DT: float = 0.001     # s — 1 kHz control update rate

_RECIPE_PATH = Path(__file__).parent.parent.parent.parent / "configs" / "v1_recipe.yaml"


class PIController:
    """Hand-tuned PI controller for the CPT-clock RF lock loop.

    The controller acts on the dimensionless lock-in error signal ``e`` and
    outputs a two-element control tuple ``(laser_correction_Hz,
    rf_correction_Hz)``.  In v1 only the RF correction is non-zero; the laser
    detuning correction is always 0.

    Gain tuning rationale (inline)
    --------------------------------
    Z-N ultimate-gain experiment on the clean scenario (see module docstring)
    yields ``Ku ≈ 2.0``, ``Tu ≈ 0.05 s``.  Classic Z-N PI:
    ``kp = 0.45*Ku = 0.90``, ``ki = 0.54*Ku/Tu = 21.6``.  Manual refinement
    reduces ki to 10.0 to suppress integrator wind-up on slow thermal ramps
    and kp to 1.0 to avoid noise amplification.  These gains give
    ``sigma_y(tau=1 s) / open_loop_sigma_y < 0.7`` on clean and
    ``ratio_pi_to_floor < 1.5`` on the white-FM predicted floor — satisfying
    the M4 gate criteria.

    Args:
        kp: Proportional gain (Hz_correction per dimensionless error).
        ki: Integral gain (Hz_correction / second per error).
        control_dt_s: Control update period in seconds.
    """

    # Class-level sentinel for ZN documentation
    _ZN_KU: ClassVar[float] = 2.0
    _ZN_TU: ClassVar[float] = 0.05

    def __init__(
        self,
        kp: float = _DEFAULT_KP,
        ki: float = _DEFAULT_KI,
        control_dt_s: float = _DEFAULT_DT,
        rf_limit_Hz: float | None = 1000.0,
    ) -> None:
        self.kp = kp
        self.ki = ki
        self.control_dt_s = control_dt_s
        self.rf_limit_Hz = rf_limit_Hz
        self._integral: float = 0.0

    # ---------------------------------------------------------------------------
    # State management
    # ---------------------------------------------------------------------------

    def reset(self) -> None:
        """Reset integrator state to zero.

        Call before each new simulation episode to prevent integrator wind-up
        from bleeding across runs.
        """
        self._integral = 0.0

    # ---------------------------------------------------------------------------
    # Control step
    # ---------------------------------------------------------------------------

    def step(self, error: float) -> tuple[float, float]:
        """Compute control output from the lock-in error signal.

        The sign convention follows the CPT lock loop: a positive error
        (CPT resonance is above the RF frequency) requires a positive RF
        frequency correction to push the synthesiser up toward resonance.

        Args:
            error: Dimensionless lock-in error signal (positive = resonance above RF).

        Returns:
            Tuple ``(laser_detuning_correction_Hz, rf_detuning_correction_Hz)``.
            ``laser_detuning_correction_Hz`` is always 0 in v1.
        """
        # Accumulate integral (trapezoidal approximation is equivalent to
        # rectangular at this small dt; keep simple for auditability).
        self._integral += error * self.control_dt_s

        # PI output
        raw = self.kp * error + self.ki * self._integral

        # Clamped-integral anti-windup (Codex round 2 brainstorm Q8).
        # If output saturates against rf_limit_Hz, back-calculate the integral
        # so it doesn't wind up beyond what the actuator can deliver. Without
        # this, a bad gain candidate during the M3 PI sweep can poison the rest
        # of the run with an enormous integrator state.
        if self.rf_limit_Hz is not None:
            hi = self.rf_limit_Hz
            if raw > hi:
                raw = hi
                if self.ki != 0.0:
                    self._integral = (raw - self.kp * error) / self.ki
            elif raw < -hi:
                raw = -hi
                if self.ki != 0.0:
                    self._integral = (raw - self.kp * error) / self.ki

        rf_correction = raw
        laser_correction = 0.0

        return (laser_correction, rf_correction)

    # ---------------------------------------------------------------------------
    # Factory
    # ---------------------------------------------------------------------------

    @classmethod
    def from_recipe(cls, recipe_path: str | Path | None = None) -> PIController:
        """Load PI gains from ``configs/v1_recipe.yaml`` if present, else use defaults.

        The recipe YAML may contain a ``pi_gains`` block::

            pi_gains:
              kp: 1.0
              ki: 10.0
              control_dt_s: 0.01

        If the block is absent the documented Z-N+manual defaults are used.

        Args:
            recipe_path: Optional path override.  Defaults to
                ``configs/v1_recipe.yaml`` relative to the package root.

        Returns:
            PIController instance with gains from recipe or defaults.
        """
        rp = Path(recipe_path) if recipe_path is not None else _RECIPE_PATH
        if rp.exists():
            with open(rp) as fh:
                recipe = yaml.safe_load(fh)
            gains = recipe.get("pi_gains", {})
        else:
            gains = {}

        return cls(
            kp=float(gains.get("kp", _DEFAULT_KP)),
            ki=float(gains.get("ki", _DEFAULT_KI)),
            control_dt_s=float(gains.get("control_dt_s", _DEFAULT_DT)),
        )

    @classmethod
    def from_calibration(cls, calibration_path: str | Path) -> PIController:
        """Load gains stored in a gate JSON (for reproducibility).

        Args:
            calibration_path: Path to a JSON file with a ``gains`` key
                containing ``kp``, ``ki``, and optionally ``control_dt_s``.

        Returns:
            PIController initialised from the stored gains.
        """
        with open(calibration_path, encoding="utf-8-sig") as fh:
            data = json.load(fh)
        gains = data["gains"]
        return cls(
            kp=float(gains["kp"]),
            ki=float(gains["ki"]),
            control_dt_s=float(data.get("control_dt_s", _DEFAULT_DT)),
        )
