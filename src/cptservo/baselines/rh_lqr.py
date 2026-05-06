"""Receding-Horizon LQR (steady-state DLQR) controller for the CPT-clock lock loop.

This module implements a 2-state discrete-time LQR controller as a drop-in
replacement for ``PIController``.  In the linear unconstrained case the
receding-horizon MPC solution collapses to the infinite-horizon LQR gain, so
"receding-horizon" here refers to the conceptual MPC framing — not iterative
online optimisation.  An iterative MPC solver with hard rf_limit_Hz constraints
is documented as future work.

Plant model (linearised around the CPT lock point)
---------------------------------------------------
State:  x = [ci_filtered, integral_ci]   (2×1)
Input:  u = rf_correction_Hz             (scalar)

Continuous-time dynamics::

    d(ci)/dt          = -(1/T_2) * ci + plant_gain * u
    d(integral_ci)/dt = ci

Discretised via zero-order hold (ZOH) at control_dt_s.

LQR objective
-------------
Minimise the infinite-horizon quadratic cost::

    J = sum_{k=0}^{inf} ( x_k' Q x_k  +  u_k' R u_k )

where Q = diag(q1, q2) and R are scalar.

Default seed (Q, R selection rationale)
----------------------------------------
Q[0,0] = q1 = 1e6
    Penalises lock-point error in ci directly.  The discriminator slope is
    ~2.4e-5 ci/Hz (M2 OBE calibration), so a 1 Hz residual error gives
    ci ~ 2.4e-5 and a cost contribution q1 * ci^2 = 1e6 * (2.4e-5)^2 ~ 0.6.
    Scaling to ~1 per Hz^2 of RF error is the right order; q1=1e6 achieves this.

Q[1,1] = q2 = 1e8
    Penalises accumulated drift (integral of ci).  A larger weight forces the
    LQR to track slow ramps aggressively — analogous to the integrator term in
    PI but optimal in the H2 sense.  q2 = 1e8 >> q1 biases the LQR toward
    eliminating steady-state offset faster, which is the primary benefit over
    PI on the ``thermal_ramp`` scenario.

R = 1.0
    Control effort penalty in Hz^2.  Keeping R = O(1) while Q[0,0] = 1e6
    makes the LQR drive the lock-point error hard; the rf_limit_Hz anti-windup
    provides the practical saturation guard.

Anti-windup
-----------
When ``rf_correction`` hits ±rf_limit_Hz the integrator state is back-calculated
to the value consistent with the saturated output, mirroring the clamped-integral
approach in ``PIController`` (see ``pi.py:135-160``).

References
----------
Anderson, B. D. O. & Moore, J. B. (1990). *Optimal Control: Linear Quadratic
    Methods*. Prentice Hall. (Discrete LQR via DARE, Chapter 4.)
Knappe, S. (2004). *MEMS atomic clocks*. Applied Physics Letters.
Kitching, J. (2018). Chip-scale atomic devices. *Applied Physics Reviews*, 5.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import ClassVar

import numpy as np
import yaml
from scipy.linalg import solve_discrete_are

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

# Q tuned to match PI's bandwidth at 1 Hz (proportional gain ~6.5e3 for pilot
# suppression) while preserving the integral-aggressive bias that gives LQR
# its thermal_ramp advantage. Bumped from initial seed (1e6, 1e8) which gave
# K=[10, 10000] — too narrow bandwidth, failed pilot probe gate.
_DEFAULT_Q: tuple[float, float] = (1.0e12, 1.0e10)
_DEFAULT_R: float = 1.0
_DEFAULT_DT: float = 0.001          # 1 kHz control rate (matches PI default)
_DEFAULT_RF_LIMIT: float = 1000.0   # Hz (from v1_recipe.yaml actuator_bounds)
_DEFAULT_PLANT_GAIN: float = 2.4e-5  # ci per Hz of RF detuning (M2 OBE slope)
_DEFAULT_PLANT_TAU: float = 1.0e-3  # T_2 coherence relaxation time (s)

_RECIPE_PATH = Path(__file__).parent.parent.parent.parent / "configs" / "v1_recipe.yaml"


def _build_discrete_plant(
    plant_gain: float,
    plant_tau_s: float,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (Ad, Bd) discrete-time ZOH matrices for the 2-state plant.

    Continuous-time system::

        A_c = [[-1/T2,  0],
               [  1,    0]]
        B_c = [[plant_gain],
               [0         ]]

    ZOH discretisation at step dt:

        Ad = expm(A_c * dt)
        Bd = integral_0^dt expm(A_c * s) ds * B_c

    For the scalar decay A_c[0,0] = -a (a = 1/T2)::

        expm(A_c*dt) = [[exp(-a*dt),          0],
                        [(1-exp(-a*dt))/a,     1]]

        Bd[0] = plant_gain * (1 - exp(-a*dt)) / a
        Bd[1] = plant_gain * (dt - (1-exp(-a*dt))/a)

    Args:
        plant_gain: Discriminator slope (ci per Hz of RF detuning).
        plant_tau_s: CPT coherence relaxation time T_2 (s).
        dt: Control sample period (s).

    Returns:
        Ad: (2, 2) discrete-time state-transition matrix.
        Bd: (2, 1) discrete-time input matrix.
    """
    a = 1.0 / plant_tau_s
    e = float(np.exp(-a * dt))

    ad = np.array(
        [
            [e, 0.0],
            [(1.0 - e) / a, 1.0],
        ],
        dtype=np.float64,
    )

    # ZOH input: integral_0^dt exp(-a*s) ds * plant_gain
    bd_ci = plant_gain * (1.0 - e) / a
    bd_int = plant_gain * (dt - (1.0 - e) / a)
    bd = np.array([[bd_ci], [bd_int]], dtype=np.float64)

    return ad, bd


def _solve_lqr(
    ad: np.ndarray,
    bd: np.ndarray,
    q: np.ndarray,
    r: np.ndarray,
) -> np.ndarray:
    """Solve the discrete-time algebraic Riccati equation and return gain K.

    Solves::

        P = Ad' P Ad - (Ad' P Bd)(Bd' P Bd + R)^{-1}(Bd' P Ad) + Q

    and returns::

        K = (Bd' P Bd + R)^{-1} (Bd' P Ad)   [shape (1, 2)]

    Args:
        ad: (n, n) discrete-time state matrix.
        bd: (n, m) discrete-time input matrix.
        q: (n, n) positive semi-definite state cost matrix.
        r: (m, m) positive definite control cost matrix.

    Returns:
        K: (m, n) optimal LQR gain such that u = -K @ x.
    """
    p = solve_discrete_are(ad, bd, q, r)
    bpb_r = bd.T @ p @ bd + r
    k = np.linalg.solve(bpb_r, bd.T @ p @ ad)  # (m, n)
    return k


class RHLQRController:
    """Steady-state discrete LQR (receding-horizon framing) for the CPT RF loop.

    The 2-state linearised plant (ci_filtered, integral_ci) is discretised via
    ZOH at ``control_dt_s``.  The DARE is solved once at construction to yield
    the steady-state gain K.  Each call to ``step()`` applies ``u = -K @ x``
    and enforces the rf_limit_Hz constraint with clamped-integral anti-windup.

    This controller is a drop-in replacement for ``PIController`` in
    ``run_fast_loop``.

    Note:
        "Receding horizon" refers to the conceptual equivalence between
        infinite-horizon LQR and MPC in the linear unconstrained case.  This
        implementation uses the steady-state (time-invariant) DLQR gain —
        not iterative online MPC.  Iterative MPC for proper constraint handling
        is future work.

    Args:
        Q: 2-tuple ``(q_ci, q_integral)`` — diagonal entries of the (2×2)
            state-cost matrix.  See module docstring for selection rationale.
        R: Scalar control-effort weight (rf_correction Hz²).
        control_dt_s: Control sample period in seconds.
        rf_limit_Hz: Symmetric saturation limit on the RF correction.
            ``None`` disables clamping.
        plant_gain: Discriminator slope — ci per Hz of RF detuning (M2
            OBE calibration: ~2.4e-5 ci/Hz).
        plant_tau_s: CPT coherence relaxation time T_2 (s); default 1 ms.
    """

    # Documented defaults for reference
    _DEFAULT_Q: ClassVar[tuple[float, float]] = _DEFAULT_Q
    _DEFAULT_R: ClassVar[float] = _DEFAULT_R

    def __init__(
        self,
        Q: tuple[float, float] = _DEFAULT_Q,
        R: float = _DEFAULT_R,
        control_dt_s: float = _DEFAULT_DT,
        rf_limit_Hz: float | None = _DEFAULT_RF_LIMIT,
        plant_gain: float = _DEFAULT_PLANT_GAIN,
        plant_tau_s: float = _DEFAULT_PLANT_TAU,
    ) -> None:
        self.Q = Q
        self.R = R
        self.control_dt_s = control_dt_s
        self.rf_limit_Hz = rf_limit_Hz
        self.plant_gain = plant_gain
        self.plant_tau_s = plant_tau_s

        # Build discrete-time plant matrices
        self._Ad, self._Bd = _build_discrete_plant(plant_gain, plant_tau_s, control_dt_s)

        # Solve DARE for steady-state LQR gain
        q_mat = np.diag([float(Q[0]), float(Q[1])])
        r_mat = np.array([[float(R)]])
        self._K: np.ndarray = _solve_lqr(self._Ad, self._Bd, q_mat, r_mat)  # (1, 2)

        # Controller state: [ci_filtered, integral_ci]
        self._x: np.ndarray = np.zeros(2, dtype=np.float64)

    # -----------------------------------------------------------------------
    # State management
    # -----------------------------------------------------------------------

    def reset(self) -> None:
        """Reset state vector to zero.

        Call before each new simulation episode to prevent state from bleeding
        across runs.
        """
        self._x[:] = 0.0

    # -----------------------------------------------------------------------
    # Control step
    # -----------------------------------------------------------------------

    def step(self, error: float) -> tuple[float, float]:
        """Advance the LQR controller by one step.

        The error signal is used directly as the observed ``ci_filtered`` state;
        the ``integral_ci`` state is maintained internally.

        Sign convention matches ``PIController``: a positive error (CPT
        resonance above RF frequency) produces a positive RF correction.

        Args:
            error: Dimensionless lock-in error signal (= ci, positive means RF
                below resonance).

        Returns:
            Tuple ``(laser_detuning_correction_Hz, rf_detuning_correction_Hz)``.
            ``laser_detuning_correction_Hz`` is always 0.0 in v1.
        """
        # Update observable state from measurement.
        self._x[0] = float(error)

        # Propagate the integrator state FIRST. x[1] += ci * dt.
        # (Doing this before computing raw so anti-windup back-calc applies
        # to the post-propagation state — otherwise the next-step propagation
        # adds an extra ci*dt on top of the back-calculated value.)
        self._x[1] += self._x[0] * self.control_dt_s

        # LQR control law: regulator u* = -K x. Sign convention matches
        # PIController (positive error → positive RF correction).
        k0 = float(self._K[0, 0])
        k1 = float(self._K[0, 1])
        raw = k0 * self._x[0] + k1 * self._x[1]

        # Clamped-integral anti-windup (mirrors pi.py).
        # When the output saturates, back-calculate the integrator state so it
        # doesn't wind up beyond what the actuator can deliver.
        if self.rf_limit_Hz is not None:
            hi = self.rf_limit_Hz
            if raw > hi:
                raw = hi
                if abs(k1) > 1e-30:
                    self._x[1] = (raw - k0 * self._x[0]) / k1
            elif raw < -hi:
                raw = -hi
                if abs(k1) > 1e-30:
                    self._x[1] = (raw - k0 * self._x[0]) / k1

        rf_correction = raw
        laser_correction = 0.0
        return (laser_correction, rf_correction)

    # -----------------------------------------------------------------------
    # Properties
    # -----------------------------------------------------------------------

    @property
    def K(self) -> np.ndarray:
        """LQR gain matrix, shape (1, 2).  ``u = -K @ x``."""
        return self._K

    # -----------------------------------------------------------------------
    # Factory
    # -----------------------------------------------------------------------

    @classmethod
    def from_recipe(cls, recipe_path: str | Path | None = None) -> RHLQRController:
        """Load RH-LQR parameters from ``configs/v1_recipe.yaml`` if present.

        The recipe YAML may contain an optional ``rh_lqr`` block::

            rh_lqr:
              Q: [1.0e6, 1.0e8]
              R: 1.0
              control_dt_s: 0.001

        If the block is absent the documented defaults are used.

        Args:
            recipe_path: Optional path override.  Defaults to
                ``configs/v1_recipe.yaml`` relative to the package root.

        Returns:
            RHLQRController with parameters from recipe or defaults.
        """
        rp = Path(recipe_path) if recipe_path is not None else _RECIPE_PATH
        if rp.exists():
            with open(rp, encoding="utf-8-sig") as fh:
                recipe = yaml.safe_load(fh)
            params = recipe.get("rh_lqr", {})
        else:
            params = {}

        raw_q = params.get("Q", list(_DEFAULT_Q))
        return cls(
            Q=(float(raw_q[0]), float(raw_q[1])),
            R=float(params.get("R", _DEFAULT_R)),
            control_dt_s=float(params.get("control_dt_s", _DEFAULT_DT)),
        )

    @classmethod
    def from_calibration(cls, calibration_path: str | Path) -> RHLQRController:
        """Load parameters stored in a gate JSON (for reproducibility).

        Args:
            calibration_path: Path to a JSON file with ``Q_diag``, ``R``, and
                optionally ``control_dt_s`` keys.

        Returns:
            RHLQRController initialised from the stored parameters.
        """
        with open(calibration_path, encoding="utf-8-sig") as fh:
            data = json.load(fh)
        q_diag = data["Q_diag"]
        return cls(
            Q=(float(q_diag[0]), float(q_diag[1])),
            R=float(data["R"]),
            control_dt_s=float(data.get("control_dt_s", _DEFAULT_DT)),
        )
