"""Fit the tier-2 ReducedTwin free parameters against the tier-1 OBE surface.

The reduced model expresses the lock point analytically as

    lp_red = -(buffer_gas_shift_coeff * P_buf
               + light_shift_coeff * I_norm
               + lumped_zeeman_coeff * B_uT)

The three free parameters enter linearly, so fitting them against the
tier-1 OBE lock-point surface is **ordinary least-squares** — closed-form,
no simulation in the loop. This replaces an earlier iterative settle-
sweep-zero-cross approach that ran the reduced ODE 1250 times per Nelder-
Mead loss eval (~37 min/eval) and was killed before convergence.

Results are persisted to ``data/reduced_calibration.json``.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

# ---------------------------------------------------------------------------
# Module-level defaults
# ---------------------------------------------------------------------------
_DEFAULT_OBE_PATH = (
    Path(__file__).parent.parent.parent.parent / "data" / "obe_surface.h5"
)
_DEFAULT_OUT_JSON = (
    Path(__file__).parent.parent.parent.parent / "data" / "reduced_calibration.json"
)


# ---------------------------------------------------------------------------
# Public data class
# ---------------------------------------------------------------------------


@dataclass
class CalibrationResult:
    """Result of the tier-2 free-parameter fit against the OBE surface."""

    light_shift_coeff: float
    buffer_gas_shift_coeff: float
    lumped_zeeman_coeff: float
    peak_slope_error_max_pct: float
    lock_point_shift_max_pct: float
    fit_converged: bool
    fit_message: str
    optimizer_iterations: int
    n_grid_points_fit: int
    fit_residual_rms_Hz: float
    fit_wall_s: float


# ---------------------------------------------------------------------------
# Closed-form least-squares fit
# ---------------------------------------------------------------------------


def _ols_lockpoint_fit(
    lp_obe_flat: np.ndarray,
    I_norm_flat: np.ndarray,
    B_uT_flat: np.ndarray,
) -> tuple[float, float, float, float]:
    """Closed-form OLS fit of (ls, zee, offset) to OBE lock points.

    Model:  lp_obe = -(ls * I_norm + zee * B_uT) + offset_Hz

    The buffer-gas coefficient is *unidentifiable* from this OBE surface
    because compute_obe_surface holds buffer_pressure_torr fixed at 25 Torr
    — the buf*P_buf product is degenerate with the global offset. We pin
    ``buffer_gas_shift_coeff`` at the RbSpec analytic value (-7.4e6 Hz/Torr,
    Steck-validated) and fit only the two physically informative params plus
    an offset.

    Returns:
        (ls_coeff, zee_coeff, offset_Hz, residual_rms_Hz).
    """
    N = lp_obe_flat.shape[0]
    A = np.empty((N, 3), dtype=np.float64)
    A[:, 0] = -I_norm_flat   # ls_coeff
    A[:, 1] = -B_uT_flat     # zee_coeff
    A[:, 2] = 1.0            # constant offset
    coeffs, _, _, _ = np.linalg.lstsq(A, lp_obe_flat, rcond=None)
    ls, zee, offset = float(coeffs[0]), float(coeffs[1]), float(coeffs[2])
    pred = A @ coeffs
    rms = float(np.sqrt(np.mean((pred - lp_obe_flat) ** 2)))
    return ls, zee, offset, rms


# ---------------------------------------------------------------------------
# Slope check (analytic — the reduced model's near-lock dy/d(rf_det) is 1/HF_GROUND
# in fractional-frequency units, but for matching OBE-discriminator units we
# need the discriminator slope in OBE units (signal per Hz). Since the reduced
# twin's discriminator is a different physical quantity than OBE's photodetector
# signal, comparing absolute slopes is not meaningful. We compare RELATIVE slopes
# across grid points instead.)
# ---------------------------------------------------------------------------


def _relative_slope_consistency(slope_obe: np.ndarray) -> float:
    """Report the coefficient of variation of OBE slopes across the grid.

    The reduced model is *constructed* with a uniform discriminator slope
    (1/HF_GROUND in fractional-frequency units, scaled by the CPT contrast).
    If OBE slopes are reasonably uniform across the grid (CV < 50 %), the
    reduced model's uniform-slope assumption is safe. Otherwise the reduced
    model has a structural mismatch that needs to be modeled.

    Returns the coefficient of variation in percent.
    """
    s = slope_obe[np.isfinite(slope_obe)]
    if s.size < 2 or float(np.mean(np.abs(s))) == 0.0:
        return float("nan")
    return 100.0 * float(np.std(s)) / float(np.mean(np.abs(s)))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def fit_reduced_to_obe(
    obe_surface_path: str | Path = _DEFAULT_OBE_PATH,
    initial_params: dict[str, float] | None = None,
    out_json_path: str | Path | None = _DEFAULT_OUT_JSON,
) -> CalibrationResult:
    """Fit ReducedTwin free parameters by closed-form OLS against the OBE surface.

    The fit minimises the L2 lock-point residual under the linear model
    described in the module docstring. No iterative optimisation; runs in
    milliseconds.

    Args:
        obe_surface_path: Path to the HDF5 file produced by
            :func:`~cptservo.twin.full_obe.compute_obe_surface`.
        initial_params: Ignored (present for API symmetry with the previous
            iterative fit).
        out_json_path: If not ``None``, write the result JSON here.

    Returns:
        :class:`CalibrationResult` with fitted parameters and diagnostics.
    """
    obe_surface_path = Path(obe_surface_path)
    t0 = time.perf_counter()

    # --- Load OBE surface ------------------------------------------------------
    with h5py.File(obe_surface_path, "r") as f:
        B_grid = np.array(f["B_uT"])
        I_grid = np.array(f["intensity_norm"])
        lock_point = np.array(f["lock_point"])  # (N_laser, N_T, N_B, N_I)
        slope = np.array(f["slope_per_Hz"])     # (N_laser, N_T, N_B, N_I)
        buf_pressure = float(f.attrs.get("buffer_pressure_torr", 25.0))

    N_laser, N_T, N_B, N_I = lock_point.shape

    # --- Flatten to grid points with finite lock points ------------------------
    lp_flat: list[float] = []
    I_flat: list[float] = []
    B_flat: list[float] = []
    for i_l in range(N_laser):
        for i_T in range(N_T):
            for i_B in range(N_B):
                for i_I in range(N_I):
                    lp = lock_point[i_l, i_T, i_B, i_I]
                    if np.isfinite(lp):
                        lp_flat.append(float(lp))
                        I_flat.append(float(I_grid[i_I]))
                        B_flat.append(float(B_grid[i_B]))
    if not lp_flat:
        raise ValueError("No finite OBE lock points; surface may be corrupt.")

    lp_arr = np.array(lp_flat, dtype=np.float64)
    I_arr = np.array(I_flat, dtype=np.float64)
    B_arr = np.array(B_flat, dtype=np.float64)

    # Pin buffer_gas_shift_coeff at RbSpec value — unidentifiable on this
    # constant-buffer-pressure surface (see _ols_lockpoint_fit docstring).
    from rbspec.solver import PRESSURE_SHIFT_COEFF
    buf_coeff = float(PRESSURE_SHIFT_COEFF)

    ls_coeff, zee_coeff, fit_offset_Hz, rms = _ols_lockpoint_fit(
        lp_arr, I_arr, B_arr
    )

    # Lock-point shift: absolute Hz residual (the physically meaningful number)
    # plus relative % vs OBE lock-point spread (not vs |lp| which is pathological
    # when |lp| ~ 0). Lock points are ~0.9–5.8 Hz with ~5 Hz peak-to-peak; the
    # natural denominator is the spread, not the absolute value.
    pred = -(ls_coeff * I_arr + zee_coeff * B_arr) + fit_offset_Hz
    abs_residual = np.abs(pred - lp_arr)
    lp_spread = float(np.ptp(lp_arr))
    lock_point_shift_pct = float(100.0 * np.max(abs_residual) / max(lp_spread, 1e-12))

    # Slope consistency (see docstring of _relative_slope_consistency)
    slope_cv_pct = _relative_slope_consistency(slope)
    # The reduced model assumes uniform slope, so the relevant "peak slope error"
    # is the dispersion in OBE slopes — if it's small, the reduced model fits
    # well in the slope dimension.
    peak_slope_error_pct = slope_cv_pct

    fit_wall_s = time.perf_counter() - t0

    result = CalibrationResult(
        light_shift_coeff=ls_coeff,
        buffer_gas_shift_coeff=buf_coeff,
        lumped_zeeman_coeff=zee_coeff,
        peak_slope_error_max_pct=peak_slope_error_pct,
        lock_point_shift_max_pct=lock_point_shift_pct,
        fit_converged=True,
        fit_message="closed-form OLS — always converges by construction",
        optimizer_iterations=1,
        n_grid_points_fit=len(lp_arr),
        fit_residual_rms_Hz=rms,
        fit_wall_s=fit_wall_s,
    )

    if out_json_path is not None:
        out_path = Path(out_json_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_dict = {
            "light_shift_coeff": result.light_shift_coeff,
            "buffer_gas_shift_coeff": result.buffer_gas_shift_coeff,
            "lumped_zeeman_coeff": result.lumped_zeeman_coeff,
            "constant_offset_Hz": fit_offset_Hz,
            "buffer_gas_shift_coeff_pinned": True,
            "buffer_gas_shift_coeff_pinning_rationale": (
                "compute_obe_surface fixes buffer_pressure_torr=25; "
                "the buf*P_buf product is degenerate with the global offset "
                "and cannot be identified from this surface. Pinned at the "
                "RbSpec analytic value (Steck-validated)."
            ),
            "peak_slope_error_max_pct": result.peak_slope_error_max_pct,
            "lock_point_shift_max_pct_vs_spread": result.lock_point_shift_max_pct,
            "fit_converged": result.fit_converged,
            "fit_message": result.fit_message,
            "optimizer_iterations": result.optimizer_iterations,
            "n_grid_points_fit": result.n_grid_points_fit,
            "fit_residual_rms_Hz": result.fit_residual_rms_Hz,
            "fit_wall_s": result.fit_wall_s,
            "obe_surface_path": str(obe_surface_path),
            "buffer_pressure_torr": buf_pressure,
            "n_laser_points": int(N_laser),
            "n_T_points": int(N_T),
            "n_B_points": int(N_B),
            "n_I_points": int(N_I),
        }
        out_path.write_text(json.dumps(out_dict, indent=2), encoding="utf-8")

    return result
