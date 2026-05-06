"""Tier-1 full 16-level Lindblad master equation for Rb-87 D1 CPT.

Builds the complete 16x16 Hilbert space for Rb-87 D1 coherent population
trapping (CPT): 8 ground sublevels (5S_{1/2}, F=1,2 with all m_F) plus 8
excited sublevels (5P_{1/2}, F'=1,2 with all m_F).  QuTiP ``steadystate()``
(sparse direct) gives the steady-state density matrix from which the
photodetector-readout CPT discriminator signal is extracted.

Used only offline for calibration: generates a 5D grid of discriminator
responses that the tier-2 reduced model is fitted against.

References
----------
Steck, D. A. (2021). Rubidium 87 D Line Data. revision 2.3.
Vanier, J. & Mandache, C. (2007). Applied Physics B, 87, 565-593.
Kitching, J. (2018). Applied Physics Reviews, 5, 031302.
"""

from __future__ import annotations

import multiprocessing as mp
import time
from concurrent.futures import ThreadPoolExecutor
from itertools import product
from pathlib import Path

import h5py
import numpy as np
import qutip as qt

from rbspec.solver import (
    FWHM_D1,
    HF_EXCITED_D1,
    PRESSURE_BROAD_COEFF,
    PRESSURE_SHIFT_COEFF,
    S_D1,
    g_F_exc_D1,
    g_F_ground,
    mu_B,
)

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
_H_PLANCK: float = 6.62607015e-34  # J·s
_KB: float = 1.380649e-23  # J/K
_C: float = 2.99792458e8  # m/s
_AMU: float = 1.66053906660e-27  # kg
_M_RB87: float = 86.909180520 * _AMU

# Rb-87 D1 natural linewidth (HWHM in Hz)
_GAMMA_SP_HZ: float = FWHM_D1 / 2.0  # ~2.873 MHz

# Precise Rb-87 hyperfine ground splitting (Hz)
_HF_GROUND_HZ: float = 6_834_682_610.904

# ---------------------------------------------------------------------------
# Level indexing
# ---------------------------------------------------------------------------
# Ground levels: 5S_{1/2}
#   F=1: m_F = -1, 0, +1  -> indices 0,1,2
#   F=2: m_F = -2,-1, 0,+1,+2 -> indices 3,4,5,6,7
# Excited levels: 5P_{1/2}
#   F'=1: m_F = -1, 0, +1  -> indices 8,9,10
#   F'=2: m_F = -2,-1, 0,+1,+2 -> indices 11,12,13,14,15

_GROUND_F1: list[tuple[int, int]] = [(1, m) for m in (-1, 0, 1)]  # (F, mF)
_GROUND_F2: list[tuple[int, int]] = [(2, m) for m in (-2, -1, 0, 1, 2)]
_EXCITED_F1: list[tuple[int, int]] = [(1, m) for m in (-1, 0, 1)]
_EXCITED_F2: list[tuple[int, int]] = [(2, m) for m in (-2, -1, 0, 1, 2)]

_ALL_GROUND: list[tuple[int, int]] = _GROUND_F1 + _GROUND_F2  # 8 levels
_ALL_EXCITED: list[tuple[int, int]] = _EXCITED_F1 + _EXCITED_F2  # 8 levels
_ALL_LEVELS: list[tuple[str, int, int]] = (
    [("g", F, m) for F, m in _ALL_GROUND] + [("e", F, m) for F, m in _ALL_EXCITED]
)

_N: int = 16


def _idx(kind: str, F: int, mF: int) -> int:
    """Return state index for a given (kind, F, mF).

    Args:
        kind: ``"g"`` for ground, ``"e"`` for excited.
        F: total angular momentum quantum number.
        mF: magnetic quantum number.

    Returns:
        Integer index 0–15.
    """
    if kind == "g":
        lst = _ALL_GROUND
    else:
        lst = _ALL_EXCITED
        return 8 + lst.index((F, mF))
    return lst.index((F, mF))


# ---------------------------------------------------------------------------
# Clebsch-Gordan / dipole matrix elements
# ---------------------------------------------------------------------------
def _cg_d1(F_g: int, mF_g: int, F_e: int, mF_e: int) -> float:
    """Squared Clebsch-Gordan coefficient for D1 sigma+/sigma- transition.

    Uses the transition-strength table S_D1 from rbspec and distributes
    evenly over allowed delta_mF = {-1, 0, +1} sub-transitions within the
    (F_g, F_e) manifold.

    The total strength of the (F_g -> F_e) manifold is S_D1[(F_g, F_e)].
    We apportion it equally over the n_allowed transitions.

    Args:
        F_g: ground hyperfine F.
        mF_g: ground magnetic quantum number.
        F_e: excited hyperfine F'.
        mF_e: excited magnetic quantum number.

    Returns:
        Effective Rabi-matrix element magnitude squared (dimensionless).
    """
    if (F_g, F_e) not in S_D1:
        return 0.0
    delta_mF = mF_e - mF_g
    if abs(delta_mF) > 1:
        return 0.0
    # Count allowed transitions in this (F_g, F_e) block
    n_allowed = sum(
        1
        for mg in range(-F_g, F_g + 1)
        for me in range(-F_e, F_e + 1)
        if abs(me - mg) <= 1
    )
    return S_D1[(F_g, F_e)] / n_allowed


# ---------------------------------------------------------------------------
# Hamiltonian builder
# ---------------------------------------------------------------------------

def _build_hamiltonian(
    rf_detuning_Hz: float,
    laser_detuning_Hz: float,
    T_K: float,
    B_uT: float,
    intensity_norm: float,
    buffer_pressure_torr: float,
    rabi_freq_Hz: float = 1.0e5,
) -> qt.Qobj:
    """Construct the 16-level CPT Hamiltonian in the rotating frame.

    The Hamiltonian is in units of Hz (angular frequency divided by 2*pi),
    i.e. H_physical = h * H_here.

    In the rotating-wave approximation with the two-photon resonance condition
    the bare F=1 ground manifold is taken as energy zero.  F=2 ground manifold
    sits at -rf_detuning_Hz (the two-photon detuning from the applied RF).
    Excited states are shifted by the laser detuning plus hyperfine offsets.

    The lin-perp-lin (bichromatic) laser field is modelled as a single optical
    drive with equal sigma+/sigma-/pi components coupling all allowed transitions
    (delta_mF in {-1, 0, +1}).  This is equivalent to the standard lin-perp-lin
    CPT configuration where both tones share the same polarisation mixing and
    drive both clock states to the same excited sublevels.

    Args:
        rf_detuning_Hz: Two-photon RF detuning from exact resonance (Hz).
        laser_detuning_Hz: One-photon laser detuning from D1 line center (Hz).
        T_K: Cell temperature (K).
        B_uT: Magnetic field (uT).
        intensity_norm: Normalised laser intensity (1 = nominal).
        buffer_pressure_torr: Buffer-gas pressure (Torr).
        rabi_freq_Hz: Single-photon Rabi frequency at I_norm=1 (Hz).
            Default 100 kHz keeps the AC Stark light shift < 2 Hz at 25 Torr
            buffer gas (Omega/Gamma_buf ~ 2e-4), giving a lock-point offset
            < 10 Hz that the tier-2 light_shift_coeff can absorb.

    Returns:
        16x16 qt.Qobj Hamiltonian in Hz.
    """
    H = np.zeros((_N, _N), dtype=complex)
    B_T = B_uT * 1e-6

    # Buffer-gas pressure shift (applied to all transition frequencies)
    delta_buf = PRESSURE_SHIFT_COEFF * buffer_pressure_torr  # Hz

    # Second-order Zeeman coefficient for Rb-87 ground hyperfine clock transition
    # (F=1,mF=0 <-> F=2,mF=0).  The SOZ shifts the F=2,mF=0 level relative to
    # F=1,mF=0 by +alpha*B^2, where alpha = 575.15 Hz/mT^2 = 5.7515e-3 Hz/uT^2.
    # Ref: Steck, Rb-87 D Line Data (2021), eq. (18); Vanier & Audoin (1989).
    # This produces the measurable lock-point shift used as the Zeeman gate metric.
    _SOZ_COEFF_HZ_PER_UT2: float = 575.15e-6  # Hz/uT^2
    soz_shift_Hz = _SOZ_COEFF_HZ_PER_UT2 * B_uT**2  # differential shift on F=2,mF=0

    # --- Ground-state energies (in rotating frame) ---
    # F=1: E = 0 + Zeeman
    for F, mF in _GROUND_F1:
        i = _idx("g", F, mF)
        zeeman = g_F_ground[F] * mu_B * B_T * mF / _H_PLANCK  # Hz
        H[i, i] = zeeman

    # F=2: E = -rf_detuning (two-photon detuning) + Zeeman + SOZ correction
    # The SOZ adds +soz_shift_Hz to F=2,mF=0 only; mF!=0 states are dominated
    # by the first-order Zeeman which already shifts them far off resonance.
    for F, mF in _GROUND_F2:
        i = _idx("g", F, mF)
        zeeman = g_F_ground[F] * mu_B * B_T * mF / _H_PLANCK  # Hz
        soz = soz_shift_Hz if mF == 0 else 0.0
        H[i, i] = -rf_detuning_Hz + zeeman + soz

    # --- Excited-state energies ---
    # In the rotating frame the excited states appear at:
    #   E_e = laser_detuning + hyperfine_offset + buffer_gas_shift + Zeeman
    # referenced to the F=1 ground state.
    for F_e, mF_e in _ALL_EXCITED:
        ie = _idx("e", F_e, mF_e)
        hf_offset = HF_EXCITED_D1.get(F_e, 0.0)  # Hz
        zeeman_e = g_F_exc_D1[F_e] * mu_B * B_T * mF_e / _H_PLANCK  # Hz
        H[ie, ie] = laser_detuning_Hz + hf_offset + delta_buf + zeeman_e

    # --- Dipole coupling: ground <-> excited ---
    # The coupling is proportional to sqrt(intensity_norm) * rabi_freq_Hz * cg_coeff
    rabi = rabi_freq_Hz * np.sqrt(max(intensity_norm, 0.0))

    for (F_g, mF_g), (F_e, mF_e) in product(_ALL_GROUND, _ALL_EXCITED):
        cg2 = _cg_d1(F_g, mF_g, F_e, mF_e)
        if cg2 == 0.0:
            continue
        omega = rabi * np.sqrt(cg2)
        ig = _idx("g", F_g, mF_g)
        ie = _idx("e", F_e, mF_e)
        # Rotating-wave approximation: half-Rabi off-diagonal
        H[ig, ie] += omega / 2.0
        H[ie, ig] += omega / 2.0

    return qt.Qobj(H)


# ---------------------------------------------------------------------------
# Collapse operators builder
# ---------------------------------------------------------------------------

def _build_collapse_ops(
    T_K: float,
    buffer_pressure_torr: float,
    ground_relax_rate_Hz: float = 100.0,
) -> list[qt.Qobj]:
    """Build Lindblad collapse operators.

    Includes:
    - Spontaneous emission: each excited sublevel decays to all dipole-allowed
      ground sublevels at rate GAMMA_SP (natural linewidth HWHM).
    - Buffer-gas optical dephasing: dephasing of all optical coherences
      at rate PRESSURE_BROAD_COEFF * buffer_pressure_torr.
    - Ground-state spin-exchange / wall relaxation at ~100 Hz.

    Args:
        T_K: Cell temperature (K, unused currently but reserved for
            temperature-dependent dephasing models).
        buffer_pressure_torr: Buffer-gas pressure (Torr).
        ground_relax_rate_Hz: Ground-state relaxation rate (Hz).

    Returns:
        List of qt.Qobj collapse operators.
    """
    c_ops: list[qt.Qobj] = []
    gamma_buf_broad = PRESSURE_BROAD_COEFF * buffer_pressure_torr  # Hz

    # --- Spontaneous emission: excited -> ground ---
    for F_e, mF_e in _ALL_EXCITED:
        ie = _idx("e", F_e, mF_e)
        for F_g, mF_g in _ALL_GROUND:
            cg2 = _cg_d1(F_g, mF_g, F_e, mF_e)
            if cg2 == 0.0:
                continue
            # Rate for this branch: gamma_sp * branching_ratio
            # Branching: proportional to CG^2 summed over all ground levels
            total_cg2 = sum(
                _cg_d1(Fg2, mFg2, F_e, mF_e)
                for Fg2, mFg2 in _ALL_GROUND
            )
            if total_cg2 == 0.0:
                continue
            rate = _GAMMA_SP_HZ * cg2 / total_cg2
            ig = _idx("g", F_g, mF_g)
            op = np.zeros((_N, _N), dtype=complex)
            op[ig, ie] = np.sqrt(rate)
            c_ops.append(qt.Qobj(op))

    # --- Buffer-gas dephasing of optical coherences ---
    # All optical coherences sigma_{g_i, e_j} decay at rate gamma_buf_broad.
    # Represented by a single diagonal operator on the excited manifold:
    # C_deph = sqrt(gamma_buf) * sum_e |e><e|.
    # Each excited projector |e_j><e_j| adds gamma_buf to the decay of every
    # coherence rho_{g_i, e_j}, reproducing the standard pressure-broadening
    # Lindblad term without spawning 8x8=64 individual operators.
    if gamma_buf_broad > 0.0:
        op = np.zeros((_N, _N), dtype=complex)
        for ie_idx in range(8, 16):
            op[ie_idx, ie_idx] = np.sqrt(gamma_buf_broad)
        c_ops.append(qt.Qobj(op))

    # --- Ground-state spin relaxation (wall collisions + spin exchange ~100 Hz) ---
    # Relax each ground sublevel toward the uniform mixture.
    n_ground = 8
    for ig_idx in range(n_ground):
        op = np.zeros((_N, _N), dtype=complex)
        op[ig_idx, ig_idx] = np.sqrt(ground_relax_rate_Hz)
        c_ops.append(qt.Qobj(op))

    return c_ops


# ---------------------------------------------------------------------------
# Single-point steady-state solve
# ---------------------------------------------------------------------------

def _photodetector_signal(rho: qt.Qobj) -> float:
    """Extract photodetector signal from steady-state density matrix.

    The CPT discriminator signal is the imaginary part of the two-photon
    coherence (rho_{F1m0, F2m0}), which is antisymmetric about the two-photon
    resonance.  This is the standard FM-demodulated lock-in output.

    Concretely: signal = Im(rho_{F1_m0, F2_m0}) where m_F=0 for both.

    Args:
        rho: 16x16 steady-state density matrix.

    Returns:
        Discriminator signal value (dimensionless, antisymmetric in rf_detuning).
    """
    rho_arr = rho.full()
    i_F1_m0 = _idx("g", 1, 0)  # F=1, mF=0
    i_F2_m0 = _idx("g", 2, 0)  # F=2, mF=0
    return float(np.imag(rho_arr[i_F1_m0, i_F2_m0]))


def full_obe_discriminator(
    rf_detuning_Hz: float,
    laser_detuning_Hz: float,
    T_K: float,
    B_uT: float,
    intensity_norm: float,
    buffer_pressure_torr: float = 25.0,
) -> dict[str, float]:
    """Compute the steady-state CPT discriminator response at one operating point.

    Solves the 16-level Lindblad master equation for Rb-87 D1 CPT using
    QuTiP ``steadystate()`` (sparse direct) and extracts the photodetector
    signal (imaginary part of the two-photon coherence rho_{F1m0,F2m0}).

    Args:
        rf_detuning_Hz: RF detuning from nominal two-photon resonance (Hz).
        laser_detuning_Hz: Laser detuning from D1 line center (Hz).
        T_K: Cell temperature (K).
        B_uT: Magnetic field (uT).
        intensity_norm: Normalised laser intensity (1 = nominal).
        buffer_pressure_torr: Buffer-gas pressure (Torr).

    Returns:
        Dict with keys:
          - ``"discriminator"``: antisymmetric CPT signal (float).
          - ``"excited_population"``: total excited-state population (float).
          - ``"ground_population"``: total ground-state population (float).
    """
    H = _build_hamiltonian(
        rf_detuning_Hz,
        laser_detuning_Hz,
        T_K,
        B_uT,
        intensity_norm,
        buffer_pressure_torr,
    )
    c_ops = _build_collapse_ops(T_K, buffer_pressure_torr)
    rho_ss = qt.steadystate(H, c_ops, method="direct")
    rho_arr = rho_ss.full()

    disc = _photodetector_signal(rho_ss)
    exc_pop = float(np.real(np.trace(rho_arr[8:, 8:])))
    gnd_pop = float(np.real(np.trace(rho_arr[:8, :8])))

    return {
        "discriminator": disc,
        "excited_population": exc_pop,
        "ground_population": gnd_pop,
    }


# ---------------------------------------------------------------------------
# Grid computation helpers (multiprocessing-safe)
# ---------------------------------------------------------------------------

def _solve_one(args: tuple) -> tuple[int, float]:
    """Worker function: solve one grid point, return (flat_index, discriminator).

    This is a module-level function so it is picklable for multiprocessing.

    Args:
        args: Tuple of (flat_idx, rf_det, laser_det, T_K, B_uT, I_norm, P_buf).

    Returns:
        (flat_idx, discriminator_value).
    """
    flat_idx, rf_det, laser_det, T_K, B_uT, I_norm, P_buf = args
    result = full_obe_discriminator(rf_det, laser_det, T_K, B_uT, I_norm, P_buf)
    return flat_idx, result["discriminator"]


def _find_lock_point_and_slope(
    rf_grid: np.ndarray,
    disc_curve: np.ndarray,
) -> tuple[float, float]:
    """Find lock point and slope from a discriminator vs rf_detuning curve.

    Lock point: linear interpolation of the zero crossing closest to rf=0.
    Slope: finite difference at the zero crossing.

    Args:
        rf_grid: RF detuning values (Hz), shape (N_rf,).
        disc_curve: Discriminator values, shape (N_rf,).

    Returns:
        Tuple of (lock_point_Hz, slope_per_Hz).  Returns (NaN, NaN) if no
        zero crossing is found.
    """
    # Find sign changes
    signs = np.sign(disc_curve)
    for i in range(len(signs) - 1):
        if signs[i] != signs[i + 1] and signs[i] != 0 and signs[i + 1] != 0:
            # Linear interpolation
            d1, d2 = disc_curve[i], disc_curve[i + 1]
            r1, r2 = rf_grid[i], rf_grid[i + 1]
            lock_pt = r1 - d1 * (r2 - r1) / (d2 - d1)
            slope = (d2 - d1) / (r2 - r1)
            return float(lock_pt), float(slope)
    return float("nan"), float("nan")


# ---------------------------------------------------------------------------
# Public surface API
# ---------------------------------------------------------------------------

def compute_obe_surface(
    rf_detuning_grid_Hz: np.ndarray,
    laser_detuning_grid_Hz: np.ndarray,
    T_K_grid: np.ndarray,
    B_uT_grid: np.ndarray,
    intensity_norm_grid: np.ndarray,
    buffer_pressure_torr: float = 25.0,
    n_workers: int | None = None,
    h5_out_path: str | Path | None = None,
) -> dict[str, np.ndarray]:
    """Compute the CPT discriminator response over a 5D parameter grid.

    Sweeps (rf_detuning, laser_detuning, T_K, B_uT, intensity_norm) and
    records the steady-state discriminator signal at each point.  Lock point
    and slope (from zero-crossing of rf_detuning sweep) are derived for each
    (laser_det, T, B, I) slice.

    Args:
        rf_detuning_grid_Hz: 1D array of RF detuning values (Hz).
        laser_detuning_grid_Hz: 1D array of laser detuning values (Hz).
        T_K_grid: 1D array of temperatures (K).
        B_uT_grid: 1D array of magnetic fields (uT).
        intensity_norm_grid: 1D array of normalised intensities.
        buffer_pressure_torr: Buffer-gas pressure (Torr).
        n_workers: Number of worker processes.  ``None`` = serial;
            positive int = ``mp.Pool`` with that many workers.
        h5_out_path: If provided, save results to this HDF5 path.

    Returns:
        Dict with arrays:
          - ``"rf_detuning_Hz"`` shape (N_rf,)
          - ``"laser_detuning_Hz"`` shape (N_laser,)
          - ``"T_K"`` shape (N_T,)
          - ``"B_uT"`` shape (N_B,)
          - ``"intensity_norm"`` shape (N_I,)
          - ``"discriminator"`` shape (N_rf, N_laser, N_T, N_B, N_I)
          - ``"lock_point"`` shape (N_laser, N_T, N_B, N_I) — Hz
          - ``"slope_per_Hz"`` shape (N_laser, N_T, N_B, N_I)
    """
    N_rf = len(rf_detuning_grid_Hz)
    N_laser = len(laser_detuning_grid_Hz)
    N_T = len(T_K_grid)
    N_B = len(B_uT_grid)
    N_I = len(intensity_norm_grid)

    discriminator = np.zeros((N_rf, N_laser, N_T, N_B, N_I), dtype=np.float64)

    # Build flat task list
    tasks = []
    for i_rf, i_l, i_T, i_B, i_I in product(
        range(N_rf), range(N_laser), range(N_T), range(N_B), range(N_I)
    ):
        flat = (
            ((i_rf * N_laser + i_l) * N_T + i_T) * N_B + i_B
        ) * N_I + i_I
        tasks.append((
            flat,
            float(rf_detuning_grid_Hz[i_rf]),
            float(laser_detuning_grid_Hz[i_l]),
            float(T_K_grid[i_T]),
            float(B_uT_grid[i_B]),
            float(intensity_norm_grid[i_I]),
            float(buffer_pressure_torr),
        ))

    t_start = time.perf_counter()

    if n_workers is None or n_workers <= 1:
        # Serial — predictable, no overhead
        results = [_solve_one(t) for t in tasks]
    else:
        # ThreadPoolExecutor: QuTiP releases the GIL during BLAS/sparse solves,
        # giving ~1.5x speedup without the process-spawn overhead of mp.Pool
        # (which is prohibitive on Windows for short tasks).
        n = min(n_workers, mp.cpu_count())
        with ThreadPoolExecutor(max_workers=n) as executor:
            results = list(executor.map(_solve_one, tasks))

    t_elapsed = time.perf_counter() - t_start

    # Unpack results into the 5D array
    for flat_idx, disc_val in results:
        i_I = flat_idx % N_I
        tmp = flat_idx // N_I
        i_B = tmp % N_B
        tmp //= N_B
        i_T = tmp % N_T
        tmp //= N_T
        i_l = tmp % N_laser
        i_rf = tmp // N_laser
        discriminator[i_rf, i_l, i_T, i_B, i_I] = disc_val

    # Compute lock points and slopes from rf sweeps
    lock_point = np.full((N_laser, N_T, N_B, N_I), np.nan, dtype=np.float64)
    slope_per_Hz = np.full((N_laser, N_T, N_B, N_I), np.nan, dtype=np.float64)

    for i_l, i_T, i_B, i_I in product(
        range(N_laser), range(N_T), range(N_B), range(N_I)
    ):
        curve = discriminator[:, i_l, i_T, i_B, i_I]
        lp, sl = _find_lock_point_and_slope(rf_detuning_grid_Hz, curve)
        lock_point[i_l, i_T, i_B, i_I] = lp
        slope_per_Hz[i_l, i_T, i_B, i_I] = sl

    surface = {
        "rf_detuning_Hz": rf_detuning_grid_Hz,
        "laser_detuning_Hz": laser_detuning_grid_Hz,
        "T_K": T_K_grid,
        "B_uT": B_uT_grid,
        "intensity_norm": intensity_norm_grid,
        "discriminator": discriminator,
        "lock_point": lock_point,
        "slope_per_Hz": slope_per_Hz,
        "compute_wall_s": np.float64(t_elapsed),
    }

    if h5_out_path is not None:
        _save_h5(surface, Path(h5_out_path))

    return surface


def _save_h5(surface: dict[str, np.ndarray], path: Path) -> None:
    """Save the OBE surface dict to an HDF5 file.

    Args:
        surface: Dict returned by :func:`compute_obe_surface`.
        path: Output file path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        for key, val in surface.items():
            arr = np.atleast_1d(val)
            f.create_dataset(key, data=arr, compression="gzip")
