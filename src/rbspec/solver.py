"""Minimal Rb-87 spectroscopy helpers vendored for CPTServo."""

from __future__ import annotations

import numpy as np
from scipy.special import wofz

kB = 1.380649e-23
c_light = 2.99792458e8
h_planck = 6.62607015e-34
amu = 1.66053906660e-27
mu_B = 9.2740100783e-24
m_Rb87 = 86.909180520 * amu
e_charge = 1.602176634e-19
m_e = 9.1093837015e-31
eps0 = 8.854187817e-12

I_Rb87 = 1.5

LAMBDA_D1 = 794.979e-9
NU_D1 = c_light / LAMBDA_D1
FWHM_D1 = 5.746e6
F_OSC_D1 = 0.34

LAMBDA_D2 = 780.241e-9
NU_D2 = c_light / LAMBDA_D2
FWHM_D2 = 6.065e6
F_OSC_D2 = 0.695

HF_GROUND = 6834.682e6
HF_EXCITED_D1 = {1: 0.0, 2: 814.0e6}
HF_EXCITED_D2 = {
    0: 0.0,
    1: 157.0e6,
    2: 157.0e6 + 267.0e6,
    3: 157.0e6 + 267.0e6 + 424.0e6,
}

S_D1 = {(1, 1): 1.0 / 6, (1, 2): 5.0 / 6, (2, 1): 5.0 / 6, (2, 2): 1.0 / 6}
S_D2 = {
    (1, 0): 1.0 / 12,
    (1, 1): 1.0 / 12,
    (1, 2): 10.0 / 12,
    (2, 1): 1.0 / 20,
    (2, 2): 5.0 / 20,
    (2, 3): 14.0 / 20,
}

PRESSURE_BROAD_COEFF = 18.0e6
PRESSURE_SHIFT_COEFF = -7.4e6

g_S = 2.00231930436256


def _g_J(J: float, L: float) -> float:
    return 1.0 + (g_S - 1.0) * 0.5 * (
        J * (J + 1) + 0.75 - L * (L + 1)
    ) / (J * (J + 1))


def _g_F(g_J: float, J: float, F: float) -> float:
    if F == 0:
        return 0.0
    return g_J * (F * (F + 1) + J * (J + 1) - I_Rb87 * (I_Rb87 + 1)) / (
        2 * F * (F + 1)
    )


_g_J_ground = _g_J(0.5, 0.0)
_g_J_exc_D1 = _g_J(0.5, 1.0)
_g_J_exc_D2 = _g_J(1.5, 1.0)

g_F_ground = {F: _g_F(_g_J_ground, 0.5, F) for F in [1, 2]}
g_F_exc_D1 = {F: _g_F(_g_J_exc_D1, 0.5, F) for F in [1, 2]}
g_F_exc_D2 = {F: _g_F(_g_J_exc_D2, 1.5, F) for F in [0, 1, 2, 3]}

SIGMA_INT_FACTOR = e_charge**2 / (4 * eps0 * m_e * c_light)


def killian_vapor_density(T_K: float | np.ndarray) -> float | np.ndarray:
    """Rb-87 vapor number density from the Nesmeianov formula."""
    log10_P = 9.318 - 4215.0 / T_K
    P_Torr = 10.0**log10_P
    P_Pa = P_Torr * 133.322
    return P_Pa / (kB * T_K)


def voigt_profile(
    nu: np.ndarray,
    nu0: float,
    sigma_D: float,
    gamma_L: float,
) -> np.ndarray:
    """Normalized Voigt profile in 1/Hz."""
    z = ((nu - nu0) + 1j * gamma_L) / (sigma_D * np.sqrt(2))
    return np.real(wofz(z)) / (sigma_D * np.sqrt(2 * np.pi))


def doppler_width(T_K: float, nu0: float) -> float:
    """One-sigma Doppler width in Hz for Rb-87 at optical frequency ``nu0``."""
    return float(nu0 * np.sqrt(kB * T_K / (m_Rb87 * c_light**2)))

