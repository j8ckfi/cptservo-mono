"""Tier-2 eight-level adiabatic-elimination reduced model for Rb-87 D1 CPT.

Implements the standard Vanier & Mandache (2007) cartoon reduction of the full
16-level Lindblad OBE.  Two ground hyperfine levels (F=1, F=2) with m_F=0 clock
states explicit, a lumped Zeeman-bath population, two lumped excited-state
populations (D1 F'=1, F'=2), CPT dark-state coherence (real + imaginary), and an
intensity proxy.  Optical coherences are adiabatically eliminated (steady-state
pumping rate).

The coherence Bloch equations are written entirely in Hz (not rad/s) so that
all rate constants have consistent units and Euler integration at dt=1e-4 s is
stable.

References
----------
Vanier, J. & Mandache, C. (2007). The passive optically pumped Rb frequency
    standard: the laser approach. *Applied Physics B*, 87, 565-593.
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch import Tensor

# ---------------------------------------------------------------------------
# RbSpec physical constants (Steck-validated)
# ---------------------------------------------------------------------------
from rbspec.solver import (
    FWHM_D1,
    NU_D1,
    PRESSURE_BROAD_COEFF,
    PRESSURE_SHIFT_COEFF,
    doppler_width,
    killian_vapor_density,
)

# ---------------------------------------------------------------------------
# Precise reference frequency from the Layer-0 recipe (Steck 2.3).
# 6 834 682 610.904 Hz — more accurate than rbspec.solver.HF_GROUND (rounded).
# ---------------------------------------------------------------------------
_RECIPE_PATH = Path(__file__).parent.parent.parent.parent / "configs" / "v1_recipe.yaml"


def _load_recipe() -> dict:
    with open(_RECIPE_PATH) as fh:
        return yaml.safe_load(fh)


_RECIPE: dict = _load_recipe()

HF_GROUND: float = float(_RECIPE["reference_frequency_Hz"])  # 6834682610.904 Hz
PHOTON_SHOT_NOISE_AMP: float = float(
    _RECIPE["noise_budget"]["photon_shot_noise_amp"]
)  # 3e-11 frac-freq / sqrt(Hz)

# ---------------------------------------------------------------------------
# Derived optical constants
# ---------------------------------------------------------------------------
_GAMMA_OPT_HZ: float = FWHM_D1 / 2.0  # natural linewidth HWHM ~2.87 MHz

_T_NOM_K: float = 333.15  # nominal cell temperature (60 C)
_N_NOM: float = float(killian_vapor_density(_T_NOM_K))  # atoms/m^3

# ---------------------------------------------------------------------------
# Torch-native physics constants (avoid GPU->CPU round-trip per step)
# ---------------------------------------------------------------------------
_KB_J_PER_K: float = 1.380649e-23
_C_LIGHT_M_PER_S: float = 2.99792458e8
_AMU_KG: float = 1.66053906660e-27
_M_RB87_KG: float = 86.909180520 * _AMU_KG
_TORR_TO_PA: float = 133.322
# Nesmeianov / Killian coefficients (log10(P_torr) = A - B / T):
_NESMEIANOV_A: float = 9.318
_NESMEIANOV_B: float = 4215.0
# Doppler-width prefactor: sigma_D = nu0 * sqrt(kB * T / (m * c^2)) = prefactor * sqrt(T_K)
_DOPPLER_PREFACTOR: float = float(NU_D1) * math.sqrt(
    _KB_J_PER_K / (_M_RB87_KG * _C_LIGHT_M_PER_S**2)
)

# Effective optical pumping rate per unit intensity (s^{-1}) at nominal conditions.
# ~1 kHz repump rate is typical for a chip-scale cell geometry.
_GAMMA_PUMP_NOM: float = 1.0e3  # Hz at I_norm=1, T_nom

# ---------------------------------------------------------------------------
# CPT coherence Bloch-equation parameters
# ---------------------------------------------------------------------------
# Coherence decay rate: T_2 ~ 1 ms for chip-scale buffer-gas cell (Kitching 2018)
_T2_S: float = 1.0e-3
_GAMMA_2_HZ: float = 1.0 / _T2_S  # ~1000 Hz

# The two-photon detuning seen by the coherence comes from the RF offset.
# Buffer-gas shift is a DC offset (shifts the lock point, not the dynamics)
# absorbed into the lock-point error.  Here we track only the residual detuning
# = rf_detuning_correction - true_resonance_offset.

# ---------------------------------------------------------------------------


class ReducedTwin(nn.Module):
    """Eight-level adiabatic-elimination reduced twin for Rb-87 D1 CPT.

    State vector (B, 8):
        [0] pop_F1m0          - F=1, m_F=0 ground population
        [1] pop_F2m0          - F=2, m_F=0 ground population
        [2] pop_zeeman_bath   - lumped non-clock m_F ground population
        [3] pop_exc1          - lumped D1 F'=1 excited population
        [4] pop_exc2          - lumped D1 F'=2 excited population
        [5] coh_real          - Re(rho_{F1m0, F2m0}) dark-state coherence (dimensionless)
        [6] coh_imag          - Im(rho_{F1m0, F2m0})
        [7] intensity_proxy   - running mean of normalised laser intensity

    Trace conservation: sum of elements [0:5] == 1.
    """

    def __init__(
        self,
        cell_temperature_K: float = 333.15,
        cell_length_mm: float = 2.0,
        buffer_pressure_torr: float = 25.0,
        b_field_uT: float = 50.0,
        light_shift_coeff: float = 0.0,
        buffer_gas_shift_coeff: float = PRESSURE_SHIFT_COEFF,
        lumped_zeeman_coeff: float = 7.0,
        temperature_coeff_Hz_per_K: float = 0.05,
        photon_shot_noise_amp: float = PHOTON_SHOT_NOISE_AMP,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float64,
    ) -> None:
        """Initialise the ReducedTwin.

        Args:
            cell_temperature_K: Nominal cell temperature (K).
            cell_length_mm: Vapour-cell path length (mm).
            buffer_pressure_torr: Buffer-gas pressure (Torr).
            b_field_uT: Nominal bias magnetic field (uT).
            light_shift_coeff: AC Stark shift coefficient (Hz per unit normalised
                intensity).  Seed=0; M2 fits this.
            buffer_gas_shift_coeff: Buffer-gas pressure-shift coefficient (Hz/Torr).
                Default from RbSpec PRESSURE_SHIFT_COEFF = -7.4e6 Hz/Torr.
            lumped_zeeman_coeff: Linearised Zeeman coefficient (Hz/uT) for the
                m_F=0 clock transition.  Second-order Zeeman ~575 Hz/G^2 ~
                0.0575 Hz/uT^2; linearised around the nominal field.  Seed=7.0.
            photon_shot_noise_amp: Shot-noise amplitude (fractional-freq/sqrt(Hz)).
            device: Torch device.
            dtype: Torch dtype for inner-loop tensors (float64 recommended).
        """
        super().__init__()

        self.T_nom = cell_temperature_K
        self.cell_length_m = cell_length_mm * 1e-3
        self.P_buf = buffer_pressure_torr
        self.B_nom_uT = b_field_uT
        self.light_shift_coeff = light_shift_coeff
        self.buffer_gas_shift_coeff = buffer_gas_shift_coeff
        self.lumped_zeeman_coeff = lumped_zeeman_coeff
        # Δν/ΔT thermal coefficient — buffer-gas pressure-shift derivative w.r.t.
        # T (P_buf scales with T at constant volume). Knappe 2005 APL reports
        # ~50 mHz/K for typical CSACs; Vanier-Mandache cite 30–80 mHz/K depending
        # on buffer-gas mix. This term DOMINATES long-tau drift in published
        # Allan curves; without it, the twin reports zero open-loop sigma_y on
        # thermal_ramp scenarios (numerically zero is unphysical).
        self.temperature_coeff_Hz_per_K = temperature_coeff_Hz_per_K
        self.photon_shot_noise_amp = photon_shot_noise_amp
        self.device = torch.device(device)
        self.dtype = dtype

        # ---------------------------------------------------------------------------
        # Derived physics rates (at nominal conditions, all in Hz)
        # ---------------------------------------------------------------------------
        # Doppler 1-sigma width at nominal T
        sigma_D_nom = doppler_width(self.T_nom, NU_D1)  # Hz

        # Buffer-gas broadening
        gamma_buf = PRESSURE_BROAD_COEFF * self.P_buf  # Hz

        # Effective optical linewidth used for absorption rate normalisation
        self._gamma_opt = math.hypot(sigma_D_nom, _GAMMA_OPT_HZ + gamma_buf)  # Hz

        # Store nominal Doppler width for temperature-scaling the pump rate
        self._sigma_D_nom = sigma_D_nom  # Hz

        # Ground-state relaxation rate (wall + buffer; ~100 Hz for CSAC cells)
        self._gamma_g = 100.0  # Hz

        # Excited-state spontaneous emission rate (natural linewidth HWHM x 2)
        # For D1: FWHM_D1 = 5.746 MHz -> gamma_sp ~ 5.7 MHz; at dt=1e-4 s
        # this would make the Euler step (gamma_sp * dt) ~ 570, hugely unstable.
        # We use the ADIABATICALLY ELIMINATED approximation: excited states
        # are in steady state with pump rate.  gamma_sp only appears in the
        # population normalisation denominator; we set it large but cap the
        # effective step size via the algebraic steady state.
        # For Euler stability: use gamma_sp = 1/dt * 0.1 (10x slower than inverse dt)
        # but large enough that pe_ss << pump_rate/gamma_sp ~ 1e-4.
        # We use 1e4 Hz (10 kHz) - much slower than real but keeps excited pops small
        # and the adiabatic-elimination approximation still valid.
        self._gamma_sp = 1.0e4  # Hz (effective, adiabatically reduced)

        # CPT coherence decay rate (Hz)
        self._gamma_2 = _GAMMA_2_HZ  # ~1000 Hz

        # Optical-pumping leakage from clock states to Zeeman bath (Hz at I=1)
        # Typical value: ~5% of pump rate -> 50 Hz at 1 kHz pump
        self._gamma_leak_per_I = 50.0  # Hz per unit intensity

        # Clock-state fractions (1 m_F=0 out of 2F+1 substates)
        self._f_clock_F1 = 1.0 / 3.0  # F=1: 1/3
        self._f_clock_F2 = 1.0 / 5.0  # F=2: 1/5

    # ---------------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------------

    def initial_state(self, batch_size: int = 1) -> Tensor:
        """Return (B, 8) initial state at thermal equilibrium.

        Args:
            batch_size: Number of parallel trajectories B.

        Returns:
            Tensor of shape (B, 8) on self.device with self.dtype.
        """
        # Clock states: 1/(2F+1) fraction of each hyperfine level.
        # F=1 and F=2 each carry half the total ground population.
        pop_F1m0 = self._f_clock_F1 * 0.5  # ~1/6
        pop_F2m0 = self._f_clock_F2 * 0.5  # ~1/10
        pop_exc1 = 1e-6
        pop_exc2 = 1e-6
        # Zeeman bath gets the rest
        pop_zeeman = 1.0 - pop_F1m0 - pop_F2m0 - pop_exc1 - pop_exc2

        state_0 = torch.tensor(
            [pop_F1m0, pop_F2m0, pop_zeeman, pop_exc1, pop_exc2, 0.0, 0.0, 1.0],
            dtype=self.dtype,
            device=self.device,
        )
        return state_0.unsqueeze(0).expand(batch_size, -1).clone()

    def step(
        self,
        state: Tensor,
        controls: Tensor,
        disturbance: Tensor,
        dt: float = 1.0e-4,
    ) -> Tensor:
        """One explicit-Euler physics step.

        Args:
            state: (B, 8) current state.
            controls: (B, 2) - [laser_detuning_correction_Hz,
                rf_detuning_correction_Hz].
            disturbance: (B, 3) - [T_K, B_uT, laser_intensity_norm].
            dt: Physics timestep in seconds (default 1e-4 = 10 kHz).

        Returns:
            (B, 8) next state.  Pure function; inputs are not mutated.
        """
        # Unpack state
        p1 = state[:, 0]  # pop_F1m0
        p2 = state[:, 1]  # pop_F2m0
        pz = state[:, 2]  # pop_zeeman_bath
        pe1 = state[:, 3]  # pop_exc1
        pe2 = state[:, 4]  # pop_exc2
        cr = state[:, 5]  # coh_real
        ci = state[:, 6]  # coh_imag

        # Unpack controls and disturbance
        delta_rf = controls[:, 1]  # Hz — RF detuning correction
        T_K = disturbance[:, 0]  # K
        B_uT = disturbance[:, 1]  # uT
        I_norm = disturbance[:, 2].clamp(min=0.0)  # dimensionless

        # ---------------------------------------------------------------------------
        # Temperature-dependent optical pumping rate (Hz)
        # ---------------------------------------------------------------------------
        # Pure-torch evaluation of vapor density and Doppler width — no
        # GPU->CPU->numpy round-trip per step. Formulas mirror rbspec.solver but
        # stay vectorised in torch on self.device.
        # Killian/Nesmeianov: log10(P_torr) = 9.318 - 4215 / T_K
        log10_P = _NESMEIANOV_A - _NESMEIANOV_B / T_K
        P_torr = torch.pow(torch.tensor(10.0, dtype=self.dtype, device=self.device), log10_P)
        # Ideal-gas density: n = P[Pa] / (kB * T) ; P[Pa] = P_torr * 133.322
        n_atoms = P_torr * _TORR_TO_PA / (_KB_J_PER_K * T_K)  # atoms / m^3
        n_ratio = n_atoms / _N_NOM

        # Doppler 1-sigma width: sigma_D = nu_D1 * sqrt(kB * T / (m_Rb87 * c^2))
        sigma_D = _DOPPLER_PREFACTOR * torch.sqrt(T_K)  # Hz
        sigma_D_ratio = self._sigma_D_nom / sigma_D.clamp(min=1.0)

        # Pump rate scales with: intensity * density * (1/Doppler_width)
        R_pump = _GAMMA_PUMP_NOM * I_norm * n_ratio * sigma_D_ratio  # Hz

        # ---------------------------------------------------------------------------
        # Two-photon (CPT) detuning seen by the coherence (Hz, not rad/s)
        # ---------------------------------------------------------------------------
        # The DC buffer-gas shift is compensated by the RF synthesiser offset and
        # does not drive coherence oscillation.  However, the Zeeman shift from the
        # ACTUAL instantaneous B field (which may differ from nominal) and the RF
        # control residual both drive the Bloch dynamics.
        # delta_bloch = RF correction + Zeeman(B_actual) - Zeeman(B_nom)
        # = delta_rf + lumped_zeeman_coeff * (B_uT - B_nom_uT)
        delta_bloch = delta_rf + self.lumped_zeeman_coeff * (B_uT - self.B_nom_uT)

        # ---------------------------------------------------------------------------
        # Excited-state populations — adiabatic steady-state
        # pe_ss = R_pump * p_ground / (gamma_sp)
        # We evolve them dynamically but with a fast return toward steady state
        # ---------------------------------------------------------------------------
        gamma_sp = self._gamma_sp  # Hz

        # Pump rates into each excited level from each clock state
        rate_pump_F1 = 0.5 * R_pump * p1
        rate_pump_F2 = 0.5 * R_pump * p2

        rate_exc1_in = rate_pump_F1 + rate_pump_F2
        rate_exc2_in = rate_pump_F1 + rate_pump_F2

        decay_exc1 = gamma_sp * pe1
        decay_exc2 = gamma_sp * pe2

        # Return flux from excited states to clock ground states (branching 50/50)
        ret_to_F1 = 0.5 * (decay_exc1 + decay_exc2)
        ret_to_F2 = 0.5 * (decay_exc1 + decay_exc2)

        # ---------------------------------------------------------------------------
        # Ground-state dynamics
        # ---------------------------------------------------------------------------
        gamma_g = self._gamma_g  # Hz

        # Zeeman-bath relaxation back to clock states
        relax_bath_F1 = gamma_g * pz * self._f_clock_F1
        relax_bath_F2 = gamma_g * pz * self._f_clock_F2

        # Leakage from clock states to Zeeman bath (driven by optical pumping)
        leak_rate = self._gamma_leak_per_I * I_norm  # Hz
        leak_F1 = leak_rate * p1
        leak_F2 = leak_rate * p2

        # d(pop_F1m0)/dt
        dp1_dt = ret_to_F1 + relax_bath_F1 - rate_pump_F1 * 2.0 - leak_F1 - gamma_g * p1

        # d(pop_F2m0)/dt
        dp2_dt = ret_to_F2 + relax_bath_F2 - rate_pump_F2 * 2.0 - leak_F2 - gamma_g * p2

        # d(pop_zeeman_bath)/dt
        dpz_dt = leak_F1 + leak_F2 - gamma_g * pz

        # d(pop_exc1)/dt
        dpe1_dt = rate_exc1_in - decay_exc1

        # d(pop_exc2)/dt
        dpe2_dt = rate_exc2_in - decay_exc2

        # ---------------------------------------------------------------------------
        # CPT dark-state coherence Bloch equations (all rates in Hz)
        #
        # d/dt rho_12 = -(gamma_2 + i*2*pi*delta_2ph) * rho_12 + R_pump/2 * sqrt(p1*p2)
        #
        # In Hz convention (divide angular freq by 2*pi):
        # d/dt [cr, ci] = [[-gamma_2, +delta_2ph], [-delta_2ph, -gamma_2]] [cr, ci]
        #                  + source
        #
        # The off-diagonal coupling is delta_2ph (Hz), NOT 2*pi*delta_2ph.
        # ---------------------------------------------------------------------------
        gamma_2 = self._gamma_2  # Hz

        # Coherence source: CPT pumping builds coherence between clock states
        coh_source = 0.5 * R_pump * torch.sqrt((p1 * p2).clamp(min=0.0))

        dcr_dt = -gamma_2 * cr + delta_bloch * ci + coh_source
        dci_dt = -gamma_2 * ci - delta_bloch * cr

        # ---------------------------------------------------------------------------
        # Intensity proxy (exponential running mean, tau=10 ms)
        # ---------------------------------------------------------------------------
        tau_I = 0.01  # s
        dI_dt = (I_norm - state[:, 7]) / tau_I

        # ---------------------------------------------------------------------------
        # Euler step
        # ---------------------------------------------------------------------------
        d_state = torch.stack(
            [dp1_dt, dp2_dt, dpz_dt, dpe1_dt, dpe2_dt, dcr_dt, dci_dt, dI_dt], dim=1
        )
        next_state = state + dt * d_state

        # ---------------------------------------------------------------------------
        # Soft trace enforcement: clamp populations >= 0 and renormalise to sum=1
        # ---------------------------------------------------------------------------
        pops = next_state[:, :5].clamp(min=0.0)
        pop_sum = pops.sum(dim=1, keepdim=True).clamp(min=1e-12)
        pops_normed = pops / pop_sum
        next_state = torch.cat(
            [pops_normed, next_state[:, 5:7], next_state[:, 7:8].clamp(min=0.0)], dim=1
        )

        return next_state

    # ---------------------------------------------------------------------------

    def photodetector_signal(self, state: Tensor) -> Tensor:
        """Compute photodetector signal from state.

        The signal increases with laser intensity and with EIT transparency
        (large dark-state coherence reduces absorption).  Shot noise is added.

        Args:
            state: (B, 8) state tensor.

        Returns:
            (B,) photodetector signal (arbitrary units, ~0-1 range).
        """
        I_proxy = state[:, 7].clamp(min=0.0)
        p1 = state[:, 0]
        p2 = state[:, 1]
        cr = state[:, 5]
        ci = state[:, 6]

        coh_sq = (cr**2 + ci**2).clamp(min=0.0)

        # Base transmitted intensity: reduced by absorption proportional to population
        absorption_depth = 0.3  # 30% peak absorption

        # EIT: coherence reduces absorption (electromagnetically induced transparency)
        # At full CPT resonance |rho_12| ~ 0.5 * sqrt(p1*p2), max coh_sq ~ 0.25
        eit_boost = 4.0 * coh_sq  # max boost ~1 at full CPT

        # Absorption signal (symmetric / Lorentzian):
        #   transmission increases with EIT transparency
        signal = I_proxy * (1.0 - absorption_depth * (p1 + p2) * (1.0 - eit_boost))

        # Dispersive contribution from imaginary coherence (antisymmetric in detuning).
        # In a real CPT spectrometer the transmission has a dispersive component
        # proportional to Im(rho_12), which changes sign with detuning.  This is
        # the component the FM lock-in extracts.  Coefficient chosen to give
        # a signal slope of ~1e-4 per 100 Hz detuning, much larger than shot noise.
        dispersive_coeff = 2.0
        signal = signal + dispersive_coeff * ci * I_proxy

        # Photon shot noise: white noise with amplitude photon_shot_noise_amp
        # Scaled to signal units (signal ~ 0-1, noise ~ 1e-5 per sample)
        noise_sigma = self.photon_shot_noise_amp * math.sqrt(1.0 / 1e-4)  # per sqrt(dt)
        noise = torch.randn_like(signal) * float(noise_sigma) * 1e-3  # scale to signal units

        return signal + noise

    # ---------------------------------------------------------------------------

    def fractional_frequency_error(self, state: Tensor, controls: Tensor) -> Tensor:
        """Compute fractional frequency error y = (nu_lock - nu_ref) / nu_ref.

        The fractional-frequency error tracks how far the lock point is from
        HF_GROUND due to uncompensated systematic shifts.  All frequency offsets
        (buffer-gas, light, Zeeman) plus the RF correction contribute.

        Args:
            state: (B, 8) state tensor.
            controls: (B, 2) controls - controls[:, 1] is RF detuning correction (Hz).

        Returns:
            (B,) fractional frequency error.
        """
        delta_rf = controls[:, 1]  # Hz
        I_proxy = state[:, 7]

        # Systematic frequency offsets (Hz)
        delta_buf = self.buffer_gas_shift_coeff * self.P_buf  # Hz (fixed P_buf)
        delta_ls = self.light_shift_coeff * I_proxy  # Hz (light shift)
        # Zeeman: use B_nom_uT as the nominal field; real B variation handled in step()
        # For the gate measurement (open-loop, B=constant=B_nom), this is exact.
        delta_B_nom = self.lumped_zeeman_coeff * self.B_nom_uT  # Hz at nominal B

        total_offset_Hz = delta_rf + delta_buf + delta_ls + delta_B_nom
        return total_offset_Hz / HF_GROUND

    def fractional_frequency_error_with_B(
        self,
        state: Tensor,
        controls: Tensor,
        B_uT: Tensor,
        T_K: Tensor | None = None,
    ) -> Tensor:
        """Fractional frequency error using actual instantaneous B field and T.

        Args:
            state: (B, 8) state tensor.
            controls: (B, 2) controls tensor.
            B_uT: (B,) actual magnetic field in uT.
            T_K: (B,) actual cell temperature in K. If None, no temperature
                drift contribution is added (legacy behaviour). Pass the
                disturbance T_K to capture the dominant long-tau drift mechanism
                in chip-scale CPT clocks (Knappe 2005 APL).

        Returns:
            (B,) fractional frequency error.
        """
        delta_rf = controls[:, 1]
        I_proxy = state[:, 7]

        delta_buf = self.buffer_gas_shift_coeff * self.P_buf
        delta_ls = self.light_shift_coeff * I_proxy
        delta_B = self.lumped_zeeman_coeff * B_uT
        if T_K is not None:
            delta_T = self.temperature_coeff_Hz_per_K * (T_K - self.T_nom)
        else:
            delta_T = torch.zeros_like(delta_rf)

        total_offset_Hz = delta_rf + delta_buf + delta_ls + delta_B + delta_T
        return total_offset_Hz / HF_GROUND
