"""Gymnasium environment wrapping ReducedTwin for PPO training.

The environment replicates the ``run_fast_loop`` inner-loop semantics exactly:

- Physics at 10 kHz, decimated to 1 kHz control steps.
- Per-substep y averaging (no aliasing).
- LO white-FM noise on ``rf_actual`` pre-step.
- Discriminator-input noise on ``error`` pre-controller.
- Anti-windup: policy action clamped to ±rf_limit_Hz.
- Noise injection point: ``rf_actual_pre_step+disc_noise_pre_controller``.

Observation (15-dim default, richer research features opt-in):
    [0:8]   Last 8 error_signal values (ci units, newest last)
    [8:12]  Last 4 rf_corr values (Hz, newest last)
    [12:15] Optional env sensors: T_K, B_uT, I_norm
    Optional: env first differences, running servo statistics, LQR hint action.

Action:
    Scalar in [-1, 1], scaled to [-rf_limit_Hz, +rf_limit_Hz].

Reward:
    Scaled negative squared fractional-frequency error.  By default this is
    the legacy per-rollout demeaned objective.  Research runs can opt into a
    10 s sliding-window Allan/variance proxy via ``reward_mode="window_var"``.

Episode:
    Runs for rollout_duration_s seconds at decimation_rate_Hz.
    Default 20 s (long-horizon, targets slow drift).

References
----------
Kitching, J. (2018). Chip-scale atomic devices.
    *Applied Physics Reviews*, 5, 031302.
Knappe, S. et al. (2004). *Applied Physics Letters*, 85, 1460.
"""

from __future__ import annotations

import math
from collections import deque
from pathlib import Path
from typing import Any

import gymnasium
import numpy as np
import torch
import yaml
from gymnasium import spaces

from cptservo.baselines.rh_lqr import RHLQRController
from cptservo.twin.disturbance import Disturbance
from cptservo.twin.reduced import ReducedTwin

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
_HF_GROUND: float = 6_834_682_610.904  # Hz, Rb-87 hyperfine frequency
_RECIPE_PATH = Path(__file__).parent.parent.parent.parent / "configs" / "v1_recipe.yaml"


def _load_recipe() -> dict:
    with open(_RECIPE_PATH, encoding="utf-8-sig") as fh:
        return yaml.safe_load(fh)


class CPTServoEnv(gymnasium.Env):
    """Gymnasium environment wrapping ReducedTwin + run_fast_loop semantics.

    Observation: 15-dim by default — 8 error history (ci units), 4 rf history
        (Hz), 3 env sensors (T_K, B_uT, I_norm).
    Action: scalar rf_correction in [-1, 1], scaled to [-rf_limit_Hz, +rf_limit_Hz].
    Reward: -squared fractional-frequency error per decimation step.
    Episode: rollout_duration_s seconds at decimation_rate_Hz.

    Args:
        twin_kwargs: Kwargs forwarded to ReducedTwin constructor. None uses
            calibrated defaults loaded from data/reduced_calibration.json.
        disturbance_recipe_name: Name of the disturbance scenario. One of
            ``clean``, ``thermal_ramp``, ``b_field_drift``,
            ``laser_intensity_drift``, ``all_stacked``.
        rollout_duration_s: Episode length in seconds.
        physics_rate_Hz: Physics simulation rate (Hz).
        decimation_rate_Hz: Control / observation rate (Hz).
        disc_noise_amp_ci: Discriminator-input noise amplitude (ci units).
            Models shot + AM + FM-to-AM noise per Knappe 2004.
        lo_noise_scale: Multiplier on the LO white-FM noise amplitude.
        rf_limit_Hz: Maximum RF correction magnitude (Hz); action anti-windup.
        n_error_history: Length of error-signal history in observation.
        n_rf_history: Length of rf-correction history in observation.
        include_env_sensors: Whether to append T_K, B_uT, I_norm to obs.
        rng_seed: Master RNG seed. Each reset increments by episode count for
            distinct noise realisations.
    """

    metadata: dict[str, Any] = {"render_modes": []}

    def __init__(
        self,
        twin_kwargs: dict | None = None,
        disturbance_recipe_name: str = "all_stacked",
        rollout_duration_s: float = 20.0,
        physics_rate_Hz: float = 10_000.0,
        decimation_rate_Hz: float = 1_000.0,
        disc_noise_amp_ci: float = 7.0e-4,
        lo_noise_scale: float = 1.0,
        rf_limit_Hz: float = 1_000.0,
        n_error_history: int = 8,
        n_rf_history: int = 4,
        include_env_sensors: bool = True,
        include_env_derivatives: bool = False,
        include_running_features: bool = False,
        include_lqr_hint: bool = False,
        reward_mode: str = "demeaned_y2",
        reward_scale: float = 1.0,
        reward_window_s: float = 10.0,
        action_mode: str = "standalone",
        residual_limit_Hz: float = 250.0,
        rng_seed: int | None = None,
    ) -> None:
        super().__init__()

        self.disturbance_recipe_name = disturbance_recipe_name
        self.rollout_duration_s = rollout_duration_s
        self.physics_rate_Hz = physics_rate_Hz
        self.decimation_rate_Hz = decimation_rate_Hz
        self.disc_noise_amp_ci = disc_noise_amp_ci
        self.lo_noise_scale = lo_noise_scale
        self.rf_limit_Hz = rf_limit_Hz
        self.n_error_history = n_error_history
        self.n_rf_history = n_rf_history
        self.include_env_sensors = include_env_sensors
        self.include_env_derivatives = include_env_derivatives
        self.include_running_features = include_running_features
        self.include_lqr_hint = include_lqr_hint
        self.reward_mode = reward_mode
        self.reward_scale = reward_scale
        self.reward_window_s = reward_window_s
        if action_mode not in {"standalone", "residual_lqr"}:
            raise ValueError(f"Unknown action_mode={action_mode!r}")
        self.action_mode = action_mode
        self.residual_limit_Hz = residual_limit_Hz
        self._base_seed = rng_seed if rng_seed is not None else 0
        self._episode_count = 0

        # Derived step counts
        self._decimation: int = int(round(physics_rate_Hz / decimation_rate_Hz))
        self._n_dec: int = int(round(rollout_duration_s * decimation_rate_Hz))
        self._dt: float = 1.0 / physics_rate_Hz

        # Build twin
        twin_kw = self._default_twin_kwargs()
        if twin_kwargs is not None:
            twin_kw.update(twin_kwargs)
        self._twin = ReducedTwin(**twin_kw)

        # LO noise std per physics step
        self._lo_noise_std = self._compute_lo_noise_std()

        # Gymnasium spaces
        obs_dim = n_error_history + n_rf_history
        obs_dim += 3 if include_env_sensors else 0
        obs_dim += 3 if include_env_derivatives else 0
        obs_dim += 3 if include_running_features else 0
        obs_dim += 1 if include_lqr_hint else 0
        self._obs_dim = obs_dim

        obs_high = np.full(obs_dim, np.inf, dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-obs_high, high=obs_high, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=np.array([-1.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
            shape=(1,),
            dtype=np.float32,
        )

        # Internal state (populated by reset)
        self._state: torch.Tensor | None = None
        self._dist_arr: np.ndarray | None = None
        self._lo_noise: np.ndarray | None = None
        self._disc_noise: np.ndarray | None = None
        self._step_k: int = 0
        self._rf_cmd: float = 0.0
        self._error_window: deque[float] = deque(
            [0.0] * n_error_history, maxlen=n_error_history
        )
        self._rf_window: deque[float] = deque(
            [0.0] * n_rf_history, maxlen=n_rf_history
        )
        # Last known env values for observation
        self._last_T_K: float = 333.15
        self._last_B_uT: float = 50.0
        self._last_I_norm: float = 1.0
        self._prev_T_K: float = 333.15
        self._prev_B_uT: float = 50.0
        self._prev_I_norm: float = 1.0
        self._integral_error: float = 0.0
        self._reward_window: deque[float] = deque(
            maxlen=max(2, int(round(reward_window_s * decimation_rate_Hz)))
        )
        self._lqr_hint = RHLQRController(
            control_dt_s=1.0 / decimation_rate_Hz,
            rf_limit_Hz=rf_limit_Hz,
        )
        self._lqr_hint_action_Hz: float = 0.0

        # Welford running mean of y for per-rollout demeaning
        # Reward = -(y - running_mean_y)^2  (per M6 APG diagnosis)
        self._y_count: int = 0
        self._y_mean: float = 0.0

    # ---------------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------------

    @staticmethod
    def _default_twin_kwargs() -> dict:
        """Load calibrated ReducedTwin kwargs from data/reduced_calibration.json."""
        import json

        project_root = Path(__file__).resolve().parents[4]
        calib_path = project_root / "data" / "reduced_calibration.json"
        if calib_path.exists():
            cal = json.loads(calib_path.read_text(encoding="utf-8-sig"))
            ls = float(cal["light_shift_coeff"])
            buf = float(cal["buffer_gas_shift_coeff"])
            zee = float(cal["lumped_zeeman_coeff"])
        else:
            ls, buf, zee = 0.0, -7.4e6, 7.0
        return {
            "light_shift_coeff": ls,
            "buffer_gas_shift_coeff": buf,
            "lumped_zeeman_coeff": zee,
            "temperature_coeff_Hz_per_K": 0.015,
            "device": "cpu",
        }

    def _compute_lo_noise_std(self) -> float:
        """Compute per-physics-step LO white-FM noise std (Hz)."""
        recipe = _load_recipe()
        nb = recipe["noise_budget"]
        white_fm_terms = [
            float(nb["photon_shot_noise_amp"]),
            float(nb["lo_white_fm_amp"]),
            float(nb.get("microwave_phase_white_amp", 0.0)),
            float(nb.get("detector_electronics_amp", 0.0)),
        ]
        total_white_fm_amp = math.sqrt(sum(t**2 for t in white_fm_terms))
        return (
            self.lo_noise_scale
            * total_white_fm_amp
            * _HF_GROUND
            * math.sqrt(self.physics_rate_Hz)
        )

    def _build_obs(self) -> np.ndarray:
        """Build the current observation vector as float32."""
        obs: list[float] = list(self._error_window) + list(self._rf_window)
        if self.include_env_sensors:
            obs += [
                (self._last_T_K - 333.15) / 10.0,
                (self._last_B_uT - 50.0) / 10.0,
                self._last_I_norm - 1.0,
            ]
        if self.include_env_derivatives:
            obs += [
                (self._last_T_K - self._prev_T_K) * self.decimation_rate_Hz / 10.0,
                (self._last_B_uT - self._prev_B_uT) * self.decimation_rate_Hz / 10.0,
                (self._last_I_norm - self._prev_I_norm) * self.decimation_rate_Hz,
            ]
        if self.include_running_features:
            y_var = float(np.var(self._reward_window)) if len(self._reward_window) > 1 else 0.0
            obs += [
                self._y_mean * 1.0e11,
                math.sqrt(max(y_var, 0.0)) * 1.0e11,
                self._integral_error,
            ]
        if self.include_lqr_hint:
            obs.append(self._lqr_hint_action_Hz / self.rf_limit_Hz)
        return np.array(obs, dtype=np.float32)

    # ---------------------------------------------------------------------------
    # Gymnasium API
    # ---------------------------------------------------------------------------

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        """Reset environment to start of a new episode.

        Args:
            seed: Optional RNG seed override (passed to gymnasium base class).
            options: Unused; accepted for API compatibility.

        Returns:
            Tuple of (observation, info).
        """
        super().reset(seed=seed)

        # Episode-specific seed so each episode sees different noise
        ep_seed = self._base_seed + self._episode_count
        if seed is not None:
            ep_seed = seed
        self._episode_count += 1

        rng = np.random.default_rng(ep_seed)

        # Generate disturbance trace
        n_physics = self._n_dec * self._decimation
        dist_gen = Disturbance.from_recipe(self.disturbance_recipe_name)
        trace = dist_gen.generate(
            duration_s=self.rollout_duration_s,
            sample_rate_Hz=self.physics_rate_Hz,
            seed=int(rng.integers(0, 2**31)),
        )
        dist_arr = np.stack(
            [trace.T_K, trace.B_uT, trace.laser_intensity_norm], axis=1
        ).astype(np.float64)
        if dist_arr.shape[0] < n_physics:
            reps = (n_physics + dist_arr.shape[0] - 1) // dist_arr.shape[0]
            dist_arr = np.tile(dist_arr, (reps, 1))[:n_physics]
        elif dist_arr.shape[0] > n_physics:
            dist_arr = dist_arr[:n_physics]
        self._dist_arr = dist_arr

        # Pre-generate LO noise
        if self._lo_noise_std > 0.0:
            self._lo_noise = rng.normal(0.0, self._lo_noise_std, size=n_physics)
        else:
            self._lo_noise = np.zeros(n_physics, dtype=np.float64)

        # Reset twin state with warmup
        self._state = self._twin.initial_state(batch_size=1)
        ctrl_zero = torch.zeros(1, 2, dtype=self._twin.dtype, device=self._twin.device)
        dist_t = torch.from_numpy(dist_arr).to(self._twin.device)
        with torch.no_grad():
            for _ in range(5_000):
                self._state = self._twin.step(
                    self._state, ctrl_zero, dist_t[:1], self._dt
                )

        # Pre-generate discriminator noise for all decimation steps
        if self.disc_noise_amp_ci > 0.0:
            self._disc_noise = rng.normal(0.0, self.disc_noise_amp_ci, size=self._n_dec)
        else:
            self._disc_noise = np.zeros(self._n_dec, dtype=np.float64)

        # Reset internal buffers
        self._step_k = 0
        self._rf_cmd = 0.0
        self._error_window = deque([0.0] * self.n_error_history, maxlen=self.n_error_history)
        self._rf_window = deque([0.0] * self.n_rf_history, maxlen=self.n_rf_history)
        self._last_T_K = float(dist_arr[0, 0])
        self._last_B_uT = float(dist_arr[0, 1])
        self._last_I_norm = float(dist_arr[0, 2])
        self._prev_T_K = self._last_T_K
        self._prev_B_uT = self._last_B_uT
        self._prev_I_norm = self._last_I_norm
        self._integral_error = 0.0
        self._reward_window.clear()
        self._lqr_hint.reset()
        self._lqr_hint_action_Hz = 0.0
        # Reset Welford running mean for per-rollout demeaning
        self._y_count = 0
        self._y_mean = 0.0

        obs = self._build_obs()
        return obs, {}

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Advance the environment by one decimation step.

        Replicates run_fast_loop inner loop:
        - action ∈ [-1, 1] is clipped then scaled to rf_correction_Hz
        - For each physics substep: rf_actual = rf_cmd + lo_noise
        - Discriminator noise added pre-controller (as in run_fast_loop)
        - Per-substep y averaging
        - Reward = -y_avg^2 (negative MSE of fractional-frequency error)

        Args:
            action: (1,) or scalar array with action in [-1, 1].

        Returns:
            (obs, reward, terminated, truncated, info)
        """
        if self._state is None or self._dist_arr is None or self._lo_noise is None \
                or self._disc_noise is None:
            raise RuntimeError("Must call reset() before step().")

        k = self._step_k

        # Clip action to [-1, 1] then scale to Hz. In residual mode the action
        # is a bounded correction around the LQR hint from the previous
        # observation; the LQR hint for the next observation is updated below.
        action_clipped = float(np.clip(action.flatten()[0], -1.0, 1.0))
        if self.action_mode == "residual_lqr":
            rf_cmd = self._lqr_hint_action_Hz + action_clipped * self.residual_limit_Hz
        else:
            rf_cmd = action_clipped * self.rf_limit_Hz
        # Additional anti-windup clamp (belt-and-suspenders)
        rf_cmd = float(np.clip(rf_cmd, -self.rf_limit_Hz, self.rf_limit_Hz))
        self._rf_cmd = rf_cmd

        dist_arr = self._dist_arr
        lo_noise = self._lo_noise
        twin = self._twin

        ctrl = torch.zeros(1, 2, dtype=twin.dtype, device=twin.device)
        dist_t = torch.from_numpy(dist_arr).to(twin.device)

        ci_sum = 0.0
        y_sum = 0.0

        with torch.no_grad():
            for j in range(self._decimation):
                idx = k * self._decimation + j
                rf_actual = rf_cmd + float(lo_noise[idx])
                ctrl[0, 1] = rf_actual

                self._state = twin.step(
                    self._state, ctrl, dist_t[idx : idx + 1], self._dt
                )

                ci_sum += float(self._state[0, 6].item())

                B_now = torch.tensor(
                    [float(dist_arr[idx, 1])], dtype=twin.dtype, device=twin.device
                )
                T_now = torch.tensor(
                    [float(dist_arr[idx, 0])], dtype=twin.dtype, device=twin.device
                )
                y_sum += float(
                    twin.fractional_frequency_error_with_B(
                        self._state, ctrl, B_now, T_now
                    ).item()
                )

        # Discriminator-input noise (pre-controller, matches run_fast_loop)
        # Uses pre-generated array from reset() — no per-step RNG construction.
        err_clean = ci_sum / self._decimation
        err_noisy = err_clean + float(self._disc_noise[k])

        y_avg = y_sum / self._decimation

        # Welford online update of per-rollout mean (for reward demeaning)
        self._y_count += 1
        delta = y_avg - self._y_mean
        self._y_mean += delta / self._y_count

        # Update env sensor cache from last substep index
        last_idx = (k + 1) * self._decimation - 1
        self._prev_T_K = self._last_T_K
        self._prev_B_uT = self._last_B_uT
        self._prev_I_norm = self._last_I_norm
        self._last_T_K = float(dist_arr[last_idx, 0])
        self._last_B_uT = float(dist_arr[last_idx, 1])
        self._last_I_norm = float(dist_arr[last_idx, 2])

        # Update history windows
        self._error_window.append(err_noisy)
        self._rf_window.append(rf_cmd)
        self._integral_error += err_noisy / self.decimation_rate_Hz
        self._lqr_hint_action_Hz = self._lqr_hint.step(err_noisy)[1]
        self._reward_window.append(y_avg)

        # Reward: scaled, variance-targeted fractional-frequency objective.
        y_demeaned = y_avg - self._y_mean
        if self.reward_mode == "demeaned_y2":
            reward_signal = y_demeaned**2
        elif self.reward_mode == "window_var":
            if len(self._reward_window) >= 2:
                reward_signal = float(np.var(self._reward_window))
            else:
                reward_signal = y_demeaned**2
        else:
            raise ValueError(f"Unknown reward_mode={self.reward_mode!r}")
        reward = float(-self.reward_scale * reward_signal)

        self._step_k += 1
        terminated = self._step_k >= self._n_dec
        truncated = False

        obs = self._build_obs()

        info: dict[str, Any] = {
            "y_avg": y_avg,
            "y_demeaned": y_demeaned,
            "y_mean_rollout": self._y_mean,
            "err_noisy": err_noisy,
            "rf_cmd_Hz": rf_cmd,
            "lqr_hint_Hz": self._lqr_hint_action_Hz,
            "reward_signal": reward_signal,
            "reward_mode": self.reward_mode,
            "step_k": k,
        }

        return obs, reward, terminated, truncated, info

    def render(self) -> None:
        """Render is not implemented (no visualisation)."""

    def close(self) -> None:
        """Clean up resources."""
