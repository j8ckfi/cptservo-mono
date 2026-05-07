"""Analytic Policy Gradient (APG) MLP policy for CPT-clock servo.

The APGPolicy is a small MLP that maps a sliding-window observation vector to
a scalar RF-frequency correction command.  The entire chain

    policy -> rf_correction -> twin.step -> state -> y

is differentiable, enabling truncated BPTT through the ReducedTwin.

Observation vector (obs_dim = n_error_history + n_rf_history + 3*include_env):
    - Last N error_signal values (ci units, sliding window, newest last)
    - Last M rf_corr values (Hz, sliding window, newest last)
    - Optional: T_K, B_uT, I_norm (3 scalars; default on; ablatable for M8)

Action: scalar rf_correction (Hz), clamped to +/-rf_limit_Hz.

Protocol compatibility:
    step(error, env) -> (laser_corr_Hz, rf_corr_Hz) matches the
    PIController / DLQRController protocol so the policy can be dropped into
    run_fast_loop without modification.

References
----------
Kitching, J. (2018). Chip-scale atomic devices.
    *Applied Physics Reviews*, 5, 031302.
"""

from __future__ import annotations

from collections import deque
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor


class APGPolicy(nn.Module):
    """MLP policy that outputs rf_correction from observation.

    Observation (input):
        - Last N error_signal values (N=8 default, ci units, sliding window)
        - Last 4 rf_corr values (Hz, sliding window)
        - Optional environmental sensors: T_K, B_uT, I_norm (3 scalars; default
          on; can be ablated for sensor-shortcut probe in M8)

    Action (output): scalar rf_correction in Hz, clamped to +/-rf_limit_Hz.

    Args:
        n_error_history: Number of past error samples in observation window.
        n_rf_history: Number of past rf_corr samples in observation window.
        include_env_sensors: Whether to include T_K, B_uT, I_norm in obs.
        hidden_dims: Sizes of hidden MLP layers.
        rf_limit_Hz: Output saturation limit (+/-Hz).  Matches PIController
            anti-windup convention.
    """

    def __init__(
        self,
        n_error_history: int = 8,
        n_rf_history: int = 4,
        include_env_sensors: bool = True,
        hidden_dims: tuple[int, ...] = (64, 64, 32),
        rf_limit_Hz: float = 1000.0,
    ) -> None:
        super().__init__()

        self.n_error_history = n_error_history
        self.n_rf_history = n_rf_history
        self.include_env_sensors = include_env_sensors
        self.rf_limit_Hz = rf_limit_Hz
        self._hidden_dims: tuple[int, ...] = tuple(hidden_dims)

        # Observation dimension
        self.obs_dim: int = n_error_history + n_rf_history + (3 if include_env_sensors else 0)

        # Build MLP: obs_dim -> hidden_dims -> 1. Tanh is used instead of
        # GELU because the Windows Torch stack used for this project can crash
        # in GELU backward on float64 policy gradients.
        layers: list[nn.Module] = []
        in_dim = self.obs_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.Tanh())
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers).float()

        # Input normalisation statistics (set during training; defaults are
        # identity: mean=0, std=1).  Stored as buffers so they are saved with
        # state_dict and are not trained by the optimiser.
        self.register_buffer(
            "obs_mean", torch.zeros(self.obs_dim, dtype=torch.float32)
        )
        self.register_buffer(
            "obs_std", torch.ones(self.obs_dim, dtype=torch.float32)
        )

        # Internal sliding windows for the step() protocol
        self._error_window: deque[float] = deque(
            [0.0] * n_error_history, maxlen=n_error_history
        )
        self._rf_window: deque[float] = deque(
            [0.0] * n_rf_history, maxlen=n_rf_history
        )

    # ---------------------------------------------------------------------------
    # Forward pass (batched; gradient-capable)
    # ---------------------------------------------------------------------------

    def forward(self, obs: Tensor) -> Tensor:
        """Compute rf_correction from batched observation.

        Args:
            obs: (B, obs_dim) observation tensor on the same device/dtype as
                policy parameters.

        Returns:
            (B,) rf_correction in Hz, clamped to +/-rf_limit_Hz via tanh.
        """
        out_dtype = obs.dtype
        obs_f = obs.to(dtype=torch.float32)
        obs_norm = (obs_f - self.obs_mean) / self.obs_std.clamp(min=1e-8)
        raw = self.net(obs_norm).squeeze(-1)  # (B,)
        rf = torch.tanh(raw / self.rf_limit_Hz) * self.rf_limit_Hz
        return rf.to(dtype=out_dtype)

    # ---------------------------------------------------------------------------
    # Single-step protocol -- drop-in for PIController / DLQRController
    # ---------------------------------------------------------------------------

    def step(
        self,
        error: float,
        env: dict[str, float] | None = None,
    ) -> tuple[float, float]:
        """Single-step (laser_corr, rf_corr) compatible with PIController protocol.

        Maintains internal sliding windows for error_history and rf_history.

        Args:
            error: Dimensionless lock-in error signal (ci units).
            env: Optional dict with keys T_K, B_uT, I_norm.  Required when
                include_env_sensors=True.  If None and env_sensors are enabled,
                defaults of (333.15, 50.0, 1.0) are used.

        Returns:
            Tuple (laser_detuning_correction_Hz, rf_detuning_correction_Hz).
            laser_detuning_correction_Hz is always 0 (v1 convention).
        """
        self._error_window.append(error)

        obs_list: list[float] = list(self._error_window) + list(self._rf_window)
        if self.include_env_sensors:
            if env is not None:
                obs_list += [
                    float(env.get("T_K", 333.15)),
                    float(env.get("B_uT", 50.0)),
                    float(env.get("I_norm", 1.0)),
                ]
            else:
                obs_list += [333.15, 50.0, 1.0]

        obs_t = torch.tensor(
            obs_list, dtype=torch.float64, device=next(self.parameters()).device
        ).unsqueeze(0)  # (1, obs_dim)

        with torch.no_grad():
            rf_corr = float(self.forward(obs_t).item())

        self._rf_window.append(rf_corr)
        return (0.0, rf_corr)

    # ---------------------------------------------------------------------------
    # State management
    # ---------------------------------------------------------------------------

    def reset(self) -> None:
        """Reset internal sliding windows to zero.

        Call before each new simulation episode.
        """
        self._error_window = deque([0.0] * self.n_error_history, maxlen=self.n_error_history)
        self._rf_window = deque([0.0] * self.n_rf_history, maxlen=self.n_rf_history)

    # ---------------------------------------------------------------------------
    # Observation builder (batched; used by training loop)
    # ---------------------------------------------------------------------------

    def build_obs(
        self,
        error_history: Tensor,
        rf_history: Tensor,
        env_tensor: Tensor | None = None,
    ) -> Tensor:
        """Assemble observation tensor from history buffers.

        Args:
            error_history: (B, n_error_history) recent error values.
            rf_history: (B, n_rf_history) recent rf_corr values.
            env_tensor: (B, 3) env scalars [T_K, B_uT, I_norm], or None.

        Returns:
            (B, obs_dim) observation tensor.
        """
        parts = [error_history, rf_history]
        if self.include_env_sensors:
            if env_tensor is None:
                B = error_history.shape[0]
                dev = error_history.device
                dtype = error_history.dtype
                env_tensor = torch.tensor(
                    [[333.15, 50.0, 1.0]], dtype=dtype, device=dev
                ).expand(B, 3)
            parts.append(env_tensor)
        return torch.cat(parts, dim=-1)

    # ---------------------------------------------------------------------------
    # Convenience helpers
    # ---------------------------------------------------------------------------

    def n_params(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def save(self, path: str) -> None:
        """Save policy weights and config to a .pt file.

        Args:
            path: Output file path.
        """
        torch.save(
            {
                "state_dict": self.state_dict(),
                "config": {
                    "n_error_history": self.n_error_history,
                    "n_rf_history": self.n_rf_history,
                    "include_env_sensors": self.include_env_sensors,
                    "hidden_dims": list(self._hidden_dims),
                    "rf_limit_Hz": self.rf_limit_Hz,
                    "obs_dim": self.obs_dim,
                },
            },
            path,
        )

    @classmethod
    def load(cls, path: str) -> APGPolicy:
        """Load policy from a checkpoint saved by save().

        Args:
            path: Path to the .pt checkpoint file.

        Returns:
            Loaded APGPolicy instance.
        """
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        cfg = ckpt["config"]
        policy = cls(
            n_error_history=int(cfg["n_error_history"]),
            n_rf_history=int(cfg["n_rf_history"]),
            include_env_sensors=bool(cfg["include_env_sensors"]),
            hidden_dims=tuple(int(h) for h in cfg["hidden_dims"]),
            rf_limit_Hz=float(cfg["rf_limit_Hz"]),
        )
        policy.load_state_dict(ckpt["state_dict"])
        return policy

    def extra_repr(self) -> str:
        return (
            f"obs_dim={self.obs_dim}, rf_limit_Hz={self.rf_limit_Hz}, "
            f"n_params={self.n_params()}"
        )


# ---------------------------------------------------------------------------
# Observation normalisation helper (called from training loop)
# ---------------------------------------------------------------------------


def compute_obs_stats(
    obs_samples: Tensor,
) -> tuple[Tensor, Tensor]:
    """Compute per-feature mean and std from a batch of observations.

    Args:
        obs_samples: (N, obs_dim) tensor of collected observations.

    Returns:
        Tuple (mean, std) each of shape (obs_dim,).
    """
    mean = obs_samples.mean(dim=0)
    std = obs_samples.std(dim=0).clamp(min=1e-8)
    return mean, std


# ---------------------------------------------------------------------------
# Module-level type alias
# ---------------------------------------------------------------------------

APGCheckpoint = dict[str, Any]
