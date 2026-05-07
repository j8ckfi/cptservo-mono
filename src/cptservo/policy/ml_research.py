"""ML controller utilities for CPT servo autoresearch.

This module intentionally separates the research controller from the legacy
PPO environment.  The controller implements the same ``step(error, env)``
protocol used by ``run_batched_loop``, so learned policies are evaluated
through the canonical M5/M8 simulator path.
"""

from __future__ import annotations

import json
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from cptservo.baselines.dlqr import DLQRController


@dataclass(frozen=True)
class MLControllerConfig:
    """Serializable controller configuration."""

    mode: str = "standalone"
    n_error_history: int = 8
    n_rf_history: int = 4
    hidden_dims: tuple[int, ...] = (64, 64, 32)
    rf_limit_Hz: float = 1000.0
    residual_limit_Hz: float = 250.0
    include_env_derivatives: bool = True
    include_running_features: bool = True
    include_lqr_hint: bool = True


class MLServoController(nn.Module):
    """Small MLP servo controller for standalone and residual research tracks."""

    def __init__(self, config: MLControllerConfig | None = None) -> None:
        super().__init__()
        self.config = config or MLControllerConfig()
        if self.config.mode not in {"standalone", "residual_lqr"}:
            raise ValueError(f"Unsupported ML mode: {self.config.mode}")

        self.obs_dim = (
            self.config.n_error_history
            + self.config.n_rf_history
            + 3
            + (3 if self.config.include_env_derivatives else 0)
            + (3 if self.config.include_running_features else 0)
            + (1 if self.config.include_lqr_hint else 0)
        )
        layers: list[nn.Module] = []
        in_dim = self.obs_dim
        for width in self.config.hidden_dims:
            layers.extend([nn.Linear(in_dim, width), nn.GELU()])
            in_dim = width
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers).float()
        self.register_buffer("obs_mean", torch.zeros(self.obs_dim, dtype=torch.float32))
        self.register_buffer("obs_std", torch.ones(self.obs_dim, dtype=torch.float32))
        self._lqr = DLQRController(rf_limit_Hz=self.config.rf_limit_Hz)
        self.reset()

    def reset(self) -> None:
        """Reset history buffers and the residual baseline controller."""
        self._error_window: deque[float] = deque(
            [0.0] * self.config.n_error_history,
            maxlen=self.config.n_error_history,
        )
        self._rf_window: deque[float] = deque(
            [0.0] * self.config.n_rf_history,
            maxlen=self.config.n_rf_history,
        )
        self._last_env = {"T_K": 333.15, "B_uT": 50.0, "I_norm": 1.0}
        self._prev_env = dict(self._last_env)
        self._y_mean_like = 0.0
        self._err_integral = 0.0
        self._step_count = 0
        self._lqr.reset()

    def _features(self, error: float, env: dict[str, float]) -> tuple[np.ndarray, float]:
        self._prev_env = dict(self._last_env)
        self._last_env = {
            "T_K": float(env.get("T_K", 333.15)),
            "B_uT": float(env.get("B_uT", 50.0)),
            "I_norm": float(env.get("I_norm", 1.0)),
        }
        _, lqr_action = self._lqr.step(error)
        self._error_window.append(float(error))
        self._err_integral += float(error) * 0.001
        self._step_count += 1
        alpha = min(1.0, 1.0 / self._step_count)
        self._y_mean_like = (1.0 - alpha) * self._y_mean_like + alpha * float(error)

        obs: list[float] = list(self._error_window) + list(self._rf_window)
        obs += [
            (self._last_env["T_K"] - 333.15) / 10.0,
            (self._last_env["B_uT"] - 50.0) / 10.0,
            self._last_env["I_norm"] - 1.0,
        ]
        if self.config.include_env_derivatives:
            obs += [
                (self._last_env["T_K"] - self._prev_env["T_K"]) * 100.0,
                (self._last_env["B_uT"] - self._prev_env["B_uT"]) * 100.0,
                (self._last_env["I_norm"] - self._prev_env["I_norm"]) * 1000.0,
            ]
        if self.config.include_running_features:
            err_arr = np.asarray(self._error_window, dtype=np.float64)
            obs += [
                self._y_mean_like,
                float(np.std(err_arr)),
                self._err_integral,
            ]
        if self.config.include_lqr_hint:
            obs.append(lqr_action / self.config.rf_limit_Hz)
        return np.asarray(obs, dtype=np.float64), float(lqr_action)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Return normalized action or normalized residual from batched obs."""
        obs_norm = (obs - self.obs_mean) / self.obs_std.clamp(min=1.0e-8)
        return torch.tanh(self.net(obs_norm)).squeeze(-1)

    def step(self, error: float, env: dict[str, float] | None = None) -> tuple[float, float]:
        """Controller protocol used by ``run_batched_loop``."""
        obs_np, lqr_action = self._features(error, env or {})
        obs_t = torch.from_numpy(obs_np).to(dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            raw = float(self.forward(obs_t).item())
        if self.config.mode == "residual_lqr":
            rf = lqr_action + raw * self.config.residual_limit_Hz
        else:
            rf = raw * self.config.rf_limit_Hz
        rf = float(np.clip(rf, -self.config.rf_limit_Hz, self.config.rf_limit_Hz))
        self._rf_window.append(rf)
        return 0.0, rf

    def save(self, path: str | Path) -> None:
        """Save config and weights."""
        torch.save(
            {"config": asdict(self.config), "state_dict": self.state_dict()},
            str(path),
        )

    @classmethod
    def load(cls, path: str | Path) -> MLServoController:
        """Load a controller saved by ``save``."""
        ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
        cfg = dict(ckpt["config"])
        cfg["hidden_dims"] = tuple(cfg["hidden_dims"])
        model = cls(MLControllerConfig(**cfg))
        model.load_state_dict(ckpt["state_dict"])
        return model


def fit_obs_stats(model: MLServoController, obs: np.ndarray) -> None:
    """Fit observation normalization buffers from collected features."""
    obs_t = torch.from_numpy(np.asarray(obs, dtype=np.float32))
    model.obs_mean.copy_(obs_t.mean(dim=0))
    model.obs_std.copy_(obs_t.std(dim=0).clamp(min=1.0e-8))


def read_rhlqr_reference(project_root: Path) -> float:
    """Read the current M5 DLQR tau=10s reference."""
    path = project_root / "data" / "eval_dlqr_vs_pi.json"
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    for key in (
        "dlqr_sigma_y_10s",
        "rhlqr_sigma_y_10s",
        "rhlqr_sigma_y_10s_thermal",
        "lqr_sigma_y_10s",
        "dlqr_sigma_y_10s",
    ):
        if key in data:
            return float(data[key])
    hth = data.get("head_to_head_thermal_ramp", {})
    for key in ("dlqr_sigma_y_10s", "lqr_sigma_y_10s", "dlqr_sigma_y_10s"):
        if key in hth:
            return float(hth[key])
    raise KeyError(f"Could not find DLQR tau=10s reference in {path}")


@dataclass(frozen=True)
class PhysicsResidualConfig:
    """Linear learned residual over DLQR using physical sensor bases."""

    k_T_Hz_per_K: float = 0.0
    k_B_Hz_per_uT: float = 0.0
    k_I_Hz_per_norm: float = 0.0
    k_dT_Hz_per_K_s: float = 0.0
    k_dB_Hz_per_uT_s: float = 0.0
    k_dI_Hz_per_norm_s: float = 0.0
    T_nom_K: float = 333.15
    B_nom_uT: float = 50.0
    I_nom: float = 1.0
    intensity_tau_s: float = 0.01
    control_dt_s: float = 0.001
    residual_limit_Hz: float = 5.0
    rf_limit_Hz: float = 1000.0
    max_sensor_step_T_K: float = 2.0
    max_sensor_step_B_uT: float = 5.0
    max_sensor_step_I_norm: float = 0.1


class PhysicsResidualController:
    """Linear residual ML controller on top of DLQR.

    The fitted model is:

        u = clip(u_LQR + clip(r_theta(phi), +/- residual_limit), +/- rf_limit)

    where the weights are learned coefficients in RF Hz per sensor unit.  This
    is intentionally small and auditable: it is a real learned model, but its
    feature basis is the physical systematic-shift basis exposed by the twin.
    """

    def __init__(self, config: PhysicsResidualConfig | None = None) -> None:
        self.config = config or PhysicsResidualConfig()
        self._lqr = DLQRController(
            control_dt_s=self.config.control_dt_s,
            rf_limit_Hz=self.config.rf_limit_Hz,
        )
        self.reset()

    def reset(self) -> None:
        """Reset base LQR, sensor filters, and derivative state."""
        self._lqr.reset()
        self._I_hat = self.config.I_nom
        self._last_T_K = self.config.T_nom_K
        self._last_B_uT = self.config.B_nom_uT
        self._last_I_hat = self.config.I_nom

    def _validated_env(self, env: dict[str, float]) -> tuple[float, float, float]:
        """Return bounded sensor values, falling back to last good values."""
        T_K = self._bounded_sensor(
            env.get("T_K"),
            self._last_T_K,
            self.config.T_nom_K,
            self.config.max_sensor_step_T_K,
        )
        B_uT = self._bounded_sensor(
            env.get("B_uT"),
            self._last_B_uT,
            self.config.B_nom_uT,
            self.config.max_sensor_step_B_uT,
        )
        I_norm = self._bounded_sensor(
            env.get("I_norm"),
            self._I_hat,
            self.config.I_nom,
            self.config.max_sensor_step_I_norm,
        )
        return T_K, B_uT, I_norm

    @staticmethod
    def _bounded_sensor(
        raw_value: float | None,
        last_good: float,
        nominal: float,
        max_step: float,
    ) -> float:
        """Reject non-finite values and clamp single-step sensor jumps."""
        if raw_value is None:
            return last_good
        value = float(raw_value)
        if not np.isfinite(value):
            return last_good
        delta = float(np.clip(value - last_good, -max_step, max_step))
        bounded = last_good + delta
        if not np.isfinite(bounded):
            return nominal
        return bounded

    def step(self, error: float, env: dict[str, float] | None = None) -> tuple[float, float]:
        """Controller protocol compatible with ``run_batched_loop``."""
        env = env or {}
        T_K, B_uT, I_env = self._validated_env(env)
        alpha = min(1.0, self.config.control_dt_s / max(self.config.intensity_tau_s, 1.0e-9))
        prev_I_hat = self._I_hat
        self._I_hat += alpha * (I_env - self._I_hat)
        dT_dt = (T_K - self._last_T_K) / self.config.control_dt_s
        dB_dt = (B_uT - self._last_B_uT) / self.config.control_dt_s
        dI_dt = (self._I_hat - prev_I_hat) / self.config.control_dt_s
        self._last_T_K = T_K
        self._last_B_uT = B_uT
        self._last_I_hat = self._I_hat

        _, u_lqr = self._lqr.step(error)
        residual = (
            self.config.k_T_Hz_per_K * (T_K - self.config.T_nom_K)
            + self.config.k_B_Hz_per_uT * (B_uT - self.config.B_nom_uT)
            + self.config.k_I_Hz_per_norm * (self._I_hat - self.config.I_nom)
            + self.config.k_dT_Hz_per_K_s * dT_dt
            + self.config.k_dB_Hz_per_uT_s * dB_dt
            + self.config.k_dI_Hz_per_norm_s * dI_dt
        )
        residual = float(
            np.clip(residual, -self.config.residual_limit_Hz, self.config.residual_limit_Hz)
        )
        rf = float(np.clip(u_lqr + residual, -self.config.rf_limit_Hz, self.config.rf_limit_Hz))
        return 0.0, rf

    def save(self, path: str | Path) -> None:
        """Save learned linear residual coefficients."""
        Path(path).write_text(json.dumps(asdict(self.config), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> PhysicsResidualController:
        """Load a saved linear residual controller."""
        data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        return cls(PhysicsResidualConfig(**data))


@dataclass(frozen=True)
class CfCFeedforwardConfig:
    """Tiny closed-form continuous-time feedforward over DLQR."""

    hidden_size: int = 8
    seed: int = 1729
    base_residual: PhysicsResidualConfig = PhysicsResidualConfig(k_T_Hz_per_K=0.05)
    cfc_residual_limit_Hz: float = 1.0


class CfCFeedforwardController:
    """Closed-form gated recurrent residual over DLQR.

    This is intentionally small and dependency-free.  The base residual starts
    from the known M5-passing physics term, while the CfC cell can learn a slow
    dynamic correction from sensors and recent servo state.
    """

    feature_dim: int = 12

    def __init__(
        self,
        config: CfCFeedforwardConfig | None = None,
        weights: dict[str, np.ndarray] | None = None,
    ) -> None:
        self.config = config or CfCFeedforwardConfig()
        self._base = PhysicsResidualController(self.config.base_residual)
        self._lqr = DLQRController(
            control_dt_s=self.config.base_residual.control_dt_s,
            rf_limit_Hz=self.config.base_residual.rf_limit_Hz,
        )
        self.weights = weights if weights is not None else self._initial_weights()
        self.reset()

    def _initial_weights(self) -> dict[str, np.ndarray]:
        """Create deterministic zero-output CfC weights."""
        rng = np.random.default_rng(self.config.seed)
        h = self.config.hidden_size
        x = self.feature_dim
        return {
            "W_c": rng.normal(0.0, 0.05, size=(h, x)),
            "U_c": rng.normal(0.0, 0.05, size=(h, h)),
            "b_c": np.zeros(h, dtype=np.float64),
            "W_tau": rng.normal(0.0, 0.02, size=(h, x)),
            "U_tau": rng.normal(0.0, 0.02, size=(h, h)),
            "b_tau": np.zeros(h, dtype=np.float64),
            "W_out": np.zeros(h, dtype=np.float64),
            "b_out": np.zeros(1, dtype=np.float64),
        }

    def reset(self) -> None:
        """Reset recurrent state and all underlying controller state."""
        self._base.reset()
        self._lqr.reset()
        self._h = np.zeros(self.config.hidden_size, dtype=np.float64)
        self._last_error = 0.0
        self._last_residual = 0.0

    def _features(
        self,
        error: float,
        env: dict[str, float] | None,
        u_lqr: float,
    ) -> tuple[np.ndarray, float]:
        """Build normalized features and return base residual in Hz."""
        env = env or {}
        T_K, B_uT, I_env = self._base._validated_env(env)
        prev_I_hat = self._base._I_hat
        alpha = min(
            1.0,
            self.config.base_residual.control_dt_s
            / max(self.config.base_residual.intensity_tau_s, 1.0e-9),
        )
        I_hat = prev_I_hat + alpha * (I_env - prev_I_hat)
        dT_dt = (T_K - self._base._last_T_K) / self.config.base_residual.control_dt_s
        dB_dt = (B_uT - self._base._last_B_uT) / self.config.base_residual.control_dt_s
        dI_dt = (I_hat - prev_I_hat) / self.config.base_residual.control_dt_s

        cfg = self.config.base_residual
        base_residual = (
            cfg.k_T_Hz_per_K * (T_K - cfg.T_nom_K)
            + cfg.k_B_Hz_per_uT * (B_uT - cfg.B_nom_uT)
            + cfg.k_I_Hz_per_norm * (I_hat - cfg.I_nom)
            + cfg.k_dT_Hz_per_K_s * dT_dt
            + cfg.k_dB_Hz_per_uT_s * dB_dt
            + cfg.k_dI_Hz_per_norm_s * dI_dt
        )
        base_residual = float(
            np.clip(base_residual, -cfg.residual_limit_Hz, cfg.residual_limit_Hz)
        )

        features = np.array(
            [
                (T_K - cfg.T_nom_K) / 10.0,
                (B_uT - cfg.B_nom_uT) / 10.0,
                I_hat - cfg.I_nom,
                dT_dt / 10.0,
                dB_dt / 10.0,
                dI_dt,
                float(error) * 1.0e3,
                (float(error) - self._last_error) * 1.0e3,
                u_lqr / cfg.rf_limit_Hz,
                base_residual / max(cfg.residual_limit_Hz, 1.0e-9),
                self._last_residual / max(cfg.residual_limit_Hz, 1.0e-9),
                1.0,
            ],
            dtype=np.float64,
        )

        self._base._I_hat = I_hat
        self._base._last_T_K = T_K
        self._base._last_B_uT = B_uT
        self._base._last_I_hat = I_hat
        return features, base_residual

    def _cfc_step(self, features: np.ndarray) -> float:
        """Advance the closed-form cell and return its residual correction."""
        w = self.weights
        cand = np.tanh(w["W_c"] @ features + w["U_c"] @ self._h + w["b_c"])
        tau_raw = w["W_tau"] @ features + w["U_tau"] @ self._h + w["b_tau"]
        tau = np.log1p(np.exp(np.clip(tau_raw, -40.0, 40.0))) + 1.0e-3
        gate = np.exp(-self.config.base_residual.control_dt_s / tau)
        self._h = gate * self._h + (1.0 - gate) * cand
        raw = float(w["W_out"] @ self._h + w["b_out"][0])
        return float(
            np.clip(
                raw,
                -self.config.cfc_residual_limit_Hz,
                self.config.cfc_residual_limit_Hz,
            )
        )

    def step(self, error: float, env: dict[str, float] | None = None) -> tuple[float, float]:
        """Controller protocol compatible with ``run_batched_loop``."""
        _, u_lqr = self._lqr.step(error)
        features, base_residual = self._features(error, env, u_lqr)
        dynamic_residual = self._cfc_step(features)
        residual = float(
            np.clip(
                base_residual + dynamic_residual,
                -self.config.base_residual.residual_limit_Hz,
                self.config.base_residual.residual_limit_Hz,
            )
        )
        cfg = self.config.base_residual
        rf = float(np.clip(u_lqr + residual, -cfg.rf_limit_Hz, cfg.rf_limit_Hz))
        self._last_error = float(error)
        self._last_residual = residual
        return 0.0, rf

    def save(self, path: str | Path) -> None:
        """Save config and NumPy weights as JSON."""
        payload = {
            "config": {
                "hidden_size": self.config.hidden_size,
                "seed": self.config.seed,
                "base_residual": asdict(self.config.base_residual),
                "cfc_residual_limit_Hz": self.config.cfc_residual_limit_Hz,
            },
            "weights": {k: v.tolist() for k, v in self.weights.items()},
        }
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> CfCFeedforwardController:
        """Load a CfC feedforward checkpoint."""
        payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        cfg_raw = payload["config"]
        cfg = CfCFeedforwardConfig(
            hidden_size=int(cfg_raw["hidden_size"]),
            seed=int(cfg_raw["seed"]),
            base_residual=PhysicsResidualConfig(**cfg_raw["base_residual"]),
            cfc_residual_limit_Hz=float(cfg_raw["cfc_residual_limit_Hz"]),
        )
        weights = {
            key: np.asarray(value, dtype=np.float64)
            for key, value in payload["weights"].items()
        }
        return cls(cfg, weights=weights)


@dataclass(frozen=True)
class CfCDirectConfig:
    """Tiny closed-form continuous-time direct-action controller.

    Unlike ``CfCFeedforwardController``, this controller does not call DLQR at
    runtime and does not add a residual to an DLQR action.  The recurrent cell
    emits the full RF correction from discriminator error, sensor features, and
    its own internal state.
    """

    hidden_size: int = 16
    seed: int = 1729
    sensor_config: PhysicsResidualConfig = PhysicsResidualConfig()
    rf_limit_Hz: float = 1000.0
    control_dt_s: float = 0.001
    output_feature_skip: bool = True


class CfCDirectController:
    """Standalone CfC-style controller that outputs the full RF action."""

    feature_dim: int = 14

    def __init__(
        self,
        config: CfCDirectConfig | None = None,
        weights: dict[str, np.ndarray] | None = None,
    ) -> None:
        self.config = config or CfCDirectConfig()
        self.weights = weights if weights is not None else self._initial_weights()
        self.reset()

    def _initial_weights(self) -> dict[str, np.ndarray]:
        """Create deterministic zero-output CfC direct weights."""
        rng = np.random.default_rng(self.config.seed)
        h = self.config.hidden_size
        x = self.feature_dim
        return {
            "W_c": rng.normal(0.0, 0.05, size=(h, x)),
            "U_c": rng.normal(0.0, 0.05, size=(h, h)),
            "b_c": np.zeros(h, dtype=np.float64),
            "W_tau": rng.normal(0.0, 0.02, size=(h, x)),
            "U_tau": rng.normal(0.0, 0.02, size=(h, h)),
            "b_tau": np.zeros(h, dtype=np.float64),
            "W_out": np.zeros(h, dtype=np.float64),
            "W_feat_out": np.zeros(x, dtype=np.float64),
            "b_out": np.zeros(1, dtype=np.float64),
        }

    def reset(self) -> None:
        """Reset recurrent, sensor, and servo memory state."""
        cfg = self.config.sensor_config
        self._h = np.zeros(self.config.hidden_size, dtype=np.float64)
        self._I_hat = cfg.I_nom
        self._last_T_K = cfg.T_nom_K
        self._last_B_uT = cfg.B_nom_uT
        self._last_error = 0.0
        self._err_integral = 0.0
        self._error_window: deque[float] = deque([0.0] * 16, maxlen=16)
        self._last_rf = 0.0
        self._last_delta_rf = 0.0

    def _validated_env(self, env: dict[str, float]) -> tuple[float, float, float]:
        """Return bounded sensor values, falling back to last good values."""
        cfg = self.config.sensor_config
        T_K = PhysicsResidualController._bounded_sensor(
            env.get("T_K"),
            self._last_T_K,
            cfg.T_nom_K,
            cfg.max_sensor_step_T_K,
        )
        B_uT = PhysicsResidualController._bounded_sensor(
            env.get("B_uT"),
            self._last_B_uT,
            cfg.B_nom_uT,
            cfg.max_sensor_step_B_uT,
        )
        I_norm = PhysicsResidualController._bounded_sensor(
            env.get("I_norm"),
            self._I_hat,
            cfg.I_nom,
            cfg.max_sensor_step_I_norm,
        )
        return T_K, B_uT, I_norm

    def _features(self, error: float, env: dict[str, float] | None) -> np.ndarray:
        """Build normalized direct-control features and update sensor filters."""
        cfg = self.config.sensor_config
        env = env or {}
        T_K, B_uT, I_env = self._validated_env(env)
        alpha = min(1.0, self.config.control_dt_s / max(cfg.intensity_tau_s, 1.0e-9))
        prev_I_hat = self._I_hat
        self._I_hat += alpha * (I_env - self._I_hat)
        dT_dt = (T_K - self._last_T_K) / self.config.control_dt_s
        dB_dt = (B_uT - self._last_B_uT) / self.config.control_dt_s
        dI_dt = (self._I_hat - prev_I_hat) / self.config.control_dt_s
        self._last_T_K = T_K
        self._last_B_uT = B_uT

        err = float(error)
        self._err_integral = float(
            np.clip(self._err_integral + err * self.config.control_dt_s, -1.0e-2, 1.0e-2)
        )
        self._error_window.append(err)
        err_arr = np.asarray(self._error_window, dtype=np.float64)
        return np.array(
            [
                (T_K - cfg.T_nom_K) / 10.0,
                (B_uT - cfg.B_nom_uT) / 10.0,
                self._I_hat - cfg.I_nom,
                dT_dt / 10.0,
                dB_dt / 10.0,
                dI_dt,
                err * 1.0e3,
                (err - self._last_error) * 1.0e3,
                self._err_integral * 1.0e6,
                float(np.mean(err_arr)) * 1.0e3,
                float(np.std(err_arr)) * 1.0e3,
                self._last_rf / max(self.config.rf_limit_Hz, 1.0e-9),
                self._last_delta_rf / 10.0,
                1.0,
            ],
            dtype=np.float64,
        )

    def _cfc_step(self, features: np.ndarray) -> None:
        """Advance the closed-form recurrent cell."""
        w = self.weights
        cand = np.tanh(w["W_c"] @ features + w["U_c"] @ self._h + w["b_c"])
        tau_raw = w["W_tau"] @ features + w["U_tau"] @ self._h + w["b_tau"]
        tau = np.log1p(np.exp(np.clip(tau_raw, -40.0, 40.0))) + 1.0e-3
        gate = np.exp(-self.config.control_dt_s / tau)
        self._h = gate * self._h + (1.0 - gate) * cand

    def _readout(self, features: np.ndarray) -> float:
        """Return unclipped direct RF action from hidden and optional feature skip."""
        w = self.weights
        raw = float(w["W_out"] @ self._h + w["b_out"][0])
        if self.config.output_feature_skip:
            raw += float(w["W_feat_out"] @ features)
        return raw

    def _commit_action(self, error: float, rf: float) -> None:
        """Update action memory after a predicted or teacher-forced action."""
        rf = float(np.clip(rf, -self.config.rf_limit_Hz, self.config.rf_limit_Hz))
        self._last_delta_rf = rf - self._last_rf
        self._last_rf = rf
        self._last_error = float(error)

    def teacher_forced_design_row(
        self,
        error: float,
        env: dict[str, float] | None,
        teacher_rf: float,
    ) -> np.ndarray:
        """Advance with a teacher action and return the linear readout row."""
        features = self._features(error, env)
        self._cfc_step(features)
        row = np.concatenate([self._h, features, np.ones(1, dtype=np.float64)])
        self._commit_action(error, teacher_rf)
        return row

    def step(self, error: float, env: dict[str, float] | None = None) -> tuple[float, float]:
        """Controller protocol compatible with ``run_batched_loop``."""
        features = self._features(error, env)
        self._cfc_step(features)
        rf = float(
            np.clip(
                self._readout(features),
                -self.config.rf_limit_Hz,
                self.config.rf_limit_Hz,
            )
        )
        self._commit_action(error, rf)
        return 0.0, rf

    def save(self, path: str | Path) -> None:
        """Save config and NumPy weights as JSON."""
        payload = {
            "config": {
                "hidden_size": self.config.hidden_size,
                "seed": self.config.seed,
                "sensor_config": asdict(self.config.sensor_config),
                "rf_limit_Hz": self.config.rf_limit_Hz,
                "control_dt_s": self.config.control_dt_s,
                "output_feature_skip": self.config.output_feature_skip,
            },
            "weights": {k: v.tolist() for k, v in self.weights.items()},
        }
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> CfCDirectController:
        """Load a standalone CfC direct checkpoint."""
        payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        cfg_raw = payload["config"]
        cfg = CfCDirectConfig(
            hidden_size=int(cfg_raw["hidden_size"]),
            seed=int(cfg_raw["seed"]),
            sensor_config=PhysicsResidualConfig(**cfg_raw["sensor_config"]),
            rf_limit_Hz=float(cfg_raw["rf_limit_Hz"]),
            control_dt_s=float(cfg_raw["control_dt_s"]),
            output_feature_skip=bool(cfg_raw.get("output_feature_skip", True)),
        )
        weights = {
            key: np.asarray(value, dtype=np.float64)
            for key, value in payload["weights"].items()
        }
        return cls(cfg, weights=weights)
