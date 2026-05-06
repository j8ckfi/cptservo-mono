"""Disturbance generators for CPT-clock simulation scenarios.

Five named scenarios from ``configs/v1_recipe.yaml`` (Layer-0 frozen artifact):

- ``clean``                 — constant T, B, I; no disturbance.
- ``thermal_ramp``          — slow sinusoidal temperature drift + white noise.
- ``b_field_drift``         — sinusoidal magnetic-field drift + white noise.
- ``laser_intensity_drift`` — sinusoidal laser-intensity drift + white noise.
- ``all_stacked``           — all three disturbances simultaneously.

Each scenario is instantiated via ``Disturbance.from_recipe(name)`` and
generates reproducible traces with ``numpy.random.default_rng(seed)``.

References
----------
configs/v1_recipe.yaml — Layer-0 frozen artifact for CPTServo.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import numpy as np
import yaml

# ---------------------------------------------------------------------------
# Locate the recipe file relative to the package install
# ---------------------------------------------------------------------------
_PACKAGE_ROOT = Path(__file__).parent.parent.parent.parent  # WIP/CPTServo/
_DEFAULT_RECIPE_PATH = _PACKAGE_ROOT / "configs" / "v1_recipe.yaml"


def _load_recipe(path: str | Path | None = None) -> dict:
    rp = Path(path) if path is not None else _DEFAULT_RECIPE_PATH
    with open(rp) as fh:
        return yaml.safe_load(fh)


# ---------------------------------------------------------------------------
# DisturbanceTrace dataclass
# ---------------------------------------------------------------------------


@dataclass
class DisturbanceTrace:
    """Container for a generated disturbance time series.

    Attributes:
        T_K: (N,) array of cell temperature samples (Kelvin).
        B_uT: (N,) array of magnetic field samples (µT).
        laser_intensity_norm: (N,) array of normalised laser intensity samples.
        sample_rate_Hz: Sample rate of the trace (Hz).
        duration_s: Duration of the trace (s).
    """

    T_K: np.ndarray
    B_uT: np.ndarray
    laser_intensity_norm: np.ndarray
    sample_rate_Hz: float
    duration_s: float

    def __post_init__(self) -> None:
        N_expected = int(round(self.duration_s * self.sample_rate_Hz))
        # Allow ±1 sample for rounding
        assert abs(len(self.T_K) - N_expected) <= 1, (
            f"T_K length {len(self.T_K)} does not match "
            f"duration_s={self.duration_s} * sample_rate_Hz={self.sample_rate_Hz} "
            f"= {N_expected}"
        )


# ---------------------------------------------------------------------------
# Disturbance class
# ---------------------------------------------------------------------------


class Disturbance:
    """Disturbance generator for a named scenario from the recipe YAML.

    Instantiate via the class method ``from_recipe``::

        d = Disturbance.from_recipe("thermal_ramp")
        trace = d.generate(duration_s=100.0, seed=99)
    """

    _SCENARIO_NAMES: ClassVar[tuple[str, ...]] = (
        "clean",
        "thermal_ramp",
        "b_field_drift",
        "laser_intensity_drift",
        "all_stacked",
    )

    # ---------------------------------------------------------------------------

    def __init__(self, scenario_name: str, params: dict) -> None:
        """Internal constructor.  Use ``from_recipe`` instead.

        Args:
            scenario_name: One of the five named scenarios.
            params: Parameter dict from the recipe YAML for this scenario.
        """
        self.name = scenario_name
        self.params = params

    # ---------------------------------------------------------------------------

    @classmethod
    def from_recipe(
        cls,
        recipe_name: str,
        recipe_path: str | Path | None = None,
    ) -> Disturbance:
        """Load a named disturbance scenario from configs/v1_recipe.yaml.

        Args:
            recipe_name: One of ``clean``, ``thermal_ramp``, ``b_field_drift``,
                ``laser_intensity_drift``, ``all_stacked``.
            recipe_path: Optional path to the recipe YAML.  Defaults to
                ``configs/v1_recipe.yaml`` relative to the package root.

        Returns:
            Disturbance instance configured for the named scenario.

        Raises:
            KeyError: If ``recipe_name`` is not found in the recipe.
            ValueError: If ``recipe_name`` is not one of the known scenarios.
        """
        if recipe_name not in cls._SCENARIO_NAMES:
            raise ValueError(
                f"Unknown scenario '{recipe_name}'. "
                f"Choose from {cls._SCENARIO_NAMES}."
            )
        recipe = _load_recipe(recipe_path)
        params = recipe["disturbance_recipes"][recipe_name]
        return cls(recipe_name, params)

    # ---------------------------------------------------------------------------

    def generate(
        self,
        duration_s: float | None = None,
        sample_rate_Hz: float | None = None,
        seed: int | None = None,
    ) -> DisturbanceTrace:
        """Generate a disturbance trace for this scenario.

        Args:
            duration_s: Trace duration in seconds.  Defaults to the recipe value.
            sample_rate_Hz: Sample rate in Hz.  Defaults to the recipe value.
            seed: RNG seed for reproducibility.  Defaults to the recipe seed.

        Returns:
            DisturbanceTrace with arrays T_K, B_uT, laser_intensity_norm and
            metadata.
        """
        p = self.params
        dur = float(duration_s) if duration_s is not None else float(p["duration_s"])
        sr = float(sample_rate_Hz) if sample_rate_Hz is not None else float(p["sample_rate_Hz"])
        rng_seed = int(seed) if seed is not None else int(p["seed"])

        N = int(round(dur * sr))
        t = np.arange(N) / sr  # time vector (s)
        rng = np.random.default_rng(rng_seed)

        # ---------------------------------------------------------------------------
        # Temperature
        # ---------------------------------------------------------------------------
        if self.name == "clean":
            T_K = np.full(N, float(p["T_K_constant"]))
        elif self.name == "thermal_ramp":
            _amp = float(p["T_K_ramp_amplitude_K"])
            _per = float(p["T_K_ramp_period_s"])
            T_K = (
                float(p["T_K_baseline"])
                + _amp * np.sin(2.0 * np.pi * t / _per)
                + float(p["T_K_white_noise_K"]) * rng.standard_normal(N)
            )
        elif self.name in ("b_field_drift", "laser_intensity_drift"):
            T_K = np.full(N, float(p["T_K_constant"]))
        else:  # all_stacked
            _amp = float(p["T_K_ramp_amplitude_K"])
            _per = float(p["T_K_ramp_period_s"])
            T_K = (
                float(p["T_K_baseline"])
                + _amp * np.sin(2.0 * np.pi * t / _per)
                + float(p["T_K_white_noise_K"]) * rng.standard_normal(N)
            )

        # ---------------------------------------------------------------------------
        # Magnetic field
        # ---------------------------------------------------------------------------
        if self.name in ("clean", "thermal_ramp", "laser_intensity_drift"):
            B_uT = np.full(N, float(p["B_uT_constant"]))
        elif self.name == "b_field_drift":
            _amp_B = float(p["B_uT_drift_amplitude_uT"])
            _per_B = float(p["B_uT_drift_period_s"])
            B_uT = (
                float(p["B_uT_baseline"])
                + _amp_B * np.sin(2.0 * np.pi * t / _per_B)
                + float(p["B_uT_white_noise_uT"]) * rng.standard_normal(N)
            )
        else:  # all_stacked
            _amp_B = float(p["B_uT_drift_amplitude_uT"])
            _per_B = float(p["B_uT_drift_period_s"])
            B_uT = (
                float(p["B_uT_baseline"])
                + _amp_B * np.sin(2.0 * np.pi * t / _per_B)
                + float(p["B_uT_white_noise_uT"]) * rng.standard_normal(N)
            )

        # ---------------------------------------------------------------------------
        # Laser intensity
        # ---------------------------------------------------------------------------
        if self.name in ("clean", "thermal_ramp", "b_field_drift"):
            laser = np.full(N, float(p["laser_intensity_norm_constant"]))
        elif self.name == "laser_intensity_drift":
            laser = (
                float(p["laser_intensity_baseline"])
                + float(p["laser_intensity_drift_amplitude"])
                * np.sin(2.0 * np.pi * t / float(p["laser_intensity_drift_period_s"]))
                + float(p["laser_intensity_white_noise"]) * rng.standard_normal(N)
            ).clip(min=0.01)
        else:  # all_stacked
            laser = (
                float(p["laser_intensity_baseline"])
                + float(p["laser_intensity_drift_amplitude"])
                * np.sin(2.0 * np.pi * t / float(p["laser_intensity_drift_period_s"]))
                + float(p["laser_intensity_white_noise"]) * rng.standard_normal(N)
            ).clip(min=0.01)

        return DisturbanceTrace(
            T_K=T_K,
            B_uT=B_uT,
            laser_intensity_norm=laser,
            sample_rate_Hz=sr,
            duration_s=dur,
        )
