"""Clean-twin reference oscillator for sigma_y ground truth.

``ReferenceOscillator`` wraps a ``ReducedTwin`` running under the ``clean``
disturbance scenario (constant T, B, I; no environmental drift).  In v1 the
reference is essentially "zero fractional-frequency error" — the measured
clock's fractional-frequency error already encodes the deviation from the true
frequency.

This class exists as a hook for M8's reality-gap test, where the reference
becomes the tier-1 OBE twin.  For now the clean twin IS the reference.

References
----------
Kitching, J. (2018). Chip-scale atomic devices.
    *Applied Physics Reviews*, 5, 031302.
"""

from __future__ import annotations

import torch
from torch import Tensor

from cptservo.twin.reduced import ReducedTwin


class ReferenceOscillator:
    """Clean-twin reference oscillator.

    Runs a ``ReducedTwin`` instance under zero-disturbance (clean) conditions.
    The fractional-frequency output is the "true" value against which a
    controlled clock is compared when computing sigma_y.

    In v1 the reference always returns y=0 (the clean twin has zero systematic
    offset by construction when controls=0 and all disturbances are at their
    nominal values).  This is a stub for the M8 tier-1 OBE upgrade.

    Args:
        twin_kwargs: Optional keyword arguments forwarded to ``ReducedTwin``.
            Defaults to the recipe nominal values.
    """

    def __init__(self, twin_kwargs: dict | None = None) -> None:
        """Initialise the reference oscillator.

        Args:
            twin_kwargs: Dict of keyword arguments for ``ReducedTwin.__init__``.
                If None, uses the recipe defaults.
        """
        kwargs = twin_kwargs or {}
        self._twin = ReducedTwin(**kwargs)
        self._state: Tensor | None = None

    # ---------------------------------------------------------------------------

    @property
    def twin(self) -> ReducedTwin:
        """The underlying ReducedTwin instance."""
        return self._twin

    # ---------------------------------------------------------------------------

    def reset(self, batch_size: int = 1) -> None:
        """Reset internal state to thermal equilibrium.

        Args:
            batch_size: Number of parallel trajectories.
        """
        self._state = self._twin.initial_state(batch_size)

    # ---------------------------------------------------------------------------

    def fractional_frequency_at(self, t: Tensor) -> Tensor:
        """Return clean-reference fractional-frequency trace (all zeros in v1).

        In v1 the reference oscillator is by definition on-frequency: the clean
        twin has zero systematic offset when controls=0 at nominal conditions.
        Returns a zero tensor matching the batch dimension of t.

        For M8, override this method to run the tier-1 OBE twin and return its
        fractional-frequency output.

        Args:
            t: Time tensor of shape (B,) or (B, N).  Used only to infer device,
               dtype, and batch shape.

        Returns:
            Tensor of zeros with the same shape and device as t.
        """
        return torch.zeros_like(t)
