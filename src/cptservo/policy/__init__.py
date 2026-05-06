"""Learned servo policies.

``apg.py`` is the primary training path: differentiable-physics analytic policy
gradient via truncated BPTT through the tier-2 twin.

``ppo.py`` is the sanity baseline via stable_baselines3.

``ml_research.py`` contains the canonical learned-controller wrapper used by
the autoresearch loop for standalone and RH-LQR-residual ML policies.
"""

from __future__ import annotations

from cptservo.policy.ml_research import (
    CfCDirectConfig,
    CfCDirectController,
    CfCFeedforwardConfig,
    CfCFeedforwardController,
    MLControllerConfig,
    MLServoController,
    PhysicsResidualConfig,
    PhysicsResidualController,
)

__all__ = [
    "CfCDirectConfig",
    "CfCDirectController",
    "CfCFeedforwardConfig",
    "CfCFeedforwardController",
    "MLControllerConfig",
    "MLServoController",
    "PhysicsResidualConfig",
    "PhysicsResidualController",
]
