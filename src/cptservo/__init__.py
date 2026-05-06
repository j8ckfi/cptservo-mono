"""CPTServo: digital-twin servo benchmarks for a chip-scale CPT-Rb87 clock.

The completed positive control result is a PI-vs-RH-LQR/DLQR benchmark. The
learned-control paths include documented APG/PPO negatives plus a direct-CfC
follow-on that improves nominal M5 but is not promoted by the M11 robustness
gate.
"""

from __future__ import annotations

__version__ = "0.1.0"
