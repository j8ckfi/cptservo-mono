"""Two-tier digital twin of a chip-scale CPT-Rb87 atomic clock.

Tier-1 (offline calibration only): full 16-level Lindblad master equation in
QuTiP — see ``full_obe.py``.

Tier-2 (in-the-loop): 8-level adiabatic-elimination reduced model in PyTorch,
autograd-compatible — see ``reduced.py``.

Helpers: ``lockin.py`` (FM modulation + lock-in demod), ``allan.py``
(overlapping Allan deviation estimator), ``disturbance.py`` (per-scenario
generators), ``reference.py`` (clean-twin reference oscillator for sigma_y
computation).
"""

from __future__ import annotations
