# CPTServo

A calibrated digital twin and servo benchmark for a chip-scale CPT-Rb87 atomic
clock. The project compares a hand-tuned PI loop, a steady-state DLQR controller
(kept under the historical name "RH-LQR"), and learned-control attempts against
named disturbance recipes, with an adversarial robustness battery validating
the headline result.

## Headline result

On the unified 100 s `thermal_ramp` benchmark:

| Controller | sigma_y(tau=10 s) | Speedup vs PI |
|---|---:|---:|
| PI baseline | 5.861e-11 | 1.00x |
| RH-LQR / DLQR | 5.074e-12 | **11.55x** |

The adversarial battery does not preserve the exact 11.55x magnitude, but the
RH-LQR controller stays ahead of PI on 5/5 perturbation probes, with speedups
from 1.77x to 36.71x including a 2.97x reality-gap win.

A learned direct-CfC controller reaches a 0.9682 ratio versus RH-LQR on the
nominal benchmark but fails the stricter 5/5 robustness requirement at 4/5
probes, so it does not displace the classical RH-LQR headline.

## Milestone summary

| Milestone | Result |
|---|---|
| M1 | Reduced twin v0 + literature scan |
| M2 | Tier-1 OBE surface + tier-2 calibration; residual RMS 0.22 Hz |
| M3 | Kitching calibration ratios 0.84 / 0.75 / 0.74 across tau in {1, 10, 100} s |
| M4 | PI noise-floor gate failed; superseded by M3/M5 evidence |
| M5 | RH-LQR/DLQR beats PI by 11.55x on `thermal_ramp` at tau=10 s |
| M6 | APG infrastructure + smoke tests; full APG training did not converge |
| M7 | PPO infrastructure works but underfits PI by ~40x at tau=10 s |
| M8 | Adversarial battery preserves a positive RH-LQR win on 5/5 probes |

Failures and partials are intentionally kept as negative results.

## Install

```bash
pip install -e .[dev]
```

Requires Python >= 3.10. PyTorch and `stable-baselines3` are pulled in for the
APG and PPO training paths.

## Run the test suite

```bash
pytest tests/ -v
ruff check src/ scripts/
```

## Reproduce the gates

```bash
# M1: literature scan + reduced twin smoke
python scripts/measure_m1_gate.py

# M2: tier-1 OBE surface + tier-2 calibration
python scripts/compute_m2_surface.py

# M3 + M4: calibration audit + original PI gate
python scripts/run_m3_m4_gates.py

# M5: RH-LQR/DLQR head-to-head vs PI
python scripts/run_m5_gate.py

# M6: APG training + gate
modal run modal_apg_train.py::train      # or run scripts/m6_apg_train.py locally
python scripts/m6_apg_gate.py

# M7: PPO training + gate
python scripts/m7_ppo_train.py
python scripts/m7_ppo_gate.py

# M8: adversarial robustness battery
python scripts/m8_adversarial.py
```

Each gate writes `data/gate_M{N}.json` with metrics, thresholds, and verdict.

## Repository layout

```text
.
├── src/
│   ├── cptservo/
│   │   ├── twin/          # tier-1 OBE + tier-2 reduced twin
│   │   ├── baselines/     # PI and RH-LQR/DLQR controllers
│   │   ├── policy/        # APG policy, PPO env, training helpers
│   │   ├── evaluation/    # closed-loop and batched runners
│   │   └── calibration/   # tier-2 fit helpers
│   └── rbspec/            # vendored Rb-87 constants
├── scripts/               # M1-M8 gate drivers and training scripts
├── tests/                 # pytest unit + integration tests
├── configs/               # YAML run recipes
├── data/                  # calibration fixtures (obe_surface.h5,
│                          # reduced_calibration.json, published_allan.json);
│                          # gate_M*.json files are generated
├── pyproject.toml
└── requirements.txt
```

`data/obe_surface.h5` and `data/reduced_calibration.json` are calibration
fixtures shipped so the rest of the gates can run without first re-deriving the
twin. To re-derive them, run `scripts/compute_m2_surface.py`.

## Calibration anchors

- Kitching, *Applied Physics Reviews* 5, 031302 (2018)
- Knappe et al., *Optics Letters* 29(7), 695 (2004)
- Knappe et al., *Applied Physics Letters* 86, 154102 (2005)
- Microsemi SA.45s CSAC datasheet sigma_y curves
- Vanier and Mandache, *Applied Physics B* 87, 565 (2007)

## License

MIT. See [LICENSE](LICENSE).
