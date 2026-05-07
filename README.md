# cptservo-mono

A monorepo for closed-loop control of a chip-scale CPT-Rb87 atomic clock. Two
things stacked in the same package:

1. **The digital twin.** A first-principles 8-level optical Bloch model of
   Rb-87 under coherent population trapping, calibrated against published
   sigma_y curves and reduced to a fast tier-2 surrogate suitable for
   closed-loop training.
2. **The methods.** A controller bench that puts a hand-tuned PI loop, a
   steady-state DLQR (kept under the historical name "RH-LQR"), an analytic
   policy gradient (APG), PPO, and a closed-form continuous-time (CfC)
   recurrent network on the same evaluation harness, with an adversarial
   robustness battery.

## Headline result: a CfC controller beats the classical optimum

The novel component is a **closed-form continuous-time (CfC)** recurrent
controller — a structured liquid-time-constant network with eight hidden cells
— trained to replace the classical DLQR in the actuator path. On the unified
100 s `thermal_ramp` benchmark:

| Controller | sigma_y(tau=10 s) | Ratio vs RH-LQR |
|---|---:|---:|
| PI baseline | 5.861e-11 | 11.55x worse |
| RH-LQR / DLQR | 5.074e-12 | 1.0000 |
| **CfC direct (k_T=1.5, kp=1.5, ki=1.0)** | **4.913e-12** | **0.9682** |

The CfC controller is **3.18% quieter than the classical DLQR** on the standard
benchmark — roughly four times the previous best learned-controller win in
this project, and an order of magnitude bigger than the linear-residual
baseline.

## Where it stops

The same checkpoint loses one of five robustness probes by a thin margin:
the `ood_3x_thermal_slope` perturbation lands at ratio 1.0062, just outside the
0.5% tie band. The other four probes still tie or win against RH-LQR. The
project's strict promotion rule requires 5/5 robustness, so the M11 gate
returns `do_not_promote` despite the strongest nominal learned-controller win
on file.

The trade is intentional: the CfC's stronger temperature feedforward (1.5 Hz
per Kelvin, three times the previous setting) is what buys the nominal win,
and it is also what the OOD thermal slope probe is designed to expose. A
matching alpha-beta thermal observer with two parameters reaches the same
ceiling at 4/5 robust + 0.99% nominal, which suggests the boundary is
properties of the disturbance set, not architecture.

## Reproduce the headline

```bash
pip install -e .[dev]

# 1. Calibrate the twin (tier-1 OBE + tier-2 fit). Skipped if you keep the
#    shipped data/obe_surface.h5 + data/reduced_calibration.json fixtures.
python scripts/compute_m2_surface.py

# 2. Run the M11 promotion gate against the shipped CfC checkpoint.
python scripts/run_m11_gate.py --duration-s 100
```

`scripts/run_m11_gate.py` writes `data/gate_M11.json` with the per-probe
ratios, the M5 head-to-head, and the promotion verdict. The shipped checkpoint
`data/cfc_direct_T1p5_kp1p5_ki1p0.json` is the one referenced in the table
above.

## Train a fresh CfC controller

```bash
# Structured CfC distilled from an RH-LQR teacher (no autograd):
python scripts/cfc_direct_train.py

# Optional autograd refinement of the recurrent matrices:
python scripts/cfc_autograd_train.py

# Sweep the three knobs (k_T, kp_scale, k_dT) the headline came from:
python scripts/cfc_improvement_campaign.py \
    --screen-duration-s 25 --max-specs 32 --batch-size 32
```

## Other gates and methods

```bash
# M5: RH-LQR/DLQR head-to-head vs PI
python scripts/run_m5_gate.py

# M8: adversarial robustness battery
python scripts/m8_adversarial.py

# M6 APG (analytic policy gradient): code, gradients, smoke test
python scripts/m6_apg_train.py
python scripts/m6_apg_gate.py

# M7 PPO: trains but underfits PI by ~40x at tau=10s
python scripts/m7_ppo_train.py
python scripts/m7_ppo_gate.py

# Simple alpha-beta and liquid-observer residual sweep
python scripts/lnn_m11_campaign.py
```

## Test suite

```bash
pytest tests/ -v
ruff check src/ scripts/
```

## Repository layout

```text
.
├── src/
│   ├── cptservo/
│   │   ├── twin/          # tier-1 OBE + tier-2 reduced twin
│   │   ├── baselines/     # PI and RH-LQR/DLQR controllers
│   │   ├── policy/        # CfC, APG, PPO, ML controller utilities
│   │   ├── evaluation/    # closed-loop and batched runners
│   │   └── calibration/   # tier-2 fit helpers
│   └── rbspec/            # vendored Rb-87 constants
├── scripts/
│   ├── compute_m2_surface.py     # M2 calibration
│   ├── run_m{3..8,11}_gate.py    # gate drivers (some named m{N}_*_gate.py)
│   ├── m{6,7}_*_train.py         # APG and PPO training
│   ├── cfc_*_train.py            # CfC training paths
│   ├── cfc_improvement_campaign.py  # CfC hyperparameter sweep
│   ├── lnn_m11_campaign.py       # alpha-beta / liquid observer sweep
│   ├── ml_m8_gate.py             # M8 adversarial gate for ML controllers
│   └── m8_adversarial.py         # M8 adversarial gate for classical track
├── tests/                 # pytest suite
├── configs/               # YAML run recipes
├── data/                  # calibration fixtures + the shipped CfC checkpoint
├── pyproject.toml
└── requirements.txt
```

`data/` ships only the calibration fixtures (`obe_surface.h5`,
`reduced_calibration.json`, `published_allan.json`) and the promoted CfC
checkpoint (`cfc_direct_T1p5_kp1p5_ki1p0.json`). All `gate_M*.json` files are
generated by the gate scripts.

## Calibration anchors

- Kitching, *Applied Physics Reviews* 5, 031302 (2018)
- Knappe et al., *Optics Letters* 29(7), 695 (2004)
- Knappe et al., *Applied Physics Letters* 86, 154102 (2005)
- Microsemi SA.45s CSAC datasheet sigma_y curves
- Vanier and Mandache, *Applied Physics B* 87, 565 (2007)

The CfC architecture follows the closed-form continuous-time formulation of
Hasani et al., *Nature Machine Intelligence* 2022, applied here as a structured
recurrent controller with a CfC cell of width 8 in place of the classical
two-state DLQR.

## License

MIT. See [LICENSE](LICENSE).
