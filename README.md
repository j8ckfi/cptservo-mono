# cptservo-mono

Closed-loop control for a chip-scale CPT-Rb87 atomic clock. The repo bundles
two things:

1. A **calibrated digital twin** of the clock (8-level optical Bloch model +
   reduced surrogate).
2. A **methods bench** that runs PI, DLQR, APG, PPO, and a closed-form
   continuous-time (CfC) recurrent controller on the same evaluator.

The novel piece is the CfC controller.

## Install

```bash
pip install -e .[dev]
```

Python >= 3.10. Pulls in PyTorch and `stable-baselines3`.

## Quick start: run the headline gate

```bash
python scripts/run_m11_gate.py --duration-s 100
```

That evaluates the shipped CfC checkpoint
(`data/cfc_direct_T1p5_kp1p5_ki1p0.json`) against DLQR on the M5 thermal-ramp
benchmark plus four robustness probes, and writes `data/gate_M11.json`.

## Headline number

| Controller | sigma_y(tau=10 s) on `thermal_ramp` | vs DLQR |
|---|---:|---:|
| PI baseline | 5.861e-11 | 11.55x worse |
| DLQR (steady-state, 2-state) | 5.074e-12 | 1.0000 |
| **CfC direct (k_T=1.5, kp=1.5, ki=1.0)** | **4.913e-12** | **0.9682** |

3.18% nominal win for the CfC over DLQR. 4/5 robustness probes tie or win;
loses `ood_3x_thermal_slope` by 0.62% (just past the 0.5% tie band), so the
strict 5/5 promotion gate returns `do_not_promote`.

## Train a new CfC

```bash
# Distill from a DLQR teacher (no autograd):
python scripts/cfc_direct_train.py

# Optional autograd refinement of the recurrent matrices:
python scripts/cfc_autograd_train.py

# Sweep (k_T, kp_scale, k_dT) — the headline came from this:
python scripts/cfc_improvement_campaign.py \
    --screen-duration-s 25 --max-specs 32 --batch-size 32
```

## Other gates

| Script | What it does |
|---|---|
| `scripts/compute_m2_surface.py` | Recompute the OBE surface + tier-2 calibration |
| `scripts/run_m5_gate.py` | DLQR vs PI head-to-head (M5) |
| `scripts/m8_adversarial.py` | Adversarial robustness battery (classical track) |
| `scripts/ml_m8_gate.py` | Same battery for an ML controller checkpoint |
| `scripts/m6_apg_train.py` / `m6_apg_gate.py` | APG attempt (documented partial) |
| `scripts/m7_ppo_train.py` / `m7_ppo_gate.py` | PPO attempt (~40x behind PI) |
| `scripts/lnn_m11_campaign.py` | Alpha-beta / liquid-observer sweep |

Each gate writes `data/gate_M{N}.json`.

## Tests

```bash
pytest tests/ -v
ruff check src/ scripts/
```

## Layout

```text
src/cptservo/
  twin/          tier-1 OBE + tier-2 reduced twin
  baselines/     PI and DLQR controllers
  policy/        CfC, APG, PPO, ML controller utilities
  evaluation/    closed-loop and batched runners
  calibration/   tier-2 fit helpers
src/rbspec/      vendored Rb-87 constants
scripts/         gate drivers, training, sweeps
tests/           pytest suite
configs/         YAML run recipes
data/            calibration fixtures + the shipped CfC checkpoint
```

`data/` ships only what the gates need to run: `obe_surface.h5`,
`reduced_calibration.json`, `published_allan.json`, the pinned `gate_M5.json`
reference, and the promoted CfC checkpoint. All other `gate_M*.json` files are
generated.

## License

MIT — see [LICENSE](LICENSE).
