"""Retry APG now that local PyTorch autograd works.

This is intentionally bounded for CPU execution:

1. Verify autograd with a tiny scalar backward.
2. Supervised-pretrain APGPolicy against the promoted direct-CfC teacher.
3. Fine-tune through the differentiable reduced twin with ``train_apg``.
4. Evaluate APG against DLQR and the current direct-CfC checkpoint on M5.

The output is a completed APG checkpoint, even if it does not win.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import torch

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_SCRIPT_DIR))

from audit_calibration import make_calibrated_twin  # noqa: E402

from cptservo.baselines.dlqr import DLQRController  # noqa: E402
from cptservo.evaluation.batched_runner import run_batched_loop  # noqa: E402
from cptservo.policy.apg import APGPolicy  # noqa: E402
from cptservo.policy.ml_research import CfCDirectController  # noqa: E402
from cptservo.policy.training import train_apg  # noqa: E402
from cptservo.twin.allan import overlapping_allan  # noqa: E402
from cptservo.twin.disturbance import Disturbance, DisturbanceTrace  # noqa: E402

PHYSICS_RATE_HZ = 10_000.0
DECIMATION_RATE_HZ = 1_000.0
DISC_NOISE_AMP_CI = 7.0e-4
RNG_SEED = 42
TAUS = [1.0, 10.0, 100.0]


def log(msg: str) -> None:
    """Timestamped console log."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _json_safe(value: Any) -> Any:
    """Convert NaN/Inf floats to null for strict JSON."""
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    return value


def _default_cfc_direct_checkpoint() -> str:
    summary_path = _PROJECT_ROOT / "data" / "cfc_direct_summary.json"
    if summary_path.exists():
        return str(json.loads(summary_path.read_text(encoding="utf-8-sig"))["checkpoint"])
    return str(
        _PROJECT_ROOT
        / "data"
        / "ml_research"
        / "cfc_direct_20260430_150420"
        / "cfc_direct.json"
    )


def _env_at(trace: DisturbanceTrace, k: int) -> dict[str, float]:
    decimation = int(round(PHYSICS_RATE_HZ / DECIMATION_RATE_HZ))
    idx = min((k + 1) * decimation - 1, len(trace.T_K) - 1)
    return {
        "T_K": float(trace.T_K[idx]),
        "B_uT": float(trace.B_uT[idx]),
        "I_norm": float(trace.laser_intensity_norm[idx]),
    }


def _run_teacher_errors(
    teacher: Any,
    trace: DisturbanceTrace,
    duration_s: float,
    rng_seed: int,
) -> np.ndarray:
    """Run teacher closed-loop and return noisy discriminator errors."""
    res = run_batched_loop(
        twin=make_calibrated_twin(),
        controller=teacher,
        disturbance_traces=[trace],
        duration_s=duration_s,
        physics_rate_Hz=PHYSICS_RATE_HZ,
        decimation_rate_Hz=DECIMATION_RATE_HZ,
        rng_seed=rng_seed,
        disc_noise_amp_ci=DISC_NOISE_AMP_CI,
        autograd=False,
    )
    return np.asarray(res["error_signal"][0], dtype=np.float64)


def collect_supervised_data(
    policy: APGPolicy,
    teacher: Any,
    scenarios: list[str],
    duration_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Collect APG observations and teacher RF actions."""
    obs_rows: list[list[float]] = []
    targets: list[float] = []
    for idx, scenario in enumerate(scenarios):
        trace = Disturbance.from_recipe(scenario).generate(
            duration_s=duration_s,
            sample_rate_Hz=PHYSICS_RATE_HZ,
            seed=RNG_SEED + idx,
        )
        if hasattr(teacher, "reset"):
            teacher.reset()
        errors = _run_teacher_errors(teacher, trace, duration_s, RNG_SEED + idx)
        if hasattr(teacher, "reset"):
            teacher.reset()

        err_window: deque[float] = deque(
            [0.0] * policy.n_error_history,
            maxlen=policy.n_error_history,
        )
        rf_window: deque[float] = deque(
            [0.0] * policy.n_rf_history,
            maxlen=policy.n_rf_history,
        )
        for k, error in enumerate(errors):
            env = _env_at(trace, k)
            err_window.append(float(error))
            obs = list(err_window) + list(rf_window)
            if policy.include_env_sensors:
                obs += [env["T_K"], env["B_uT"], env["I_norm"]]
            _, rf = teacher.step(float(error), env)
            obs_rows.append(obs)
            targets.append(float(rf))
            rf_window.append(float(rf))

    return np.asarray(obs_rows, dtype=np.float32), np.asarray(targets, dtype=np.float32)


def fit_obs_stats(policy: APGPolicy, obs: np.ndarray) -> None:
    """Set policy normalization buffers from supervised observations."""
    obs_t = torch.from_numpy(obs)
    policy.obs_mean.copy_(obs_t.mean(dim=0))
    policy.obs_std.copy_(obs_t.std(dim=0).clamp(min=1.0e-8))


def supervised_pretrain(
    policy: APGPolicy,
    obs: np.ndarray,
    target_rf: np.ndarray,
    epochs: int,
    batch_size: int,
    lr: float,
) -> dict[str, Any]:
    """Behavior-clone APGPolicy to teacher RF actions."""
    fit_obs_stats(policy, obs)
    obs_t = torch.from_numpy(obs)
    target_t = torch.from_numpy(target_rf)
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    rng = np.random.default_rng(RNG_SEED)
    losses: list[float] = []
    for epoch in range(epochs):
        perm = rng.permutation(len(obs))
        batch_losses: list[float] = []
        for start in range(0, len(obs), batch_size):
            idx = torch.from_numpy(perm[start : start + batch_size])
            pred = policy(obs_t[idx])
            loss = torch.mean((pred.float() - target_t[idx]) ** 2)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step()
            batch_losses.append(float(loss.item()))
        epoch_loss = float(np.mean(batch_losses))
        losses.append(epoch_loss)
        if epoch == 0 or (epoch + 1) % max(1, epochs // 5) == 0:
            log(f"supervised epoch {epoch + 1}/{epochs} mse_Hz2={epoch_loss:.6g}")
    return {
        "epochs": epochs,
        "samples": int(len(obs)),
        "initial_mse_Hz2": losses[0] if losses else None,
        "final_mse_Hz2": losses[-1] if losses else None,
        "loss_history": losses,
    }


def evaluate_controller(controller: Any, duration_s: float) -> dict[str, float]:
    """Evaluate controller on M5 thermal ramp."""
    trace = Disturbance.from_recipe("thermal_ramp").generate(
        duration_s=duration_s,
        sample_rate_Hz=PHYSICS_RATE_HZ,
        seed=RNG_SEED,
    )
    if hasattr(controller, "reset"):
        controller.reset()
    res = run_batched_loop(
        twin=make_calibrated_twin(),
        controller=controller,
        disturbance_traces=[trace],
        duration_s=duration_s,
        physics_rate_Hz=PHYSICS_RATE_HZ,
        decimation_rate_Hz=DECIMATION_RATE_HZ,
        rng_seed=RNG_SEED,
        disc_noise_amp_ci=DISC_NOISE_AMP_CI,
        autograd=False,
    )
    y = res["y"][0] - float(np.mean(res["y"][0]))
    allan = overlapping_allan(y, DECIMATION_RATE_HZ, TAUS)
    return {
        "sigma_y_1s": float(allan.get(1.0, float("nan"))),
        "sigma_y_10s": float(allan.get(10.0, float("nan"))),
        "sigma_y_100s": float(allan.get(100.0, float("nan"))),
        "rf_abs_max_Hz": float(np.max(np.abs(res["rf_cmd"][0]))),
        "wall_s": float(res["wall_s"]),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Run APG retry."""
    x = torch.tensor(2.0, requires_grad=True)
    (x * x).backward()
    log(f"autograd smoke grad={x.grad.item():.3f}")

    run_dir = _PROJECT_ROOT / "data" / "ml_research" / time.strftime("apg_retry_%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    models_dir = _PROJECT_ROOT / "models"
    models_dir.mkdir(exist_ok=True)

    policy = APGPolicy(
        n_error_history=8,
        n_rf_history=4,
        include_env_sensors=True,
        hidden_dims=tuple(args.hidden_dims),
        rf_limit_Hz=1000.0,
    )
    teacher_path = args.teacher_checkpoint or _default_cfc_direct_checkpoint()
    teacher = CfCDirectController.load(teacher_path)
    scenarios = [item.strip() for item in args.supervised_scenarios.split(",") if item.strip()]
    log(f"collecting supervised data scenarios={scenarios} duration={args.supervised_duration_s}s")
    obs, target_rf = collect_supervised_data(policy, teacher, scenarios, args.supervised_duration_s)
    sup = supervised_pretrain(
        policy,
        obs,
        target_rf,
        epochs=args.supervised_epochs,
        batch_size=args.batch_size,
        lr=args.supervised_lr,
    )

    fine_tune_metrics: list[dict[str, Any]] = []
    if args.finetune_epochs > 0:
        twin = make_calibrated_twin()
        finetune_scenarios = [
            item.strip() for item in args.finetune_scenarios.split(",") if item.strip()
        ]
        for scenario in finetune_scenarios:
            log(f"fine-tune scenario={scenario} epochs={args.finetune_epochs}")
            metrics = train_apg(
                policy=policy,
                twin=twin,
                disturbance_recipe_name=scenario,
                n_rollouts=args.finetune_rollouts,
                rollout_duration_s=args.finetune_duration_s,
                truncation_window=args.truncation_window,
                n_epochs=args.finetune_epochs,
                lr=args.finetune_lr,
                grad_clip=1.0,
                disc_noise_amp_ci=DISC_NOISE_AMP_CI,
                lo_noise_scale=1.0,
                rng_seed=RNG_SEED + len(fine_tune_metrics) * 100,
                log_every=1,
                checkpoint_dir=run_dir / "checkpoints",
                checkpoint_every=max(1, args.finetune_epochs),
                verbose=True,
            )
            fine_tune_metrics.append(metrics)

    ckpt_path = models_dir / "apg_autograd_retry.pt"
    policy.save(str(ckpt_path))
    metrics: dict[str, Any] = {
        "controller": "apg_autograd_retry",
        "checkpoint": str(ckpt_path),
        "teacher_checkpoint": teacher_path,
        "supervised": sup,
        "fine_tune": fine_tune_metrics,
        "config": {
            "hidden_dims": list(args.hidden_dims),
            "supervised_scenarios": scenarios,
            "supervised_duration_s": args.supervised_duration_s,
            "finetune_scenarios": args.finetune_scenarios,
            "finetune_duration_s": args.finetune_duration_s,
        },
    }
    if args.eval_duration_s > 0:
        metrics["eval_duration_s"] = args.eval_duration_s
        metrics["dlqr"] = evaluate_controller(DLQRController.from_recipe(), args.eval_duration_s)
        metrics["teacher_cfc_direct"] = evaluate_controller(
            CfCDirectController.load(teacher_path),
            args.eval_duration_s,
        )
        metrics["apg"] = evaluate_controller(APGPolicy.load(str(ckpt_path)), args.eval_duration_s)
        metrics["apg_over_rhlqr_10s"] = (
            metrics["apg"]["sigma_y_10s"] / metrics["dlqr"]["sigma_y_10s"]
            if metrics["dlqr"]["sigma_y_10s"] > 0 else float("nan")
        )

    (run_dir / "metrics.json").write_text(
        json.dumps(_json_safe(metrics), indent=2),
        encoding="utf-8",
    )
    (_PROJECT_ROOT / "data" / "apg_autograd_retry_summary.json").write_text(
        json.dumps(_json_safe(metrics), indent=2),
        encoding="utf-8",
    )
    log(f"wrote {ckpt_path}")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-checkpoint", default="")
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[64, 64, 32])
    parser.add_argument("--supervised-scenarios", default="thermal_ramp")
    parser.add_argument("--supervised-duration-s", type=float, default=25.0)
    parser.add_argument("--supervised-epochs", type=int, default=60)
    parser.add_argument("--supervised-lr", type=float, default=1.0e-3)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--finetune-scenarios", default="thermal_ramp")
    parser.add_argument("--finetune-epochs", type=int, default=3)
    parser.add_argument("--finetune-rollouts", type=int, default=1)
    parser.add_argument("--finetune-duration-s", type=float, default=0.2)
    parser.add_argument("--finetune-lr", type=float, default=1.0e-4)
    parser.add_argument("--truncation-window", type=int, default=50)
    parser.add_argument("--eval-duration-s", type=float, default=25.0)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
