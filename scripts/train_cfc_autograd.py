"""End-to-end autograd training for the CfC direct recurrent dynamics.

The earlier CfC direct checkpoint used a fixed recurrent feature map and a
structured/readout solution.  This script makes the CfC matrices trainable under
Torch autograd, then exports back to the small JSON checkpoint format consumed
by ``CfCDirectController``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_SCRIPT_DIR))

from train_apg_autograd import _env_at, _run_teacher_errors, evaluate_controller  # noqa: E402
from train_cfc_direct import initialize_structured_direct  # noqa: E402

from cptservo.baselines.dlqr import DLQRController  # noqa: E402
from cptservo.policy.ml_research import (  # noqa: E402
    CfCDirectConfig,
    CfCDirectController,
    PhysicsResidualConfig,
)
from cptservo.twin.disturbance import Disturbance  # noqa: E402

PHYSICS_RATE_HZ = 10_000.0
DECIMATION_RATE_HZ = 1_000.0
RNG_SEED = 42


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


class TorchCfCDirect(nn.Module):
    """Torch trainable version of ``CfCDirectController``."""

    def __init__(self, source: CfCDirectController) -> None:
        super().__init__()
        self.config = source.config
        self.W_c = nn.Parameter(torch.tensor(source.weights["W_c"], dtype=torch.float32))
        self.U_c = nn.Parameter(torch.tensor(source.weights["U_c"], dtype=torch.float32))
        self.b_c = nn.Parameter(torch.tensor(source.weights["b_c"], dtype=torch.float32))
        self.W_tau = nn.Parameter(torch.tensor(source.weights["W_tau"], dtype=torch.float32))
        self.U_tau = nn.Parameter(torch.tensor(source.weights["U_tau"], dtype=torch.float32))
        self.b_tau = nn.Parameter(torch.tensor(source.weights["b_tau"], dtype=torch.float32))
        self.W_out = nn.Parameter(torch.tensor(source.weights["W_out"], dtype=torch.float32))
        self.W_feat_out = nn.Parameter(
            torch.tensor(source.weights["W_feat_out"], dtype=torch.float32)
        )
        self.b_out = nn.Parameter(torch.tensor(source.weights["b_out"], dtype=torch.float32))

    def forward_chunk(
        self,
        features: torch.Tensor,
        h0: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run one feature chunk, returning RF predictions and final hidden."""
        h = h0
        outputs: list[torch.Tensor] = []
        dt = float(self.config.control_dt_s)
        for x in features:
            cand = torch.tanh(self.W_c @ x + self.U_c @ h + self.b_c)
            tau_raw = self.W_tau @ x + self.U_tau @ h + self.b_tau
            tau = torch.nn.functional.softplus(torch.clamp(tau_raw, -40.0, 40.0)) + 1.0e-3
            gate = torch.exp(torch.tensor(-dt, dtype=x.dtype) / tau)
            h = gate * h + (1.0 - gate) * cand
            raw = self.W_out @ h + self.W_feat_out @ x + self.b_out[0]
            outputs.append(torch.clamp(raw, -self.config.rf_limit_Hz, self.config.rf_limit_Hz))
        return torch.stack(outputs), h

    def export(self, template: CfCDirectController) -> CfCDirectController:
        """Export trainable tensors back to NumPy checkpoint controller."""
        weights = {
            "W_c": self.W_c.detach().double().numpy(),
            "U_c": self.U_c.detach().double().numpy(),
            "b_c": self.b_c.detach().double().numpy(),
            "W_tau": self.W_tau.detach().double().numpy(),
            "U_tau": self.U_tau.detach().double().numpy(),
            "b_tau": self.b_tau.detach().double().numpy(),
            "W_out": self.W_out.detach().double().numpy(),
            "W_feat_out": self.W_feat_out.detach().double().numpy(),
            "b_out": self.b_out.detach().double().numpy(),
        }
        return CfCDirectController(template.config, weights=weights)


def collect_features(
    controller: CfCDirectController,
    teacher: CfCDirectController,
    scenario: str,
    duration_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Collect direct-CfC features and teacher RF targets."""
    trace = Disturbance.from_recipe(scenario).generate(
        duration_s=duration_s,
        sample_rate_Hz=PHYSICS_RATE_HZ,
        seed=RNG_SEED,
    )
    errors = _run_teacher_errors(teacher, trace, duration_s, RNG_SEED)
    teacher.reset()
    controller.reset()
    rows: list[np.ndarray] = []
    targets: list[float] = []
    for k, error in enumerate(errors):
        env = _env_at(trace, k)
        _, rf = teacher.step(float(error), env)
        features = controller._features(float(error), env)
        rows.append(features.copy())
        targets.append(float(rf))
        controller._commit_action(float(error), float(rf))
    return np.asarray(rows, dtype=np.float32), np.asarray(targets, dtype=np.float32)


def _json_safe(value: Any) -> Any:
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    return value


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Train CfC recurrent matrices with autograd."""
    run_dir = _PROJECT_ROOT / "data" / "ml_research" / time.strftime("cfc_autograd_%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    teacher_path = args.teacher_checkpoint or json.loads(
        (_PROJECT_ROOT / "data" / "cfc_direct_summary.json").read_text(encoding="utf-8-sig")
    )["checkpoint"]
    teacher = CfCDirectController.load(teacher_path)
    base = CfCDirectController(
        CfCDirectConfig(
            hidden_size=args.hidden_size,
            seed=args.seed,
            sensor_config=PhysicsResidualConfig(),
            output_feature_skip=True,
        )
    )
    initialize_structured_direct(base, args.structured_k_T, None, None, 0.0)
    features, targets = collect_features(base, teacher, "thermal_ramp", args.duration_s)
    model = TorchCfCDirect(base)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    x = torch.from_numpy(features)
    y = torch.from_numpy(targets)
    losses: list[float] = []
    for epoch in range(args.epochs):
        h = torch.zeros(args.hidden_size, dtype=torch.float32)
        chunk_losses: list[float] = []
        for start in range(0, len(x), args.chunk_len):
            xb = x[start : start + args.chunk_len]
            yb = y[start : start + args.chunk_len]
            pred, h_next = model.forward_chunk(xb, h.detach())
            loss = torch.mean((pred - yb) ** 2)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            h = h_next.detach()
            chunk_losses.append(float(loss.item()))
        epoch_loss = float(np.mean(chunk_losses))
        losses.append(epoch_loss)
        if epoch == 0 or (epoch + 1) % max(1, args.epochs // 5) == 0:
            log(f"epoch {epoch + 1}/{args.epochs} mse_Hz2={epoch_loss:.6g}")
    trained = model.export(base)
    ckpt_path = run_dir / "cfc_autograd_direct.json"
    trained.save(ckpt_path)
    metrics: dict[str, Any] = {
        "controller": "cfc_autograd_direct",
        "checkpoint": str(ckpt_path),
        "teacher_checkpoint": teacher_path,
        "loss_history": losses,
        "final_mse_Hz2": losses[-1] if losses else None,
        "training_samples": int(len(features)),
        "config": {
            "duration_s": args.duration_s,
            "epochs": args.epochs,
            "chunk_len": args.chunk_len,
            "hidden_size": args.hidden_size,
            "structured_k_T": args.structured_k_T,
        },
    }
    if args.eval_duration_s > 0:
        metrics["eval_duration_s"] = args.eval_duration_s
        metrics["dlqr"] = evaluate_controller(DLQRController.from_recipe(), args.eval_duration_s)
        metrics["teacher"] = evaluate_controller(
            CfCDirectController.load(teacher_path),
            args.eval_duration_s,
        )
        metrics["cfc_autograd"] = evaluate_controller(
            CfCDirectController.load(ckpt_path),
            args.eval_duration_s,
        )
        metrics["cfc_autograd_over_rhlqr_10s"] = (
            metrics["cfc_autograd"]["sigma_y_10s"] / metrics["dlqr"]["sigma_y_10s"]
            if metrics["dlqr"]["sigma_y_10s"] > 0 else float("nan")
        )
    (run_dir / "metrics.json").write_text(
        json.dumps(_json_safe(metrics), indent=2),
        encoding="utf-8",
    )
    (_PROJECT_ROOT / "data" / "cfc_autograd_summary.json").write_text(
        json.dumps(_json_safe(metrics), indent=2),
        encoding="utf-8",
    )
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-checkpoint", default="")
    parser.add_argument("--duration-s", type=float, default=25.0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--chunk-len", type=int, default=1000)
    parser.add_argument("--hidden-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--structured-k-T", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-6)
    parser.add_argument("--eval-duration-s", type=float, default=25.0)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
