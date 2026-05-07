"""APG training loop: analytic policy gradient via truncated BPTT.

The entire chain

    policy → rf_correction → ReducedTwin.step → state → y

is differentiable.  Loss = mean squared fractional-frequency error over the
rollout window (equivalent to -log sigma_y for white-FM dominated noise).
Truncated BPTT releases the graph every ``truncation_window`` physics steps
to prevent memory explosion.

Curriculum (from scripts/train_apg.py):
    ``clean`` → ``thermal_ramp`` → ``b_field_drift`` → ``laser_intensity_drift``
    → ``all_stacked``

Each curriculum phase trains for a specified number of epochs.  The loss is
the mean-squared y (fractional frequency error) averaged over all batch
elements and all decimation steps in the rollout.

References
----------
Kitching, J. (2018). Chip-scale atomic devices.
    *Applied Physics Reviews*, 5, 031302.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from cptservo.evaluation.batched_runner import run_batched_loop
from cptservo.policy.apg import APGPolicy, compute_obs_stats  # noqa: F401
from cptservo.twin.disturbance import Disturbance
from cptservo.twin.reduced import ReducedTwin

# ---------------------------------------------------------------------------
# Observation normalisation (collect stats from a short clean rollout)
# ---------------------------------------------------------------------------


def _collect_obs_stats(
    policy: APGPolicy,
    twin: ReducedTwin,
    n_samples: int = 4096,
    rng_seed: int = 99999,
) -> None:
    """Run a brief open-loop rollout to estimate observation normalisation stats.

    Updates ``policy.obs_mean`` and ``policy.obs_std`` in-place.

    Args:
        policy: APGPolicy to update.
        twin: ReducedTwin to use for the short rollout.
        n_samples: Number of obs samples to collect.
        rng_seed: RNG seed.
    """
    from cptservo.twin.disturbance import Disturbance

    dist = Disturbance.from_recipe("clean")
    trace = dist.generate(duration_s=float(n_samples) / 1000.0 + 1.0, seed=rng_seed)

    # Single batch element, open-loop, no disc noise
    res = run_batched_loop(
        twin=twin,
        controller=None,
        disturbance_traces=[trace],
        duration_s=float(n_samples) / 1000.0,
        physics_rate_Hz=10_000.0,
        decimation_rate_Hz=1_000.0,
        n_warmup_steps=1_000,
        rng_seed=rng_seed,
        disc_noise_amp_ci=0.0,
        autograd=False,
    )

    # Build obs vectors from the y and error traces
    # For normalisation we need the actual obs distribution.
    # Approximate: construct obs from the error signal history.
    err_np = res["error_signal"][0]  # (n_dec,)
    n_err = policy.n_error_history
    n_rf = policy.n_rf_history
    n_env = 3 if policy.include_env_sensors else 0

    obs_list = []
    for i in range(n_err, len(err_np)):
        err_win = err_np[i - n_err : i]
        rf_win = np.zeros(n_rf)  # zeros for warm-up; OK for scale estimation
        obs = np.concatenate([err_win, rf_win])
        if n_env > 0:
            obs = np.concatenate([obs, [333.15, 50.0, 1.0]])
        obs_list.append(obs)
        if len(obs_list) >= n_samples:
            break

    if not obs_list:
        return

    obs_arr = np.array(obs_list, dtype=np.float64)
    obs_t = torch.from_numpy(obs_arr).to(dtype=twin.dtype, device=twin.device)
    mean, std = compute_obs_stats(obs_t)
    policy.obs_mean.copy_(mean)
    policy.obs_std.copy_(std)


# ---------------------------------------------------------------------------
# Core training function
# ---------------------------------------------------------------------------


def train_apg(
    policy: APGPolicy,
    twin: ReducedTwin,
    disturbance_recipe_name: str,
    n_rollouts: int,
    rollout_duration_s: float,
    batch_size: int = 1024,
    truncation_window: int = 200,
    n_epochs: int = 200,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    grad_clip: float = 1.0,
    physics_rate_Hz: float = 10_000.0,
    decimation_rate_Hz: float = 1_000.0,
    disc_noise_amp_ci: float = 7.0e-4,
    lo_noise_scale: float = 1.0,
    rng_seed: int = 12345,
    log_every: int = 10,
    checkpoint_dir: Path | None = None,
    checkpoint_every: int = 10,
    verbose: bool = True,
) -> dict[str, Any]:
    """Train APG policy via truncated BPTT through the differentiable twin.

    Loss: mean-squared fractional-frequency error over the rollout window.
    This is proportional to sigma_y^2 at the rollout timescale and is
    differentiable through the ReducedTwin physics.

    Truncated BPTT: backprop through ``truncation_window`` physics steps, then
    detach state.  Avoids gradient explosion through long rollouts.

    Args:
        policy: APGPolicy to train (modified in-place).
        twin: ReducedTwin (must be on CPU; parameters are NOT trained).
        disturbance_recipe_name: Named scenario from ``configs/v1_recipe.yaml``.
        n_rollouts: Number of parallel rollout traces per epoch.
        rollout_duration_s: Duration of each rollout (seconds).
        batch_size: Batch size for the differentiable rollout (number of
            parallel batch elements).  Set to ``n_rollouts`` for simplicity;
            larger values amortise the controller overhead.
        truncation_window: Physics steps between BPTT detaches.
        n_epochs: Number of training epochs for this curriculum phase.
        lr: Adam learning rate.
        weight_decay: Adam weight decay.
        grad_clip: Gradient clipping norm (applied per epoch step).
        physics_rate_Hz: Physics integration rate.
        decimation_rate_Hz: Controller update rate.
        disc_noise_amp_ci: Discriminator-input noise std (ci units).
        lo_noise_scale: LO noise amplitude multiplier.
        rng_seed: Base RNG seed; each epoch increments by 1 to get fresh noise.
        log_every: Log progress every N epochs.
        checkpoint_dir: Directory for saving epoch checkpoints.  ``None``
            skips saving.
        checkpoint_every: Save a checkpoint every N epochs.
        verbose: Whether to print epoch-level progress.

    Returns:
        Dict with keys:
            ``loss_history`` — list of per-epoch mean losses.
            ``epoch_count`` — total epochs trained.
            ``final_loss`` — loss at last epoch.
            ``wall_time_s`` — total training wall time.
            ``checkpoints_saved`` — number of checkpoints written.
            ``scenario`` — disturbance scenario name.
    """
    dist_gen = Disturbance.from_recipe(disturbance_recipe_name)

    optimiser = torch.optim.Adam(
        policy.parameters(), lr=lr, weight_decay=weight_decay
    )

    loss_history: list[float] = []
    checkpoints_saved = 0
    t_train_start = time.perf_counter()

    if verbose:
        _log(
            f"  APG train: scenario={disturbance_recipe_name}, "
            f"n_epochs={n_epochs}, n_rollouts={n_rollouts}, "
            f"duration={rollout_duration_s}s, batch_size={batch_size}"
        )
    if checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(n_epochs):
        epoch_seed = rng_seed + epoch

        # Generate n_rollouts disturbance traces
        traces = [
            dist_gen.generate(
                duration_s=rollout_duration_s,
                sample_rate_Hz=physics_rate_Hz,
                seed=epoch_seed + i,
            )
            for i in range(n_rollouts)
        ]

        # Run batched loop with autograd enabled
        # We run n_rollouts as batch elements: batch_size = n_rollouts
        try:
            res = run_batched_loop(
                twin=twin,
                controller=policy,
                disturbance_traces=traces,
                duration_s=rollout_duration_s,
                physics_rate_Hz=physics_rate_Hz,
                decimation_rate_Hz=decimation_rate_Hz,
                n_warmup_steps=2_000,
                rng_seed=epoch_seed,
                disc_noise_amp_ci=disc_noise_amp_ci,
                lo_noise_scale=lo_noise_scale,
                autograd=True,
                truncation_window=truncation_window,
            )
        except RuntimeError as exc:
            # Surface numerical instability (NaN / gradient explosion) rather than
            # silently continuing.  Caller can reduce lr or truncation_window.
            _log(f"  [WARN] epoch {epoch}: RuntimeError during rollout: {exc}")
            break

        y_t: Tensor = res["y_tensor"]  # (B, n_dec) — differentiable

        # Check for NaN
        if torch.isnan(y_t).any():
            _log(
                f"  [WARN] epoch {epoch}: NaN in y_tensor — "
                "reducing lr or truncation_window recommended"
            )
            break

        # Loss: per-rollout demeaned squared fractional-frequency error.
        # Targets the VARIANCE the policy can shape — equivalent to sigma_y^2.
        # The raw mean(y^2) is dominated by the deterministic DC systematic
        # offset (delta_buf + delta_ls + delta_B + delta_T), which is ~14x
        # larger than the policy's ±rf_limit_Hz actuator can compensate, so
        # the affactable gradient is buried under unaffactable noise. Demean
        # along the time axis per rollout to isolate the variance.
        y_demeaned = y_t - y_t.mean(dim=-1, keepdim=True)
        loss: Tensor = (y_demeaned**2).mean()

        optimiser.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_clip)

        # Check for NaN gradients before step
        has_nan_grad = any(
            p.grad is not None and torch.isnan(p.grad).any()
            for p in policy.parameters()
        )
        if has_nan_grad:
            _log(
                f"  [WARN] epoch {epoch}: NaN gradient — skipping optimiser step. "
                "Suggest: reduce lr, smaller truncation_window, or gradient clipping."
            )
            optimiser.zero_grad()
            break

        optimiser.step()

        epoch_loss = float(loss.item())
        loss_history.append(epoch_loss)

        if verbose and (epoch % log_every == 0 or epoch == n_epochs - 1):
            elapsed = time.perf_counter() - t_train_start
            _log(
                f"  epoch {epoch:4d}/{n_epochs}  "
                f"loss={epoch_loss:.4e}  "
                f"elapsed={elapsed:.1f}s"
            )

        # Checkpoint
        if checkpoint_dir is not None and (
            (epoch + 1) % checkpoint_every == 0 or epoch == n_epochs - 1
        ):
            ckpt_path = checkpoint_dir / f"apg_epoch_{epoch + 1}.pt"
            policy.save(str(ckpt_path))
            checkpoints_saved += 1

    wall_time_s = time.perf_counter() - t_train_start
    final_loss = loss_history[-1] if loss_history else float("nan")

    return {
        "loss_history": loss_history,
        "epoch_count": len(loss_history),
        "final_loss": final_loss,
        "wall_time_s": wall_time_s,
        "checkpoints_saved": checkpoints_saved,
        "scenario": disturbance_recipe_name,
    }


# ---------------------------------------------------------------------------
# Curriculum training: multi-phase
# ---------------------------------------------------------------------------


def train_curriculum(
    policy: APGPolicy,
    twin: ReducedTwin,
    phases: list[dict[str, Any]],
    checkpoint_dir: Path | None = None,
    checkpoint_every: int = 10,
    verbose: bool = True,
) -> dict[str, Any]:
    """Run multi-phase curriculum training.

    Each phase is a dict with keys matching ``train_apg`` arguments plus
    ``disturbance_recipe_name`` and ``n_epochs``.

    Args:
        policy: APGPolicy to train.
        twin: ReducedTwin instance.
        phases: List of phase dicts.  Required keys: ``disturbance_recipe_name``,
            ``n_epochs``.  Optional keys: any ``train_apg`` kwarg.
        checkpoint_dir: Directory for checkpoints.
        checkpoint_every: Checkpoint save interval (epochs within each phase).
        verbose: Print progress.

    Returns:
        Dict with ``phases`` (list of per-phase metrics), ``total_epochs``,
        ``total_wall_time_s``, ``final_loss``, ``all_loss_history``.
    """
    all_loss: list[float] = []
    phase_results: list[dict[str, Any]] = []
    total_wall = 0.0
    epoch_offset = 0

    for phase_idx, phase in enumerate(phases):
        scenario = phase["disturbance_recipe_name"]
        n_epochs = int(phase["n_epochs"])

        if verbose:
            _log(
                f"\n=== Curriculum phase {phase_idx + 1}/{len(phases)}: "
                f"{scenario} ({n_epochs} epochs) ==="
            )

        # Extract train_apg kwargs from phase dict
        train_kwargs: dict[str, Any] = {
            k: v
            for k, v in phase.items()
            if k not in ("disturbance_recipe_name", "n_epochs")
        }
        train_kwargs["n_epochs"] = n_epochs
        if checkpoint_dir is not None:
            train_kwargs["checkpoint_dir"] = checkpoint_dir
        train_kwargs["checkpoint_every"] = checkpoint_every
        train_kwargs["verbose"] = verbose
        # Offset rng_seed so each phase uses different noise
        base_seed = int(train_kwargs.get("rng_seed", 12345))
        train_kwargs["rng_seed"] = base_seed + epoch_offset * 7

        metrics = train_apg(
            policy=policy,
            twin=twin,
            disturbance_recipe_name=scenario,
            **train_kwargs,
        )

        all_loss.extend(metrics["loss_history"])
        phase_results.append(metrics)
        total_wall += metrics["wall_time_s"]
        epoch_offset += n_epochs

        if verbose:
            _log(
                f"  Phase {phase_idx + 1} done: "
                f"final_loss={metrics['final_loss']:.4e}, "
                f"wall={metrics['wall_time_s']:.1f}s"
            )

    final_loss = all_loss[-1] if all_loss else float("nan")

    return {
        "phases": phase_results,
        "total_epochs": len(all_loss),
        "total_wall_time_s": total_wall,
        "final_loss": final_loss,
        "all_loss_history": all_loss,
    }


# ---------------------------------------------------------------------------
# Logging helper
# ---------------------------------------------------------------------------


def _log(msg: str) -> None:
    """Timestamped log line."""
    import sys

    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    try:
        print(line, flush=True)
    except UnicodeEncodeError:
        print(line.encode("ascii", errors="replace").decode("ascii"), flush=True)
    sys.stdout.flush()
