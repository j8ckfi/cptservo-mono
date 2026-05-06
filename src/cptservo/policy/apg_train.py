"""APG training loop: analytic policy gradient via truncated BPTT.

The entire chain

    obs (error_window) -> policy.forward(obs) -> rf_correction
    -> ReducedTwin.step(state, ctrl, disturbance) -> state -> y

is differentiable.  Loss = ``mean(y^2)`` over the BPTT window (equivalent to
sigma_y^2 at the rollout timescale).

Truncated BPTT releases the graph every ``bptt_window`` controller-steps to
prevent gradient explosion.  Noise tensors are drawn outside the graph so
randomness does not contribute spurious gradients (per spec rule: use
``torch.no_grad()`` for noise generation).

The ``train_apg`` function is self-contained: it owns the rollout loop, noise
generation, and optimizer step.  It calls
``ReducedTwin.step`` and ``ReducedTwin.fractional_frequency_error_with_B``
directly rather than going through ``run_batched_loop``.

Noise budget (M3-calibrated, spec rule 6):
    disc_noise_amp_ci  = 7.0e-4
    lo_noise_scale     = 1.0
    Full LO white-FM spectrum from configs/v1_recipe.yaml.

References
----------
Kitching, J. (2018). Chip-scale atomic devices.
    *Applied Physics Reviews*, 5, 031302.
Vanier, J. & Mandache, C. (2007). Applied Physics B, 87, 565-593.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import Tensor

from cptservo.policy.apg import APGPolicy
from cptservo.twin.disturbance import Disturbance
from cptservo.twin.reduced import HF_GROUND, ReducedTwin

# ---------------------------------------------------------------------------
# Internal helpers
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


def _load_noise_budget(project_root: Path) -> dict[str, float]:
    """Load noise budget from configs/v1_recipe.yaml.

    Args:
        project_root: Root of the CPTServo project.

    Returns:
        Dict with ``total_white_fm_amp`` and individual budget entries.
    """
    recipe_path = project_root / "configs" / "v1_recipe.yaml"
    with open(recipe_path, encoding="utf-8-sig") as fh:
        recipe = yaml.safe_load(fh)
    nb = recipe["noise_budget"]
    terms = [
        float(nb["photon_shot_noise_amp"]),
        float(nb["lo_white_fm_amp"]),
        float(nb.get("microwave_phase_white_amp", 0.0)),
        float(nb.get("detector_electronics_amp", 0.0)),
    ]
    return {
        "total_white_fm_amp": float(np.sqrt(sum(t**2 for t in terms))),
    }


def _make_twin_from_project_root(project_root: Path) -> ReducedTwin:
    """Build a ReducedTwin with M2-calibrated free parameters.

    Args:
        project_root: Root of the CPTServo project.

    Returns:
        ReducedTwin on CPU with float64.
    """
    import json

    calib_path = project_root / "data" / "reduced_calibration.json"
    if calib_path.exists():
        cal = json.loads(calib_path.read_text(encoding="utf-8-sig"))
        ls = float(cal["light_shift_coeff"])
        buf = float(cal["buffer_gas_shift_coeff"])
        zee = float(cal["lumped_zeeman_coeff"])
    else:
        ls, buf, zee = 0.0, -7.4e6, 7.0
    return ReducedTwin(
        light_shift_coeff=ls,
        buffer_gas_shift_coeff=buf,
        lumped_zeeman_coeff=zee,
        temperature_coeff_Hz_per_K=0.015,
        device="cpu",
        dtype=torch.float64,
    )


# ---------------------------------------------------------------------------
# Core training function (public API)
# ---------------------------------------------------------------------------


def train_apg(
    policy: APGPolicy,
    n_episodes_per_stage: int = 200,
    bptt_window: int = 200,
    learning_rate: float = 3.0e-4,
    grad_clip: float = 1.0,
    curriculum: list[str] | None = None,
    save_path: str | Path = "models/apg_best.pt",
    rng_seed: int = 42,
    physics_rate_Hz: float = 10_000.0,
    decimation_rate_Hz: float = 1_000.0,
    disc_noise_amp_ci: float = 7.0e-4,
    lo_noise_scale: float = 1.0,
    n_warmup_steps: int = 2_000,
    verbose: bool = True,
) -> dict[str, Any]:
    """Train the APGPolicy via analytic policy gradient (truncated BPTT).

    Each episode is a ``bptt_window``-step differentiable rollout through the
    ReducedTwin.  The loss is ``mean(y^2)`` over the window (minimising sigma_y^2
    at the rollout timescale).

    Noise is injected per the M3-calibrated budget (spec rule 6):
        - LO white-FM noise on the RF command (pre-drawn, no autograd through
          randomness).
        - Discriminator-input noise on the error signal (pre-drawn).
    ``torch.no_grad()`` is used for all noise-generation calls.

    Args:
        policy: APGPolicy to train.  Modified in-place.
        n_episodes_per_stage: Number of episodes per curriculum stage.
        bptt_window: Controller-steps per BPTT segment.  At 1 kHz, 200 steps
            = 200 ms simulated time.
        learning_rate: Adam learning rate.
        grad_clip: Gradient clipping norm.
        curriculum: Ordered list of disturbance scenario names.  Defaults to
            ``["clean", "thermal_ramp", "b_field_drift",
               "laser_intensity_drift", "all_stacked"]``.
        save_path: Path for the best-model checkpoint.  Parent directory is
            created if it does not exist.
        rng_seed: Base RNG seed; incremented per episode for fresh noise.
        physics_rate_Hz: Physics integration rate (Hz).
        decimation_rate_Hz: Controller update rate (Hz).
        disc_noise_amp_ci: Discriminator-input noise std (ci units per step).
        lo_noise_scale: Multiplier on LO white-FM noise amplitude.
        n_warmup_steps: No-grad warm-up steps before training rollout starts.
        verbose: Print per-episode progress.

    Returns:
        Dict with keys:

            ``loss_per_episode``    -- list of per-episode mean squared y.
            ``sigma_y_per_episode`` -- list of per-episode sqrt(mean y^2).
            ``stage_final_loss``    -- dict mapping scenario -> final episode loss.
            ``total_wall_s``        -- total wall time in seconds.
            ``curriculum``          -- list of stage names.
            ``final_loss``          -- loss at final episode.
            ``best_loss``           -- best loss seen during training.
            ``n_episodes_total``    -- total episodes run.
            ``save_path``           -- path to the saved checkpoint.
    """
    if curriculum is None:
        curriculum = [
            "clean",
            "thermal_ramp",
            "b_field_drift",
            "laser_intensity_drift",
            "all_stacked",
        ]

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Locate project root (3 levels up: policy -> cptservo -> src -> root)
    _here = Path(__file__).resolve()
    project_root = _here.parent.parent.parent.parent

    # Build calibrated twin
    twin = _make_twin_from_project_root(project_root)

    # Load noise budget
    noise = _load_noise_budget(project_root)
    total_white_fm_amp = noise["total_white_fm_amp"]

    dt = 1.0 / physics_rate_Hz
    decimation = int(round(physics_rate_Hz / decimation_rate_Hz))

    # Per-physics-step LO noise std (Hz of RF detuning)
    lo_noise_std = (
        lo_noise_scale * total_white_fm_amp * HF_GROUND * float(np.sqrt(physics_rate_Hz))
    )

    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)

    # Determine obs dimension components from policy
    n_err_hist = policy.n_error_history
    n_rf_hist = policy.n_rf_history

    loss_per_episode: list[float] = []
    sigma_y_per_episode: list[float] = []
    stage_final_loss: dict[str, float] = {}
    t_start = time.perf_counter()
    best_loss = float("inf")
    global_ep = 0

    for stage_idx, scenario in enumerate(curriculum):
        if verbose:
            _log(
                f"\n=== APG stage {stage_idx + 1}/{len(curriculum)}: {scenario} "
                f"({n_episodes_per_stage} episodes) ==="
            )

        dist_gen = Disturbance.from_recipe(scenario)
        stage_losses: list[float] = []

        for ep_in_stage in range(n_episodes_per_stage):
            ep_seed = rng_seed + global_ep
            global_ep += 1

            # Duration covers warmup + bptt_window controller steps
            ep_ctrl_steps = bptt_window
            ep_duration_s = ep_ctrl_steps / decimation_rate_Hz
            warmup_duration_s = n_warmup_steps / physics_rate_Hz
            total_ep_duration_s = warmup_duration_s + ep_duration_s

            trace = dist_gen.generate(
                duration_s=total_ep_duration_s,
                sample_rate_Hz=physics_rate_Hz,
                seed=ep_seed,
            )

            n_warmup_phys = n_warmup_steps
            n_main_phys = ep_ctrl_steps * decimation
            n_total_phys = n_warmup_phys + n_main_phys

            # Build disturbance tensor (n_total_phys, 3) -- no grad
            dist_arr = np.stack(
                [trace.T_K, trace.B_uT, trace.laser_intensity_norm], axis=1
            ).astype(np.float64)
            if dist_arr.shape[0] < n_total_phys:
                reps = (n_total_phys + dist_arr.shape[0] - 1) // dist_arr.shape[0]
                dist_arr = np.tile(dist_arr, (reps, 1))[:n_total_phys]
            else:
                dist_arr = dist_arr[:n_total_phys]

            dist_t = torch.from_numpy(dist_arr).to(
                dtype=twin.dtype, device=twin.device
            )  # (N, 3), no grad

            # Pre-generate noise outside autograd graph
            rng_np = np.random.default_rng(ep_seed + 10_000)
            with torch.no_grad():
                if lo_noise_std > 0.0:
                    lo_noise_np = rng_np.normal(0.0, lo_noise_std, size=n_total_phys)
                else:
                    lo_noise_np = np.zeros(n_total_phys, dtype=np.float64)
                disc_noise_np = rng_np.normal(
                    0.0, disc_noise_amp_ci, size=ep_ctrl_steps
                )
                lo_noise_t = torch.from_numpy(lo_noise_np).to(
                    dtype=twin.dtype, device=twin.device
                )  # no grad

            # Warm-up: no-grad, no controller
            state = twin.initial_state(batch_size=1)
            ctrl_zero = torch.zeros(1, 2, dtype=twin.dtype, device=twin.device)
            with torch.no_grad():
                for wi in range(n_warmup_phys):
                    state = twin.step(state, ctrl_zero, dist_t[wi : wi + 1], dt)
            state = state.detach()

            # Reset policy history
            policy.reset()

            # Sliding history buffers (no grad -- history is bookkeeping only)
            err_hist = torch.zeros(1, n_err_hist, dtype=twin.dtype, device=twin.device)
            rf_hist = torch.zeros(1, n_rf_hist, dtype=twin.dtype, device=twin.device)

            # Differentiable BPTT rollout
            y_list: list[Tensor] = []
            phys_step_count = 0

            for k in range(ep_ctrl_steps):
                phys_offset = n_warmup_phys + k * decimation

                # Build obs from history buffers (no grad in history)
                # Env scalars from disturbance at start of decimation window
                dist_k = dist_t[phys_offset : phys_offset + 1].detach()  # (1, 3)
                env_k = dist_k if policy.include_env_sensors else None
                obs = policy.build_obs(err_hist, rf_hist, env_k)
                # obs is (1, obs_dim) -- no grad in inputs

                # Forward pass through policy (differentiable through policy params)
                rf_corr_t = policy.forward(obs)  # (1,) with grad through policy weights

                # LO noise (non-differentiable)
                lo_k = lo_noise_t[phys_offset : phys_offset + 1].detach()
                ctrl_rf = rf_corr_t + lo_k  # (1,)

                ctrl = torch.stack(
                    [torch.zeros(1, dtype=twin.dtype, device=twin.device), ctrl_rf],
                    dim=1,
                )  # (1, 2)

                y_sum: Tensor = torch.zeros(1, dtype=twin.dtype, device=twin.device)

                for j in range(decimation):
                    idx = phys_offset + j
                    dist_step = dist_t[idx : idx + 1].detach()  # (1, 3)

                    if j == 0:
                        ctrl_j = ctrl
                    else:
                        # Inner substeps: keep gradient through the same RF
                        # command while adding fresh non-differentiable LO noise.
                        extra_lo = lo_noise_t[idx : idx + 1].detach()
                        ctrl_j = torch.stack(
                            [
                                torch.zeros(1, dtype=twin.dtype, device=twin.device),
                                rf_corr_t + extra_lo,
                            ],
                            dim=1,
                        )

                    state = twin.step(state, ctrl_j, dist_step, dt)

                    B_now = dist_t[idx, 1].unsqueeze(0).detach()  # (1,)
                    T_now = dist_t[idx, 0].unsqueeze(0).detach()  # (1,)
                    y_sum = y_sum + twin.fractional_frequency_error_with_B(
                        state, ctrl_j, B_now, T_now
                    )

                    phys_step_count += 1

                    # Truncated BPTT: detach state to release accumulated graph
                    if phys_step_count % bptt_window == 0:
                        state = state.detach()

                y_dec = y_sum / decimation  # (1,)
                y_list.append(y_dec)

                # Update history windows (detached -- not part of gradient path)
                ci_now = float(state[0, 6].detach().item())
                ci_noisy = ci_now + float(disc_noise_np[k])
                rf_now = float(rf_corr_t.detach().item())

                with torch.no_grad():
                    # Shift error history left, insert new sample at end
                    err_hist = torch.roll(err_hist, shifts=-1, dims=1).clone()
                    err_hist[0, -1] = ci_noisy
                    # Shift RF history left, insert new rf command at end
                    if n_rf_hist > 0:
                        rf_hist = torch.roll(rf_hist, shifts=-1, dims=1).clone()
                        rf_hist[0, -1] = rf_now

            # Loss: mean squared y over the BPTT window
            y_tensor = torch.cat(y_list, dim=0)  # (ep_ctrl_steps,)

            # Check for NaN before backward
            if torch.isnan(y_tensor).any():
                if verbose:
                    _log(
                        f"  [WARN] stage={scenario} ep={ep_in_stage}: NaN in y_tensor"
                        " -- skipping episode"
                    )
                loss_per_episode.append(float("nan"))
                sigma_y_per_episode.append(float("nan"))
                continue

            loss = (y_tensor**2).mean()

            optimizer.zero_grad()
            loss.backward()

            # Check for NaN gradients
            has_nan_grad = any(
                p.grad is not None and torch.isnan(p.grad).any()
                for p in policy.parameters()
            )
            if has_nan_grad:
                if verbose:
                    _log(
                        f"  [WARN] stage={scenario} ep={ep_in_stage}: NaN gradient"
                        " -- skipping optimizer step"
                    )
                optimizer.zero_grad()
                loss_per_episode.append(float("nan"))
                sigma_y_per_episode.append(float("nan"))
                continue

            torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_clip)
            optimizer.step()

            ep_loss = float(loss.item())
            ep_sigma_y = float(ep_loss**0.5)
            loss_per_episode.append(ep_loss)
            sigma_y_per_episode.append(ep_sigma_y)
            stage_losses.append(ep_loss)

            # Save checkpoint if best so far
            if ep_loss < best_loss:
                best_loss = ep_loss
                policy.save(str(save_path))

            if verbose and (ep_in_stage % 20 == 0 or ep_in_stage == n_episodes_per_stage - 1):
                elapsed = time.perf_counter() - t_start
                _log(
                    f"  stage={scenario} ep={ep_in_stage:4d}/{n_episodes_per_stage}"
                    f"  loss={ep_loss:.4e}  sigma_y={ep_sigma_y:.4e}"
                    f"  elapsed={elapsed:.1f}s"
                )

        # Record final loss for this stage
        finite_losses = [v for v in stage_losses if np.isfinite(v)]
        stage_final_loss[scenario] = finite_losses[-1] if finite_losses else float("nan")
        if verbose:
            _log(
                f"  Stage {scenario} done: "
                f"final_loss={stage_final_loss[scenario]:.4e}, "
                f"n_episodes={len(stage_losses)}"
            )

    total_wall_s = time.perf_counter() - t_start

    # Ensure we have a saved checkpoint (save final if no improvement recorded)
    if not save_path.exists():
        policy.save(str(save_path))

    finite_all = [v for v in loss_per_episode if np.isfinite(v)]
    final_loss = finite_all[-1] if finite_all else float("nan")

    if verbose:
        _log(
            f"\n=== APG training done: {global_ep} episodes, "
            f"final_loss={final_loss:.4e}, wall={total_wall_s:.1f}s "
            f"({total_wall_s / 60.0:.1f} min) ==="
        )

    return {
        "loss_per_episode": loss_per_episode,
        "sigma_y_per_episode": sigma_y_per_episode,
        "stage_final_loss": stage_final_loss,
        "total_wall_s": total_wall_s,
        "curriculum": curriculum,
        "final_loss": final_loss,
        "best_loss": best_loss,
        "n_episodes_total": global_ep,
        "save_path": str(save_path),
    }
