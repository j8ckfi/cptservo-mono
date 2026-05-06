"""Batched closed-loop runner for CPT-clock servo evaluation and APG training.

Supports two operating modes:

``autograd=False`` (evaluation, default)
    Runs with ``torch.no_grad()`` for maximum speed.  Identical noise
    architecture to ``run_fast_loop`` in ``scripts/run_m3_m4_gates.py``:
    LO white-FM noise on ``rf_actual`` + discriminator-input Gaussian noise
    on the error signal.  Provenance metadata:
    ``noise_injection_point = "rf_actual_pre_step+disc_noise_pre_controller"``.

``autograd=True`` (APG training)
    Keeps the autograd graph alive so gradients can flow from the loss back
    through the twin physics to the policy parameters.  Implements truncated
    BPTT: every ``truncation_window`` substeps, ``state.detach_()`` is called
    to release the accumulated graph and prevent memory/gradient explosion.
    Noise tensors for LO noise are pre-drawn from NumPy and converted to
    non-gradient tensors so they don't contribute spurious gradients.

Per-substep y averaging (boxcar over decimation window) is applied in both
modes, matching the M3 / M4 anti-fudge architecture.

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
import yaml
from torch import Tensor

from cptservo.twin.disturbance import DisturbanceTrace
from cptservo.twin.reduced import HF_GROUND, ReducedTwin

# ---------------------------------------------------------------------------
# Internal helper: load noise budget from recipe
# ---------------------------------------------------------------------------


def _load_total_white_fm_amp(project_root: Path) -> float:
    """Compute total white-FM noise amplitude from configs/v1_recipe.yaml.

    Args:
        project_root: Root directory of the CPTServo project.

    Returns:
        Total white-FM amplitude in fractional-freq / sqrt(Hz).
    """
    recipe_path = project_root / "configs" / "v1_recipe.yaml"
    with open(recipe_path, encoding="utf-8") as fh:
        recipe = yaml.safe_load(fh)
    nb = recipe["noise_budget"]
    terms = [
        float(nb["photon_shot_noise_amp"]),
        float(nb["lo_white_fm_amp"]),
        float(nb.get("microwave_phase_white_amp", 0.0)),
        float(nb.get("detector_electronics_amp", 0.0)),
    ]
    return float(np.sqrt(sum(t**2 for t in terms)))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_batched_loop(
    twin: ReducedTwin,
    controller: Any,  # PIController | RHLQRController | APGPolicy
    disturbance_traces: list[DisturbanceTrace],
    duration_s: float,
    physics_rate_Hz: float = 10_000.0,
    decimation_rate_Hz: float = 1_000.0,
    n_warmup_steps: int = 5_000,
    rng_seed: int = 12345,
    pilot_freq_Hz: float = 0.0,
    pilot_amp_Hz: float = 0.0,
    lo_noise_scale: float = 1.0,
    disc_noise_amp_ci: float = 7.0e-4,
    shared_noise_across_batch: bool = False,
    autograd: bool = False,
    truncation_window: int = 200,
) -> dict[str, Any]:
    """Batched closed-loop runner.

    Runs ``len(disturbance_traces)`` parallel trajectories through the
    ReducedTwin, each with its own disturbance trace.  All trajectories share
    the same controller (policy parameters broadcast across the batch).

    Noise architecture (identical to ``run_fast_loop``):
        - LO white-FM noise on ``rf_actual`` (Hz, per physics step).
        - Discriminator-input noise on error signal (ci units, per decimation
          step) when ``disc_noise_amp_ci > 0`` and a controller is active.
        - ``noise_injection_point = "rf_actual_pre_step+disc_noise_pre_controller"``

    Args:
        twin: ReducedTwin instance.
        controller: Controller object with ``.step(error, env)`` or ``.step(error)``
            returning ``(laser_Hz, rf_Hz)``.  ``None`` runs open-loop.
        disturbance_traces: List of ``DisturbanceTrace`` objects, one per batch
            element.  All must have the same sample rate.
        duration_s: Simulation duration in seconds.
        physics_rate_Hz: Physics integration rate (Hz).
        decimation_rate_Hz: Controller update and output decimation rate (Hz).
        n_warmup_steps: Warm-up steps (no-grad, no-controller) before the
            main loop.
        rng_seed: NumPy RNG seed for reproducible noise.
        pilot_freq_Hz: Deterministic pilot tone frequency (Hz).  0 disables.
        pilot_amp_Hz: Deterministic pilot tone amplitude (Hz).
        lo_noise_scale: Multiplier on LO white-FM noise amplitude.
        disc_noise_amp_ci: Discriminator-input noise amplitude (ci units,
            per decimation step std).
        shared_noise_across_batch: If ``True``, use the first generated LO and
            discriminator-noise row for every batch element.  This is useful for
            paired controller comparisons.
        autograd: If ``True``, keep autograd graph alive for APG training.
        truncation_window: Number of physics substeps between BPTT detaches
            (only used when ``autograd=True``).

    Returns:
        Dict with keys:
            ``y`` — (B, n_dec) fractional-frequency error (boxcar-averaged).
            ``error_signal`` — (B, n_dec) discriminated error (ci + disc noise).
            ``rf_cmd`` — (B, n_dec) controller-commanded RF correction.
            ``rf_actual`` — (B, n_dec) actual RF applied (cmd + LO noise).
            ``wall_s`` — elapsed wall time.
            ``noise_injection_point`` — provenance string.
    """
    B = len(disturbance_traces)
    dt = 1.0 / physics_rate_Hz
    decimation = int(round(physics_rate_Hz / decimation_rate_Hz))
    n_dec = int(round(duration_s * decimation_rate_Hz))
    n_physics = n_dec * decimation

    # ---------------------------------------------------------------------------
    # Locate project root for recipe loading
    # ---------------------------------------------------------------------------
    _here = Path(__file__).resolve()
    project_root = _here.parent.parent.parent.parent  # src/cptservo/evaluation -> root

    # ---------------------------------------------------------------------------
    # Noise budget
    # ---------------------------------------------------------------------------
    total_white_fm_amp = _load_total_white_fm_amp(project_root)
    lo_noise_std_per_physics = (
        lo_noise_scale * total_white_fm_amp * HF_GROUND * float(np.sqrt(physics_rate_Hz))
    )

    # ---------------------------------------------------------------------------
    # Pre-allocate disturbance tensors: (B, n_physics, 3)
    # ---------------------------------------------------------------------------
    dist_arrays: list[np.ndarray] = []
    for trace in disturbance_traces:
        arr = np.stack(
            [trace.T_K, trace.B_uT, trace.laser_intensity_norm], axis=1
        ).astype(np.float64)
        if arr.shape[0] < n_physics:
            reps = (n_physics + arr.shape[0] - 1) // arr.shape[0]
            arr = np.tile(arr, (reps, 1))[:n_physics]
        elif arr.shape[0] > n_physics:
            arr = arr[:n_physics]
        dist_arrays.append(arr)
    dist_np = np.stack(dist_arrays, axis=0)  # (B, n_physics, 3)
    # Convert to torch tensor (no grad needed for disturbances)
    dist_tensor = torch.from_numpy(dist_np).to(twin.device)  # (B, n_physics, 3)

    # ---------------------------------------------------------------------------
    # Pre-generate LO noise: (B, n_physics)
    #
    # In shared-noise mode generate a single trajectory first, then repeat it.
    # That preserves the exact RNG stream used by a canonical B=1 evaluation.
    # ---------------------------------------------------------------------------
    rng = np.random.default_rng(rng_seed)
    noise_batch = 1 if shared_noise_across_batch and B > 1 else B
    if lo_noise_std_per_physics > 0.0:
        lo_noise_np = rng.normal(
            0.0,
            lo_noise_std_per_physics,
            size=(noise_batch, n_physics),
        )
    else:
        lo_noise_np = np.zeros((noise_batch, n_physics), dtype=np.float64)
    if shared_noise_across_batch and B > 1:
        lo_noise_np = np.repeat(lo_noise_np[:1, :], B, axis=0)

    # Pilot tone: (n_physics,) — same for all batch elements
    if pilot_amp_Hz != 0.0 and pilot_freq_Hz != 0.0:
        t_axis = np.arange(n_physics, dtype=np.float64) * dt
        pilot_np = pilot_amp_Hz * np.sin(2.0 * np.pi * pilot_freq_Hz * t_axis)
    else:
        pilot_np = np.zeros(n_physics, dtype=np.float64)

    # Discriminator noise: (B, n_dec)
    if disc_noise_amp_ci > 0.0 and controller is not None:
        disc_noise_np = rng.normal(0.0, disc_noise_amp_ci, size=(noise_batch, n_dec))
    else:
        disc_noise_np = np.zeros((noise_batch, n_dec), dtype=np.float64)
    if shared_noise_across_batch and B > 1:
        disc_noise_np = np.repeat(disc_noise_np[:1, :], B, axis=0)

    # ---------------------------------------------------------------------------
    # Warm-up (always no-grad)
    # ---------------------------------------------------------------------------
    state = twin.initial_state(batch_size=B)  # (B, 8)
    ctrl_zero = torch.zeros(B, 2, dtype=twin.dtype, device=twin.device)

    with torch.no_grad():
        for i in range(n_warmup_steps):
            # Use first disturbance step for warm-up (constant environment)
            state = twin.step(state, ctrl_zero, dist_tensor[:, 0, :], dt)

    # ---------------------------------------------------------------------------
    # Reset controller
    # ---------------------------------------------------------------------------
    if controller is not None and hasattr(controller, "reset"):
        controller.reset()

    # ---------------------------------------------------------------------------
    # Output buffers (NumPy; converted from tensors after loop)
    # ---------------------------------------------------------------------------
    y_out = np.zeros((B, n_dec), dtype=np.float64)
    err_out = np.zeros((B, n_dec), dtype=np.float64)
    rf_cmd_out = np.zeros((B, n_dec), dtype=np.float64)
    rf_actual_out = np.zeros((B, n_dec), dtype=np.float64)

    # Per-batch rf_cmd (Hz): shape (B,), updated each decimation step
    rf_cmd = np.zeros(B, dtype=np.float64)

    # For autograd mode: track y as a tensor for gradient accumulation
    # We accumulate the (B, n_dec) y tensor to compute the loss.
    if autograd:
        y_tensor_list: list[Tensor] = []
        # Detach state from warm-up graph
        state = state.detach()
        # Sliding history windows for APG policy obs: (B, n_error_history) and (B, n_rf_history)
        # These are float tensors that carry gradients through the policy forward pass.
        n_err_hist = getattr(controller, "n_error_history", 8) if controller is not None else 8
        n_rf_hist = getattr(controller, "n_rf_history", 4) if controller is not None else 4
        err_hist = torch.zeros(B, n_err_hist, dtype=twin.dtype, device=twin.device)
        rf_hist = torch.zeros(B, n_rf_hist, dtype=twin.dtype, device=twin.device)

    t0 = time.perf_counter()

    # ---------------------------------------------------------------------------
    # Main simulation loop
    # ---------------------------------------------------------------------------
    if autograd:
        _run_autograd_loop(
            twin=twin,
            controller=controller,
            state=state,
            dist_tensor=dist_tensor,
            lo_noise_np=lo_noise_np,
            pilot_np=pilot_np,
            disc_noise_np=disc_noise_np,
            rf_cmd=rf_cmd,
            n_dec=n_dec,
            decimation=decimation,
            dt=dt,
            truncation_window=truncation_window,
            y_out=y_out,
            err_out=err_out,
            rf_cmd_out=rf_cmd_out,
            rf_actual_out=rf_actual_out,
            y_tensor_list=y_tensor_list,
            err_hist=err_hist,
            rf_hist=rf_hist,
        )
    else:
        with torch.no_grad():
            _run_nograd_loop(
                twin=twin,
                controller=controller,
                state=state,
                dist_tensor=dist_tensor,
                lo_noise_np=lo_noise_np,
                pilot_np=pilot_np,
                disc_noise_np=disc_noise_np,
                rf_cmd=rf_cmd,
                n_dec=n_dec,
                decimation=decimation,
                dt=dt,
                y_out=y_out,
                err_out=err_out,
                rf_cmd_out=rf_cmd_out,
                rf_actual_out=rf_actual_out,
            )

    wall_s = time.perf_counter() - t0

    result: dict[str, Any] = {
        "y": y_out,
        "error_signal": err_out,
        "rf_cmd": rf_cmd_out,
        "rf_actual": rf_actual_out,
        "wall_s": wall_s,
        "decimation_rate_Hz": decimation_rate_Hz,
        "physics_rate_Hz": physics_rate_Hz,
        "duration_s": duration_s,
        "noise_injection_point": "rf_actual_pre_step+disc_noise_pre_controller",
        "total_white_fm_amp": total_white_fm_amp,
        "lo_noise_std_per_physics": lo_noise_std_per_physics,
    }

    if autograd:
        # Stack list of (B,) tensors into (B, n_dec) tensor for loss computation
        result["y_tensor"] = torch.stack(y_tensor_list, dim=1)  # (B, n_dec)

    return result


# ---------------------------------------------------------------------------
# Internal: no-grad loop (evaluation)
# ---------------------------------------------------------------------------


def _run_nograd_loop(
    twin: ReducedTwin,
    controller: Any,
    state: Tensor,
    dist_tensor: Tensor,
    lo_noise_np: np.ndarray,
    pilot_np: np.ndarray,
    disc_noise_np: np.ndarray,
    rf_cmd: np.ndarray,
    n_dec: int,
    decimation: int,
    dt: float,
    y_out: np.ndarray,
    err_out: np.ndarray,
    rf_cmd_out: np.ndarray,
    rf_actual_out: np.ndarray,
) -> None:
    """No-grad inner loop (used in evaluation mode).

    Args:
        twin: ReducedTwin instance.
        controller: Controller object or None.
        state: (B, 8) initial state.
        dist_tensor: (B, n_physics, 3) disturbance tensor.
        lo_noise_np: (B, n_physics) LO noise array.
        pilot_np: (n_physics,) pilot tone array.
        disc_noise_np: (B, n_dec) discriminator noise array.
        rf_cmd: (B,) current rf command (Hz) — mutated in-place.
        n_dec: Number of decimation steps.
        decimation: Physics steps per decimation step.
        dt: Physics timestep (s).
        y_out: (B, n_dec) output buffer — filled in-place.
        err_out: (B, n_dec) output buffer — filled in-place.
        rf_cmd_out: (B, n_dec) output buffer — filled in-place.
        rf_actual_out: (B, n_dec) output buffer — filled in-place.
    """
    B = state.shape[0]
    ctrl = torch.zeros(B, 2, dtype=twin.dtype, device=twin.device)

    for k in range(n_dec):
        y_sum = torch.zeros(B, dtype=twin.dtype, device=twin.device)
        ci_sum = torch.zeros(B, dtype=twin.dtype, device=twin.device)
        rf_actual_sum = np.zeros(B, dtype=np.float64)

        for j in range(decimation):
            idx = k * decimation + j
            # LO noise + pilot: rf_actual per batch element
            rf_actual_batch = rf_cmd + lo_noise_np[:, idx] + float(pilot_np[idx])

            # Update ctrl tensor for all batch elements
            ctrl[:, 1] = torch.from_numpy(rf_actual_batch).to(
                dtype=twin.dtype, device=twin.device
            )
            ctrl[:, 0] = 0.0

            state = twin.step(state, ctrl, dist_tensor[:, idx, :], dt)

            # y: fractional frequency error using actual B and T
            B_now = dist_tensor[:, idx, 1]  # (B,)
            T_now = dist_tensor[:, idx, 0]  # (B,)
            y_sum += twin.fractional_frequency_error_with_B(state, ctrl, B_now, T_now)
            ci_sum += state[:, 6]
            rf_actual_sum += rf_actual_batch

        # Decimation-averaged outputs
        ci_avg = ci_sum / decimation
        err_clean = ci_avg.cpu().numpy()  # (B,)

        # Discriminator-input noise
        err_noisy = err_clean + disc_noise_np[:, k]

        y_out[:, k] = (y_sum / decimation).cpu().numpy()
        err_out[:, k] = err_noisy
        rf_cmd_out[:, k] = rf_cmd.copy()
        rf_actual_out[:, k] = rf_actual_sum / decimation

        # Controller update
        if controller is not None:
            for b in range(B):
                last_idx = (k + 1) * decimation - 1
                env = {
                    "T_K": float(dist_tensor[b, last_idx, 0].item()),
                    "B_uT": float(dist_tensor[b, last_idx, 1].item()),
                    "I_norm": float(dist_tensor[b, last_idx, 2].item()),
                }
                _, rf_corr = _controller_step(controller, float(err_noisy[b]), env)
                rf_cmd[b] = rf_corr


def _controller_step(
    controller: Any,
    error: float,
    env: dict[str, float],
) -> tuple[float, float]:
    """Call a controller with optional environment-sensor support."""
    try:
        return controller.step(error, env)
    except TypeError:
        return controller.step(error)


# ---------------------------------------------------------------------------
# Internal: autograd loop (APG training)
# ---------------------------------------------------------------------------


def _run_autograd_loop(
    twin: ReducedTwin,
    controller: Any,
    state: Tensor,
    dist_tensor: Tensor,
    lo_noise_np: np.ndarray,
    pilot_np: np.ndarray,
    disc_noise_np: np.ndarray,
    rf_cmd: np.ndarray,
    n_dec: int,
    decimation: int,
    dt: float,
    truncation_window: int,
    y_out: np.ndarray,
    err_out: np.ndarray,
    rf_cmd_out: np.ndarray,
    rf_actual_out: np.ndarray,
    y_tensor_list: list[Tensor],
    err_hist: Tensor,
    rf_hist: Tensor,
) -> None:
    """Autograd inner loop with truncated BPTT.

    The differentiable chain is:
        obs (err_hist, rf_hist) → policy.forward() → rf_corr_t →
        twin.step(state, ctrl) → state → y

    Every ``truncation_window`` physics substeps, ``state.detach_()`` and
    ``err_hist.detach_()`` / ``rf_hist.detach_()`` are called to release the
    accumulated graph and prevent gradient explosion.

    Args:
        twin: ReducedTwin instance.
        controller: APGPolicy (must have ``build_obs`` and ``forward`` methods).
        state: (B, 8) initial state (already detached from warm-up).
        dist_tensor: (B, n_physics, 3) disturbance tensor.
        lo_noise_np: (B, n_physics) LO noise (non-differentiable).
        pilot_np: (n_physics,) pilot tone.
        disc_noise_np: (B, n_dec) discriminator noise (non-differentiable).
        rf_cmd: (B,) current rf command (Hz) — NumPy array used for
            bookkeeping only; actual gradient-carrying rf_corr is a tensor.
        n_dec: Number of decimation steps.
        decimation: Physics steps per decimation step.
        dt: Physics timestep (s).
        truncation_window: Physics steps between BPTT detaches.
        y_out: (B, n_dec) NumPy output buffer.
        err_out: (B, n_dec) NumPy output buffer.
        rf_cmd_out: (B, n_dec) NumPy output buffer.
        rf_actual_out: (B, n_dec) NumPy output buffer.
        y_tensor_list: Accumulator for differentiable y tensors (one per dec step).
        err_hist: (B, n_error_history) sliding error history tensor.
        rf_hist: (B, n_rf_history) sliding rf history tensor.
    """
    B = state.shape[0]

    # LO noise as non-grad tensor
    lo_noise_t = torch.from_numpy(lo_noise_np).to(dtype=twin.dtype, device=twin.device)

    # Scalar for BPTT step counter
    physics_step_global = 0

    for k in range(n_dec):
        # Build observation and get rf_corr from policy (differentiable)
        if controller is not None:
            # Env scalars from disturbance at start of this decimation window
            idx_start = k * decimation
            env_t = dist_tensor[:, idx_start, :].detach()  # (B, 3): T_K, B_uT, I_norm
            obs = controller.build_obs(err_hist, rf_hist, env_t)  # (B, obs_dim)
            rf_corr_t = controller.forward(obs)  # (B,) — differentiable
        else:
            rf_corr_t = torch.zeros(B, dtype=twin.dtype, device=twin.device)

        # Build ctrl tensor: rf channel carries the policy output
        # LO noise at the start of this decimation window (non-grad)
        ctrl_rf = rf_corr_t + lo_noise_t[:, k * decimation]  # (B,) — still differentiable

        y_sum: Tensor = torch.zeros(B, dtype=twin.dtype, device=twin.device)
        ci_sum: Tensor = torch.zeros(B, dtype=twin.dtype, device=twin.device)
        rf_actual_sum = np.zeros(B, dtype=np.float64)

        for j in range(decimation):
            idx = k * decimation + j

            # For inner substeps, add fresh LO noise (non-grad) to the base cmd
            if j == 0:
                rf_for_step = ctrl_rf
            else:
                # Inner substep noise: non-differentiable additive noise
                extra_noise = lo_noise_t[:, idx].detach()
                rf_for_step = rf_corr_t + extra_noise

            ctrl_j = torch.stack(
                [torch.zeros(B, dtype=twin.dtype, device=twin.device), rf_for_step], dim=1
            )

            state = twin.step(state, ctrl_j, dist_tensor[:, idx, :].detach(), dt)

            B_now = dist_tensor[:, idx, 1].detach()
            T_now = dist_tensor[:, idx, 0].detach()
            y_sum = y_sum + twin.fractional_frequency_error_with_B(state, ctrl_j, B_now, T_now)
            ci_sum = ci_sum + state[:, 6]

            rf_actual_sum += (
                rf_corr_t.detach().cpu().numpy()
                + lo_noise_np[:, idx]
                + float(pilot_np[idx])
            )

            physics_step_global += 1
            # Truncated BPTT: detach state to release graph
            if physics_step_global % truncation_window == 0:
                state = state.detach()

        # Decimation-averaged y (differentiable)
        y_dec = y_sum / decimation  # (B,)
        y_tensor_list.append(y_dec)

        # Error signal: boxcar-averaged ci over the decimation window.
        err_clean = (ci_sum / decimation).detach().cpu().numpy()  # (B,)
        err_noisy = err_clean + disc_noise_np[:, k]

        y_out[:, k] = y_dec.detach().cpu().numpy()
        err_out[:, k] = err_noisy
        rf_cmd_out[:, k] = rf_corr_t.detach().cpu().numpy()
        rf_actual_out[:, k] = rf_actual_sum / decimation

        # Update sliding history windows (detached — history is bookkeeping,
        # not part of the gradient path for the NEXT decimation step; the
        # gradient only flows through y_dec in the current step)
        err_t_detached = torch.tensor(err_noisy, dtype=twin.dtype, device=twin.device)
        err_hist = torch.roll(err_hist, shifts=-1, dims=1)
        err_hist[:, -1] = err_t_detached

        rf_t_detached = rf_corr_t.detach()
        rf_hist = torch.roll(rf_hist, shifts=-1, dims=1)
        rf_hist[:, -1] = rf_t_detached
