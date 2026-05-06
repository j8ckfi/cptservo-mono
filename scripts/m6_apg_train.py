"""M6 APG training driver: curriculum training of the APGPolicy.

Curriculum (adjusted for CPU tractability):
    Phase 1: clean             20 epochs
    Phase 2: thermal_ramp      20 epochs
    Phase 3: b_field_drift     20 epochs
    Phase 4: laser_intensity   20 epochs
    Phase 5: all_stacked       20 epochs
    Total: 100 epochs

Per epoch: n_rollouts=4, rollout_duration_s=5s, physics_rate_Hz=10_000.
Sample seconds = 100 × 4 × 5 × 10_000 = 20M physics steps.

Checkpoints saved every 10 epochs to models/apg_epoch_{N}.pt.
Best (final) model saved to models/apg_best.pt.

Usage
-----
    cd /c/Users/Jack/Documents/Research/WIP/CPTServo
    python scripts/m6_apg_train.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Project path setup
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_SCRIPT_DIR))

from run_m3_m4_gates import log, make_calibrated_twin  # noqa: E402

from cptservo.policy.apg import APGPolicy  # noqa: E402
from cptservo.policy.training import _collect_obs_stats, train_curriculum  # noqa: E402

# ---------------------------------------------------------------------------
# Curriculum definition
# ---------------------------------------------------------------------------

PHASES: list[dict] = [
    {
        "disturbance_recipe_name": "clean",
        "n_epochs": 20,
        "n_rollouts": 4,
        "rollout_duration_s": 5.0,
        "lr": 1e-3,
        "weight_decay": 1e-5,
        "grad_clip": 1.0,
        "truncation_window": 50,
        "batch_size": 256,
        "disc_noise_amp_ci": 7.0e-4,
        "lo_noise_scale": 1.0,
        "rng_seed": 12345,
        "log_every": 1,
    },
    {
        "disturbance_recipe_name": "thermal_ramp",
        "n_epochs": 20,
        "n_rollouts": 4,
        "rollout_duration_s": 5.0,
        "lr": 5e-4,
        "weight_decay": 1e-5,
        "grad_clip": 1.0,
        "truncation_window": 50,
        "batch_size": 256,
        "disc_noise_amp_ci": 7.0e-4,
        "lo_noise_scale": 1.0,
        "rng_seed": 22345,
        "log_every": 1,
    },
    {
        "disturbance_recipe_name": "b_field_drift",
        "n_epochs": 20,
        "n_rollouts": 4,
        "rollout_duration_s": 5.0,
        "lr": 5e-4,
        "weight_decay": 1e-5,
        "grad_clip": 1.0,
        "truncation_window": 50,
        "batch_size": 256,
        "disc_noise_amp_ci": 7.0e-4,
        "lo_noise_scale": 1.0,
        "rng_seed": 32345,
        "log_every": 1,
    },
    {
        "disturbance_recipe_name": "laser_intensity_drift",
        "n_epochs": 20,
        "n_rollouts": 4,
        "rollout_duration_s": 5.0,
        "lr": 5e-4,
        "weight_decay": 1e-5,
        "grad_clip": 1.0,
        "truncation_window": 50,
        "batch_size": 256,
        "disc_noise_amp_ci": 7.0e-4,
        "lo_noise_scale": 1.0,
        "rng_seed": 42345,
        "log_every": 1,
    },
    {
        "disturbance_recipe_name": "all_stacked",
        "n_epochs": 20,
        "n_rollouts": 4,
        "rollout_duration_s": 5.0,
        "lr": 2e-4,
        "weight_decay": 1e-5,
        "grad_clip": 1.0,
        "truncation_window": 50,
        "batch_size": 256,
        "disc_noise_amp_ci": 7.0e-4,
        "lo_noise_scale": 1.0,
        "rng_seed": 52345,
        "log_every": 1,
    },
]


def main() -> None:
    models_dir = _PROJECT_ROOT / "models"
    models_dir.mkdir(exist_ok=True)
    data_dir = _PROJECT_ROOT / "data"
    data_dir.mkdir(exist_ok=True)

    log("=== M6 APG training start ===")
    log(f"  Project root: {_PROJECT_ROOT}")
    log(f"  Models dir:   {models_dir}")

    # Build policy
    policy = APGPolicy(
        n_error_history=8,
        n_rf_history=4,
        include_env_sensors=True,
        hidden_dims=(64, 64, 32),
        rf_limit_Hz=1000.0,
    )
    log(f"  Policy: obs_dim={policy.obs_dim}, n_params={policy.n_params()}")

    # Build twin
    twin = make_calibrated_twin()

    # Initialise observation normalisation from a short clean rollout
    log("  Collecting obs normalisation stats from clean rollout...")
    _collect_obs_stats(policy, twin, n_samples=2048, rng_seed=99999)
    log(
        f"  obs_mean range: [{policy.obs_mean.min().item():.4e}, "
        f"{policy.obs_mean.max().item():.4e}]"
    )
    log(
        f"  obs_std range:  [{policy.obs_std.min().item():.4e}, "
        f"{policy.obs_std.max().item():.4e}]"
    )

    # Run curriculum
    t_total_start = time.perf_counter()
    try:
        curriculum_result = train_curriculum(
            policy=policy,
            twin=twin,
            phases=PHASES,
            checkpoint_dir=models_dir,
            checkpoint_every=10,
            verbose=True,
        )
    except Exception as exc:
        import traceback
        log(f"!!! Curriculum CRASH: {type(exc).__name__}: {exc}")
        log(traceback.format_exc())
        # Save whatever we have to an emergency checkpoint
        try:
            emergency_path = models_dir / "apg_emergency.pt"
            policy.save(str(emergency_path))
            log(f"  Saved emergency checkpoint: {emergency_path}")
        except Exception as save_exc:
            log(f"  Emergency save failed: {save_exc}")
        raise
    total_wall = time.perf_counter() - t_total_start

    log("\n=== Training complete ===")
    log(f"  Total epochs:    {curriculum_result['total_epochs']}")
    log(f"  Final loss:      {curriculum_result['final_loss']:.4e}")
    log(f"  Wall time:       {total_wall:.1f} s ({total_wall/60:.1f} min)")
    log(f"  Checkpoints:     {sum(p['checkpoints_saved'] for p in curriculum_result['phases'])}")

    # Save final model
    best_path = models_dir / "apg_best.pt"
    policy.save(str(best_path))
    log(f"  Saved best model: {best_path}")

    # Save training metrics JSON
    metrics_path = data_dir / "m6_training_metrics.json"
    metrics = {
        "total_epochs": curriculum_result["total_epochs"],
        "final_loss": curriculum_result["final_loss"],
        "wall_time_s": total_wall,
        "checkpoints_saved": sum(p["checkpoints_saved"] for p in curriculum_result["phases"]),
        "n_params": policy.n_params(),
        "obs_dim": policy.obs_dim,
        "phases": [
            {
                "scenario": p["scenario"],
                "epoch_count": p["epoch_count"],
                "final_loss": p["final_loss"],
                "wall_time_s": p["wall_time_s"],
            }
            for p in curriculum_result["phases"]
        ],
        "all_loss_history": curriculum_result["all_loss_history"],
    }
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    log(f"  Saved training metrics: {metrics_path}")

    log(f"\n  DONE — model ready at {best_path}")


if __name__ == "__main__":
    main()
