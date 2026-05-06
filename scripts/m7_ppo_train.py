"""M7 PPO training driver for the CPT-clock servo.

Trains a PPO (MLP policy) agent on the CPTServoEnv using stable-baselines3.
Long-horizon curriculum (M7 spec): episode length 20 s to target slow drift
where the controller demonstrates real wins over PI.

    Stage 1: clean                   50 k steps  (sanity — confirms loop alive)
    Stage 2: thermal_ramp           200 k steps  (slow drift the controller must track)
    Stage 3: b_field_drift          100 k steps
    Stage 4: laser_intensity_drift  100 k steps
    Stage 5: all_stacked            200 k steps  (final compounding)

Total ~650 k env steps.  4 parallel envs (DummyVecEnv; memory-light).

Memory safety
-------------
- N_ENVS = 4 by default; drops to 2 if RSS > 5 GB after stage 1.
- no_grad rollouts throughout (SB3 PPO default).
- Episode length = 20 s × 1 kHz = 20 000 steps per env reset.

Outputs
-------
models/ppo_best.zip   — best model by mean episode reward
logs/ppo_train.log    — training log with stage metrics

Usage
-----
    cd /c/Users/Jack/Documents/Research/WIP/CPTServo
    python scripts/m7_ppo_train.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Project root on path
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_SCRIPT_DIR))

from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.callbacks import BaseCallback  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv  # noqa: E402

from cptservo.policy.cpt_env import CPTServoEnv  # noqa: E402

# ---------------------------------------------------------------------------
# Constants — change only here, not in individual runs
# ---------------------------------------------------------------------------
N_ENVS: int = 8  # SubprocVecEnv: real parallelism across 8 cores
ROLLOUT_DURATION_S: float = 20.0  # LONG-HORIZON: 20s targets slow drift
DISC_NOISE_AMP_CI: float = 7.0e-4
RF_LIMIT_HZ: float = 1_000.0
BASE_SEED: int = 42

# Curriculum: (scenario_name, total_timesteps)
# Long-horizon design per M7 spec — thermal_ramp gets the most steps because
# it is the primary gate scenario (PPO must beat PI on thermal_ramp at tau=10s).
CURRICULUM: list[tuple[str, int]] = [
    ("clean", 50_000),            # sanity: confirms loop is alive
    ("thermal_ramp", 200_000),    # primary gate scenario — slow drift tracking
    ("b_field_drift", 100_000),
    ("laser_intensity_drift", 100_000),
    ("all_stacked", 200_000),     # final compounding disturbance
]

# PPO hyperparameters (conservative; memory-light)
PPO_KWARGS: dict = {
    "learning_rate": 3e-4,
    "n_steps": 512,       # steps per env before update
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.01,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "verbose": 1,  # per-rollout SB3 logger output (visibility on hangs)
    "policy_kwargs": {"net_arch": [64, 64, 32]},
}


def log(msg: str, log_fh=None) -> None:
    """Timestamped log to stdout and optional file handle."""
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    try:
        print(line, flush=True)
    except UnicodeEncodeError:
        print(line.encode("ascii", errors="replace").decode("ascii"), flush=True)
    if log_fh is not None:
        log_fh.write(line + "\n")
        log_fh.flush()


def get_rss_gb() -> float:
    """Return current process RSS in GB (0 if psutil not available)."""
    try:
        import os

        import psutil
        proc = psutil.Process(os.getpid())
        return proc.memory_info().rss / (1024**3)
    except ImportError:
        return 0.0


class StageRewardCallback(BaseCallback):
    """Collects mean episode reward during a training stage.

    Args:
        stage_name: Name of the current curriculum stage.
        log_fn: Logging function.
        log_fh: Optional file handle for log output.
    """

    def __init__(self, stage_name: str, log_fn, log_fh=None) -> None:
        super().__init__(verbose=0)
        self.stage_name = stage_name
        self.log_fn = log_fn
        self.log_fh = log_fh
        self.episode_rewards: list[float] = []
        self._ep_reward_sum: float = 0.0
        self._ep_steps: int = 0

    def _on_step(self) -> bool:
        # SB3 DummyVecEnv provides 'rewards' and 'dones' in locals
        rewards = self.locals.get("rewards", [])
        dones = self.locals.get("dones", [])
        for i, (rew, done) in enumerate(zip(rewards, dones)):
            self._ep_reward_sum += float(rew)
            self._ep_steps += 1
            if done:
                self.episode_rewards.append(self._ep_reward_sum)
                self._ep_reward_sum = 0.0
                self._ep_steps = 0
        return True

    def mean_reward(self) -> float:
        if not self.episode_rewards:
            return float("nan")
        return float(np.mean(self.episode_rewards[-20:]))  # last 20 episodes


def make_env(scenario: str, seed: int, env_idx: int):
    """Factory for a single CPTServoEnv instance."""
    def _init():
        return CPTServoEnv(
            disturbance_recipe_name=scenario,
            rollout_duration_s=ROLLOUT_DURATION_S,
            disc_noise_amp_ci=DISC_NOISE_AMP_CI,
            rf_limit_Hz=RF_LIMIT_HZ,
            rng_seed=seed + env_idx * 1000,
        )
    return _init


def train_ppo(
    models_dir: Path,
    logs_dir: Path,
    n_envs: int = N_ENVS,
) -> dict:
    """Run the full PPO curriculum and return training metrics.

    Args:
        models_dir: Directory to save model checkpoints.
        logs_dir: Directory to write ppo_train.log.
        n_envs: Number of parallel environments.

    Returns:
        Dict with training metrics for embedding in gate_M7.json.
    """
    models_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    log_path = logs_dir / "ppo_train.log"
    log_fh = open(log_path, "w", encoding="utf-8")  # noqa: SIM115

    def _log(msg: str) -> None:
        log(msg, log_fh)

    _log("=== M7 PPO Training Start ===")
    _log(f"  n_envs={n_envs}, curriculum={[s for s, _ in CURRICULUM]}")
    _log(f"  rollout_duration_s={ROLLOUT_DURATION_S}, disc_noise={DISC_NOISE_AMP_CI:.0e}")

    wall_t0 = time.perf_counter()

    # Build initial env on stage-1 scenario
    first_scenario = CURRICULUM[0][0]
    env_fns = [make_env(first_scenario, BASE_SEED, i) for i in range(n_envs)]
    vec_env = SubprocVecEnv(env_fns) if len(env_fns) > 1 else DummyVecEnv(env_fns)

    # Initialise PPO
    model = PPO("MlpPolicy", vec_env, seed=BASE_SEED, **PPO_KWARGS)
    _log(f"  PPO policy: {model.policy}")

    total_steps_so_far = 0
    stage_metrics: list[dict] = []
    best_mean_reward = float("-inf")
    best_model_path = str(models_dir / "ppo_best.zip")

    for stage_idx, (scenario, timesteps) in enumerate(CURRICULUM, start=1):
        _log(f"  Stage {stage_idx}/5: scenario={scenario}, timesteps={timesteps}")

        # Check memory pressure before each stage
        rss_gb = get_rss_gb()
        if rss_gb > 5.0 and n_envs > 2:
            _log(f"  [WARN] RSS={rss_gb:.2f} GB > 5 GB; dropping to n_envs=2")
            n_envs = 2

        # Rebuild vec_env for this scenario
        env_fns = [make_env(scenario, BASE_SEED + stage_idx * 100, i) for i in range(n_envs)]
        new_vec_env = SubprocVecEnv(env_fns) if len(env_fns) > 1 else DummyVecEnv(env_fns)
        model.set_env(new_vec_env)
        vec_env = new_vec_env

        callback = StageRewardCallback(stage_name=scenario, log_fn=_log, log_fh=log_fh)
        t_stage_start = time.perf_counter()

        model.learn(
            total_timesteps=timesteps,
            callback=callback,
            reset_num_timesteps=False,
            progress_bar=False,
        )

        t_stage_wall = time.perf_counter() - t_stage_start
        total_steps_so_far += timesteps
        mean_rew = callback.mean_reward()

        _log(
            f"    stage {stage_idx} done: wall={t_stage_wall:.1f}s, "
            f"mean_rew={mean_rew:.4e}, total_steps={total_steps_so_far}"
        )

        stage_metrics.append({
            "stage": stage_idx,
            "scenario": scenario,
            "timesteps": timesteps,
            "wall_s": round(t_stage_wall, 2),
            "mean_reward_last20ep": round(mean_rew, 8) if np.isfinite(mean_rew) else None,
        })

        # Save best model (by stage-5 / final reward; otherwise save each improvement)
        if np.isfinite(mean_rew) and mean_rew > best_mean_reward:
            best_mean_reward = mean_rew
            model.save(best_model_path)
            _log(f"    [best] saved model (mean_rew={mean_rew:.4e})")

        rss_gb = get_rss_gb()
        if rss_gb > 0:
            _log(f"    RSS after stage {stage_idx}: {rss_gb:.2f} GB")

    # Always save final model (overwrite if last stage is best)
    final_path = str(models_dir / "ppo_final.zip")
    model.save(final_path)
    # Ensure ppo_best.zip exists even if no improvement tracked
    if not Path(best_model_path).exists():
        model.save(best_model_path)

    total_wall = time.perf_counter() - wall_t0
    _log(f"=== Training complete: total_wall={total_wall:.1f}s ===")
    log_fh.close()

    # Final stage-5 all_stacked mean reward
    final_mean_rew = stage_metrics[-1]["mean_reward_last20ep"]

    return {
        "total_timesteps": total_steps_so_far,
        "stages": stage_metrics,
        "wall_time_s": round(total_wall, 2),
        "final_mean_reward": final_mean_rew if final_mean_rew is not None else float("nan"),
        "best_model_path": best_model_path,
        "n_envs_used": n_envs,
    }


def main() -> None:
    models_dir = _PROJECT_ROOT / "models"
    logs_dir = _PROJECT_ROOT / "logs"
    data_dir = _PROJECT_ROOT / "data"
    data_dir.mkdir(exist_ok=True)

    train_metrics = train_ppo(models_dir=models_dir, logs_dir=logs_dir)

    # Write a quick summary JSON for inspection (data/ for gate script to read)
    summary_path = data_dir / "ppo_train_summary.json"
    summary_path.write_text(json.dumps(train_metrics, indent=2), encoding="utf-8")
    log(f"Wrote {summary_path}")
    log(f"Best model: {train_metrics['best_model_path']}")
    log(f"Total wall time: {train_metrics['wall_time_s']:.1f}s")


if __name__ == "__main__":
    main()
