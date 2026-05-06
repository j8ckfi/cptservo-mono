"""Tests for CPTServoEnv (M7 PPO environment).

Required by M7 spec (4 tests minimum):

1. test_env_reset_observation_shape
   -- reset() returns obs matching observation_space.shape.
2. test_env_step_returns_5tuple
   -- step() returns (obs, reward, terminated, truncated, info).
3. test_env_episode_terminates
   -- after rollout_duration_s seconds, terminated=True.
4. test_env_action_clamp
   -- action outside [-1, 1] is clipped before reaching the controller.

Additional tests:
5. test_env_obs_dtype_float32
   -- observation is float32 (SB3 requirement).
6. test_env_reward_negative
   -- reward is <= 0 (negative squared error).
7. test_env_reset_reproducible
   -- same seed -> same first step reward.
"""

from __future__ import annotations

import numpy as np
import pytest

from cptservo.policy.cpt_env import CPTServoEnv

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def short_env():
    """A CPTServoEnv with a very short episode for fast tests (0.05 s = 50 steps)."""
    return CPTServoEnv(
        disturbance_recipe_name="clean",
        rollout_duration_s=0.05,
        disc_noise_amp_ci=0.0,
        lo_noise_scale=0.0,
        rng_seed=7,
    )


@pytest.fixture
def standard_env():
    """Standard CPTServoEnv with default observation (15-dim)."""
    return CPTServoEnv(
        disturbance_recipe_name="clean",
        rollout_duration_s=0.1,
        disc_noise_amp_ci=7.0e-4,
        rng_seed=42,
    )


# ---------------------------------------------------------------------------
# Test 1 (spec): reset returns obs matching observation_space
# ---------------------------------------------------------------------------


def test_env_reset_observation_shape(standard_env: CPTServoEnv) -> None:
    """reset() must return observation whose shape matches observation_space.shape.

    Spec test 1.
    """
    obs, info = standard_env.reset()

    assert obs.shape == standard_env.observation_space.shape, (
        f"obs.shape={obs.shape} != observation_space.shape="
        f"{standard_env.observation_space.shape}"
    )
    assert standard_env.observation_space.contains(obs), (
        "obs returned by reset() is not contained in observation_space"
    )
    assert isinstance(info, dict), f"info must be dict, got {type(info)}"


# ---------------------------------------------------------------------------
# Test 2 (spec): step returns 5-tuple
# ---------------------------------------------------------------------------


def test_env_step_returns_5tuple(standard_env: CPTServoEnv) -> None:
    """step() must return a 5-tuple (obs, reward, terminated, truncated, info).

    Spec test 2.
    """
    standard_env.reset(seed=1)
    action = standard_env.action_space.sample()

    result = standard_env.step(action)

    assert len(result) == 5, f"step() must return 5-tuple, got {len(result)}-tuple"
    obs, reward, terminated, truncated, info = result

    assert obs.shape == standard_env.observation_space.shape, (
        f"step obs.shape={obs.shape} != obs_space.shape="
        f"{standard_env.observation_space.shape}"
    )
    assert isinstance(reward, float), f"reward must be float, got {type(reward)}"
    assert isinstance(terminated, (bool, np.bool_)), (
        f"terminated must be bool, got {type(terminated)}"
    )
    assert isinstance(truncated, (bool, np.bool_)), (
        f"truncated must be bool, got {type(truncated)}"
    )
    assert isinstance(info, dict), f"info must be dict, got {type(info)}"


# ---------------------------------------------------------------------------
# Test 3 (spec): episode terminates after rollout_duration_s
# ---------------------------------------------------------------------------


def test_env_episode_terminates(short_env: CPTServoEnv) -> None:
    """terminated becomes True exactly after n_dec = rollout_duration_s * 1000 steps.

    Spec test 3.  Uses a 0.05-s episode (50 steps at 1 kHz decimation).
    """
    short_env.reset(seed=5)

    n_steps = int(round(short_env.rollout_duration_s * short_env.decimation_rate_Hz))
    last_terminated = False
    last_truncated = False

    for i in range(n_steps):
        action = np.array([0.0], dtype=np.float32)
        _, _, terminated, truncated, _ = short_env.step(action)
        last_terminated = terminated
        last_truncated = truncated
        if terminated or truncated:
            assert i == n_steps - 1, (
                f"Episode terminated at step {i} but expected step {n_steps - 1}"
            )
            break

    assert last_terminated, (
        f"Episode did not terminate after {n_steps} steps "
        f"(rollout_duration_s={short_env.rollout_duration_s}s)"
    )
    assert not last_truncated, "Episode should be terminated (done), not truncated"


# ---------------------------------------------------------------------------
# Test 4 (spec): action outside [-1, 1] is clipped
# ---------------------------------------------------------------------------


def test_env_action_clamp(standard_env: CPTServoEnv) -> None:
    """Action outside [-1, 1] must be clipped; rf_cmd_Hz must stay within ±rf_limit_Hz.

    Spec test 4.  Sends extreme actions (+5, -5, +1e6) and checks info['rf_cmd_Hz'].
    """
    standard_env.reset(seed=10)
    rf_limit = standard_env.rf_limit_Hz

    extreme_actions = [
        np.array([5.0], dtype=np.float32),
        np.array([-5.0], dtype=np.float32),
        np.array([1e6], dtype=np.float32),
        np.array([-1e6], dtype=np.float32),
    ]

    for action in extreme_actions:
        _, _, _, _, info = standard_env.step(action)
        rf_cmd_hz = info["rf_cmd_Hz"]
        assert abs(rf_cmd_hz) <= rf_limit + 1e-6, (
            f"action={action[0]:.0f} produced rf_cmd_Hz={rf_cmd_hz:.2f} "
            f"which exceeds ±rf_limit_Hz={rf_limit}"
        )


def test_env_residual_lqr_action_mode_uses_lqr_hint() -> None:
    """Residual-LQR mode applies actions around the previous LQR hint."""
    env = CPTServoEnv(
        disturbance_recipe_name="thermal_ramp",
        rollout_duration_s=0.05,
        disc_noise_amp_ci=0.0,
        lo_noise_scale=0.0,
        action_mode="residual_lqr",
        residual_limit_Hz=2.0,
        rng_seed=21,
    )
    env.reset(seed=21)

    _, _, _, _, first_info = env.step(np.array([0.0], dtype=np.float32))
    first_hint = float(first_info["lqr_hint_Hz"])

    _, _, _, _, zero_info = env.step(np.array([0.0], dtype=np.float32))
    assert zero_info["rf_cmd_Hz"] == pytest.approx(first_hint)

    _, _, _, _, pos_info = env.step(np.array([5.0], dtype=np.float32))
    assert abs(pos_info["rf_cmd_Hz"]) <= env.rf_limit_Hz + 1e-6


# ---------------------------------------------------------------------------
# Test 5: observation dtype is float32
# ---------------------------------------------------------------------------


def test_env_obs_dtype_float32(standard_env: CPTServoEnv) -> None:
    """Observation arrays must be float32 (SB3 requires this for MlpPolicy)."""
    obs_reset, _ = standard_env.reset(seed=99)
    assert obs_reset.dtype == np.float32, (
        f"Reset obs dtype={obs_reset.dtype}, expected float32"
    )

    action = np.array([0.1], dtype=np.float32)
    obs_step, _, _, _, _ = standard_env.step(action)
    assert obs_step.dtype == np.float32, (
        f"Step obs dtype={obs_step.dtype}, expected float32"
    )


# ---------------------------------------------------------------------------
# Test 6: reward is non-positive
# ---------------------------------------------------------------------------


def test_env_reward_negative(standard_env: CPTServoEnv) -> None:
    """Reward must be <= 0 at every step (reward = -y_avg^2)."""
    standard_env.reset(seed=77)
    for _ in range(20):
        action = standard_env.action_space.sample()
        _, reward, terminated, _, _ = standard_env.step(action)
        assert reward <= 0.0 + 1e-12, (
            f"Reward must be <= 0, got {reward:.4e}"
        )
        if terminated:
            break


# ---------------------------------------------------------------------------
# Test 7: reset with same seed gives reproducible first-step reward
# ---------------------------------------------------------------------------


def test_env_reset_reproducible() -> None:
    """Same seed in reset() must yield the same first-step reward."""
    env1 = CPTServoEnv(
        disturbance_recipe_name="clean",
        rollout_duration_s=0.05,
        disc_noise_amp_ci=0.0,
        lo_noise_scale=0.0,
        rng_seed=123,
    )
    env2 = CPTServoEnv(
        disturbance_recipe_name="clean",
        rollout_duration_s=0.05,
        disc_noise_amp_ci=0.0,
        lo_noise_scale=0.0,
        rng_seed=123,
    )

    env1.reset(seed=123)
    env2.reset(seed=123)

    action = np.array([0.3], dtype=np.float32)
    _, rew1, _, _, _ = env1.step(action)
    _, rew2, _, _, _ = env2.step(action)

    assert abs(rew1 - rew2) < 1e-10, (
        f"Same seed should give same reward: rew1={rew1:.6e}, rew2={rew2:.6e}"
    )


# ---------------------------------------------------------------------------
# Test 8 (spec): reward depends on demeaned y, not raw y
# ---------------------------------------------------------------------------


def test_env_reward_demeaned() -> None:
    """Reward must be -(y - running_mean_y)^2, not -y^2.

    Spec test 5 (reward_demeaned).  Verifies that the per-rollout Welford mean
    is accumulated and subtracted, so the reward is variance-targeted rather
    than raw-offset-targeted (per M6 APG diagnosis).

    Strategy: run two identical envs.  In env_a we take zero actions so the
    loop converges to a non-zero y offset (DC component from disturbance).
    The reward from env_a after many steps must be closer to 0 than -(y^2)
    would be (because the mean is subtracted).  Concretely: check that
    info['y_demeaned'] != info['y_avg'] after several steps (mean has drifted).
    """
    env = CPTServoEnv(
        disturbance_recipe_name="thermal_ramp",
        rollout_duration_s=0.5,
        disc_noise_amp_ci=0.0,
        lo_noise_scale=0.0,
        rng_seed=55,
    )
    env.reset(seed=55)
    action = np.array([0.0], dtype=np.float32)

    # Run several steps to accumulate a non-trivial running mean
    y_avgs: list[float] = []
    y_demeaned_vals: list[float] = []
    rewards: list[float] = []

    for _ in range(30):
        _, reward, terminated, _, info = env.step(action)
        y_avgs.append(float(info["y_avg"]))
        y_demeaned_vals.append(float(info["y_demeaned"]))
        rewards.append(reward)
        if terminated:
            break

    # After >=2 steps, the running mean should be non-zero if y is non-zero,
    # so y_demeaned should differ from y_avg for most steps.
    assert len(y_avgs) >= 2, "Expected at least 2 steps"

    # The reward must equal -(y_demeaned)^2 at each step (not -(y_avg)^2)
    for i, (rew, yd) in enumerate(zip(rewards, y_demeaned_vals)):
        expected = -(yd**2)
        assert abs(rew - expected) < 1e-20, (
            f"Step {i}: reward={rew:.6e} but -(y_demeaned)^2={expected:.6e}; "
            "reward must use demeaned y"
        )

    # After several steps the running mean should have moved: y_demeaned != y_avg
    # (This would fail if demeaning were disabled.)
    diffs = [abs(ya - yd) for ya, yd in zip(y_avgs[5:], y_demeaned_vals[5:])]
    assert max(diffs) > 0.0, (
        "y_demeaned equals y_avg at all steps — demeaning appears disabled"
    )


# ---------------------------------------------------------------------------
# Test 9 (spec): env passes stable_baselines3 check_env
# ---------------------------------------------------------------------------


def test_env_compatible_with_sb3() -> None:
    """CPTServoEnv must pass stable_baselines3.common.env_checker.check_env.

    Spec test 6 (compatible_with_sb3).  check_env verifies observation/action
    space consistency, dtype requirements, and step/reset API shape contracts.
    """
    from stable_baselines3.common.env_checker import check_env

    env = CPTServoEnv(
        disturbance_recipe_name="clean",
        rollout_duration_s=0.05,
        disc_noise_amp_ci=0.0,
        lo_noise_scale=0.0,
        rng_seed=11,
    )
    # check_env raises AssertionError or warns on API violations.
    # warn=False turns warnings into errors so we catch them.
    check_env(env, warn=True)  # passes silently if compliant
