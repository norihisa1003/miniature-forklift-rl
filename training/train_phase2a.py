"""
Phase 2a Training Script: Fork Insertion (Fixed Position)
=========================================================
Uses PPO from Stable-Baselines3 to train the forklift to
insert forks into a pallet slot from a fixed start position.

Usage:
    source ~/rl_env/bin/activate
    cd ~/projects/miniature-forklift-rl
    python3 training/train_phase2a.py

Outputs:
    training/logs/phase2a/          ← TensorBoard logs
    training/models/phase2a_final   ← trained model
"""

import os
import time
import numpy as np
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import (
    EvalCallback,
    StopTrainingOnRewardThreshold,
    CheckpointCallback,
)
from stable_baselines3.common.monitor import Monitor

# Import our environment
import sys
sys.path.insert(0, str(Path(__file__).parent))
from forklift_mujoco_env import ForkliftMujocoEnv


# ── Paths ──────────────────────────────────────────────────

BASE_DIR   = Path(__file__).parent
LOG_DIR    = BASE_DIR / "logs" / "phase2a"
MODEL_DIR  = BASE_DIR / "models"
MODEL_NAME = "phase2a_final"

LOG_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ── Hyperparameters ────────────────────────────────────────
#
# These are reasonable starting values for a short-episode contact task.
# Adjust based on TensorBoard results:
#   - If loss diverges:    lower learning_rate (e.g. 1e-4)
#   - If converges slowly: increase n_steps or n_envs
#   - If reward plateaus:  check reward scaling in env

HYPERPARAMS = dict(
    # Number of parallel environments.
    # MuJoCo has no ROS2 overhead, so multiple envs are safe and fast.
    # 4 envs × n_steps=1024 = 4096 samples per update (standard PPO batch).
    n_envs=4,

    # PPO-specific
    n_steps=1024,           # steps per env before each update
    batch_size=256,         # minibatch size for gradient update
    n_epochs=10,            # number of passes over the collected data
    learning_rate=3e-4,     # Adam learning rate
    gamma=0.99,             # discount factor (long-horizon task)
    gae_lambda=0.95,        # GAE lambda for advantage estimation
    clip_range=0.2,         # PPO clip range
    ent_coef=0.01,          # entropy coefficient (encourages exploration)
    vf_coef=0.5,            # value function loss coefficient
    max_grad_norm=0.5,      # gradient clipping

    # Total training budget.
    # Phase 2a is simple (fixed start, short episode) so 200k steps should
    # be enough to see convergence. Increase to 500k if not converged.
    total_timesteps=200_000,
)


def make_env():
    """Factory function for a single monitored environment."""
    env = ForkliftMujocoEnv(render_mode=None)
    env = Monitor(env)
    return env


def train():
    print("=" * 60)
    print("Phase 2a Training: Fork Insertion (Fixed Position)")
    print("=" * 60)
    print(f"Log dir  : {LOG_DIR}")
    print(f"Model dir: {MODEL_DIR}")
    print()

    # ── Create vectorized training environments ──
    vec_env = make_vec_env(make_env, n_envs=HYPERPARAMS["n_envs"])

    # ── Create separate eval environment ──
    # EvalCallback needs its own env (not the training vec_env).
    # Single env is enough for evaluation.
    eval_env = Monitor(ForkliftMujocoEnv(render_mode=None))

    # ── Callbacks ──

    # Stop early if mean reward reaches threshold.
    # Success bonus=100, so consistent success → mean_reward ≈ 90+
    stop_callback = StopTrainingOnRewardThreshold(
        reward_threshold=150.0,
        verbose=1,
    )

    # Evaluate every 10k steps, save best model.
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(MODEL_DIR / "phase2a_best"),
        log_path=str(LOG_DIR),
        eval_freq=10_000 // HYPERPARAMS["n_envs"],  # per-env steps
        n_eval_episodes=20,
        deterministic=True,
        verbose=1,
        callback_on_new_best=stop_callback,
    )

    # Save checkpoint every 50k steps (recovery if training crashes).
    checkpoint_callback = CheckpointCallback(
        save_freq=50_000 // HYPERPARAMS["n_envs"],
        save_path=str(MODEL_DIR / "checkpoints"),
        name_prefix="phase2a",
        verbose=1,
    )

    # ── Build PPO model ──
    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        n_steps=HYPERPARAMS["n_steps"],
        batch_size=HYPERPARAMS["batch_size"],
        n_epochs=HYPERPARAMS["n_epochs"],
        learning_rate=HYPERPARAMS["learning_rate"],
        gamma=HYPERPARAMS["gamma"],
        gae_lambda=HYPERPARAMS["gae_lambda"],
        clip_range=HYPERPARAMS["clip_range"],
        ent_coef=HYPERPARAMS["ent_coef"],
        vf_coef=HYPERPARAMS["vf_coef"],
        max_grad_norm=HYPERPARAMS["max_grad_norm"],
        tensorboard_log=str(LOG_DIR),
        verbose=1,
    )

    print(f"Policy network: {model.policy}")
    print(f"Total timesteps: {HYPERPARAMS['total_timesteps']:,}")
    print()

    # ── Train ──
    start_time = time.time()

    model.learn(
        total_timesteps=HYPERPARAMS["total_timesteps"],
        callback=[eval_callback, checkpoint_callback],
        tb_log_name="PPO",
        reset_num_timesteps=True,
        progress_bar=True,
    )

    elapsed = time.time() - start_time
    print(f"\nTraining completed in {elapsed/60:.1f} minutes")

    # ── Save final model ──
    final_path = MODEL_DIR / MODEL_NAME
    model.save(str(final_path))
    print(f"Final model saved: {final_path}.zip")

    vec_env.close()
    eval_env.close()


if __name__ == "__main__":
    train()