"""
Phase 2d Training Script
========================
Task: Fork insertion with randomized pallet height.
      Agent must control fork height via action[2] to align with slot.

Randomization ranges:
  X distance:   0.20 ~ 0.50m from pallet front
  Yaw offset:   ±30°
  Y offset:     ±2cm
  Pallet height: 0.00 ~ 0.09m  (slot Z = pallet_z + 0.0225)

Changes from Phase 2c:
  - MODEL_SAVE_PATH: phase2c_ppo → phase2d_ppo
  - INITIAL_MODEL:   phase2c best_model as starting weights

Usage:
    source ~/rl_env/bin/activate
    cd ~/projects/miniature-forklift-rl
    python training/train_phase2d.py

Outputs:
    training/models/phase2d_ppo/best_model.zip   ← best model during training
    training/models/phase2d_ppo/final_model.zip  ← final model
"""

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import (
    EvalCallback,
    StopTrainingOnNoModelImprovement
)
from forklift_mujoco_env import ForkliftMujocoEnv
import os

# ── Config ──────────────────────────────────────────────────

TOTAL_TIMESTEPS = 1_000_000
N_ENVS          = 4
MODEL_SAVE_PATH = "models/phase2d_ppo"

# Phase 2c の best_model を初期重みとして使用。
# action[2] が追加されてアーキテクチャが変わるため、重みは引き継げない。
# → None にしてスクラッチから学習する。
INITIAL_MODEL = None

# ── Environment ─────────────────────────────────────────────

train_env = make_vec_env(ForkliftMujocoEnv, n_envs=N_ENVS)
eval_env  = make_vec_env(ForkliftMujocoEnv, n_envs=1)

# ── Model ────────────────────────────────────────────────────

if INITIAL_MODEL and os.path.exists(INITIAL_MODEL + ".zip"):
    print(f"Loading weights from {INITIAL_MODEL}")
    model = PPO.load(INITIAL_MODEL, env=train_env, device="cpu")
else:
    print("Starting from scratch")
    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=1,
        device="cpu",
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        learning_rate=3e-4,
        gamma=0.99,
        ent_coef=0.01,
    )

# ── Callback ─────────────────────────────────────────────────

os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

stop_callback = StopTrainingOnNoModelImprovement(
    max_no_improvement_evals=20,
    min_evals=30,
    verbose=1
)

eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=MODEL_SAVE_PATH,
    log_path=MODEL_SAVE_PATH,
    eval_freq=10_000 // N_ENVS,
    n_eval_episodes=10,
    deterministic=True,
    callback_after_eval=stop_callback,
    verbose=1,
)

# ── Training ─────────────────────────────────────────────────

print(f"\nStarting Phase 2d training: {TOTAL_TIMESTEPS:,} timesteps")
model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=eval_callback)

model.save(f"{MODEL_SAVE_PATH}/final_model")
print(f"\nTraining complete. Model saved to {MODEL_SAVE_PATH}/")

train_env.close()
eval_env.close()