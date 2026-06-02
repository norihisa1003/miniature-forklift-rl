"""
Phase 2c Training Script
========================
Task: Fork insertion from fully randomized start position.
      (random X distance, yaw offset, Y offset)

Randomization ranges:
  X distance: 0.20 ~ 0.50m from pallet front
  Yaw offset: ±30°
  Y offset:   ±15cm
  Fork height: fixed (slot-aligned)

Usage:
    source ~/rl_env/bin/activate
    cd ~/projects/miniature-forklift-rl
    python training/train_phase2c.py

Outputs:
    training/models/phase2c_ppo/best_model.zip   ← best model during training
    training/models/phase2c_ppo/final_model.zip  ← final model
"""

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from forklift_mujoco_env import ForkliftMujocoEnv
import os

# ── Config ──────────────────────────────────────────────────

TOTAL_TIMESTEPS = 1_000_000
N_ENVS          = 4
MODEL_SAVE_PATH = "models/phase2c_ppo"

# Set to a model path to continue training from a checkpoint.
# e.g. "models/phase2b_ppo/best_model"
# Set to None to train from scratch.
INITIAL_MODEL = "models/phase2b_ppo/best_model"

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
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=MODEL_SAVE_PATH,
    log_path=MODEL_SAVE_PATH,
    eval_freq=10_000 // N_ENVS,
    n_eval_episodes=10,
    deterministic=True,
    verbose=1,
)

# ── Training ─────────────────────────────────────────────────

print(f"\nStarting Phase 2c training: {TOTAL_TIMESTEPS:,} timesteps")
model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=eval_callback)

model.save(f"{MODEL_SAVE_PATH}/final_model")
print(f"\nTraining complete. Model saved to {MODEL_SAVE_PATH}/")

train_env.close()
eval_env.close()