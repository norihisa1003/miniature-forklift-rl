"""
Phase 2b Training Script
========================
Task: Fork insertion from randomized start position.
      (random X distance, yaw offset, fork height)

Key differences from Phase 2a:
- More timesteps (500k vs 20k) due to harder task
- Curriculum: optionally load Phase 2a weights as starting point
- EvalCallback to save best model during training
"""

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from forklift_mujoco_env import ForkliftMujocoEnv
import os

# ── Config ──────────────────────────────────────────────────
TOTAL_TIMESTEPS   = 1_000_000
N_ENVS            = 4          # parallel environments
MODEL_SAVE_PATH   = "models/phase2b_ppo"
# PHASE2A_MODEL = "training/models/phase2b_ppo/best_model"
PHASE2A_MODEL = None

# ── Environment ─────────────────────────────────────────────
train_env = make_vec_env(ForkliftMujocoEnv, n_envs=N_ENVS)
eval_env  = make_vec_env(ForkliftMujocoEnv, n_envs=1)

# ── Model ────────────────────────────────────────────────────
if PHASE2A_MODEL and os.path.exists(PHASE2A_MODEL + ".zip"):
    print(f"Loading Phase 2a weights from {PHASE2A_MODEL}")
    model = PPO.load(PHASE2A_MODEL, env=train_env, device="cpu")
else:
    print("Starting from scratch (no Phase 2a weights found)")
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
        ent_coef=0.01,         # exploration encouragement
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
print(f"\nStarting Phase 2b training: {TOTAL_TIMESTEPS:,} timesteps")
model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=eval_callback)

model.save(f"{MODEL_SAVE_PATH}/final_model")
print(f"\nTraining complete. Model saved to {MODEL_SAVE_PATH}/")

train_env.close()
eval_env.close()