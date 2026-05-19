# training/train.py

import os
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import (
    EvalCallback,
    StopTrainingOnRewardThreshold,
)
from forklift_env import ForkliftEnv

LOG_DIR   = os.path.expanduser("~/projects/miniature-forklift-rl/training/logs")
MODEL_DIR = os.path.expanduser("~/projects/miniature-forklift-rl/training/models")
os.makedirs(LOG_DIR,   exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

env      = Monitor(ForkliftEnv(), filename=os.path.join(LOG_DIR, "train"))
eval_env = Monitor(ForkliftEnv(), filename=os.path.join(LOG_DIR, "eval"))

stop_callback = StopTrainingOnRewardThreshold(
    reward_threshold=80.0, verbose=1
)
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=MODEL_DIR,
    log_path=LOG_DIR,
    eval_freq=5_000,
    n_eval_episodes=5,
    callback_on_new_best=stop_callback,
    verbose=1,
)

model = PPO(
    policy="MlpPolicy",
    env=env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    verbose=1,
    tensorboard_log=LOG_DIR,
    device="cpu",   # MlpPolicy runs faster on CPU than GPU
)

print("Starting training...")
model.learn(
    total_timesteps=200_000,
    callback=eval_callback,
)

final_path = os.path.join(MODEL_DIR, "forklift_ppo_final")
model.save(final_path)
print(f"Training complete. Model saved to {final_path}")

eval_env.close()
env.close()
