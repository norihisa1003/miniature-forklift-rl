# training/evaluate.py
# Run the trained PPO model in Gazebo and visualize the result.

import os
import time
from stable_baselines3 import PPO
from forklift_env import ForkliftEnv

MODEL_PATH = os.path.expanduser(
    "~/projects/miniature-forklift-rl/training/models/best_model"
)

env = ForkliftEnv()
model = PPO.load(MODEL_PATH, env=env, device="cpu")

for episode in range(5):
    obs, _ = env.reset()
    total_reward = 0.0
    steps = 0

    print(f"\n── Episode {episode + 1} ──")

    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(int(action))
        total_reward += reward
        steps += 1

        print(
            f"  step {steps:03d} | action={action} "
            f"reward={reward:+.2f} | "
            f"pos=({obs[0]:.2f}, {obs[1]:.2f}) "
            f"yaw={obs[2]:.2f}"
        )

        if terminated or truncated:
            result = "GOAL" if reward > 50 else "TIMEOUT/WALL"
            print(f"  → {result} | total_reward={total_reward:.1f} steps={steps}")
            break

    time.sleep(1.0)

env.close()
