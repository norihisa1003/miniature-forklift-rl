"""
Phase 2a Evaluation Script: Visual playback of trained policy
=============================================================
Loads the trained model and runs it in the MuJoCo viewer.

Usage:
    source ~/rl_env/bin/activate
    cd ~/projects/miniature-forklift-rl
    python3 training/evaluate_phase2a.py

Controls (MuJoCo Viewer):
    Space : pause / resume
    Mouse : rotate / zoom view
"""

import time
import numpy as np
from pathlib import Path

from stable_baselines3 import PPO
import mujoco
import mujoco.viewer

import sys
sys.path.insert(0, str(Path(__file__).parent))
from forklift_mujoco_env import ForkliftMujocoEnv, STEPS_PER_ACTION


# ── Settings ───────────────────────────────────────────────

MODEL_PATH = Path(__file__).parent / "models" / "phase2a_best" / "best_model.zip"
N_EPISODES = 10       # number of episodes to play back
REALTIME   = True     # True = realtime display, False = as fast as possible


def evaluate():
    print("=== Phase 2a Visual Evaluation ===")
    print(f"Model: {MODEL_PATH}")

    # Load trained model
    model = PPO.load(str(MODEL_PATH))
    print("Model loaded.")

    # Create environment (no render_mode — we drive the viewer manually)
    env = ForkliftMujocoEnv(render_mode=None)

    success_count = 0

    # Launch MuJoCo interactive viewer
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:

        # Camera: side view to see fork insertion clearly
        viewer.cam.azimuth   = 160
        viewer.cam.elevation = -20
        viewer.cam.distance  = 1.5
        viewer.cam.lookat[:] = [0.3, 0.0, 0.05]

        for ep in range(N_EPISODES):
            obs, _ = env.reset()
            ep_reward = 0.0
            step = 0

            print(f"\nEpisode {ep + 1}/{N_EPISODES}")

            while viewer.is_running():
                # Get deterministic action from trained policy
                action, _ = model.predict(obs, deterministic=True)

                obs, reward, terminated, truncated, info = env.step(action)
                ep_reward += reward
                step += 1

                # Sync viewer with current physics state
                viewer.sync()

                # Realtime pacing:
                # STEPS_PER_ACTION physics steps × timestep = wall time per action
                if REALTIME:
                    time.sleep(STEPS_PER_ACTION * 0.002)

                if terminated or truncated:
                    success = info.get("success", False)
                    if success:
                        success_count += 1
                        print(f"  SUCCESS in {step} steps, reward={ep_reward:.1f}")
                    else:
                        reason = "timeout" if truncated else "fallen"
                        print(f"  {reason.upper()} at step {step}, reward={ep_reward:.1f}")

                    # Pause briefly so you can see the final state
                    time.sleep(1.0)
                    break

    print(f"\n=== Results ===")
    print(f"Success rate: {success_count}/{N_EPISODES} ({success_count/N_EPISODES*100:.0f}%)")
    env.close()


if __name__ == "__main__":
    evaluate()