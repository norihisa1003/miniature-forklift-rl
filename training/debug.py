"""
Debug Script: Step-by-step inspection of trained policy behavior
================================================================
Usage:
    source ~/rl_env/bin/activate
    cd ~/projects/miniature-forklift-rl

    # Debug default model (phase2c best)
    python3 training/debug.py

    # Debug a specific model
    python3 training/debug.py --model models/phase2b_ppo/best_model.zip

    # More episodes
    python3 training/debug.py --episodes 10
"""

import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from stable_baselines3 import PPO
from forklift_mujoco_env import ForkliftMujocoEnv


DEFAULT_MODEL    = Path(__file__).parent / "models" / "phase2c_ppo" / "best_model.zip"
DEFAULT_EPISODES = 5
LOG_INTERVAL     = 100  # print every N steps


def debug(model_path: Path, n_episodes: int):
    print("=== Forklift Policy Debug ===")
    print(f"Model: {model_path}")
    print()

    if not model_path.exists():
        print(f"ERROR: Model not found: {model_path}")
        return

    model = PPO.load(str(model_path))
    env   = ForkliftMujocoEnv()

    for ep in range(n_episodes):
        obs, _ = env.reset()

        fork_tip  = env.data.site_xpos[env._fork_tip_site_id].copy()
        fork_root = env.data.site_xpos[env._fork_root_site_id].copy()
        slot      = env.data.site_xpos[env._slot_entry_site_id].copy()
        fork_tip_z = env.data.site_xpos[env._fork_tip_site_id][2]
        slot_z     = env.data.site_xpos[env._slot_entry_site_id][2]
        
        print(f"Ep{ep+1} start:")
        print(f"  fork_tip  = ({fork_tip[0]:.3f}, {fork_tip[1]:.3f})")
        print(f"  fork_root = ({fork_root[0]:.3f}, {fork_root[1]:.3f})")
        print(f"  slot      = ({slot[0]:.3f}, {slot[1]:.3f})")
        print(f"  root_dy   = {abs(fork_root[1] - slot[1]):.3f}m")
        print(f"  fork_joint= {env.data.qpos[env._fork_qposadr]:.4f}m")
        print(f"  fork_tip_z = {fork_tip_z:.4f}m (slot_z={slot_z:.4f}m, diff={fork_tip_z-slot_z:.4f}m)")

        for step in range(500):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            if step % LOG_INTERVAL == 0:
                fork_tip = env.data.site_xpos[env._fork_tip_site_id].copy()
                print(f"  step{step:3d}: "
                    f"fork_x={fork_tip[0]:.3f} "
                    f"root_dy={info['root_dy']:.3f}m "
                    f"insertion={info['insertion_depth']:.3f}m "
                    f"align={info['alignment_error']:.3f}rad "
                    f"ncon={env.data.ncon}")
            
            # debug.pyのstep100あたりに追加
            if step == 100:
                print(f"  action: lw={action[0]:.3f} rw={action[1]:.3f} fork={action[2]:.3f}")

            if terminated or truncated:
                result = "SUCCESS" if info.get("success") else \
                         "TIMEOUT" if truncated else "FALLEN"
                print(f"  → {result} at step {step+1}, reward={reward:.1f}")
                break

        print()

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Debug a trained forklift policy.")
    parser.add_argument(
        "--model", type=Path, default=DEFAULT_MODEL,
        help="Path to model .zip file"
    )
    parser.add_argument(
        "--episodes", type=int, default=DEFAULT_EPISODES,
        help="Number of episodes to debug"
    )
    args = parser.parse_args()

    debug(model_path=args.model, n_episodes=args.episodes)