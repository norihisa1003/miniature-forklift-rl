"""
Evaluation Script: Visual playback of trained policy
=====================================================
Loads a trained model and runs it in the MuJoCo viewer.

Usage:
    source ~/rl_env/bin/activate
    cd ~/projects/miniature-forklift-rl

    # Evaluate default model (phase2c best)
    python3 training/evaluate.py

    # Evaluate a specific model
    python3 training/evaluate.py --model models/phase2b_ppo/best_model.zip

    # Run without viewer (faster, for CI)
    python3 training/evaluate.py --no-viewer

Controls (MuJoCo Viewer):
    Space : pause / resume
    Mouse : rotate / zoom view
"""

import argparse
import time
from pathlib import Path

from stable_baselines3 import PPO
import mujoco
import mujoco.viewer

import sys
sys.path.insert(0, str(Path(__file__).parent))
from forklift_mujoco_env import ForkliftMujocoEnv, STEPS_PER_ACTION


# ── Defaults ────────────────────────────────────────────────

DEFAULT_MODEL = Path(__file__).parent / "models" / "phase2d_ppo" / "best_model.zip"
DEFAULT_N_EPISODES = 10
DEFAULT_REALTIME   = True


# ── Evaluate ────────────────────────────────────────────────

def evaluate(model_path: Path, n_episodes: int, realtime: bool, use_viewer: bool):
    print("=== Forklift Policy Evaluation ===")
    print(f"Model     : {model_path}")
    print(f"Episodes  : {n_episodes}")
    print(f"Viewer    : {'on' if use_viewer else 'off'}")
    print()

    if not model_path.exists():
        print(f"ERROR: Model not found: {model_path}")
        return

    model = PPO.load(str(model_path))
    env   = ForkliftMujocoEnv(render_mode=None)

    results = []  # list of (success, steps, reward)

    def run_episodes(viewer=None):
        for ep in range(n_episodes):
            obs, _ = env.reset()
            ep_reward = 0.0
            step = 0

            print(f"Episode {ep + 1}/{n_episodes}")

            while True:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                ep_reward += reward
                step += 1

                if viewer is not None:
                    viewer.sync()
                    if not viewer.is_running():
                        return
                    if realtime:
                        time.sleep(STEPS_PER_ACTION * 0.002)

                if terminated or truncated:
                    success = info.get("success", False)
                    results.append((success, step, ep_reward))

                    if success:
                        print(f"  SUCCESS in {step} steps, reward={ep_reward:.1f}")
                    else:
                        reason = "TIMEOUT" if truncated else "FALLEN"
                        print(f"  {reason} at step {step}, reward={ep_reward:.1f}")

                    if viewer is not None:
                        time.sleep(1.0)
                    break

    if use_viewer:
        with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
            viewer.cam.azimuth   = 160
            viewer.cam.elevation = -20
            viewer.cam.distance  = 1.5
            viewer.cam.lookat[:] = [0.3, 0.0, 0.05]
            run_episodes(viewer)
    else:
        run_episodes()

    env.close()

    # ── Summary ─────────────────────────────────────────────
    if not results:
        return

    n_success = sum(1 for s, _, _ in results if s)
    avg_steps  = sum(st for _, st, _ in results if results) / len(results)
    avg_reward = sum(r for _, _, r in results) / len(results)

    print()
    print("=== Results ===")
    print(f"Success rate : {n_success}/{n_episodes} ({n_success/n_episodes*100:.0f}%)")
    print(f"Avg steps    : {avg_steps:.1f}")
    print(f"Avg reward   : {avg_reward:.1f}")


# ── Entry point ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained forklift policy.")
    parser.add_argument(
        "--model", type=Path, default=DEFAULT_MODEL,
        help="Path to model .zip file"
    )
    parser.add_argument(
        "--episodes", type=int, default=DEFAULT_N_EPISODES,
        help="Number of episodes to evaluate"
    )
    parser.add_argument(
        "--no-viewer", action="store_true",
        help="Run without MuJoCo viewer (faster)"
    )
    parser.add_argument(
        "--no-realtime", action="store_true",
        help="Run as fast as possible (viewer only)"
    )
    args = parser.parse_args()

    evaluate(
        model_path  = args.model,
        n_episodes  = args.episodes,
        realtime    = not args.no_realtime,
        use_viewer  = not args.no_viewer,
    )