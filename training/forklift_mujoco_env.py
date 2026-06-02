"""
ForkliftMujocoEnv - Phase 2c: Fork Insertion (Randomized Position)
==================================================================
Task: Forklift starts at a randomized position in front of the pallet.
      Agent learns to align and insert the forks into the pallet slot.

Randomization (Phase 2c):
  - X distance to pallet: 0.20 ~ 0.50m
  - Yaw offset:           ±30°
  - Y offset:             ±15cm
  - Fork height:          fixed at FORK_INIT_HEIGHT (slot-aligned)

Observation space (7 dims):
  [0]  dx:           fork_tip → slot_center, X direction
  [1]  dy:           fork_tip → slot_center, Y direction
  [2]  dz:           fork_tip → slot_center, Z direction
  [3]  yaw:          forklift heading angle (radians)
  [4]  fork_height:  fork lift joint position (m)
  [5]  lw_vel:       left wheel angular velocity (normalized)
  [6]  rw_vel:       right wheel angular velocity (normalized)

Action space (3 dims, continuous [-1, 1]):
  [0]  left_wheel_vel  → scaled to ±5.0 rad/s
  [1]  right_wheel_vel → scaled to ±5.0 rad/s
  [2]  fork_lift_vel   → scaled to ±0.3 m/s

Reward:
  - Time penalty:       -0.1 per step
  - Alignment reward:   improvement in fork-slot alignment × 50
  - Lateral reward:     Y-distance reduction × 50
  - Insertion reward:   tip/root distance reduction × 50 (gated by alignment)
  - Collision penalty:  -1.0 per step when fork contacts pallet
  - Success bonus:      +1000 (fork fully inserted)
  - Fall penalty:       -50 (base_z too low)
"""

import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces
from pathlib import Path


# ── Simulation constants ────────────────────────────────────

SCENE_XML        = Path(__file__).parent.parent / "sim" / "worlds" / "scene.xml"
TIMESTEP         = 0.002   # matches scene.xml
STEPS_PER_ACTION = 10      # control frequency = 50Hz
MAX_EPISODE_STEPS = 500

WHEEL_VEL_MAX = 5.0        # rad/s
FORK_VEL_MAX  = 0.3        # m/s
MIN_BASE_Z    = 0.05       # below this = fallen


# ── Forklift geometry ───────────────────────────────────────

FORKLIFT_START_Z   = 0.13   # base_link height
FORK_INIT_HEIGHT   = 0.0095  # fork joint position (aligned to slot)
FORK_TIP_OFFSET_X  = 0.50   # fork tip to base_link (measured)
PALLET_FRONT_X     = 0.40   # world X of pallet slot entry face
FORK_LENGTH        = 0.32   # m


# ── Randomization ranges ────────────────────────────────────

RAND_DIST_MIN = 0.20   # m (distance from pallet front to fork tip)
RAND_DIST_MAX = 0.50   # m
RAND_YAW_MAX  = np.pi / 6   # ±30°
RAND_Y_MAX    = 0.02   # m (±15cm)


class ForkliftMujocoEnv(gym.Env):
    """MuJoCo Gymnasium environment for forklift pallet insertion (Phase 2c)."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, render_mode=None):
        super().__init__()

        self.render_mode = render_mode

        self.model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
        self.data  = mujoco.MjData(self.model)
        self._cache_ids()

        obs_low  = np.array([-2.0, -2.0, -0.5, -np.pi, 0.0, -1.0, -1.0, -2.0], dtype=np.float32)
        obs_high = np.array([ 2.0,  2.0,  0.5,  np.pi, 0.25, 1.0,  1.0,  2.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

        self._renderer = None
        self._step_count = 0

    # ── ID cache ────────────────────────────────────────────

    def _cache_ids(self):
        def body_id(n):  return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY,  n)
        def site_id(n):  return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE,  n)
        def joint_id(n): return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)
        def geom_id(n):  return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM,  n)

        self._base_body_id     = body_id("base_link")
        self._fork_tip_site_id = site_id("fork_tip_site")
        self._fork_root_site_id = site_id("fork_root_site")
        self._slot_entry_site_id = site_id("slot_center")
        self._slot_exit_site_id  = site_id("slot_center_exit")

        self._root_qposadr = self.model.jnt_qposadr[joint_id("root")]
        self._fork_qposadr = self.model.jnt_qposadr[joint_id("fork_lift_joint")]

        self._pallet_geom_ids = {
            geom_id(n) for n in
            ["bottom_deck", "leg_left_outer", "leg_inner", "leg_right_outer", "top_deck"]
        }
        self._fork_geom_ids = {geom_id(n) for n in ["left_fork_geom", "right_fork_geom"]}

    # ── Reset ───────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        # Randomize start pose
        distance = self.np_random.uniform(RAND_DIST_MIN, RAND_DIST_MAX)
        start_x  = PALLET_FRONT_X - distance - FORK_TIP_OFFSET_X
        start_y  = self.np_random.uniform(-RAND_Y_MAX, RAND_Y_MAX)
        yaw      = self.np_random.uniform(-RAND_YAW_MAX, RAND_YAW_MAX)
        qw, qz   = np.cos(yaw / 2.0), np.sin(yaw / 2.0)

        q = self._root_qposadr
        self.data.qpos[q + 0] = start_x
        self.data.qpos[q + 1] = start_y
        self.data.qpos[q + 2] = FORKLIFT_START_Z
        self.data.qpos[q + 3] = qw
        self.data.qpos[q + 4] = 0.0
        self.data.qpos[q + 5] = 0.0
        self.data.qpos[q + 6] = qz

        self.data.qpos[self._fork_qposadr] = FORK_INIT_HEIGHT
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

        # Initialize previous-state variables for incremental rewards
        fork_tip   = self.data.site_xpos[self._fork_tip_site_id].copy()
        fork_root  = self.data.site_xpos[self._fork_root_site_id].copy()
        slot_entry = self.data.site_xpos[self._slot_entry_site_id].copy()
        slot_exit  = self.data.site_xpos[self._slot_exit_site_id].copy()

        self._prev_alignment_error  = self._calc_alignment_error(
            fork_tip, fork_root, slot_entry, slot_exit
        )
        self._prev_root_dy          = abs(fork_root[1] - slot_entry[1])
        self._prev_fork_tip_x       = fork_tip[0]
        self._prev_insertion_depth  = max(0.0, fork_tip[0] - slot_entry[0])
        self._step_count = 0
        return self._get_obs(), {}

    # ── Step ────────────────────────────────────────────────

    def step(self, action):
        self.data.ctrl[0] = float(action[0]) * WHEEL_VEL_MAX
        self.data.ctrl[1] = float(action[1]) * WHEEL_VEL_MAX
        self.data.ctrl[2] = float(action[2]) * FORK_VEL_MAX

        for _ in range(STEPS_PER_ACTION):
            mujoco.mj_step(self.model, self.data)

        self._step_count += 1
        reward, terminated, info = self._compute_reward()
        truncated = self._step_count >= MAX_EPISODE_STEPS
        obs = self._get_obs()

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    # ── Observation ─────────────────────────────────────────

    def _get_obs(self):
        fork_tip = self.data.site_xpos[self._fork_tip_site_id]
        slot_ctr = self.data.site_xpos[self._slot_entry_site_id]
        delta    = slot_ctr - fork_tip

        fork_root  = self.data.site_xpos[self._fork_root_site_id]
        slot_entry = self.data.site_xpos[self._slot_entry_site_id]
        root_dy    = slot_entry[1] - fork_root[1]

        q   = self._root_qposadr
        qw  = self.data.qpos[q + 3]
        qz  = self.data.qpos[q + 6]
        yaw = np.arctan2(2.0 * qw * qz, 1.0 - 2.0 * qz * qz)

        fork_height = self.data.qpos[self._fork_qposadr]
        lw_vel = self.data.sensordata[2] / WHEEL_VEL_MAX
        rw_vel = self.data.sensordata[3] / WHEEL_VEL_MAX

        return np.array(
            [delta[0], delta[1], delta[2], yaw, fork_height, lw_vel, rw_vel, root_dy],
            dtype=np.float32
        )

    # ── Reward ──────────────────────────────────────────────

    def _calc_alignment_error(self, fork_tip, fork_root, slot_entry, slot_exit):
        fork_angle = np.arctan2(fork_tip[1] - fork_root[1], fork_tip[0] - fork_root[0])
        slot_angle = np.arctan2(slot_exit[1] - slot_entry[1], slot_exit[0] - slot_entry[0])
        err = abs(fork_angle - slot_angle)
        if err > np.pi:
            err = 2 * np.pi - err
        return err

    def _compute_reward(self):
        info = {}
        terminated = False

        fork_tip   = self.data.site_xpos[self._fork_tip_site_id].copy()
        fork_root  = self.data.site_xpos[self._fork_root_site_id].copy()
        slot_entry = self.data.site_xpos[self._slot_entry_site_id].copy()
        slot_exit  = self.data.site_xpos[self._slot_exit_site_id].copy()

        alignment_error = self._calc_alignment_error(fork_tip, fork_root, slot_entry, slot_exit)
        root_dy         = abs(fork_root[1] - slot_entry[1])
        insertion_depth = max(0.0, fork_tip[0] - slot_entry[0])

        reward = -0.1  # time penalty

        # 1. Root Y alignment (main task)
        reward += (self._prev_root_dy - root_dy) * 100.0
        self._prev_root_dy = root_dy

        # 2. Alignment improvement
        reward += (self._prev_alignment_error - alignment_error) * 50.0
        self._prev_alignment_error = alignment_error

        # 3a. Forward progress (small: encourages moving toward pallet)
        forward_progress = fork_tip[0] - self._prev_fork_tip_x
        reward += forward_progress * 50.0
        self._prev_fork_tip_x = fork_tip[0]

        # 3b. Insertion depth (large: reward for actually entering slot)
        insertion_improvement = insertion_depth - self._prev_insertion_depth
        reward += insertion_improvement * 200.0
        self._prev_insertion_depth = insertion_depth

        # Height penalty (encourage keeping forks low during insertion)
        fork_tip_z = fork_tip[2]
        slot_z     = slot_entry[2]
        z_error    = abs(fork_tip_z - slot_z)
        reward -= z_error * 50.0  # Height penalty (encourage keeping forks low during insertion)

        # 4. Collision penalty
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            if (c.geom1 in self._pallet_geom_ids or c.geom2 in self._pallet_geom_ids):
                if (c.geom1 in self._fork_geom_ids or c.geom2 in self._fork_geom_ids):
                    reward -= 1.0
                    break

        # 5. Success
        tip_past_exit   = fork_tip[0] >= slot_exit[0] and abs(fork_tip[1] - slot_exit[1]) < 0.03
        root_near_entry = abs(fork_root[1] - slot_entry[1]) < 0.03
        if tip_past_exit and root_near_entry:
            reward += 1000.0
            terminated = True
            info["success"] = True

        # 6. Fall
        base_z = self.data.xpos[self._base_body_id][2]
        if base_z < MIN_BASE_Z:
            reward -= 100.0
            terminated = True
            info["fallen"] = True
        else:
            info["fallen"] = False

        info.update({
            "alignment_error":  float(alignment_error),
            "root_dy":          float(root_dy),
            "insertion_depth":  float(insertion_depth),
            "fork_tip_x":       float(fork_tip[0]),
            "base_z":           float(base_z),
        })

        return reward, terminated, info
    # ── Render / Close ──────────────────────────────────────

    def render(self):
        if self.render_mode == "rgb_array":
            if self._renderer is None:
                self._renderer = mujoco.Renderer(self.model, height=480, width=640)
            self._renderer.update_scene(self.data)
            return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


# ── Sanity check ─────────────────────────────────────────

if __name__ == "__main__":
    print("=== ForkliftMujocoEnv sanity check ===")

    env = ForkliftMujocoEnv()
    print(f"Observation space: {env.observation_space}")
    print(f"Action space     : {env.action_space}")

    obs, _ = env.reset()
    fork_tip = env.data.site_xpos[env._fork_tip_site_id]
    base_x   = env.data.qpos[env._root_qposadr]
    print(f"offset check: fork_tip_x - base_x = {fork_tip[0] - base_x:.4f} m")
    print(f"fork_tip_x = {fork_tip[0]:.4f} m")

    print(f"\nInitial observation:")
    for label, val in zip(["dx","dy","dz","yaw","fork_h","lw_vel","rw_vel", "root_dy"], obs):
        print(f"  {label:8s}: {val:+.4f}")

    total_reward = 0.0
    for step in range(200):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            print(f"\nEpisode ended at step {step+1}")
            break

    print(f"\nTotal reward (200 random steps): {total_reward:.2f}")
    print(f"Final alignment_err: {info['alignment_error']:.4f} rad")
    print(f"Final insertion    : {info['insertion_depth']:.4f} m")
    print(f"Final fork_tip_x   : {info['fork_tip_x']:.4f} m")
    print(f"Final base_z       : {info['base_z']:.4f} m")

    env.close()
    print("\nSanity check PASSED")