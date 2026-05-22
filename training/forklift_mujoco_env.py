"""
ForkliftMujocoEnv - Phase 2a: Fork Insertion (Fixed Position)
=============================================================
Task: Forklift starts 0.3m directly in front of the pallet,
      perfectly aligned. Agent learns only to drive forward
      and insert the forks into the pallet slot.

Phase 2a constraints (intentionally simple):
- Fixed start: forklift at X=-0.3 from pallet front face, yaw=0
- Fork height pre-set to slot height (no height learning yet)
- Success = fork tip reaches slot center within XY threshold

Observation space (7 dims):
  [0]   dx: fork_tip to slot_center, X direction (forward)
  [1]   dy: fork_tip to slot_center, Y direction (lateral)
  [2]   dz: fork_tip to slot_center, Z direction (height)
  [3]   yaw: forklift heading angle (radians)
  [4]   fork_height: current fork lift joint position (m)
  [5]   left_wheel_vel: left wheel angular velocity (rad/s)
  [6]   right_wheel_vel: right wheel angular velocity (rad/s)

Action space (3 dims, continuous [-1, 1]):
  [0]   left_wheel_vel  → scaled to [-5.0, 5.0] rad/s
  [1]   right_wheel_vel → scaled to [-5.0, 5.0] rad/s
  [2]   fork_lift_vel   → scaled to [-0.3, 0.3] m/s
  (In Phase 2a, the agent will learn to ignore action[2]
   since fork height is pre-set)

Reward:
  - Dense: reduction in XY distance to slot center
  - Success bonus: +100 when fork tip is within threshold
  - Timeout: 0 (no penalty, just episode ends)
  - Fallen: -50 (base_z too low)
"""

import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces
from pathlib import Path


# ── Constants ──────────────────────────────────────────────

# Scene XML path (relative to this file)
SCENE_XML = Path(__file__).parent.parent / "sim" / "worlds" / "scene.xml"

# Simulation
TIMESTEP        = 0.002   # matches scene.xml option timestep
STEPS_PER_ACTION = 10     # physics steps per RL action (control frequency = 50Hz)
MAX_EPISODE_STEPS = 500   # max steps before timeout

# Actuator scaling (maps [-1,1] action to physical units)
WHEEL_VEL_MAX = 5.0       # rad/s
FORK_VEL_MAX  = 0.3       # m/s

# Phase 2a fixed start configuration
# Pallet body is at X=0.55 in world frame (from scene.xml).
# Pallet front face = 0.55 - 0.15 = 0.40m
# Forklift fork tip at start = pallet_front - gap = 0.40 - 0.30 = 0.10m
# Forklift base_link X = fork_tip_X - 0.38 (fork tip offset from base) = -0.28m
FORKLIFT_START_X   = -0.28   # world X of base_link at episode start
FORKLIFT_START_Y   =  0.0
FORKLIFT_START_Z   =  0.13   # base_link height (from forklift.xml)
FORKLIFT_START_YAW =  0.0    # facing +X (directly toward pallet)

# Fork height at start: pre-set to match slot center Z
# slot_center world Z = 0.0225m
# fork_tip world Z at joint=0 is ~0.013m (from validation)
# Needed lift = 0.0225 - 0.013 = ~0.010m
FORK_INIT_HEIGHT = 0.0095      # m (pre-set, not learned in Phase 2a)

# Pallet slot center in world frame (from scene.xml)
SLOT_CENTER_WORLD = np.array([0.50, 0.0, 0.0225])

# Success threshold
SUCCESS_XY_THRESHOLD = 0.015  # m (fork tip within 2cm of slot center in XY)
SUCCESS_X_THRESHOLD  = 0.050  # m (fork tip has passed slot entry in X)

# Termination conditions
MIN_BASE_Z = 0.05   # below this = fallen over


class ForkliftMujocoEnv(gym.Env):
    """
    MuJoCo-based Gymnasium environment for forklift pallet insertion.
    Phase 2a: fixed start position, fork height pre-set, learn forward insertion.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, render_mode=None):
        super().__init__()

        self.render_mode = render_mode

        # Load MuJoCo model
        self.model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
        self.data  = mujoco.MjData(self.model)

        # Cache frequently used IDs (avoids repeated string lookups during training)
        self._cache_ids()

        # Observation space: 7 continuous values, all normalized to roughly [-1, 1]
        obs_low  = np.array([-2.0, -2.0, -0.5, -np.pi, 0.0,  -1.0, -1.0], dtype=np.float32)
        obs_high = np.array([ 2.0,  2.0,  0.5,  np.pi, 0.25,  1.0,  1.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        # Action space: 3 continuous values in [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float32
        )

        # Renderer (lazy init)
        self._renderer = None

        # Episode state
        self._step_count = 0
        self._prev_dist_xy = None

    def _cache_ids(self):
        """Cache MuJoCo object IDs to avoid repeated lookups in step()."""

        def body_id(name):
            return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)

        def site_id(name):
            return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)

        def joint_id(name):
            return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)

        def geom_id(name):
            return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)

        self._base_body_id      = body_id("base_link")
        self._fork_tip_site_id  = site_id("fork_tip_site")
        self._slot_center_id    = site_id("slot_center")
        self._fork_lift_joint_id = joint_id("fork_lift_joint")
        self._fork_root_site_id  = site_id("fork_root_site")
        self._slot_exit_site_id  = site_id("slot_center_exit")

        # qpos address for each joint
        self._root_qposadr      = self.model.jnt_qposadr[joint_id("root")]
        self._fork_qposadr      = self.model.jnt_qposadr[joint_id("fork_lift_joint")]

        # Pallet geom IDs (for collision detection)
        pallet_geom_names = [
            "bottom_deck", "leg_left_outer", "leg_inner",
            "leg_right_outer", "top_deck"
        ]
        self._pallet_geom_ids = set(geom_id(n) for n in pallet_geom_names)

        # Fork geom IDs (to detect fork-pallet contact specifically)
        fork_geom_names = ["left_fork_geom", "right_fork_geom"]
        self._fork_geom_ids = set(geom_id(n) for n in fork_geom_names)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        mujoco.mj_resetData(self.model, self.data)

        # ── Set forklift start pose ──
        # freejoint qpos layout: [x, y, z, qw, qx, qy, qz]
        qposadr = self._root_qposadr
        self.data.qpos[qposadr + 0] = FORKLIFT_START_X
        self.data.qpos[qposadr + 1] = FORKLIFT_START_Y
        self.data.qpos[qposadr + 2] = FORKLIFT_START_Z
        # yaw=0 → quaternion = [1, 0, 0, 0]
        self.data.qpos[qposadr + 3] = 1.0  # qw
        self.data.qpos[qposadr + 4] = 0.0  # qx
        self.data.qpos[qposadr + 5] = 0.0  # qy
        self.data.qpos[qposadr + 6] = 0.0  # qz

        # ── Set fork initial height ──
        self.data.qpos[self._fork_qposadr] = FORK_INIT_HEIGHT

        # Zero all velocities
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = 0.0

        # Forward kinematics to update site positions
        mujoco.mj_forward(self.model, self.data)

        # Initialize previous distance for dense reward
        fork_tip  = self.data.site_xpos[self._fork_tip_site_id].copy()
        slot_ctr  = self.data.site_xpos[self._slot_center_id].copy()
        self._prev_dist_xy = np.linalg.norm(fork_tip[:2] - slot_ctr[:2])

        self._step_count = 0

        obs = self._get_obs()
        info = {}
        self._prev_fork_x = 0.0
        return obs, info

    def step(self, action):
        # ── Apply action ──
        # Scale from [-1, 1] to physical units
        self.data.ctrl[0] = float(action[0]) * WHEEL_VEL_MAX   # left wheel
        self.data.ctrl[1] = float(action[1]) * WHEEL_VEL_MAX   # right wheel
        self.data.ctrl[2] = float(action[2]) * FORK_VEL_MAX    # fork lift

        # Step physics multiple times per RL action
        for _ in range(STEPS_PER_ACTION):
            mujoco.mj_step(self.model, self.data)

        self._step_count += 1

        # ── Compute reward ──
        reward, terminated, info = self._compute_reward()

        # ── Timeout ──
        truncated = self._step_count >= MAX_EPISODE_STEPS

        obs = self._get_obs()

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        """Build observation vector from current simulation state."""

        fork_tip = self.data.site_xpos[self._fork_tip_site_id]
        slot_ctr = self.data.site_xpos[self._slot_center_id]

        # Relative displacement: fork tip → slot center
        delta = slot_ctr - fork_tip
        dx, dy, dz = delta[0], delta[1], delta[2]

        # Yaw angle from base_link quaternion
        # qpos layout for freejoint: [x,y,z, qw,qx,qy,qz]
        qposadr = self._root_qposadr
        qw = self.data.qpos[qposadr + 3]
        qz = self.data.qpos[qposadr + 6]
        # yaw = atan2(2*(qw*qz), 1 - 2*qz^2)  (simplified for planar motion)
        yaw = np.arctan2(2.0 * qw * qz, 1.0 - 2.0 * qz * qz)

        # Fork height
        fork_height = self.data.qpos[self._fork_qposadr]

        # Wheel velocities (normalized by max speed)
        lw_vel = self.data.sensordata[2] / WHEEL_VEL_MAX
        rw_vel = self.data.sensordata[3] / WHEEL_VEL_MAX

        obs = np.array([dx, dy, dz, yaw, fork_height, lw_vel, rw_vel],
                       dtype=np.float32)
        return obs

    def _compute_reward(self):
        """
        Compute reward, terminated flag, and info dict.

        Reward structure (Phase 2a):
          1. Dense distance reward: reward proportional to XY distance reduction
          2. Success bonus: +100 when fork tip reaches slot center
          3. Fall penalty: -50 when forklift tips over
        """
        info = {}
        terminated = False
        reward = 0.0

        fork_tip = self.data.site_xpos[self._fork_tip_site_id]
        slot_ctr = self.data.site_xpos[self._slot_center_id]

        # Current XY distance
        dist_xy = np.linalg.norm(fork_tip[:2] - slot_ctr[:2])

        # ── 1. Dense reward: XY distance reduction ──
        # Positive when getting closer, negative when moving away.
        # Scale factor 100: makes the per-step reward comparable to success bonus.
        dist_reduction = self._prev_dist_xy - dist_xy
        reward += dist_reduction * 100.0
        self._prev_dist_xy = dist_xy

        # ── 1b. Insertion depth reward ──
        slot_entry_x = slot_ctr[0]
        current_insertion = max(0.0, fork_tip[0] - slot_entry_x)
        prev_insertion = max(0.0, self._prev_fork_x - slot_entry_x)
        insertion_increase = current_insertion - prev_insertion
        reward += insertion_increase * 200.0
        self._prev_fork_x = fork_tip[0]

        # ── 1c. Collision penalty ──
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            if (c.geom1 in self._pallet_geom_ids or c.geom2 in self._pallet_geom_ids):
                if (c.geom1 in self._fork_geom_ids or c.geom2 in self._fork_geom_ids):
                    reward -= 1.0
                    break

        # # ── 2. Success: fork tip within threshold of slot center ──
        # fork_passed_entry = fork_tip[0] >= (slot_ctr[0] + SUCCESS_X_THRESHOLD)
        # fork_aligned_xy   = dist_xy <= SUCCESS_XY_THRESHOLD
        # if fork_passed_entry and fork_aligned_xy:
        #     reward += 100.0
        #     terminated = True
        #     info["success"] = True
        # else:
        #     info["success"] = False

        # ── 2. Success ──
        fork_tip  = self.data.site_xpos[self._fork_tip_site_id]
        fork_root = self.data.site_xpos[self._fork_root_site_id]
        slot_exit = self.data.site_xpos[self._slot_exit_site_id]
        slot_entry = self.data.site_xpos[self._slot_center_id]

        tip_near_exit = (
            abs(fork_tip[1] - slot_exit[1]) < 0.03 and   # Y within 3cm
            fork_tip[0] >= slot_exit[0]                    # X has passed the exit
        )
        root_near_entry = (
            abs(fork_root[1] - slot_entry[1]) < 0.03      # Y within 3cm
        )

        if tip_near_exit and root_near_entry:
            reward += 100.0
            terminated = True
            info["success"] = True

        # ── 3. Fall detection ──
        base_z = self.data.xpos[self._base_body_id][2]
        if base_z < MIN_BASE_Z:
            reward -= 50.0
            terminated = True
            info["fallen"] = True
        else:
            info["fallen"] = False

        # Diagnostic info
        info["dist_xy"] = float(dist_xy)
        info["fork_tip_x"] = float(fork_tip[0])
        info["base_z"] = float(base_z)

        return reward, terminated, info

    def render(self):
        if self.render_mode == "human":
            if self._renderer is None:
                self._renderer = mujoco.Renderer(self.model)
            self._renderer.update_scene(self.data)
            # human mode: use MuJoCo viewer externally
        elif self.render_mode == "rgb_array":
            if self._renderer is None:
                self._renderer = mujoco.Renderer(self.model, height=480, width=640)
            self._renderer.update_scene(self.data)
            return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


# ── Sanity check (run directly) ──────────────────────────

if __name__ == "__main__":
    print("=== ForkliftMujocoEnv sanity check ===")

    env = ForkliftMujocoEnv()

    # Check spaces
    print(f"Observation space: {env.observation_space}")
    print(f"Action space     : {env.action_space}")

    # Reset
    obs, info = env.reset()
    print(f"\nInitial observation:")
    labels = ["dx", "dy", "dz", "yaw", "fork_h", "lw_vel", "rw_vel"]
    for label, val in zip(labels, obs):
        print(f"  {label:8s}: {val:+.4f}")

    # Run random actions for 200 steps
    total_reward = 0.0
    for step in range(200):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            print(f"\nEpisode ended at step {step+1}")
            print(f"  success={info.get('success')}, fallen={info.get('fallen')}")
            break

    print(f"\nTotal reward (200 random steps): {total_reward:.2f}")
    print(f"Final dist_xy: {info['dist_xy']:.4f} m")
    print(f"Final base_z : {info['base_z']:.4f} m")

    env.close()
    print("\nSanity check PASSED")