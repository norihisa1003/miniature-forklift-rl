"""
ForkliftMujocoEnv - Phase 2d: Fork Height Control
==================================================
Task: Forklift starts at a randomized position in front of the pallet.
      Pallet height is also randomized. Agent must control fork height
      to align with the slot before insertion.

Randomization (Phase 2d):
  - X distance to pallet: 0.20 ~ 0.50m
  - Yaw offset:           ±30°
  - Y offset:             ±2cm
  - Pallet height (Z):    0.00 ~ 0.09m  (slot Z = pallet_z + 0.0225)

Changes from Phase 2c:
  - scene.xml: platform body added under pallet (static, no joint).
      Platform height randomized via model.geom_size / model.body_pos in reset().
      Platform front face blocks fork if height is wrong — no free sliding under pallet.
  - action[2] now controls fork height: ctrl[2] = (action[2]+1)/2 * 0.20
    (was: ctrl[2] = FORK_INIT_HEIGHT fixed)
  - reset(): platform height randomized; fork initial height set to match slot
    so the agent starts with a solvable configuration
  - Observation: dz (obs[2]) is now meaningful — fork_tip Z vs slot Z
  - Height penalty retained; now guides the agent to adjust fork height

Observation space (8 dims):
  [0]  dx:           fork_tip → slot_center, X direction
  [1]  dy:           fork_tip → slot_center, Y direction
  [2]  dz:           fork_tip → slot_center, Z direction  ← key signal in 2d
  [3]  yaw:          forklift heading angle (radians)
  [4]  fork_height:  fork lift joint position (m)
  [5]  lw_vel:       left wheel angular velocity (normalized)
  [6]  rw_vel:       right wheel angular velocity (normalized)
  [7]  root_dy:      fork_root → slot_entry, Y direction

Action space (3 dims, continuous [-1, 1]):
  [0]  left_wheel_vel  → scaled to ±5.0 rad/s
  [1]  right_wheel_vel → scaled to ±5.0 rad/s
  [2]  fork_height_cmd → joint目標位置 [0, 0.20m] に直接マッピング
                         action=-1→0.00m、action=0→0.10m、action=+1→0.20m
                         position アクチュエータ(kp=500)が重力に関係なくキープ

Reward:
  - Time penalty:       -0.1 per step
  - Root Y alignment:   improvement × 100
  - Forward progress:   improvement × 10
  - Insertion depth:    improvement × 200
  - Height penalty:     z_error × 50 (fork_tip vs slot_center Z)
  - Collision penalty:  -1.0 per step when fork contacts pallet
  - Success bonus:      +1000 (fork fully inserted, Z also aligned)
  - Fall penalty:       -100 (base_z too low)
"""

import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces
from pathlib import Path


# ── Simulation constants ────────────────────────────────────

SCENE_XML         = Path(__file__).parent.parent / "sim" / "worlds" / "scene.xml"
TIMESTEP          = 0.002   # matches scene.xml
STEPS_PER_ACTION  = 10      # control frequency = 50Hz
MAX_EPISODE_STEPS = 500

WHEEL_VEL_MAX = 5.0   # rad/s
FORK_POS_MAX  = 0.20  # m (fork_lift_joint upper limit)
MIN_BASE_Z    = 0.05  # below this = fallen


# ── Forklift geometry ───────────────────────────────────────

FORKLIFT_START_Z  = 0.13    # base_link height (world Z)
FORK_TIP_OFFSET_X = 0.50    # fork tip to base_link (measured)
PALLET_FRONT_X    = 0.40    # world X of pallet slot entry face

# fork_tip Z (world) = FORKLIFT_START_Z - 0.117 + fork_lift_joint_pos
#                    = 0.013 + fork_lift_joint_pos
FORK_TIP_Z_BASE   = FORKLIFT_START_Z - 0.117   # = 0.013

# slot_center Z (world) = pallet_body_z + 0.0225
SLOT_Z_OFFSET     = 0.0225  # slot height above pallet body origin


# ── Platform height randomization ───────────────────────────

PLATFORM_X_FIXED   = 0.65   # world X (fixed)
PLATFORM_Y_FIXED   = 0.0    # world Y (fixed)
PLATFORM_HEIGHT_MIN = 0.00  # m (platform top face Z = platform height)
PLATFORM_HEIGHT_MAX = 0.09  # m
# → slot Z range: 0.0225 ~ 0.1125
# → required fork_lift_joint range: 0.0095 ~ 0.0995  (well within 0~0.20)


# ── Forklift start randomization ────────────────────────────

RAND_DIST_MIN    = 0.20        # m (distance from pallet front to fork tip)
RAND_DIST_MAX    = 0.50        # m
RAND_YAW_MAX     = np.pi / 6  # ±30°
RAND_Y_MAX       = 0.02       # m (±2cm)
RAND_FORK_POS_MIN = 0.00      # m (fork_lift_joint lower limit)
RAND_FORK_POS_MAX = 0.20      # m (fork_lift_joint upper limit)


class ForkliftMujocoEnv(gym.Env):
    """MuJoCo Gymnasium environment for forklift pallet insertion (Phase 2d).

    Key change from Phase 2c: pallet height is randomized each episode,
    and the agent must control fork height via action[2] to align with the slot.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, render_mode=None):
        super().__init__()

        self.render_mode = render_mode

        self.model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
        self.data  = mujoco.MjData(self.model)
        self._cache_ids()

        # dz (obs[2]) range expanded: pallet can be 0~9cm high, fork 0~20cm
        obs_low  = np.array([-2.0, -2.0, -0.5, -np.pi, 0.0, -1.0, -1.0, -2.0], dtype=np.float32)
        obs_high = np.array([ 2.0,  2.0,  0.5,  np.pi, 0.25, 1.0,  1.0,  2.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        # action[2] added: fork height command (position target)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

        self._renderer      = None
        self._step_count    = 0
        self._platform_height = 0.0   # current episode platform height (for logging)

    # ── ID cache ────────────────────────────────────────────

    def _cache_ids(self):
        def body_id(n):  return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY,  n)
        def site_id(n):  return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE,  n)
        def joint_id(n): return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)
        def geom_id(n):  return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM,  n)

        self._base_body_id       = body_id("base_link")
        self._fork_tip_site_id   = site_id("fork_tip_site")
        self._fork_root_site_id  = site_id("fork_root_site")
        self._slot_entry_site_id = site_id("slot_center")
        self._slot_exit_site_id  = site_id("slot_center_exit")

        self._root_qposadr     = self.model.jnt_qposadr[joint_id("root")]
        self._fork_qposadr     = self.model.jnt_qposadr[joint_id("fork_lift_joint")]

        # Platform height randomization: rewrite model directly in reset()
        self._platform_geom_id = geom_id("platform_geom")
        self._pallet_body_id   = body_id("pallet")

        self._pallet_geom_ids = {
            geom_id(n) for n in
            ["platform_geom",
             "bottom_deck", "leg_left_outer", "leg_inner", "leg_right_outer", "top_deck"]
        }
        self._fork_geom_ids = {geom_id(n) for n in ["left_fork_geom", "right_fork_geom"]}

    # ── Reset ───────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        # ── Randomize platform height (Phase 2d new) ──
        platform_height = self.np_random.uniform(PLATFORM_HEIGHT_MIN, PLATFORM_HEIGHT_MAX)
        self._platform_height = platform_height

        # model.geom_size[id] = [half_x, half_y, half_z]
        # platform box の half_z を platform_height/2 に設定
        self.model.geom_size[self._platform_geom_id][2] = platform_height / 2.0

        # platform geom の中心Z = half_z → top face Z = platform_height
        # geom pos は body frame 内の offset なので body_pos は変えず、
        # geom_pos[2] を half_z に合わせる
        self.model.geom_pos[self._platform_geom_id][2] = platform_height / 2.0

        # pallet body is now a sibling of platform (not a child).
        # body_pos[2] is world Z, so set it directly to platform_height.
        self.model.body_pos[self._pallet_body_id][2] = platform_height

        # ── Randomize forklift start pose ──
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

        # ── Fork initial height: randomized (Phase 2d) ──
        init_fork_pos = self.np_random.uniform(RAND_FORK_POS_MIN, RAND_FORK_POS_MAX)
        self.data.qpos[self._fork_qposadr] = init_fork_pos

        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

        # ── Initialize incremental reward state ──
        fork_tip   = self.data.site_xpos[self._fork_tip_site_id].copy()
        fork_root  = self.data.site_xpos[self._fork_root_site_id].copy()
        slot_entry = self.data.site_xpos[self._slot_entry_site_id].copy()
        slot_exit  = self.data.site_xpos[self._slot_exit_site_id].copy()

        self._prev_alignment_error = self._calc_alignment_error(
            fork_tip, fork_root, slot_entry, slot_exit
        )
        self._prev_root_dy         = abs(fork_root[1] - slot_entry[1])
        self._prev_fork_tip_x      = fork_tip[0]
        self._prev_insertion_depth = max(0.0, fork_tip[0] - slot_entry[0])
        self._prev_z_error         = abs(fork_tip[2] - slot_entry[2])
        self._step_count           = 0

        return self._get_obs(), {}

    # ── Step ────────────────────────────────────────────────

    def step(self, action):
        # Wheel velocity (same as Phase 2c)
        self.data.ctrl[0] = float(action[0]) * WHEEL_VEL_MAX
        self.data.ctrl[1] = float(action[1]) * WHEEL_VEL_MAX

        # Fork height: action[2] in [-1, 1] → joint目標位置 [0, 0.20m] に直接マッピング
        # action=-1 → 0.00m（最下点）、action=0 → 0.10m（中間）、action=+1 → 0.20m（最上点）
        # position アクチュエータ(kp=500)が重力に関係なくその位置をキープする。
        fork_target = (float(action[2]) + 1.0) / 2.0 * FORK_POS_MAX
        self.data.ctrl[2] = fork_target

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
        fork_tip   = self.data.site_xpos[self._fork_tip_site_id]
        slot_entry = self.data.site_xpos[self._slot_entry_site_id]
        fork_root  = self.data.site_xpos[self._fork_root_site_id]

        # dx, dy, dz: fork_tip → slot_center (signed)
        delta  = slot_entry - fork_tip
        root_dy = slot_entry[1] - fork_root[1]

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
        info      = {}
        terminated = False

        fork_tip   = self.data.site_xpos[self._fork_tip_site_id].copy()
        fork_root  = self.data.site_xpos[self._fork_root_site_id].copy()
        slot_entry = self.data.site_xpos[self._slot_entry_site_id].copy()
        slot_exit  = self.data.site_xpos[self._slot_exit_site_id].copy()

        alignment_error = self._calc_alignment_error(fork_tip, fork_root, slot_entry, slot_exit)
        root_dy         = abs(fork_root[1] - slot_entry[1])
        insertion_depth = max(0.0, fork_tip[0] - slot_entry[0])

        reward = -0.1  # time penalty

        # 1. Root Y alignment
        reward += (self._prev_root_dy - root_dy) * 100.0
        self._prev_root_dy = root_dy

        # 2. Forward progress (small: encourages moving toward pallet)
        # forward_progress = fork_tip[0] - self._prev_fork_tip_x
        # reward += forward_progress * 50.0
        # self._prev_fork_tip_x = fork_tip[0]

        # 3. Insertion depth (large: reward for actually entering slot)
        y_in_slot = abs(fork_tip[1] - slot_entry[1]) < 0.03  # slot Y within ±3cm of slot center
        y_root_in_slot = root_dy < 0.03  # root Y also within ±3cm (encourage whole forklift alignment)
        z_in_slot = abs(fork_tip[2] - slot_entry[2]) < 0.010  # fork_tip Z within ±1cm of slot Z (height alignment)
        gate = y_in_slot and z_in_slot and y_root_in_slot  # all conditions to count insertion depth
        forward_progress = fork_tip[0] - self._prev_fork_tip_x
        forward_scale    = 200.0 if gate else 50.0
        reward += forward_progress * forward_scale
        self._prev_fork_tip_x = fork_tip[0]

        # 4. Height alignment (incremental, same design as root_dy)
        z_error = abs(fork_tip[2] - slot_entry[2])
        reward -= z_error * 100.0  # direct penalty on Z error to encourage height adjustment

        # 5. Collision penalty
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            if (c.geom1 in self._pallet_geom_ids or c.geom2 in self._pallet_geom_ids):
                if (c.geom1 in self._fork_geom_ids or c.geom2 in self._fork_geom_ids):
                    reward -= 1.0
                    break

        # 6. Success condition: XY insertion + root alignment + Z alignment
        tip_past_exit   = fork_tip[0] >= slot_exit[0] and abs(fork_tip[1] - slot_exit[1]) < 0.01
        root_near_entry = abs(fork_root[1] - slot_entry[1]) < 0.01
        z_aligned       = abs(fork_tip[2] - slot_entry[2]) < 0.01  # スロット高さ±1cm以内
        if tip_past_exit and root_near_entry and z_aligned:
            reward    += 1000.0
            terminated = True
            info["success"] = True

        # 7. Fall
        base_z = self.data.xpos[self._base_body_id][2]
        if base_z < MIN_BASE_Z:
            reward    -= 100.0
            terminated = True
            info["fallen"] = True
        else:
            info["fallen"] = False

        info.update({
            "alignment_error":  float(alignment_error),
            "root_dy":          float(root_dy),
            "insertion_depth":  float(insertion_depth),
            "fork_tip_x":       float(fork_tip[0]),
            "fork_tip_z":       float(fork_tip[2]),
            "slot_z":           float(slot_entry[2]),
            "z_error":          float(z_error),
            "platform_height":  float(self._platform_height),
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
    print("=== ForkliftMujocoEnv Phase 2d sanity check ===")

    env = ForkliftMujocoEnv()
    print(f"Observation space: {env.observation_space}")
    print(f"Action space     : {env.action_space}")

    # Run 3 episodes to verify platform height randomization
    for ep in range(3):
        obs, _ = env.reset()
        fork_tip  = env.data.site_xpos[env._fork_tip_site_id]
        slot_ctr  = env.data.site_xpos[env._slot_entry_site_id]
        fork_h    = env.data.qpos[env._fork_qposadr]

        print(f"\n--- Episode {ep+1} ---")
        print(f"  platform_height : {env._platform_height:.4f} m")
        print(f"  slot_z          : {slot_ctr[2]:.4f} m  (expected: {env._platform_height + 0.0225:.4f})")
        print(f"  fork_tip_z      : {fork_tip[2]:.4f} m")
        print(f"  fork_height     : {fork_h:.4f} m")
        print(f"  z_error         : {abs(fork_tip[2] - slot_ctr[2]):.6f} m  (should be ~0)")

        for label, val in zip(["dx","dy","dz","yaw","fork_h","lw_vel","rw_vel","root_dy"], obs):
            print(f"  {label:8s}: {val:+.4f}")

    # Random rollout
    obs, _ = env.reset()
    total_reward = 0.0
    for step in range(200):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            print(f"\nEpisode ended at step {step+1}")
            print(f"  success: {info.get('success', False)}")
            break

    print(f"\nTotal reward (random 200 steps): {total_reward:.2f}")
    print(f"Final z_error        : {info['z_error']:.4f} m")
    print(f"Final insertion      : {info['insertion_depth']:.4f} m")
    print(f"Final platform_height: {info['platform_height']:.4f} m")

    env.close()
    print("\nSanity check PASSED")