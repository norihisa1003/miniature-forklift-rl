# Phase 2 System Architecture

**Project**: miniature-forklift-rl  
**Phase**: Phase 2 — MuJoCo Simulation & Fork Insertion  
**Created**: 2026-05-31

---

## Overview

Migrated from the Gazebo/ROS2 stack used in Phase 1 to MuJoCo. The ROS2 middleware layer is no longer needed — SB3 connects directly to the simulator, resulting in a simpler and faster architecture.

---

## Component Roles

### RL Layer

| Component | Role |
|---|---|
| `Stable-Baselines3` | Trains the agent using PPO. Has no knowledge of the simulator internals. |
| `train_phase2c.py` | Entry point for the training loop. Manages model saving and evaluation callbacks. |

### Interface Layer

| Component | Role |
|---|---|
| `ForkliftMujocoEnv` | Subclass of `gymnasium.Env`. Implements `reset()` / `step()`. Handles reward calculation, observation construction, and success detection. |

> **Difference from Phase 1**: The ROS2 node is gone. Direct MuJoCo API calls eliminate the communication latency that existed between Gazebo and the RL layer.

### Simulation Layer

| Component | Role |
|---|---|
| `scene.xml` | Complete scene definition including the forklift and pallet. **Single source of truth** — all geometry changes must be made here. |
| MuJoCo physics | Handles contact, friction, and rigid body dynamics internally. |

---

## Data Flow

```
train_phase2c.py
  └─ SB3.learn()
       └─ ForkliftMujocoEnv.step(action)
            ├─ mujoco.mj_step()          (physics step × 10)
            ├─ site_xpos[]               (fork tip / root world positions)
            ├─ qpos[] / sensordata[]     (pose, wheel velocities)
            └─ return obs, reward, terminated, truncated, info
```

---

## Phase 2 Sub-phases

| Sub-phase | X distance | Yaw | Y offset | Fork height | Result |
|---|---|---|---|---|---|
| 2a | Fixed (0.30m) | Fixed 0° | Fixed 0 | Fixed | 10/10 (100%) |
| 2b | Random 0.20–0.50m | Random ±30° | Fixed 0 | Fixed | 10/10 (100%) |
| 2c | Random 0.20–0.50m | Random ±30° | Random ±15cm | Fixed | In progress |

---

## Observation Space (8 dimensions)

| Index | Content | Notes |
|---|---|---|
| [0] | dx: fork_tip → slot_entry, X direction | |
| [1] | dy: fork_tip → slot_entry, Y direction | |
| [2] | dz: fork_tip → slot_entry, Z direction | |
| [3] | yaw: forklift heading angle (rad) | |
| [4] | fork_height: fork lift joint position (m) | |
| [5] | lw_vel: left wheel angular velocity (normalized) | |
| [6] | rw_vel: right wheel angular velocity (normalized) | |
| [7] | root_dy: fork_root → slot_entry, Y direction | Added in Phase 2c |

## Action Space (3 dimensions, [-1, 1])

| Index | Content | Scaled range |
|---|---|---|
| [0] | Left wheel velocity | ±5.0 rad/s |
| [1] | Right wheel velocity | ±5.0 rad/s |
| [2] | Fork lift velocity | ±0.3 m/s |

---

## Reward Design (Phase 2c)

| Type | Formula | Notes |
|---|---|---|
| Time penalty | −0.1 / step | Encourages faster completion |
| Alignment improvement | Δalignment_error × 50 | Angle between fork axis and slot axis |
| Lateral improvement | Δdist_y × 50 | Y-direction error of fork_tip |
| Insertion improvement | Δ(tip_dist + root_dist) × 50 × yaw_gate | Active only when alignment < 30° |
| Collision penalty | −1.0 / step | When fork contacts pallet |
| Success bonus | +1000 | Fork fully inserted |
| Fall penalty | −50 | base_z < 0.05m |

---

## Success Condition

```python
tip_past_exit   = fork_tip[0] >= slot_exit[0] and abs(fork_tip[1]  - slot_exit[1])  < 0.03
root_near_entry = abs(fork_root[1] - slot_entry[1]) < 0.03
success = tip_past_exit and root_near_entry
```

---

## Key File Structure

```
training/
├── forklift_mujoco_env.py   ← Gymnasium environment (core of Phase 2)
├── train_phase2c.py         ← Training script
├── evaluate_phase2a.py      ← Evaluation and visualization script
└── models/
    ├── phase2a_final.zip    ← Phase 2a trained model
    ├── phase2b_ppo/
    │   └── best_model.zip   ← Phase 2b trained model
    └── phase2c_ppo/
        └── best_model.zip   ← Phase 2c (in training)

sim/
└── worlds/
    └── scene.xml            ← Scene definition (single source of truth)
```

---

## Tech Stack

| Item | Version |
|---|---|
| Ubuntu | 24.04.4 LTS |
| Python | 3.12.3 |
| PyTorch | 2.5.1+cu121 |
| Stable-Baselines3 | 2.8.0 |
| Gymnasium | 1.2.3 |
| MuJoCo | 3.8.1 |