# Miniature Forklift RL

An autonomous miniature forklift trained with reinforcement learning (sim-to-real).

## Goal
Build a physical miniature forklift that can navigate, pick up, and transport
cargo autonomously — trained via RL and transferred from simulation to real hardware.

## Tech Stack
| Component      | Technology                     |
|----------------|--------------------------------|
| Simulator      | Gazebo Harmonic                |
| Middleware     | ROS2 Jazzy                     |
| RL Framework   | Stable-Baselines3 2.8.0        |
| RL Interface   | Gymnasium 1.2.3                |
| Training GPU   | NVIDIA GTX 1080 Ti (CUDA 12.2) |

## Project Structure
| Directory    | Description                          |
|--------------|--------------------------------------|
| `sim/`       | Gazebo simulation environment        |
| `training/`  | RL training code                     |
| `hardware/`  | Disassembly notes, wiring diagrams   |
| `docs/`      | Research notes, design decisions     |

## Phases
- **Phase 0**: Environment setup ✅
- **Phase 1**: Gazebo environment + basic RL (navigation)
- **Phase 2**: Fork operation + full task completion in sim
- **Phase 3**: Real hardware implementation (sim-to-real)

## Demo Videos
Coming soon on YouTube.
