# training/forklift_env.py

import math
import time
import threading

import gymnasium
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry



# ---------------------------------------------------------------------------
# Global ROS2 executor: shared across all ForkliftEnv instances
# ---------------------------------------------------------------------------
_executor: SingleThreadedExecutor | None = None
_executor_thread: threading.Thread | None = None
_rclpy_lock = threading.Lock()


def _ensure_rclpy_running() -> None:
    """Initialize rclpy and start the global executor thread if not running."""
    global _executor, _executor_thread
    with _rclpy_lock:
        if not rclpy.ok():
            rclpy.init()
        if _executor is None:
            _executor = SingleThreadedExecutor()
            _executor_thread = threading.Thread(
                target=_executor.spin, daemon=True
            )
            _executor_thread.start()


def _add_node(node: Node) -> None:
    """Add a node to the global executor."""
    global _executor
    with _rclpy_lock:
        _executor.add_node(node)


def _remove_node(node: Node) -> None:
    """Remove a node from the global executor."""
    global _executor
    with _rclpy_lock:
        _executor.remove_node(node)


# ---------------------------------------------------------------------------
# ROS2 node
# ---------------------------------------------------------------------------
class ForkliftROSNode(Node):
    """Handles all ROS2 pub/sub communication for one ForkliftEnv instance."""

    def __init__(self, node_name: str = "forklift_env_node"):
        super().__init__(node_name)

        self._cmd_vel_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self._odom_sub = self.create_subscription(
            Odometry, "/odom", self._odom_callback, 10
        )

        # No ROS2 service client needed for reset.
        # gz service is called directly via subprocess (see reset_world()).

        self._odom_lock = threading.Lock()
        self._x = 0.0
        self._y = 0.0
        self._yaw = 0.0

    def _odom_callback(self, msg: Odometry) -> None:
        with self._odom_lock:
            self._x = msg.pose.pose.position.x
            self._y = msg.pose.pose.position.y
            self._yaw = self._quaternion_to_yaw(msg.pose.pose.orientation)

    def get_pose(self) -> tuple[float, float, float]:
        with self._odom_lock:
            return self._x, self._y, self._yaw

    def publish_cmd_vel(self, linear: float, angular: float) -> None:
        msg = Twist()
        msg.linear.x = linear
        msg.angular.z = angular
        self._cmd_vel_pub.publish(msg)

    def reset_world(self) -> None:
        """Reset the entire Gazebo world via gz service.

        This resets simulation time and all model states including the
        DiffDrive odometry counter, which cannot be reset via set_pose alone.
        """
        import subprocess
        result = subprocess.run(
            [
                "gz", "service",
                "-s", "/world/warehouse/control",
                "--reqtype", "gz.msgs.WorldControl",
                "--reptype", "gz.msgs.Boolean",
                "--req", "reset: {all: true}",
                "--timeout", "2000",
            ],
            capture_output=True,
            text=True,
        )
        if "true" not in result.stdout:
            self.get_logger().warning(
                f"World reset may have failed: {result.stdout.strip()}"
            )

    @staticmethod
    def _quaternion_to_yaw(orientation) -> float:
        x, y, z, w = (
            orientation.x, orientation.y,
            orientation.z, orientation.w,
        )
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return math.atan2(siny_cosp, cosy_cosp)


# ---------------------------------------------------------------------------
# Gymnasium environment
# ---------------------------------------------------------------------------
class ForkliftEnv(gymnasium.Env):
    """Differential-drive forklift navigation environment for Phase 1 RL.

    Observation space (5 values):
        [x, y, yaw, goal_x, goal_y]

    Action space (discrete, 5 actions):
        0: forward   linear=+0.3  angular= 0.0
        1: backward  linear=-0.3  angular= 0.0
        2: turn left  linear= 0.0  angular=+0.5
        3: turn right linear= 0.0  angular=-0.5
        4: stop       linear= 0.0  angular= 0.0

    Reward:
        +100  goal reached  (dist < GOAL_THRESHOLD)
        - 50  wall hit      (|x| or |y| > WALL_LIMIT)
        -  0.1 every step   (time penalty)
        + 10 * (prev_dist - curr_dist)  (distance shaping)
    """

    GOAL_X: float = 2.0
    GOAL_Y: float = 0.0
    GOAL_THRESHOLD: float = 0.3
    WALL_LIMIT: float = 2.4
    MAX_STEPS: int = 500

    ACTION_TABLE: list[tuple[float, float]] = [
        ( 0.3,  0.0),  # 0: forward
        (-0.3,  0.0),  # 1: backward
        ( 0.0,  0.5),  # 2: turn left
        ( 0.0, -0.5),  # 3: turn right
        ( 0.0,  0.0),  # 4: stop
    ]

    # Instance counter for unique node names
    _instance_count = 0

    def __init__(self):
        super().__init__()

        obs_low  = np.array([-3.0, -3.0, -math.pi, -3.0, -3.0], dtype=np.float32)
        obs_high = np.array([ 3.0,  3.0,  math.pi,  3.0,  3.0], dtype=np.float32)
        self.observation_space = gymnasium.spaces.Box(
            low=obs_low, high=obs_high, dtype=np.float32
        )
        self.action_space = gymnasium.spaces.Discrete(5)

        self._step_count: int = 0
        self._prev_dist: float = 0.0

        # Give each instance a unique node name to avoid ROS2 name conflicts
        ForkliftEnv._instance_count += 1
        node_name = f"forklift_env_node_{ForkliftEnv._instance_count}"

        # Start shared executor if not running
        _ensure_rclpy_running()

        self._ros_node = ForkliftROSNode(node_name)
        _add_node(self._ros_node)

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        self._ros_node.publish_cmd_vel(0.0, 0.0)
        time.sleep(0.1)
        self._ros_node.reset_world()
        time.sleep(0.5)

        self._step_count = 0
        self._prev_dist = self._distance_to_goal()

        return self._get_obs(), {}

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        linear, angular = self.ACTION_TABLE[action]
        self._ros_node.publish_cmd_vel(linear, angular)
        time.sleep(0.1)  # 10 Hz control loop

        obs = self._get_obs()
        curr_dist = self._distance_to_goal()
        reward, terminated = self._compute_reward(curr_dist)

        self._prev_dist = curr_dist
        self._step_count += 1
        truncated = self._step_count >= self.MAX_STEPS

        return obs, reward, terminated, truncated, {}

    def close(self) -> None:
        self._ros_node.publish_cmd_vel(0.0, 0.0)
        _remove_node(self._ros_node)
        self._ros_node.destroy_node()

    def _get_obs(self) -> np.ndarray:
        x, y, yaw = self._ros_node.get_pose()
        return np.array(
            [x, y, yaw, self.GOAL_X, self.GOAL_Y], dtype=np.float32
        )

    def _distance_to_goal(self) -> float:
        x, y, _ = self._ros_node.get_pose()
        return math.hypot(self.GOAL_X - x, self.GOAL_Y - y)

    def _compute_reward(self, curr_dist: float) -> tuple[float, bool]:
        if curr_dist < self.GOAL_THRESHOLD:
            return 100.0, True

        x, y, _ = self._ros_node.get_pose()
        if abs(x) > self.WALL_LIMIT or abs(y) > self.WALL_LIMIT:
            return -50.0, True

        reward = 10.0 * (self._prev_dist - curr_dist) - 0.1
        return reward, False
