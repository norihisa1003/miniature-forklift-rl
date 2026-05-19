#!/bin/bash
# launch_sim.sh
# Usage: bash sim/launch_sim.sh

set -e

source /opt/ros/jazzy/setup.bash
unset CYCLONEDDS_URI

export GZ_SIM_RESOURCE_PATH=~/projects/miniature-forklift-rl/sim/models

# Launch Gazebo in the background
gz sim -r ~/projects/miniature-forklift-rl/sim/worlds/warehouse.sdf &
GZ_PID=$!
echo "Gazebo started (PID: $GZ_PID)"

sleep 3

# Launch ROS2 bridge (topics + set_pose service)
echo "Starting ROS2 bridge..."
ros2 run ros_gz_bridge parameter_bridge \
  /cmd_vel@geometry_msgs/msg/Twist@gz.msgs.Twist \
  /odom@nav_msgs/msg/Odometry@gz.msgs.Odometry \
  /world/warehouse/set_pose@ros_gz_interfaces/srv/SetEntityPose &
BRIDGE_PID=$!
echo "Bridge started (PID: $BRIDGE_PID)"

echo ""
echo "Simulation ready."
echo "Press Ctrl+C to stop all processes."

trap "kill $GZ_PID $BRIDGE_PID 2>/dev/null; exit 0" INT
wait
