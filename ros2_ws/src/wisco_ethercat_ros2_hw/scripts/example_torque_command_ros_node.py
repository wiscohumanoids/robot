#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState
import time

class StictionBreakerNode(Node):
    def __init__(self):
        super().__init__('example_torque_command_ros_node')
        self.publisher_ = self.create_publisher(Float64MultiArray, '/effort_controller/commands', 10)
        self.subscription = self.create_subscription(JointState, '/joint_states', self.joint_state_callback, 10)
        
        self.current_position = 0.0
        self.current_velocity = 0.0
        self.target_torque = 0.0
        
        self.state = "PULSE" # PULSE or COAST
        self.state_start_time = time.time()
        
        # Tuned for 10:1 reducer stiction: needs higher output torque briefly
        self.pulse_torque = 1.35    # N·m output torque to break stiction cleanly
        self.pulse_duration = 0.25 # 150ms sharp kick
        self.coast_duration = 1.3  # 2 seconds coasting/paused
        
        self.timer = self.create_timer(0.01, self.control_loop) # 100 Hz
        self.get_logger().info("Stiction Breaker Torque Node Initialized.")

    def joint_state_callback(self, msg):
        if len(msg.position) > 0:
            self.current_position = msg.position[0]
        if len(msg.velocity) > 0:
            self.current_velocity = msg.velocity[0]

    def control_loop(self):
        now = time.time()
        elapsed = now - self.state_start_time

        if self.state == "PULSE":
            self.target_torque = self.pulse_torque
            if int(now * 20) % 2 == 0:
                self.get_logger().info(f"[PULSE] Torque: {self.target_torque} N·m | Pos: {self.current_position:.2f}")

            if elapsed >= self.pulse_duration:
                self.state = "COAST"
                self.state_start_time = now

        elif self.state == "COAST":
            self.target_torque = 0.0
            if int(now * 10) % 2 == 0:
                self.get_logger().info(f"[COAST] Torque: 0.0 | Pos: {self.current_position:.2f} | Vel: {self.current_velocity:.4f}")

            if elapsed >= self.coast_duration:
                self.state = "PULSE"
                self.state_start_time = now

        msg = Float64MultiArray()
        msg.data = [self.target_torque]
        self.publisher_.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = StictionBreakerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        pub = node.create_publisher(Float64MultiArray, '/effort_controller/commands', 10)
        pub.publish(Float64MultiArray(data=[0.0]))
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
