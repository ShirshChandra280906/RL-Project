#!/usr/bin/env python3
"""
GPU Client Node for Remote Neural Network Inference

This node acts as a bridge between the ROS2 system and a remote GPU server.
It sends state observations to the remote server and receives action predictions.

The remote server is accessed via an SSH tunnel:
    ssh -L 8000:localhost:8000 rl-group13@192.168.3.214

Usage:
    1. Start SSH tunnel to GPU server
    2. Run this node: ros2 run turtlebot3_drl gpu_client
    3. The node will handle all inference requests via HTTP
"""

import numpy as np
import requests
import json
from typing import List, Tuple, Optional
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray


# ============================================================================
# CONFIGURATION - Modify these values as needed
# ============================================================================

# Remote GPU server endpoint (accessed via SSH tunnel)
GPU_SERVER_URL = "http://localhost:8000/infer"

# Request timeout in seconds
REQUEST_TIMEOUT = 5.0

# Retry configuration
MAX_RETRIES = 3
RETRY_DELAY = 0.5  # seconds


class GPUInferenceClient:
    """
    HTTP client for remote GPU inference.
    
    Sends state vectors to the remote server and receives action predictions.
    The server is expected to have a /infer endpoint that accepts JSON:
        Request:  {"values": [float, ...]}
        Response: {"result": [float, ...]}
    """
    
    def __init__(self, server_url: str = GPU_SERVER_URL, timeout: float = REQUEST_TIMEOUT):
        """
        Initialize the GPU inference client.
        
        Args:
            server_url: Full URL to the inference endpoint
            timeout: Request timeout in seconds
        """
        self.server_url = server_url
        self.timeout = timeout
        self.session = requests.Session()
        
        # Set up headers for JSON communication
        self.headers = {"Content-Type": "application/json"}
        
        # Track connection status
        self.connected = False
        self.last_error = None
    
    def check_connection(self) -> bool:
        """
        Test connection to the GPU server.
        
        Returns:
            True if server is reachable, False otherwise
        """
        try:
            # Send a small test request
            test_data = {"values": [0.0]}
            response = self.session.post(
                self.server_url,
                headers=self.headers,
                json=test_data,
                timeout=self.timeout
            )
            self.connected = response.status_code == 200
            return self.connected
        except Exception as e:
            self.last_error = str(e)
            self.connected = False
            return False
    
    def infer(self, state: List[float], retries: int = MAX_RETRIES) -> Optional[List[float]]:
        """
        Send state to remote GPU server and get action prediction.
        
        This is where the actual remote inference happens!
        The state vector is sent to the GPU server, which runs the neural network
        forward pass and returns the predicted action.
        
        Args:
            state: State vector as list of floats (e.g., LiDAR + goal info)
            retries: Number of retry attempts on failure
            
        Returns:
            Action vector as list of floats, or None on failure
        """
        # Prepare request payload
        # The remote server expects {"values": [float, ...]}
        payload = {"values": state}
        
        for attempt in range(retries):
            try:
                # ===========================================================
                # REMOTE INFERENCE CALL - This is the key HTTP request
                # that offloads computation to the GPU server
                # ===========================================================
                response = self.session.post(
                    self.server_url,
                    headers=self.headers,
                    json=payload,
                    timeout=self.timeout
                )
                
                if response.status_code == 200:
                    result = response.json()
                    # Server returns {"result": [float, ...]}
                    action = result.get("result", None)
                    self.connected = True
                    return action
                else:
                    self.last_error = f"HTTP {response.status_code}: {response.text}"
                    
            except requests.exceptions.ConnectionError as e:
                self.last_error = f"Connection error: {e}"
                self.connected = False
            except requests.exceptions.Timeout as e:
                self.last_error = f"Request timeout: {e}"
            except json.JSONDecodeError as e:
                self.last_error = f"Invalid JSON response: {e}"
            except Exception as e:
                self.last_error = f"Unexpected error: {e}"
            
            # Wait before retry
            if attempt < retries - 1:
                time.sleep(RETRY_DELAY)
        
        return None
    
    def infer_batch(self, states: List[List[float]]) -> Optional[List[List[float]]]:
        """
        Send batch of states for inference.
        
        Note: This requires server support for batch inference.
        Currently sends states one by one if batch not supported.
        
        Args:
            states: List of state vectors
            
        Returns:
            List of action vectors, or None on failure
        """
        # For now, process individually
        # TODO: Add batch support to server for efficiency
        actions = []
        for state in states:
            action = self.infer(state)
            if action is None:
                return None
            actions.append(action)
        return actions


class GPUClientNode(Node):
    """
    ROS2 Node that provides remote GPU inference as a service.
    
    This node:
    1. Subscribes to state observations
    2. Sends them to the remote GPU server for inference
    3. Publishes the resulting actions
    
    Topics:
        Subscribes: /drl_state (Float32MultiArray) - State observations
        Publishes:  /drl_action (Float32MultiArray) - Action predictions
    """
    
    def __init__(self):
        super().__init__('gpu_client_node')
        
        # Declare parameters
        self.declare_parameter('server_url', GPU_SERVER_URL)
        self.declare_parameter('timeout', REQUEST_TIMEOUT)
        
        # Get parameters
        server_url = self.get_parameter('server_url').get_parameter_value().string_value
        timeout = self.get_parameter('timeout').get_parameter_value().double_value
        
        # Initialize GPU client
        self.gpu_client = GPUInferenceClient(server_url, timeout)
        
        # Check initial connection
        self.get_logger().info(f"Connecting to GPU server at {server_url}...")
        if self.gpu_client.check_connection():
            self.get_logger().info("✓ Connected to GPU server")
        else:
            self.get_logger().warn(f"✗ Could not connect to GPU server: {self.gpu_client.last_error}")
            self.get_logger().warn("Make sure SSH tunnel is active: ssh -L 8000:localhost:8000 rl-group13@192.168.3.214")
        
        # Create subscriber for state input
        self.state_subscription = self.create_subscription(
            Float32MultiArray,
            '/drl_state',
            self.state_callback,
            10
        )
        
        # Create publisher for action output
        self.action_publisher = self.create_publisher(
            Float32MultiArray,
            '/drl_action',
            10
        )
        
        # Connection status timer
        self.create_timer(10.0, self.check_connection_callback)
        
        self.get_logger().info("GPU Client Node initialized")
    
    def state_callback(self, msg: Float32MultiArray):
        """
        Handle incoming state observations.
        
        Sends state to GPU server and publishes resulting action.
        """
        state = list(msg.data)
        
        # ===========================================================
        # Call remote GPU server for inference
        # ===========================================================
        action = self.gpu_client.infer(state)
        
        if action is not None:
            # Publish action
            action_msg = Float32MultiArray()
            action_msg.data = action
            self.action_publisher.publish(action_msg)
        else:
            self.get_logger().error(f"Inference failed: {self.gpu_client.last_error}")
    
    def check_connection_callback(self):
        """Periodic connection check."""
        if not self.gpu_client.connected:
            if self.gpu_client.check_connection():
                self.get_logger().info("Reconnected to GPU server")
            else:
                self.get_logger().warn(f"GPU server unavailable: {self.gpu_client.last_error}")


def main(args=None):
    rclpy.init(args=args)
    node = GPUClientNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
