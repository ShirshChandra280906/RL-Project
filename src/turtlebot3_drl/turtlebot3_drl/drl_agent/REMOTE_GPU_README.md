# Remote GPU Inference for TurtleBot3 DRL Navigation

This document explains how to offload neural network computations to a remote GPU server.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Local Robot/Computer                         │
│  ┌──────────────┐    ┌──────────────┐    ┌────────────────────────┐ │
│  │ Environment  │◄──►│ RemoteDrl    │◄──►│ RemoteInferenceClient │ │
│  │    Node      │    │   Agent      │    │                        │ │
│  └──────────────┘    └──────────────┘    └──────────┬─────────────┘ │
│                                                      │               │
│                          SSH Tunnel (port 8000)      │               │
└──────────────────────────────────────────────────────┼───────────────┘
                                                       │
                                                       ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    Remote GPU Server (192.168.3.214)                │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │                    FastAPI Server (port 8000)                  │ │
│  │                                                                │ │
│  │   POST /infer                                                  │ │
│  │   Input:  {"values": [float, float, ...]}                      │ │
│  │   Output: {"result": [float, float, ...]}                      │ │
│  │                                                                │ │
│  │   ┌──────────────────────────────────────────────────────┐    │ │
│  │   │              Neural Network (GPU)                    │    │ │
│  │   │   - DDPG Actor Network                               │    │ │
│  │   │   - TD3 Actor Network                                │    │ │
│  │   │   - DQN Network                                      │    │ │
│  │   └──────────────────────────────────────────────────────┘    │ │
│  └────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

## Prerequisites

1. **Remote GPU Server**: A GPU server with:
   - PyTorch with CUDA support
   - FastAPI server running the inference endpoint
   - Network accessibility from your local machine

2. **Local Machine**: Running ROS2 with:
   - `turtlebot3_drl` package installed
   - SSH client for tunneling

## Setup Instructions

### Step 1: Establish SSH Tunnel

Open a terminal and create an SSH tunnel to forward port 8000:

```bash
ssh -L 8000:localhost:8000 rl-group13@192.168.3.214
```

Keep this terminal open while running inference.

### Step 2: Verify Connection

Test the connection to the inference server:

```bash
curl -X POST http://localhost:8000/infer \
  -H "Content-Type: application/json" \
  -d '{"values": [0.1, 0.2, 0.3]}'
```

You should receive a response like:
```json
{"result": [0.5, -0.3]}
```

### Step 3: Build the ROS2 Workspace

```bash
cd ~/turtlebot3_drlnav
colcon build --packages-select turtlebot3_drl
source install/setup.bash
```

### Step 4: Launch the Remote DRL System

Option A: Using the launch file (starts environment + GPU client):
```bash
ros2 launch turtlebot3_gazebo turtlebot3_drl_remote.launch.py stage:=stage1
```

Then in another terminal, start the remote agent:
```bash
ros2 run turtlebot3_drl remote_agent
```

Option B: Start nodes individually:
```bash
# Terminal 1: GPU Client Node
ros2 run turtlebot3_drl gpu_client

# Terminal 2: Environment Node  
ros2 run turtlebot3_drl environment

# Terminal 3: Gazebo Goals Node
ros2 run turtlebot3_drl gazebo_goals

# Terminal 4: Remote Agent
ros2 run turtlebot3_drl remote_agent
```

## Configuration

### GPU Client Parameters

You can configure the GPU client via ROS2 parameters:

```bash
ros2 run turtlebot3_drl gpu_client --ros-args \
  -p gpu_server_url:=http://localhost:8000 \
  -p request_timeout:=5.0 \
  -p retry_count:=3 \
  -p retry_delay:=0.5
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `gpu_server_url` | `http://localhost:8000` | URL of the FastAPI inference server |
| `request_timeout` | `5.0` | Timeout for inference requests (seconds) |
| `retry_count` | `3` | Number of retry attempts on failure |
| `retry_delay` | `0.5` | Delay between retries (seconds) |

### Remote Agent Parameters

```bash
ros2 run turtlebot3_drl remote_agent --ros-args \
  -p gpu_server_url:=http://localhost:8000 \
  -p request_timeout:=5.0 \
  -p action_size:=2 \
  -p exploration_noise:=0.1 \
  -p training:=true
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `gpu_server_url` | `http://localhost:8000` | URL of the FastAPI inference server |
| `request_timeout` | `5.0` | Timeout for inference requests (seconds) |
| `action_size` | `2` | Size of action vector (linear_vel, angular_vel) |
| `exploration_noise` | `0.1` | Exploration noise during training |
| `training` | `true` | Whether agent is in training mode |

## API Specification

### Inference Endpoint

**URL**: `POST http://localhost:8000/infer`

**Request Body**:
```json
{
  "values": [float, float, ...]
}
```

The `values` array contains the state observation:
- LiDAR scan data (24 values, normalized 0-1)
- Goal distance (1 value, normalized)
- Goal angle (1 value, in radians)
- Previous action (2 values, linear_vel and angular_vel)

**Response Body**:
```json
{
  "result": [float, float]
}
```

The `result` array contains the action:
- `result[0]`: Linear velocity command
- `result[1]`: Angular velocity command

### Health Check Endpoint

**URL**: `GET http://localhost:8000/health`

**Response**: 
```json
{"status": "healthy"}
```

## Troubleshooting

### Connection Refused
1. Verify SSH tunnel is active
2. Check if FastAPI server is running on remote host
3. Ensure port 8000 is not blocked by firewall

### Timeout Errors
1. Increase `request_timeout` parameter
2. Check network latency to remote server
3. Verify GPU server is not overloaded

### Invalid Response Format
1. Verify FastAPI server returns correct JSON format
2. Check that model output dimensions match expected action size

### SSH Tunnel Drops
1. Use `ssh -o ServerAliveInterval=60` to keep connection alive
2. Consider using `autossh` for automatic reconnection:
   ```bash
   autossh -M 0 -o "ServerAliveInterval 30" -o "ServerAliveCountMax 3" \
     -L 8000:localhost:8000 rl-group13@192.168.3.214
   ```

## File Structure

```
turtlebot3_drl/
├── drl_agent/
│   ├── __init__.py           # Module exports
│   ├── drl_agent.py          # Original local agent
│   ├── drl_agent_remote.py   # Remote inference agent (NEW)
│   ├── gpu_client.py         # GPU client node (NEW)
│   ├── ddpg.py               # DDPG algorithm
│   ├── td3.py                # TD3 algorithm
│   └── dqn.py                # DQN algorithm
└── ...

turtlebot3_gazebo/
└── launch/
    └── turtlebot3_drl_remote.launch.py  # Remote DRL launch (NEW)
```

## Performance Considerations

1. **Latency**: Remote inference adds network latency (~1-10ms locally, more over WAN)
2. **Batching**: Consider batching multiple inference requests if possible
3. **Caching**: The inference server can cache models to reduce load time
4. **Connection Pooling**: Uses HTTP keep-alive for efficient connections

## Security Notes

1. SSH tunnel provides encrypted communication
2. Only expose inference endpoint, not training endpoints
3. Consider using SSH keys instead of passwords
4. Limit server access to known IP addresses
