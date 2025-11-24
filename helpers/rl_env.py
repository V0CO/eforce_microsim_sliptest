"""
Gym-like reinforcement learning environment for cone-following driving task.

This environment wraps the simulation state and provides:
- Normalized observations (cone positions, IMU, velocity)
- Continuous action space (steering, throttle)
- Reward function based on progress and penalties
- Episode termination on collision or checkpoint completion
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from pathlib import Path
from typing import Tuple, Dict, Any
from helpers.sim import State
from config import state_config, ConeClasses


class ConeDriveEnv(gym.Env):
    """
    Gym environment for learning to drive between cones.
    
    Observation: Normalized cone positions (local frame) + IMU + velocity
    Action: [steering ∈ [-1,1], throttle ∈ [0,1]]
    Reward: Speed + CTE penalty + steering smoothness + collision/checkpoint rewards
    """
    
    metadata = {"render_modes": []}
    
    def __init__(
        self,
        map_path: str | Path = "maps/map1.json",
        max_episode_steps: int = 1000,
        n_cones: int = 20,
        max_cone_range: float = 30.0,
        action_smoothing: float = 0.7,
        verbose: bool = False,
    ):
        """
        Initialize the cone-driving environment.
        
        Args:
            map_path: Path to the track JSON file
            max_episode_steps: Maximum steps per episode
            n_cones: Number of nearest cones to track
            max_cone_range: Maximum range for cone visibility (meters)
            action_smoothing: EMA smoothing factor for steering (0=no smooth, 1=full smooth)
            verbose: Print debug info
        """
        super().__init__()
        
        self.map_path = Path(map_path)
        self.max_episode_steps = max_episode_steps
        self.n_cones = n_cones
        self.max_cone_range = max_cone_range
        self.action_smoothing = action_smoothing
        self.verbose = verbose
        
        # Initialize simulator state
        self.state = State(self.map_path, state_config)
        self.start_pose = self.state.car_pose.copy()
        
        # Episode tracking
        self.step_count = 0
        self.last_checkpoint_distance = 0.0
        self.prev_steering = 0.0
        self.prev_throttle = 0.0
        
        # Define observation and action spaces
        # Observation: n_cones*2 (x,y) + 4 IMU values = n_cones*2 + 4
        obs_dim = self.n_cones * 2 + 4
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(obs_dim,),
            dtype=np.float32,
        )
        
        # Action space: [steering, throttle]
        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        
        if self.verbose:
            print(f"ConeDriveEnv initialized:")
            print(f"  Map: {self.map_path}")
            print(f"  Obs space: {self.observation_space.shape}")
            print(f"  Action space: {self.action_space.shape}")
    
    def reset(self, seed: int | None = None, options: Dict[str, Any] | None = None) -> Tuple[np.ndarray, Dict]:
        """
        Reset the environment to initial state.
        
        Returns:
            obs: Initial observation (normalized)
            info: Metadata dict
        """
        super().reset(seed=seed)
        
        # Reinitialize simulator state
        self.state = State(self.map_path, state_config)
        self.start_pose = self.state.car_pose.copy()
        self.step_count = 0
        self.last_checkpoint_distance = 0.0
        self.prev_steering = 0.0
        self.prev_throttle = 0.0
        
        obs = self._get_observation()
        info = {"reset": True}
        
        return obs, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one environment step.
        
        Args:
            action: [steering ∈ [-1,1], throttle ∈ [0,1]]
        
        Returns:
            obs: Next observation
            reward: Scalar reward
            terminated: Whether episode ended (collision/checkpoint)
            truncated: Whether episode truncated (timeout)
            info: Metadata dict
        """
        self.step_count += 1
        
        # Parse action
        steering_cmd = float(action[0])  # [-1, 1]
        throttle_cmd = float(action[1])  # [0, 1]
        
        # Apply action smoothing (EMA)
        self.prev_steering = (
            self.action_smoothing * self.prev_steering +
            (1.0 - self.action_smoothing) * steering_cmd
        )
        self.prev_throttle = (
            self.action_smoothing * self.prev_throttle +
            (1.0 - self.action_smoothing) * throttle_cmd
        )
        
        # Convert steering to [-30, 30] degrees and throttle to [0, 8] m/s
        steering_angle_deg = self.prev_steering * 30.0
        speed_setpoint = self.prev_throttle * 8.0
        
        # Update simulator state
        self.state.set_new_setpoints(steering_angle_deg, speed_setpoint)
        self.state.update_state(1.0 / 90.0)  # 90 Hz simulation
        
        # Check termination conditions
        terminated = False
        collision = False
        checkpoint_reached = False
        
        # Collision detection
        if self.state.cones_hit.sum() > 0:
            terminated = True
            collision = True
        
        # Timeout
        truncated = self.step_count >= self.max_episode_steps
        
        # Compute reward
        reward = self._compute_reward(collision, checkpoint_reached)
        
        # Get next observation
        obs = self._get_observation()
        
        info = {
            "step": self.step_count,
            "collision": collision,
            "speed": float(self.state.noisy_speed),
            "steering": float(self.state.steering_angle),
            "cones_hit": int(self.state.cones_hit.sum()),
        }
        
        return obs, reward, terminated, truncated, info
    
    def _get_observation(self) -> np.ndarray:
        """
        Construct normalized observation vector.
        
        Returns:
            obs: Flattened array [normalized_cones (x,y), IMU (4,)]
        """
        obs_list = []
        
        # Get visible cones in local frame and normalize
        cones_local = self.state.get_detections()  # Already in local frame from simulator
        
        if len(cones_local) > 0:
            # Sort by distance from origin
            distances = np.sqrt(cones_local[:, 0]**2 + cones_local[:, 1]**2)
            sorted_indices = np.argsort(distances)
            cones_sorted = cones_local[sorted_indices]
            
            # Keep only nearest n_cones and within range
            cones_filtered = cones_sorted[
                (distances[sorted_indices] <= self.max_cone_range) &
                (distances[sorted_indices] > 0.1)
            ]
            cones_filtered = cones_filtered[:self.n_cones]
        else:
            cones_filtered = np.array([])
        
        # Normalize cone positions to [-1, 1]
        for i in range(self.n_cones):
            if i < len(cones_filtered):
                x_norm = np.clip(cones_filtered[i, 0] / self.max_cone_range, -1.0, 1.0)
                y_norm = np.clip(cones_filtered[i, 1] / self.max_cone_range, -1.0, 1.0)
                obs_list.append(x_norm)
                obs_list.append(y_norm)
            else:
                # Pad with zeros
                obs_list.append(0.0)
                obs_list.append(0.0)
        
        # Add IMU data (normalized to [-1, 1])
        obs = self.state.get_obs()
        speed_norm = np.clip(obs["actual_speed"] / 10.0, -1.0, 1.0)
        yaw_rate_norm = np.clip(obs["ins_imu_gyro"][2] / 180.0, -1.0, 1.0)  # deg/s to [-1,1]
        
        # Use a small accel estimate from velocity change
        accel_x_norm = np.clip(self.state.velocity[0] / 10.0, -1.0, 1.0)
        accel_y_norm = np.clip(self.state.velocity[1] / 10.0, -1.0, 1.0)
        
        obs_list.extend([speed_norm, yaw_rate_norm, accel_x_norm, accel_y_norm])
        
        return np.array(obs_list, dtype=np.float32)
    
    def _compute_reward(self, collision: bool, checkpoint_reached: bool) -> float:
        """
        Compute reward signal.
        
        Rewards:
            +1.0 * forward_speed
            -2.0 * abs(cross_track_error)
            -0.1 * steering^2
            -5.0 * collision
            +20.0 * checkpoint
        
        Returns:
            reward: Scalar reward value
        """
        reward = 0.0
        
        # Forward speed reward (encourage movement)
        forward_speed = self.state.noisy_speed
        reward += 1.0 * forward_speed
        
        # Cross-track error penalty (lateral deviation from centerline)
        # Approximate CTE using lateral position in local frame
        obs = self.state.get_obs()
        lateral_pos = obs["ins_position"][1]  # y in local frame
        cte = abs(lateral_pos)
        reward -= 2.0 * cte
        
        # Steering smoothness penalty
        steering_norm = self.state.steering_angle / 30.0
        reward -= 0.1 * (steering_norm ** 2)
        
        # Collision penalty
        if collision:
            reward -= 5.0
        
        # Checkpoint reward
        if checkpoint_reached:
            reward += 20.0
        
        return float(reward)
    
    def render(self, mode: str = "human") -> None:
        """Render is handled by simulator StateRenderer if needed."""
        pass
    
    def close(self) -> None:
        """Clean up resources."""
        if hasattr(self, "state"):
            del self.state


class ConeDriveEnvWrapper(gym.Wrapper):
    """
    Wrapper to add episode monitoring and stats tracking.
    """
    
    def __init__(self, env: ConeDriveEnv):
        super().__init__(env)
        self.episode_returns = 0.0
        self.episode_length = 0
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.episode_returns = 0.0
        self.episode_length = 0
        return obs, info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.episode_returns += reward
        self.episode_length += 1
        
        if terminated or truncated:
            info["episode_return"] = self.episode_returns
            info["episode_length"] = self.episode_length
        
        return obs, reward, terminated, truncated, info
