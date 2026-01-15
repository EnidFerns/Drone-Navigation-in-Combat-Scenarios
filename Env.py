import gym
from gym import spaces
import pybullet as p
import pybullet_data
import numpy as np
import time
import math
import random
from typing import Dict, Tuple, List, Optional


class DroneEnv(gym.Env):
    """
    3D drone environment for training reinforcement learning agents in drone navigation, obstacle avoidance, and combat
    """

    def __init__(self, render: bool = True, num_obstacles: int = 10, enemy_drones: int = 1):
        super(DroneEnv, self).__init__()

        # Environment parameters
        self.render_mode = render
        self.num_obstacles = num_obstacles
        self.num_enemy_drones = enemy_drones
        self.max_steps = 1000
        self.current_step = 0

        # Action space (throttle, rotation-pitch, rotation-yaw, rotation-roll, fire)
        # Each action is a continuous value in range [-1, 1], fire is a binary action [0, 1]
        self.action_space = spaces.Box(
            low=np.array([-1, -1, -1, -1, 0]),
            high=np.array([1, 1, 1, 1, 1]),
            dtype=np.float32
        )

        # Observation space
        # [x, y, z, velocity x, velocity y, velocity z, pitch, yaw, roll,
        #  obstacle1 distance, obstacle1 direction x, obstacle1 direction y, obstacle1 direction z,
        #  ...(for each detected obstacle),
        #  enemy1 distance, enemy1 direction x, enemy1 direction y, enemy1 direction z, enemy1 relative speed,
        #  ...(for each enemy)]

        # We assume that 5 nearest obstacles and all enemies can be detected
        obstacles_dims = 4 * 5  # 4 values per obstacle (distance + 3D direction)
        enemies_dims = 5 * self.num_enemy_drones  # 5 values per enemy (distance + 3D direction + relative speed)

        # 9 drone state values + obstacle information + enemy information
        obs_dim = 9 + obstacles_dims + enemies_dims

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )

        # PyBullet setup
        self.physics_client = None
        self.drone_id = None
        self.obstacle_ids = []
        self.enemy_drone_ids = []
        self.ground_id = None

        # Drone parameters
        self.drone_size = 0.3  # Drone size (meters)
        self.max_drone_speed = 5.0  # Maximum speed (meters/second)
        self.drone_lifepoints = 100  # Drone life points

        # Weapon parameters
        self.bullet_ids = []
        self.enemy_bullet_ids = []
        self.fire_cooldown = 0  # Weapon cooldown counter
        self.fire_cooldown_time = 10  # Number of frames between shots

        # Enemy AI parameters
        self.enemy_fire_cooldowns = np.zeros(self.num_enemy_drones)

    def reset(self) -> np.ndarray:
        """Reset the environment to initial state and return initial observation"""
        # Reset counters
        self.current_step = 0
        self.drone_lifepoints = 100
        self.bullet_ids = []
        self.enemy_bullet_ids = []
        self.fire_cooldown = 0
        self.enemy_fire_cooldowns = np.zeros(self.num_enemy_drones)

        # Disconnect from physics engine if already connected
        if self.physics_client is not None:
            p.disconnect(self.physics_client)

        # Connect to PyBullet physics engine
        if self.render_mode:
            self.physics_client = p.connect(p.GUI)  # GUI mode
        else:
            self.physics_client = p.connect(p.DIRECT)  # Headless mode without GUI

        # Load PyBullet data
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        # Set gravity and time step
        p.setGravity(0, 0, -9.8)
        p.setTimeStep(1. / 240.)

        # Load ground
        self.ground_id = p.loadURDF("plane.urdf")

        # Create our drone (using a simple cube to represent it)
        base_position = [0, 0, 2]  # Initial position
        self.drone_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=[self.drone_size / 2, self.drone_size / 2,
                                                                        self.drone_size / 2])
        self.drone_id = p.createMultiBody(
            baseMass=1.0,
            baseCollisionShapeIndex=self.drone_id,
            basePosition=base_position,
            baseOrientation=[0, 0, 0, 1]
        )
        p.changeVisualShape(self.drone_id, -1, rgbaColor=[0, 0, 1, 1])  # [R, G, B, Alpha] Blue color
        # Create obstacles (randomly placed)
        self.obstacle_ids = []
        for i in range(self.num_obstacles):
            # Random position, but not too close to the drone
            position = [
                random.uniform(-10, 10),
                random.uniform(-10, 10),
                random.uniform(1, 5)
            ]

            # Ensure obstacles are not directly at the drone's starting position
            while np.linalg.norm(np.array(position) - np.array(base_position)) < 2:
                position = [
                    random.uniform(-10, 10),
                    random.uniform(-10, 10),
                    random.uniform(1, 5)
                ]

            obstacle_size = random.uniform(0.3, 1.0)
            obstacle_shape = p.createCollisionShape(
                p.GEOM_BOX,
                halfExtents=[obstacle_size / 2, obstacle_size / 2, obstacle_size / 2]
            )
            obstacle_id = p.createMultiBody(
                baseMass=0,  # Static obstacle
                baseCollisionShapeIndex=obstacle_shape,
                basePosition=position
            )
            self.obstacle_ids.append(obstacle_id)

        # Create enemy drones
        self.enemy_drone_ids = []
        for i in range(self.num_enemy_drones):
            # Random position, but keep a certain distance from our drone
            position = [
                random.uniform(-10, 10),
                random.uniform(-10, 10),
                random.uniform(1, 5)
            ]

            # Ensure enemy drones are not directly at our drone's starting position
            while np.linalg.norm(np.array(position) - np.array(base_position)) < 5:
                position = [
                    random.uniform(-10, 10),
                    random.uniform(-10, 10),
                    random.uniform(1, 5)
                ]

            enemy_shape = p.createCollisionShape(
                p.GEOM_BOX,
                halfExtents=[self.drone_size / 2, self.drone_size / 2, self.drone_size / 2]
            )
            enemy_id = p.createMultiBody(
                baseMass=1.0,
                baseCollisionShapeIndex=enemy_shape,
                basePosition=position,
                baseOrientation=[0, 0, 0, 1]
            )

            # Set enemy drone color to red
            p.changeVisualShape(enemy_id, -1, rgbaColor=[1, 0, 0, 1])

            self.enemy_drone_ids.append(enemy_id)

        # Get initial observation
        observation = self._get_observation()

        return observation

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute one step with the action and return next observation, reward, done flag, and additional info"""
        self.current_step += 1

        # Parse action
        throttle = action[0]  # Throttle control (-1 to 1)
        pitch = action[1]  # Pitch control (-1 to 1)
        yaw = action[2]  # Yaw control (-1 to 1)
        roll = action[3]  # Roll control (-1 to 1)
        fire = action[4] > 0.5  # Fire control (binary)

        # Get current drone state
        drone_pos, drone_orn = p.getBasePositionAndOrientation(self.drone_id)
        drone_vel, drone_ang_vel = p.getBaseVelocity(self.drone_id)

        # Enforce height limit - more aggressive method
        min_height = 2.0  # Minimum height 2 meters
        if drone_pos[2] < min_height:
            # 1. Apply strong upward force
            upward_force = [0, 0, 50.0]
            p.applyExternalForce(self.drone_id, -1, upward_force, [0, 0, 0], p.WORLD_FRAME)

            # 2. Reset vertical velocity to positive value
            current_vel = list(drone_vel)
            if current_vel[2] <= 0:  # If descending or stationary
                current_vel[2] = 5.0  # Force upward velocity
                p.resetBaseVelocity(self.drone_id, linearVelocity=current_vel)

        # Calculate drone's forward direction
        drone_orn_quat = drone_orn
        forward_vector = self._get_forward_vector(drone_orn_quat)

        # Apply thrust and torque to control drone movement
        # Map [-1,1] range to appropriate physics engine force and torque values
        max_force = 20.0
        max_torque = 5.0

        # Calculate thrust (along z-axis positive direction)
        thrust_force = [0, 0, max_force * (throttle + 1) / 2]  # Map to [0, max_force]

        # Convert force to drone coordinate system
        force_pos = [0, 0, 0]  # Force application point at drone center
        p.applyExternalForce(self.drone_id, -1, thrust_force, force_pos, p.LINK_FRAME)

        # Apply torque to control rotation
        torque = [
            max_torque * pitch,  # Pitch
            max_torque * roll,  # Roll
            max_torque * yaw  # Yaw
        ]
        p.applyExternalTorque(self.drone_id, -1, torque, p.LINK_FRAME)

        # Weapon system - handle shooting
        if fire and self.fire_cooldown <= 0:
            self._fire_bullet(drone_pos, forward_vector)
            self.fire_cooldown = self.fire_cooldown_time

        if self.fire_cooldown > 0:
            self.fire_cooldown -= 1

        # Update bullet positions
        self._update_bullets()

        # Enemy drone AI behavior
        self._update_enemy_drones()

        # Detect collisions
        is_collision = self._check_collisions()

        # Step simulation forward
        p.stepSimulation()

        # Get new observation
        observation = self._get_observation()

        # Calculate reward
        reward = self._compute_reward(is_collision)

        # Check if done
        done = self._is_done(is_collision)

        # Additional info
        info = {
            "drone_lifepoints": self.drone_lifepoints,
            "step": self.current_step
        }

        return observation, reward, done, info

    def _compute_reward(self, is_collision: bool) -> float:
        """Compute reward function"""
        reward = 0.0

        # Survival reward
        reward += 0.1

        # Get drone position
        drone_pos, _ = p.getBasePositionAndOrientation(self.drone_id)
        current_height = drone_pos[2]

        # Height reward/penalty - harsh penalty for low height
        min_height = 2.0
        optimal_min = 3.0
        optimal_max = 5.0

        if current_height < min_height:
            # Very severe low height penalty
            reward -= 50.0  # Significantly increase penalty strength
        elif current_height < optimal_min:
            # Mild penalty for slightly low height
            height_penalty = (optimal_min - current_height) * 2.0
            reward -= height_penalty
        elif current_height <= optimal_max:
            # Reward for ideal height
            reward += 0.5  # Increase reward for ideal height
        else:
            # Mild penalty for too high
            height_penalty = (current_height - optimal_max) * 0.5
            reward -= height_penalty

        # Other reward conditions
        if is_collision["obstacle"]:
            reward -= 10.0
        if is_collision["hit_by_enemy"]:
            reward -= 5.0
        if is_collision["hit_enemy"]:
            reward += 10.0
        if is_collision["defeated_enemy"]:
            reward += 50.0

        # Boundary exit penalty
        if abs(drone_pos[0]) > 20 or abs(drone_pos[1]) > 20 or drone_pos[2] > 10:
            reward -= 5.0

        return reward

    def _is_done(self, is_collision: Dict[str, bool]) -> bool:
        """Check if termination conditions are met"""
        # If maximum steps exceeded
        if self.current_step >= self.max_steps:
            return True

        # If drone life points are 0
        if self.drone_lifepoints <= 0:
            return True

        # If all enemies are eliminated
        if len(self.enemy_drone_ids) == 0:
            return True

        # If drone flies too far out of bounds
        drone_pos, _ = p.getBasePositionAndOrientation(self.drone_id)
        if abs(drone_pos[0]) > 25 or abs(drone_pos[1]) > 25 or drone_pos[2] < -5 or drone_pos[2] > 15:
            return True

        return False

    def _get_observation(self) -> np.ndarray:
        """Get observation vector of current environment state"""
        # Get drone state
        drone_pos, drone_orn = p.getBasePositionAndOrientation(self.drone_id)
        drone_vel, drone_ang_vel = p.getBaseVelocity(self.drone_id)

        # Convert quaternion to Euler angles (roll, pitch, yaw)
        euler_angles = p.getEulerFromQuaternion(drone_orn)

        # Basic drone state
        obs = list(drone_pos) + list(drone_vel[:3]) + list(euler_angles)

        # Get information about the nearest 5 obstacles
        obstacle_info = self._get_obstacles_info()
        obs.extend(obstacle_info)

        # Get enemy drone information
        enemy_info = self._get_enemies_info()
        obs.extend(enemy_info)

        return np.array(obs, dtype=np.float32)

    def _get_obstacles_info(self) -> List[float]:
        """Get information about the nearest obstacles"""
        drone_pos, _ = p.getBasePositionAndOrientation(self.drone_id)
        obstacles_data = []

        # Calculate distance to each obstacle
        distances = []
        for obstacle_id in self.obstacle_ids:
            obstacle_pos, _ = p.getBasePositionAndOrientation(obstacle_id)
            distance = np.linalg.norm(np.array(drone_pos) - np.array(obstacle_pos))
            distances.append((distance, obstacle_id, obstacle_pos))

        # Sort by distance
        distances.sort(key=lambda x: x[0])

        # Get data for the 5 nearest obstacles
        for i in range(min(5, len(distances))):
            distance, _, obstacle_pos = distances[i]

            # Calculate direction vector
            direction = np.array(obstacle_pos) - np.array(drone_pos)
            if np.linalg.norm(direction) > 0:
                direction = direction / np.linalg.norm(direction)

            # Add distance and direction information
            obstacles_data.extend([distance, direction[0], direction[1], direction[2]])

        # If fewer than 5 obstacles, fill with zeros
        while len(obstacles_data) < 5 * 4:
            obstacles_data.extend([20.0, 0.0, 0.0, 0.0])  # Default large distance and zero vector

        return obstacles_data

    def _get_enemies_info(self) -> List[float]:
        """Get enemy drone information"""
        drone_pos, _ = p.getBasePositionAndOrientation(self.drone_id)
        drone_vel, _ = p.getBaseVelocity(self.drone_id)
        enemies_data = []

        for enemy_id in self.enemy_drone_ids:
            enemy_pos, _ = p.getBasePositionAndOrientation(enemy_id)
            enemy_vel, _ = p.getBaseVelocity(enemy_id)

            # Calculate distance
            distance = np.linalg.norm(np.array(drone_pos) - np.array(enemy_pos))

            # Calculate direction vector
            direction = np.array(enemy_pos) - np.array(drone_pos)
            if np.linalg.norm(direction) > 0:
                direction = direction / np.linalg.norm(direction)

            # Calculate relative speed
            relative_speed = np.linalg.norm(np.array(enemy_vel[:3]) - np.array(drone_vel[:3]))

            # Add information
            enemies_data.extend([distance, direction[0], direction[1], direction[2], relative_speed])

        # If fewer enemies than expected, fill with zeros
        while len(enemies_data) < self.num_enemy_drones * 5:
            enemies_data.extend([20.0, 0.0, 0.0, 0.0, 0.0])  # Default large distance, zero vector and zero relative speed

        return enemies_data

    def _fire_bullet(self, position: List[float], direction: np.ndarray) -> None:
        """Fire a bullet from the drone position"""
        # Bullet initial position slightly forward
        bullet_pos = np.array(position) + direction * self.drone_size

        # Create a small sphere as the bullet
        bullet_radius = 0.05
        bullet_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=bullet_radius)
        bullet_id = p.createMultiBody(
            baseMass=0.1,  # Lightweight
            baseCollisionShapeIndex=bullet_shape,
            basePosition=bullet_pos.tolist()
        )

        # Set bullet color to yellow
        p.changeVisualShape(bullet_id, -1, rgbaColor=[1, 1, 0, 1])

        # Give bullet an initial velocity
        bullet_speed = 20.0
        bullet_velocity = direction * bullet_speed
        p.resetBaseVelocity(bullet_id, bullet_velocity.tolist())

        # Record bullet id and creation time
        self.bullet_ids.append((bullet_id, self.current_step))

    def _enemy_fire_bullet(self, enemy_id: int, target_pos: List[float]) -> None:
        """Enemy drone fires a bullet"""
        enemy_pos, enemy_orn = p.getBasePositionAndOrientation(enemy_id)

        # Calculate direction towards target
        direction = np.array(target_pos) - np.array(enemy_pos)
        if np.linalg.norm(direction) > 0:
            direction = direction / np.linalg.norm(direction)

        # Bullet initial position slightly forward
        bullet_pos = np.array(enemy_pos) + direction * self.drone_size

        # Create a small sphere as the bullet
        bullet_radius = 0.05
        bullet_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=bullet_radius)
        bullet_id = p.createMultiBody(
            baseMass=0.1,  # Lightweight
            baseCollisionShapeIndex=bullet_shape,
            basePosition=bullet_pos.tolist()
        )

        # Set bullet color to red
        p.changeVisualShape(bullet_id, -1, rgbaColor=[1, 0, 0, 1])

        # Give bullet an initial velocity
        bullet_speed = 15.0
        bullet_velocity = direction * bullet_speed
        p.resetBaseVelocity(bullet_id, bullet_velocity.tolist())

        # Record enemy bullet id and creation time
        self.enemy_bullet_ids.append((bullet_id, self.current_step))

    def _update_bullets(self) -> None:
        """Update all bullet states, remove expired bullets"""
        # Maximum bullet lifetime (steps)
        max_bullet_lifetime = 100

        # Update our bullets
        active_bullets = []
        for bullet_id, creation_step in self.bullet_ids:
            if self.current_step - creation_step < max_bullet_lifetime:
                active_bullets.append((bullet_id, creation_step))
            else:
                p.removeBody(bullet_id)
        self.bullet_ids = active_bullets

        # Update enemy bullets
        active_enemy_bullets = []
        for bullet_id, creation_step in self.enemy_bullet_ids:
            if self.current_step - creation_step < max_bullet_lifetime:
                active_enemy_bullets.append((bullet_id, creation_step))
            else:
                p.removeBody(bullet_id)
        self.enemy_bullet_ids = active_enemy_bullets

    def _update_enemy_drones(self) -> None:
        """Update enemy drone behavior"""
        if not self.enemy_drone_ids:
            return

        drone_pos, _ = p.getBasePositionAndOrientation(self.drone_id)

        for i, enemy_id in enumerate(self.enemy_drone_ids[:]):  # Use a copy for iteration, as elements may be deleted
            enemy_pos, enemy_orn = p.getBasePositionAndOrientation(enemy_id)

            # Simple AI: Move towards our drone and occasionally shoot

            # Calculate direction
            direction = np.array(drone_pos) - np.array(enemy_pos)
            distance = np.linalg.norm(direction)

            if distance > 0:
                direction = direction / distance

            # Apply force to move enemy drone
            # Keep distance if too close
            if distance < 5:
                force = -direction * 5.0  # Move away
            else:
                force = direction * 10.0  # Approach

            # Add some randomness
            force += np.random.normal(0, 2, 3)

            # Apply force
            p.applyExternalForce(enemy_id, -1, force, [0, 0, 0], p.WORLD_FRAME)

            # Shooting logic
            if self.enemy_fire_cooldowns[i] <= 0:
                # 30% chance to shoot
                if random.random() < 0.3 and distance < 10:
                    self._enemy_fire_bullet(enemy_id, drone_pos)
                    self.enemy_fire_cooldowns[i] = self.fire_cooldown_time * 1.5  # Enemy shooting interval slightly longer
            else:
                self.enemy_fire_cooldowns[i] -= 1

    def _check_collisions(self) -> Dict[str, bool]:
        """Check various collision and hit scenarios"""
        result = {
            "obstacle": False,
            "hit_by_enemy": False,
            "hit_enemy": False,
            "defeated_enemy": False
        }

        # Get all collision pairs
        contact_points = p.getContactPoints()

        # Check collisions between our drone and obstacles
        for contact in contact_points:
            if contact[1] == self.drone_id and contact[2] in self.obstacle_ids:
                result["obstacle"] = True
                self.drone_lifepoints -= 10
                break
            elif contact[2] == self.drone_id and contact[1] in self.obstacle_ids:
                result["obstacle"] = True
                self.drone_lifepoints -= 10
                break

        # Check collisions between our bullets and enemy drones
        for contact in contact_points:
            for bullet_id, _ in self.bullet_ids:
                for enemy_id in self.enemy_drone_ids[:]:  # Use a copy for iteration
                    if (contact[1] == bullet_id and contact[2] == enemy_id) or \
                            (contact[2] == bullet_id and contact[1] == enemy_id):
                        result["hit_enemy"] = True

                        # Remove the hit bullet
                        try:
                            p.removeBody(bullet_id)
                            self.bullet_ids = [(bid, step) for bid, step in self.bullet_ids if bid != bullet_id]
                        except:
                            pass

                        # Enemy has 50% chance to be destroyed when hit
                        if random.random() < 0.5:
                            try:
                                p.removeBody(enemy_id)
                                self.enemy_drone_ids.remove(enemy_id)
                                result["defeated_enemy"] = True
                            except:
                                pass

                        break

        # Check collisions between enemy bullets and our drone
        for contact in contact_points:
            for bullet_id, _ in self.enemy_bullet_ids:
                if (contact[1] == bullet_id and contact[2] == self.drone_id) or \
                        (contact[2] == bullet_id and contact[1] == self.drone_id):
                    result["hit_by_enemy"] = True
                    self.drone_lifepoints -= 20

                    # Remove the hit bullet
                    try:
                        p.removeBody(bullet_id)
                        self.enemy_bullet_ids = [(bid, step) for bid, step in self.enemy_bullet_ids if bid != bullet_id]
                    except:
                        pass

                    break

        return result

    def _get_forward_vector(self, quaternion: List[float]) -> np.ndarray:
        """Calculate forward direction vector from quaternion"""
        # Initial forward vector (pointing along positive x-axis when quaternion is [0,0,0,1])
        initial_forward = np.array([1, 0, 0])

        # Create rotation matrix
        rotation_matrix = np.array(p.getMatrixFromQuaternion(quaternion)).reshape(3, 3)

        # Apply rotation
        forward = np.dot(rotation_matrix, initial_forward)

        return forward

    def render(self, mode='human'):
        """Render the environment"""
        if self.render_mode and mode == 'human':
            # Adjust camera view (optional)
            drone_pos, _ = p.getBasePositionAndOrientation(self.drone_id)
            # Set camera position behind and above the drone
            p.resetDebugVisualizerCamera(
                cameraDistance=5.0,
                cameraYaw=50,
                cameraPitch=-35,
                cameraTargetPosition=drone_pos
            )

        # PyBullet already renders in step(), so no additional work needed here
        return None

    def close(self):
        """Close the environment"""
        if self.physics_client is not None:
            p.disconnect(self.physics_client)