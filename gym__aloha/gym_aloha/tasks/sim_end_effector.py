import collections

import numpy as np
from dm_control.suite import base

from gym_aloha.constants import (
    PUPPET_GRIPPER_POSITION_CLOSE,
    START_ARM_POSE,
    normalize_puppet_gripper_position,
    normalize_puppet_gripper_velocity,
    unnormalize_puppet_gripper_position,
)

# Define fixed pose for the right arm outside the class or as class attributes
FIXED_RIGHT_ARM_POS = np.array([0.3, 0.5, 0.29525084])
# Quaternion for 90 degrees rotation around Z-axis (looking right)
FIXED_RIGHT_ARM_QUAT = np.array([0.7071068, 0, 0, 0.7071068]) 
# Define a fixed state for the right gripper (e.g., closed)
# Use the raw control value for closed state
FIXED_RIGHT_GRIPPER_CTRL_VAL = PUPPET_GRIPPER_POSITION_CLOSE 


class BimanualViperXEndEffectorTask(base.Task):
    def __init__(self, random=None):
        super().__init__(random=random)

    def before_step(self, action, physics):
        action_left = action[:2]  # x, y for left end effector

        # set mocap position and quat
        # left
        np.copyto(physics.data.mocap_pos[0], action_left[:3])
        np.copyto(physics.data.mocap_quat[0], action_left[3:7])
        # right
        np.copyto(physics.data.mocap_pos[1], action_right[:3])
        np.copyto(physics.data.mocap_quat[1], action_right[3:7])

        # set gripper
        g_left_ctrl = unnormalize_puppet_gripper_position(action_left[7])
        g_right_ctrl = unnormalize_puppet_gripper_position(action_right[7])
        np.copyto(physics.data.ctrl, np.array([g_left_ctrl, -g_left_ctrl, g_right_ctrl, -g_right_ctrl]))

    def initialize_robots(self, physics):
        # Reset mocap positions and orientations
        np.copyto(physics.data.mocap_pos[0], [-0.31718881, 0.5, 0.29525084])  # Left
        np.copyto(physics.data.mocap_quat[0], [1, 0, 0, 0])
        np.copyto(physics.data.mocap_pos[1], [0.31718881, 0.5, 0.29525084])  # Right
        np.copyto(physics.data.mocap_quat[1], [1, 0, 0, 0])

        # Reset gripper control
        close_gripper_control = np.array(
            [
                PUPPET_GRIPPER_POSITION_CLOSE,
                -PUPPET_GRIPPER_POSITION_CLOSE,
                PUPPET_GRIPPER_POSITION_CLOSE,
                -PUPPET_GRIPPER_POSITION_CLOSE,
            ]
        )
        np.copyto(physics.data.ctrl, close_gripper_control)

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        super().initialize_episode(physics)

    @staticmethod
    def get_qpos(physics):
        qpos_raw = physics.data.qpos.copy()
        left_qpos_raw = qpos_raw[:8]
        right_qpos_raw = qpos_raw[8:16]
        left_arm_qpos = left_qpos_raw[:6]
        right_arm_qpos = right_qpos_raw[:6]
        left_gripper_qpos = [normalize_puppet_gripper_position(left_qpos_raw[6])]
        right_gripper_qpos = [normalize_puppet_gripper_position(right_qpos_raw[6])]
        return np.concatenate([left_arm_qpos, left_gripper_qpos, right_arm_qpos, right_gripper_qpos])

    @staticmethod
    def get_qvel(physics):
        qvel_raw = physics.data.qvel.copy()
        left_qvel_raw = qvel_raw[:8]
        right_qvel_raw = qvel_raw[8:16]
        left_arm_qvel = left_qvel_raw[:6]
        right_arm_qvel = right_qvel_raw[:6]
        left_gripper_qvel = [normalize_puppet_gripper_velocity(left_qvel_raw[6])]
        right_gripper_qvel = [normalize_puppet_gripper_velocity(right_qvel_raw[6])]
        return np.concatenate([left_arm_qvel, left_gripper_qvel, right_arm_qvel, right_gripper_qvel])

    @staticmethod
    def get_env_state(physics):
        raise NotImplementedError

    def get_observation(self, physics):
        obs = collections.OrderedDict()
        obs["mocap_pose_left"] = np.concatenate(
            [physics.data.mocap_pos[0], physics.data.mocap_quat[0]]
        ).copy()
        obs["mocap_pose_right"] = np.concatenate(
            [physics.data.mocap_pos[1], physics.data.mocap_quat[1]]
        ).copy()

        # Add gripper control state
        obs["gripper_ctrl"] = physics.data.ctrl.copy()
        return obs

    def get_reward(self, physics):
        raise NotImplementedError


class SimpleEndEffectorTask(BimanualViperXEndEffectorTask):
    def __init__(self, random=None, target_position=None):
        super().__init__(random=random)
        self.target_position = target_position if target_position is not None else np.array([0.5, 0.5])  # Default target

    def before_step(self, action, physics):
        # Extract x and y positions from the action
        action_left = action[:2]  # x, y for left end effector

        # Set fixed z-axis and orientation
        fixed_z = 0.29525084
        fixed_orientation = [1, 0, 0, 0]

        # Update mocap positions for left and right end effectors
        np.copyto(physics.data.mocap_pos[0], [action_left[0], action_left[1], fixed_z])
        np.copyto(physics.data.mocap_quat[0], fixed_orientation)

        # Keep the right end effector fixed or remove its control logic if unused
        # Example: Keep right EE fixed at its initial pose (or another desired pose)
        # np.copyto(physics.data.mocap_pos[1], [0.3, 0.5, fixed_z]) 
        # np.copyto(physics.data.mocap_quat[1], fixed_orientation)

        # Set gripper control (optional, can be fixed or part of the action)
        # Gripper control might need adjustment if only controlling one arm
        g_left_ctrl = unnormalize_puppet_gripper_position(0.5)  # Example: fixed halfway open
        # g_right_ctrl = unnormalize_puppet_gripper_position(0.5) # Remove or adjust if right arm isn't controlled
        # np.copyto(physics.data.ctrl, np.array([g_left_ctrl, -g_left_ctrl, g_right_ctrl, -g_right_ctrl]))
        np.copyto(physics.data.ctrl, np.array([g_left_ctrl, -g_left_ctrl, 0, 0])) # Example: Only control left gripper

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        # Initialize robots with fixed z-axis and orientation
        self.initialize_robots(physics)

        # Set initial positions for left and right end effectors
        np.copyto(physics.data.mocap_pos[0], [-0.3, 0.5, 0.29525084])  # Left
        np.copyto(physics.data.mocap_pos[1], [0.3, 0.5, 0.29525084])  # Right

        # Optionally, randomize the target position on the plane
        self.target_position = np.random.uniform(low=[-0.5, 0.3], high=[0.5, 0.7])

        super().initialize_episode(physics)

    def get_observation(self, physics):
        obs = collections.OrderedDict()
        # Only include left EE pose if that's what's controlled/relevant
        obs["mocap_pose_left"] = np.concatenate(
            [physics.data.mocap_pos[0][:2], physics.data.mocap_quat[0]] 
        ).copy().astype(np.float32) # Cast to float32
        
        # obs["mocap_pose_right"] = ... # Remove if right EE isn't relevant

        # obs["gripper_ctrl"] = physics.data.ctrl.copy().astype(np.float32) # Cast to float32

        # Add target position to the observation
        obs["target_position"] = self.target_position.copy().astype(np.float32) # Cast to float32
        return obs

    def get_reward(self, physics):
        # Reward should likely only depend on the left end effector's distance
        current_left_pos = physics.data.mocap_pos[0][:2]  # x, y
        # current_right_pos = physics.data.mocap_pos[1][:2]  # x, y # Remove if right EE isn't relevant

        # Calculate distance to the target position
        left_distance = np.linalg.norm(current_left_pos - self.target_position)
        # right_distance = np.linalg.norm(current_right_pos - self.target_position) # Remove if right EE isn't relevant

        # Reward is based on minimizing the distance to the target
        reward = -left_distance # Only use left distance
        return reward
