import gymnasium as gym
import numpy as np
from dm_control import mujoco
from dm_control.rl import control
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch
import torch.nn.functional as F
from torchvision.models import convnext_base

from gym_aloha.constants import (
    ACTIONS,
    ASSETS_DIR,
    DT,
    JOINTS,
    START_ARM_POSE,
    START_ARM_POSE_SIMPLE,
)
from gym_aloha.tasks.sim import BOX_POSE, InsertionTask, TransferCubeTask, SimpleTask
from gym_aloha.tasks.sim_end_effector import (
    InsertionEndEffectorTask,
    TransferCubeEndEffectorTask,
)
from gym_aloha.utils import sample_box_pose, sample_insertion_pose

class AlohaEnv(gym.Env):
    # TODO(aliberts): add "human" render_mode
    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(
        self,
        task,
        obs_type="agent_pos", #"pixels", #"pixels_agent_pos",
        render_mode="rgb_array",
        observation_width=640,
        observation_height=480,
        visualization_width=640,
        visualization_height=480,
        #feature_extractor = ConvNeXtFeatureExtractor(observation_space=None),

    ):
        super().__init__()
        print("running AlohaEnv init")
        self.task = task
        self.obs_type = obs_type
        self.render_mode = render_mode
        self.observation_width = observation_width
        self.observation_height = observation_height
        self.visualization_width = visualization_width
        self.visualization_height = visualization_height
        #self.feature_extractor = feature_extractor.to("cuda")
        
        self._env = self._make_env_task(self.task)
        self.last_action = None
        self.speed_limit = 0.1 # m/s
    
        # fetch max_episode_steps from the environment registry
        self.max_episode_steps = gym.envs.registry["gym_aloha/AlohaSimple"].max_episode_steps
        
        if self.obs_type == "pixels_agent_pos":
            self.observation_space = spaces.Dict(
                {
                    "pixels": spaces.Dict(
                        {
                            "top": spaces.Box(
                                low=0,
                                high=255,
                                shape=(self.observation_height, self.observation_width, 3),
                                dtype=np.uint8,
                            )
                        }
                    ),
                    "agent_posi": spaces.Box(
                        low=-10.0,
                        high=10.0,
                        shape=(len(JOINTS),),
                        dtype=np.float64,
                    ),
                }
            )
        elif self.obs_type == "agent_pos":
            self.observation_space = spaces.Dict(
                {
                    "agent_pos": spaces.Box(
                        low=-10.0,
                        high=10.0,
                        shape=(len(JOINTS),),
                        dtype=np.float64,
                    ),
                }
            )

        self.action_space = spaces.Box(low=-0.15, high=0.15, shape=(len(ACTIONS),), dtype=np.float32)

    def render(self):
        return self._render(visualize=True)

    def _render(self, visualize=False):
        assert self.render_mode == "rgb_array"
        width, height = (
            (self.visualization_width, self.visualization_height)
            if visualize
            else (self.observation_width, self.observation_height)
        )
        # if mode in ["visualize", "human"]:
        #     height, width = self.visualize_height, self.visualize_width
        # elif mode == "rgb_array":
        #     height, width = self.observation_height, self.observation_width
        # else:
        #     raise ValueError(mode)
        # TODO(rcadene): render and visualizer several cameras (e.g. angle, front_close)
        image = self._env.physics.render(height=height, width=width, camera_id="top")
        return image

    def _make_env_task(self, task_name):
        # time limit is controlled by StepCounter in env factory
        time_limit = float("inf")

        if task_name == "transfer_cube":
            xml_path = ASSETS_DIR / "bimanual_viperx_transfer_cube.xml"
            physics = mujoco.Physics.from_xml_path(str(xml_path))
            task = TransferCubeTask()
        elif task_name == "simple":
            xml_path = ASSETS_DIR / "bimanual_viperx_transfer_cube.xml"
            physics = mujoco.Physics.from_xml_path(str(xml_path))
            task = SimpleTask()
        elif task_name == "insertion":
            xml_path = ASSETS_DIR / "bimanual_viperx_insertion.xml"
            physics = mujoco.Physics.from_xml_path(str(xml_path))
            task = InsertionTask()
        elif task_name == "end_effector_transfer_cube":
            raise NotImplementedError()
            xml_path = ASSETS_DIR / "bimanual_viperx_end_effector_transfer_cube.xml"
            physics = mujoco.Physics.from_xml_path(str(xml_path))
            task = TransferCubeEndEffectorTask()
        elif task_name == "end_effector_insertion":
            raise NotImplementedError()
            xml_path = ASSETS_DIR / "bimanual_viperx_end_effector_insertion.xml"
            physics = mujoco.Physics.from_xml_path(str(xml_path))
            task = InsertionEndEffectorTask()
        else:
            raise NotImplementedError(task_name)

        env = control.Environment(
            physics, task, time_limit, control_timestep=DT, n_sub_steps=None, flat_observation=False
        )
        return env

    def _format_raw_obs(self, raw_obs):
        if self.obs_type == "state":
            raise NotImplementedError()
        elif self.obs_type == "pixels":
            obs = {"top": raw_obs["images"]["top"].copy()}
        elif self.obs_type == "pixels_agent_pos":
            obs = {
                "pixels": {"top": raw_obs["images"]["top"].copy()},
                "agent_pos": raw_obs["qpos"],
            }
        elif self.obs_type == "agent_pos":
            obs = {"agent_pos": raw_obs["qpos"]}
        elif self.obs_type == "features":
            obs = {"top": raw_obs["images"]["top"].copy()}["top"]
            #obs = self.feature_extractor.forward(obs)
        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # TODO(rcadene): how to seed the env?
        if seed is not None:
            self._env.task.random.seed(seed)
            self._env.task._random = np.random.RandomState(seed)

        # TODO(rcadene): do not use global variable for this
        if self.task == "transfer_cube":
            BOX_POSE[0] = sample_box_pose(seed)  # used in sim reset
        elif self.task == "insertion":
            BOX_POSE[0] = np.concatenate(sample_insertion_pose(seed))  # used in sim reset
        elif self.task == "simple":
            BOX_POSE[0] = sample_box_pose(seed)  # used in sim reset
        else:
            raise ValueError(self.task)

        raw_obs = self._env.reset()

        observation = self._format_raw_obs(raw_obs.observation)
        

        info = {"is_success": False}
        return observation, info

    def clip_speed(self, action):
        delta = action - self.last_action if self.last_action is not None else 0
        delta = np.clip(delta, -self.speed_limit, self.speed_limit)
        action = self.last_action + delta if self.last_action is not None else action
        self.last_action = action
        return action
                       
    def tanH_speed(self, action):
        # Calculate delta between current and last action
        delta = action - self.last_action if self.last_action is not None else 0
        # Scale delta using tanh to limit its range within [-speed_limit, speed_limit]
        scaled_delta = np.tanh(delta / self.speed_limit) * self.speed_limit
        # Update action by adding the scaled delta
        action = self.last_action + scaled_delta if self.last_action is not None else action
        # Store the last action
        self.last_action = action
        return action
        
    def step(self, action):
        assert action.ndim == 1
        # TODO(rcadene): add info["is_success"] and info["success"] ?
        
        # speed limit if wanted
        # action = self.clip_speed(action)
        action = self.tanH_speed(action)
        _, reward, _, raw_obs = self._env.step(action)

        # TODO(rcadene): add an enum
        terminated = is_success = reward == 4

        info = {"is_success": is_success}

        observation = self._format_raw_obs(raw_obs)

        # check if episode is truncated after max_episode_steps
        truncated = False
        if self._env._step_count >= self.max_episode_steps:
            truncated = True
            
        return observation, reward, terminated, truncated, info

    def close(self):
        pass

class SimpleAlohaEnv(gym.Env):
    # TODO(aliberts): add "human" render_mode
    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(
        self,
        task,
        obs_type="agent_pos", #"pixels_agent_pos",
        render_mode="rgb_array",
        observation_width=640,
        observation_height=480,
        visualization_width=640,
        visualization_height=480,

    ):
        super().__init__()
        print("running SimpleAlohaEnv init")
        self.task = task        
        self.obs_type = obs_type
        self.render_mode = render_mode
        self.observation_width = observation_width
        self.observation_height = observation_height
        self.visualization_width = visualization_width
        self.visualization_height = visualization_height
        
        self._env = self._make_env_task(self.task)
        self.last_action = None
        self.speed_limit = 0.80 # rad/s
        self.max_delta_per_step = self.speed_limit * DT
        self.last_reward = 0
        self.tolleranz   = 0.05
        self.max_reward_since=0
        self.action_is_vel_not_pos = True 
        
        # fetch max_episode_steps from the environment registry
        self.max_episode_steps = gym.envs.registry["gym_aloha/AlohaSimple"].max_episode_steps
        
        if self.obs_type == "pixels_agent_pos":
            self.observation_space = spaces.Dict(
                {
                    "pixels": spaces.Dict(
                        {
                            "top": spaces.Box(
                                low=0,
                                high=255,
                                shape=(self.observation_height, self.observation_width, 3),
                                dtype=np.uint8,
                            )
                        }
                    ),
                    "agent_posi": spaces.Box(
                        low=-np.pi,
                        high=np.pi,
                        shape=(len(JOINTS),),
                        dtype=np.float64,
                    ),
                }
            )
        elif self.obs_type == "agent_pos":
            self.observation_space = spaces.Dict(
                {
                    "agent_pos": spaces.Box(
                        low=-np.pi,
                        high=np.pi,
                        shape=(len(JOINTS),),
                        dtype=np.float64,
                    ),
                }
            )

        self.action_space = spaces.Box(low=-0.9, high=0.9, shape=(len(ACTIONS),), dtype=np.float32)

    def render(self):
        return self._render(visualize=True)

    def _render(self, visualize=False):
        assert self.render_mode == "rgb_array"
        width, height = (
            (self.visualization_width, self.visualization_height)
            if visualize
            else (self.observation_width, self.observation_height)
        )
        image = self._env.physics.render(height=height, width=width, camera_id="angle")
        return image

    def _make_env_task(self, task_name):
        # time limit is controlled by StepCounter in env factory
        time_limit = float("inf")

        if task_name == "simple":
            xml_path = ASSETS_DIR / "bimanual_viperx_simple.xml"
            physics = mujoco.Physics.from_xml_path(str(xml_path))
            task = SimpleTask()
        else:
            raise NotImplementedError(task_name)

        env = control.Environment(
            physics, task, time_limit, control_timestep=DT, n_sub_steps=None, flat_observation=False
        )
        return env

    def _format_raw_obs(self, raw_obs):
        if self.obs_type == "pixels_agent_pos":
            obs = {
                "pixels": {"top": raw_obs["images"]["top"].copy()},
                "agent_pos": raw_obs["qpos"],
            }
        elif self.obs_type == "agent_pos":
            obs = {"agent_pos": raw_obs["qpos"]}

        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # TODO(rcadene): how to seed the env?
        if seed is not None:
            self._env.task.random.seed(seed)
            self._env.task._random = np.random.RandomState(seed)

        # TODO(rcadene): do not use global variable for this
        if self.task == "transfer_cube":
            BOX_POSE[0] = sample_box_pose(seed)  # used in sim reset
        elif self.task == "insertion":
            BOX_POSE[0] = np.concatenate(sample_insertion_pose(seed))  # used in sim reset
        elif self.task == "simple":
            pass
        else:
            raise ValueError(self.task)

        raw_obs = self._env.reset()
        observation = self._format_raw_obs(raw_obs.observation)
        info = {"is_success": False}
        return observation, info

    def clip_speed(self, action):
        delta = action - self.last_action if self.last_action is not None else 0
        delta = np.clip(delta, -self.speed_limit, self.speed_limit)
        action = self.last_action + delta if self.last_action is not None else action
        self.last_action = action
        return action
                       
    def tanH_speed(self, action):
        # Get the current position of the robot
        pos = self._env._task.get_qpos(self._env.physics)
        # Calculate delta between proposed action (as a position) and last pos
        delta = action - pos
        # Scale delta using tanh to limit its range within [-speed_limit, speed_limit]
        scaled_delta = np.tanh(delta / self.max_delta_per_step) * self.max_delta_per_step
        # Update action by adding the scaled delta
        action = pos + scaled_delta
        return action
    
    def tanH(self, action):
        action = np.tanh(action/self.speed_limit) * self.speed_limit
        return action
        
    def step(self, action):
        assert action.ndim == 1
        # TODO(rcadene): add info["is_success"] and info["success"] ?
        
        if self.action_is_vel_not_pos:
            pos = self._env._task.get_qpos(self._env.physics)
            action = pos + self.tanH(action)*DT
        else:
            # speed limit if wanted
            # action = self.clip_speed(action)
            action = self.tanH_speed(action)

        # set every action value after position 5 to 0
        action[6:] = 0
        action[7] = np.pi

        _, reward, _, raw_obs = self._env.step(action)
        
        # check if episode is truncated after max_episode_steps
        truncated = False
        if self._env._step_count >= self.max_episode_steps:
            truncated = True
            
        # TODO(rcadene): add an enum
        terminated = is_success = False
        # if reward >= 6-self.tolleranz:
        #     self.max_reward_since+=1
        #     # if self.max_reward_since >= 10:
        #     #     terminated = is_success = True
        # else:
        #     if self.max_reward_since >=1:
        #         reward = -10
        #     self.max_reward_since=0
            
            
            
        # if reward >= 6-self.tolleranz and self.last_reward >= 6-self.tolleranz:
        #     reward = 100*self.max_reward_since
            

        info = {"is_success": is_success}

        observation = self._format_raw_obs(raw_obs)

        # self.last_reward = reward
        return observation, reward, terminated, truncated, info

    def close(self):
        pass

