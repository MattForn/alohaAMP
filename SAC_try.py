import gym_aloha.constants
import imageio
import gymnasium as gym
import numpy as np
import gym_aloha
import stable_baselines3
from stable_baselines3.common.logger import configure
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from torchvision.transforms import Compose, Normalize, ToTensor, Resize
import torch
import torch.nn.functional as F
from torchvision.models import convnext_base
from dm_control.rl.control import PhysicsError
import os
import glob
import pprint

# Custom policy
from stable_baselines3.sac.policies import SACPolicy
from gymnasium import spaces
from stable_baselines3.common.type_aliases import Schedule
from typing import Any, Dict, List, Optional, Type, Union


env = gym.make("gym_aloha/AlohaInsertion-features-v0")
#env = make_vec_env("gym_aloha/AlohaInsertion-features-v0", n_envs=4)

#observation, info = env.reset()

learning_rate = 0.0003
tau = 0.005
gamma = 0.99
batch_size = 100
verbose = 1
target_entropy = env.action_space.shape[0]
buffer_size = 2**9# 2**21
create_new_model = True
load_saved_model = False
load_saved_replay_buffer = False
save_freq = 10000
total_timesteps = 1600
save_replay_buffer=False
del_old_checkpoints=False
make_video_after_learning = False
video_length = 100 # number of frames in the video

#TODO: the model cant use the saved Buffer if more than one env's are used

#load the last saved model in models with the graetest amounts of steps
if load_saved_model and not create_new_model:
    try:
        #find the last saved model
        directory = "models/*"
        list_of_files = glob.glob(directory)
        list_of_files_zip = [file for file in list_of_files if file.endswith('.zip')]
        list_of_files_pkl = [file for file in list_of_files if file.endswith('.pkl')]
        latest_zip_file = max(list_of_files_zip, key=os.path.getctime)
        latest_pkl_file = max(list_of_files_pkl, key=os.path.getctime)
        latest_zip_file = max(list_of_files_zip, key=os.path.getctime)
        print(f"trying to load a model: {latest_zip_file}")
        model = stable_baselines3.SAC.load(latest_zip_file, env=env, verbose=1)
        print('------- successfully loaded Model -------')
            
        #chainge batch size of the model
        model.batch_size = batch_size
        model.verbose = verbose
        model.device="cuda" if torch.cuda.is_available() else "cpu"
    except:
        print('------- can not loaded Model -------')
        load_saved_model = False
        
if not load_saved_model or create_new_model:
        """    The ``net_arch`` parameter allows to specify the amount and size of the hidden layers.
        It can be in either of the following forms:
        1. ``dict(vf=[<list of layer sizes>], pi=[<list of layer sizes>])``: to specify the amount and size of the layers in the
            policy and value nets individually. If it is missing any of the keys (pi or vf),
            zero layers will be considered for that key.
        2. ``[<list of layer sizes>]``: "shortcut" in case the amount and size of the layers
            in the policy and value nets are the same. Same as ``dict(vf=int_list, pi=int_list)``
            where int_list is the same for the actor and critic.
        Usage:
         policy_kwargs = dict(net_arch=dict(pi=[256, 256], vf=[256, 256])"""
         
        print('------- creating new Model -------')
        model = stable_baselines3.SAC("MultiInputPolicy", 
                                        env, 
                                        verbose=verbose, 
                                        learning_rate=learning_rate,
                                        buffer_size=buffer_size,
                                        batch_size=batch_size,
                                        device="cuda",
                                        tau=tau,
                                        gamma=gamma,
                                        target_entropy=target_entropy)
        
        new_logger = configure("models/logs", ["stdout", "csv", "tensorboard"])
        model.set_logger(new_logger)

if load_saved_replay_buffer:
    try:
        # Load the replay buffer
        if os.path.exists(latest_pkl_file):
            print(f"trying to load replay buffer: {latest_pkl_file}")
            with open(latest_pkl_file, 'rb') as f:
                model.load_replay_buffer(f)
            print('------- successfully loaded Replay Buffer -------')
        else:
            print('------- Replay Buffer not found -------')
    except Exception as e:
            print('------- cant load Replay Buffer -------')
            print(e)
    
model.actor

# Custom callback to catch errors during learning
class ErrorCatchingCallback(CheckpointCallback):
    def __init__(self, save_freq: int, save_path: str,
                 name_prefix: str, save_replay_buffer=True,
                 del_old_checkpoints=True):
        super().__init__(save_freq, save_path, 
                         name_prefix, save_replay_buffer, 
                         del_old_checkpoints=del_old_checkpoints)
    
    def _on_step(self) -> bool:
        try:
            a_loss = self.model.logger.name_to_value["train/actor_loss"]
            c_loss = self.model.logger.name_to_value["train/critic_loss"]
            e_c = self.model.logger.name_to_value["train/ent_coef"]
            e_c_loss = self.model.logger.name_to_value["train/ent_coef_loss"]
            l_rate = self.model.logger.name_to_value["train/learning_rate"]
            
            return super()._on_step()
        except Exception as e:
            print("################################")
            if e == PhysicsError:
                print(f"PhysicsError: {e}")
            else:
                print(f"Error during learning: {e}")
            return False

# Use the custom callback
ecc = ErrorCatchingCallback(save_freq=save_freq, save_path='./models/',
                                                name_prefix='sac_ConvNext_aloha',
                                                save_replay_buffer=True,
                                                del_old_checkpoints=False)
print("starting to learn")
model.learn(total_timesteps=total_timesteps, callback=ecc)
# save the model
#model.save("sac_ConvNext_aloha")

#---------------Animation----------------
if make_video_after_learning:
    frames = []
    start_pose = np.asarray(gym_aloha.constants.START_ARM_POSE.copy())
    start_pose = np.delete(start_pose, [8, 15])
    # Move the Grippers closer to each other
    close_pose = start_pose.copy()
    close_pose[1],close_pose[8] = -0.5,-0.5
    close_pose[2],close_pose[9] = 0.9,0.9
    close_pose[6],close_pose[13] = 0.5,0.5

    observation, info = env.reset()
    # loop for acting
    for i in range(100):
        # get model predicted action
        action, _states = model.predict(observation, deterministic=True)
        
        #action = close_pose
        observation, reward, terminated, truncated, info = env.step(action)
        print(np.max(observation["top"]))
        #print(np.shape(observation["top"]))
        #print(type(observation["top"]))
        #print("reward: ", reward)
        image = env.render()
        frames.append(image)

        if terminated or truncated:
            observation, info = env.reset()

    filename = "videos/example" + str(model._total_timesteps) + ".mp4"
    imageio.mimsave(filename, np.stack(frames), fps=25)
env.close()
