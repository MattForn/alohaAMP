import gym_aloha.constants
import imageio
import gymnasium as gym
import numpy as np
import gym_aloha
import stable_baselines3
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import CheckpointCallback
from torchvision.transforms import Compose, Normalize, ToTensor, Resize
import torch
import torch.nn.functional as F
from torchvision.models import convnext_base
from dm_control.rl.control import PhysicsError
import os
import glob

# Custom policy
from stable_baselines3.sac.policies import SACPolicy
from gymnasium import spaces
from stable_baselines3.common.type_aliases import Schedule
from typing import Any, Dict, List, Optional, Type, Union


env = gym.make("gym_aloha/AlohaInsertion-features-v0")

observation, info = env.reset()

batch_size = 256
#load the last saved model in models with the graetest amounts of steps
try:
    #find the last saved model
    print('trying to load a model')
    directory = "models/*"
    list_of_files = glob.glob(directory)
    list_of_files_zip = [file for file in list_of_files if file.endswith('.zip')]
    list_of_files_pkl = [file for file in list_of_files if file.endswith('.pkl')]
    latest_zip_file = max(list_of_files_zip, key=os.path.getctime)
    latest_pkl_file = max(list_of_files_pkl, key=os.path.getctime)
    model = stable_baselines3.SAC.load(latest_zip_file, env=env, verbose=1)
    print('------- successfully loaded Model -------')
        
    #chainge batch size of the model
    model.batch_size = batch_size
except:
    print('------- can not loaded Model -------')
    model = stable_baselines3.SAC("MultiInputPolicy", env, verbose=1, buffer_size=2**21, batch_size=batch_size)

try:

    # Load the replay buffer
    if os.path.exists(latest_pkl_file):
        print("trying to load replay buffer")
        with open(latest_pkl_file, 'rb') as f:
            model.load_replay_buffer(f)
        print('------- successfully loaded Replay Buffer -------')
    else:
        print('------- Replay Buffer not found -------')
except Exception as e:
        print('------- cant load Replay Buffer -------')
        print(e)
    
    

# model.learn(total_timesteps=10000, callback=CheckpointCallback(save_freq=1000, save_path='./models/', name_prefix='sac_ConvNext_aloha'))
# Custom callback to catch errors during learning
class ErrorCatchingCallback(CheckpointCallback):
    def __init__(self, save_freq: int, save_path: str, name_prefix: str, save_replay_buffer=True, del_old_checkpoints=True):
        super().__init__(save_freq, save_path, name_prefix, save_replay_buffer, del_old_checkpoints=del_old_checkpoints)

    def _on_step(self) -> bool:
        try:
            return super()._on_step()
        except Exception as e:
            
            print("################################")
            if e == PhysicsError:
                print(f"PhysicsError: {e}")
            else:
                print(f"Error during learning: {e}")
            return False

# Use the custom callback
ecc = ErrorCatchingCallback(save_freq=10000, save_path='./models/',
                                                name_prefix='sac_ConvNext_aloha',
                                                save_replay_buffer=True,
                                                del_old_checkpoints=True)
print("starting to learn")
model.learn(total_timesteps=1000000, callback=ecc)
# save the model
model.save("sac_ConvNext_aloha")


#---------------Animation----------------
frames = []
start_pose = np.asarray(gym_aloha.constants.START_ARM_POSE.copy())
start_pose = np.delete(start_pose, [8, 15])
# Move the Grippers closer to each other
close_pose = start_pose.copy()
close_pose[1],close_pose[8] = -0.5,-0.5
close_pose[2],close_pose[9] = 0.9,0.9
close_pose[6],close_pose[13] = 0.5,0.5

'''
# Bring grippers into position
for i in range(10):
    # interpolate between start_pose and close_pose
    action = start_pose + (close_pose - start_pose) * i / 10
    observation, reward, terminated, truncated, info = env.step(action)
    image = env.render()
    frames.append(image)

    if terminated or truncated:
        observation, info = env.reset()
        
'''
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

env.close()
filename = "videos/example" + str(model._total_timesteps) + ".mp4"
imageio.mimsave(filename, np.stack(frames), fps=25)


'''
# example.py
import imageio
import gymnasium as gym
import numpy as np
import gym_aloha
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch as th
from torchvision import models

env = gym.make("gym_aloha/AlohaInsertion-v0")
observation, info = env.reset()
frames = []

for _ in range(1000):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    image = env.render()
    frames.append(image)

    if terminated or truncated:
        observation, info = env.reset()

env.close()
imageio.mimsave("example.mp4", np.stack(frames), fps=25)
'''