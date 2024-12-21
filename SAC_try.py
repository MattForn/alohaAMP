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

# Custom policy
from stable_baselines3.sac.policies import SACPolicy
from gymnasium import spaces
from stable_baselines3.common.type_aliases import Schedule
from typing import Any, Dict, List, Optional, Type, Union


env = gym.make("gym_aloha/AlohaInsertion-features-v0")

observation, info = env.reset()
print(type(observation["top"]))
print(np.shape(observation["top"]))
frames = []
start_pose = np.asarray(gym_aloha.constants.START_ARM_POSE.copy())
start_pose = np.delete(start_pose, [8, 15])

# Move the Grippers closer to each other
close_pose = start_pose.copy()
close_pose[1],close_pose[8] = -0.5,-0.5
close_pose[2],close_pose[9] = 0.9,0.9
close_pose[6],close_pose[13] = 0.5,0.5


model = stable_baselines3.SAC("MultiInputPolicy", env, verbose=1, buffer_size=1024, batch_size=64)

model.learn(total_timesteps=3600, callback=CheckpointCallback(save_freq=1000, save_path='./models/', name_prefix='sac_ConvNext_aloha'))

# save the model
model.save("sac_ConvNext_aloha")

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
filename = "example" + str(model._total_timesteps) + ".mp4"
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