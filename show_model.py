import os
import imageio
import gymnasium as gym
import gym_aloha
import numpy as np
import stable_baselines3
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.env_util import make_vec_env
import torch
import glob

try:
    model_path = "models/sac_Speed5.zip"
    env = gym.make("gym_aloha/AlohaSimple")
    video_length = 300 # number of frames in the video

    print('------- trying to load Model -------')
    try:
        model = stable_baselines3.SAC.load(model_path, env=env, verbose=1)
        print('------- successfully loaded Model -------')
        model.device="cuda" if torch.cuda.is_available() else "cpu"
    except:
        print('------- can not loaded Model -------')
        load_saved_model = False

    
        #---------------Animation----------------
    frames = []
    observation, info = env.reset()
    # loop for acting
    for i in range(video_length):
        # get model predicted action
        action, _states = model.predict(observation, deterministic=True)
        #physics = env.unwrapped._physics
        #print("qpos:", physics.data.qpos)
        action[6:] = 0
        #print("action:", action[:6])
        observation, reward, terminated, truncated, info = env.step(action)
        #print(reward)
        image = env.render()
        frames.append(image)

        if terminated or truncated:
            observation, info = env.reset()
    
    a = model._total_timesteps
    scalar = 0
    list_scalar = ["", "K", "M", "B", "T", "P", "E", "Z", "Y"]
    while a > 1000:
        scalar += 1
        a = round(a/1000)
    scalar_Symbol = list_scalar[scalar]
        
    filename = "videos/example" + str(a)+ str(scalar_Symbol)+".mp4"
    imageio.mimsave(filename, np.stack(frames), fps=25)

    env.close()

finally:
    print("Weeeerbung Eeeende")
