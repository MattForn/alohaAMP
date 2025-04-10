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
    model_path = "models/sac_aloha_Simple2.zip"
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
        action[6:] = 0
        observation, reward, terminated, truncated, info = env.step(action)
        image = env.render()
        frames.append(image)

        if terminated or truncated:
            observation, info = env.reset()
    filename = "videos/example" + str(model._total_timesteps) + ".mp4"
    imageio.mimsave(filename, np.stack(frames), fps=25)

    env.close()

finally:
    print("Weeeerbung Eeeende")
