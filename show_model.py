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
from xvfbwrapper import Xvfb

# Start Xvfb
vdisplay = Xvfb()
vdisplay.start()

# Set the DISPLAY environment variable
os.environ['DISPLAY'] = ':{}'.format(vdisplay.new_display)

# Set up EGL for headless rendering
os.environ['MUJOCO_GL'] = 'egl'

try:
    env = gym.make("gym_aloha/AlohaSimple")
    video_length = 300 # number of frames in the video

    try:
        #find the last saved model
        directory = "/media/local/fornepaetz/models/*"
        list_of_files = glob.glob(directory)
        list_of_files_zip = [file for file in list_of_files if file.endswith('.zip')]
        latest_zip_file = max(list_of_files_zip, key=os.path.getctime)
        print(f"trying to load a model: {latest_zip_file}")
        model = stable_baselines3.SAC.load(latest_zip_file, env=env, verbose=1)
        print('------- successfully loaded Model -------')
        model.device="cuda" if torch.cuda.is_available() else "cpu"
    except:
        print('------- can not loaded Model -------')
    
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
    filename = "/media/local/fornepaetz/videos/AlohaSimple.mp4"
    imageio.mimsave(filename, np.stack(frames), fps=25)
    print(f"Video saved to {filename}")

    env.close()

finally:
    print("Weeeerbung Eeeende")
    vdisplay.stop()
