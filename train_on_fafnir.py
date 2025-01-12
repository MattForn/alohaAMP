import os
from xvfbwrapper import Xvfb

# Start Xvfb
vdisplay = Xvfb()
vdisplay.start()

# Set the DISPLAY environment variable
os.environ['DISPLAY'] = ':{}'.format(vdisplay.new_display)

# Set up EGL for headless rendering
os.environ['MUJOCO_GL'] = 'egl'

import wandb
from stable_baselines3.common.logger import HParam
import gym_aloha.constants
import imageio
import gymnasium as gym
import numpy as np
import gym_aloha
import stable_baselines3
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from torchvision.transforms import Compose, Normalize, ToTensor, Resize
import torch
import torch.nn.functional as F
from torchvision.models import convnext_base
from dm_control.rl.control import PhysicsError
import glob

# Custom policy
from stable_baselines3.sac.policies import SACPolicy
from gymnasium import spaces
from stable_baselines3.common.type_aliases import Schedule
from typing import Any, Dict, List, Optional, Type, Union

# Parameters
batch_size = 256
verbose = 0
buffer_size = 2**22
load_saved_model = True
load_saved_replay_buffer = False
total_learning_timesteps = 10000000
save_freq = 10000
make_video_after_learning = False
video_length = 100 # number of frames in the video


# Initialize W&B project
wandb.init(
    project="aloha-insertion",  # Replace with your project name
    config={
        "algorithm": "SAC",
        "env": "AlohaInsertion-features-v0",
        "batch_size": batch_size,
        "buffer_size": buffer_size,
        "learning_timesteps": total_learning_timesteps,
    }
)

try:    
    #env = gym.make("gym_aloha/AlohaInsertion-features-v0")
    env = make_vec_env("gym_aloha/AlohaInsertion-features-v0", n_envs=4)

    #observation, info = env.reset()
    
    
    #TODO: the model cant use the saved Buffer if more than one env's are used

    #load the last saved model in models with the graetest amounts of steps
    if load_saved_model:
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

    if not load_saved_model:
            print('------- creating new Model -------')
            model = stable_baselines3.SAC("MultiInputPolicy", 
                                        env, 
                                        verbose=verbose, 
                                        buffer_size=buffer_size,
                                        batch_size=batch_size,
                                        device="cuda")

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

    class WandbCallback(BaseCallback):
        def __init__(self, verbose=0):
            super(WandbCallback, self).__init__(verbose)

        def _on_step(self) -> bool:
            # Log training metrics to W&B
            wandb.log({
                "step": self.num_timesteps,
                "reward": self.locals["rewards"].mean(),
                #"episode_length": self.locals["episode_lengths"].mean(),
                "loss": self.locals.get("loss", 0),
            })
            return True
    
    # Custom callback to catch errors during learning
    class ErrorCatching_Wandb_Callback(CheckpointCallback):
        def __init__(self, save_freq: int, save_path: str,
                    name_prefix: str, save_replay_buffer=True,
                    del_old_checkpoints=True):
            super().__init__(save_freq, save_path, 
                            name_prefix, save_replay_buffer, 
                            del_old_checkpoints=del_old_checkpoints)

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
    ecc = ErrorCatching_Wandb_Callback(save_freq=save_freq, save_path='./models/',
                                                    name_prefix='sac_ConvNext_aloha',
                                                    save_replay_buffer=True,
                                                    del_old_checkpoints=False)
    wandb_callback = WandbCallback(verbose=1)
    
    print("starting to learn")
    model.learn(total_timesteps=total_learning_timesteps, 
                callback=[ecc, wandb_callback])
    # Save the model and log it to W&B as an artifact
    model_path = "model/sac_ConvNext_aloha.zip"
    model.save(model_path)

    artifact = wandb.Artifact('trained-model', type='model')
    artifact.add_file(model_path)
    wandb.log_artifact(artifact)

    # End the W&B run at the end of the training
    wandb.finish()
    
    if make_video_after_learning:
        #---------------Animation----------------
        frames = []
        observation, info = env.reset()
        # loop for acting
        for i in range(video_length):
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

finally:
    # Stop Xvfb
    print("stopping Xvfb_____BECHAUSE YOU STOPPED MEEEEEEEE!")
    vdisplay.stop()
