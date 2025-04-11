import os
#from xvfbwrapper import Xvfb

# Start Xvfb
#vdisplay = Xvfb()
#vdisplay.start()

# Set the DISPLAY environment variable
#os.environ['DISPLAY'] = ':{}'.format(vdisplay.new_display)

# Set up EGL for headless rendering
#os.environ['MUJOCO_GL'] = 'egl'

import wandb
from stable_baselines3.common.logger import HParam
import gym_aloha.constants
import imageio
import gymnasium as gym
import numpy as np
import gym_aloha
import stable_baselines3
from stable_baselines3.common.logger import configure
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from torchvision.transforms import Compose, Normalize, ToTensor, Resize
from typing import Tuple
import torch
import torch.nn.functional as F
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
learning_rate = 0.0006
tau = 0.005
gamma = 0.99
target_entropy = -14
ent_coef = 0.2
buffer_size = 2**14
gradient_steps = 1
train_freq: Union[int, Tuple[int, str]] = (1, "step")
model_path_save = "models/sac_1devX3.zip"
model_path_load = "models/sac_1devX3.zip"
load_saved_model = False
load_saved_replay_buffer = False
total_learning_timesteps = 100000000
save_freq = 10000
save_replay_buffer=True
make_video_after_learning = True
video_length = 100 # number of frames in the video



# Initialize W&B project
wandb.init(
    project="aloha-simple-1devX",  # Replace with your project name
    config={
        "algorithm": "SAC",
        "env": "SimpleAloha",
        "batch_size": batch_size,
        "buffer_size": buffer_size,
        "learning_timesteps": total_learning_timesteps,
    }
)

try:    
    env = gym.make("gym_aloha/AlohaSimple")
    #env = make_vec_env("gym_aloha/AlohaSimple", n_envs=4)

    target_entropy = -env.action_space.shape[0]
    print(f"setting target_entropy: {target_entropy}")
    
    #load the last saved model in models with the graetest amounts of steps
    if load_saved_model:
        try:
            model = stable_baselines3.SAC.load(model_path_load, env=env, verbose=1)
            print('------- successfully loaded Model -------')
            model.batch_size = batch_size
            model.learning_rate = learning_rate
            model.device="cuda" if torch.cuda.is_available() else "cpu"
        except:
            print('------- can not loaded Model -------')
            load_saved_model = False

    if not load_saved_model:
        print('------- creating new Model -------')
            
        # Berechne Target Entropy
        action_space_dim = np.prod(env.action_space.shape)
        target_entropy = -action_space_dim
    
        model = stable_baselines3.SAC("MultiInputPolicy",
                                    env,
                                    verbose=verbose,
                                    buffer_size=buffer_size,
                                    batch_size=batch_size,
                                    train_freq=train_freq,
                                    learning_rate=learning_rate,
                                    gradient_steps=int(gradient_steps),
                                    tau=tau,
                                    gamma=gamma,
                                    target_entropy=target_entropy,
                                    ent_coef='auto',
                                    device="cuda",
                                    )
        
        
        # Setze die gewünschte Target Entropy
        model.target_entropy = target_entropy

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

    new_logger = configure("models/logs", ["stdout", "csv", "tensorboard"])
    model.set_logger(new_logger)

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
                "actor loss": self.model.logger.name_to_value["train/actor_loss"],
                "critic loss": self.model.logger.name_to_value["train/critic_loss"],
                "ent_coef": self.model.logger.name_to_value["train/ent_coef"],
                "ent_coef_loss": self.model.logger.name_to_value["train/ent_coef_loss"],
                "learning_rate": self.model.logger.name_to_value["train/learning_rate"],
                "q_values": self.model.logger.name_to_value["train/q_values"],
            })
            return True
    
    wandb_callback = WandbCallback(verbose=1)
    
    print("starting to learn")
    model.learn(total_timesteps=total_learning_timesteps, 
                callback=[wandb_callback])
                
    # Save the model and log it to W&B as an artifact
    model.save(model_path_save)

    artifact = wandb.Artifact('trained-model', type='model')
    artifact.add_file(model_path_save)
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
            observation, reward, terminated, truncated, info = env.step(action)
            print(np.max(observation["top"]))
            image = env.render()
            frames.append(image)

            if terminated or truncated:
                observation, info = env.reset()
        filename = "videos/example" + str(model._total_timesteps) + ".mp4"
        imageio.mimsave(filename, np.stack(frames), fps=25)

    env.close()

finally:
    try:
        model.save(model_path_save)
        artifact = wandb.Artifact('trained-model', type='model')
        artifact.add_file(model_path_save)
        wandb.log_artifact(artifact)

        # End the W&B run at the end of the training
        wandb.finish()
        print("------- saved Model -------")
    except:
        print("------- can not save Model -------")
        pass
    print("Weeeerbung Eeeende")
