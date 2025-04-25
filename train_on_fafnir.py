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
from stable_baselines3.common.logger import configure
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
from typing import Any, Dict, List, Optional, Type, Union, Tuple

# Parameters
batch_size = 256
verbose = 0
buffer_size = 2**22
learning_rate = 0.0002
tau = 0.005
gamma = 0.99
#target_entropy = -14
ent_coef = "auto" #0.2
load_saved_model = False
load_saved_replay_buffer = False
total_learning_timesteps = 50000
gradient_steps = 9
train_freq = (3, "step")
save_freq = 1000
save_replay_buffer=False
del_old_checkpoints=True

# Initialize W&B project
wandb.init(
    project="simple_aloha_env_ee",
    config={
        "algorithm": "SAC",
        "env": "SimpleAlohaEndEffector",
        "batch_size": batch_size,
        "buffer_size": buffer_size,
        "learning_timesteps": total_learning_timesteps,
    }
)

try:    
    env = gym.make("gym_aloha/SimpleAlohaEndEffector")
    #env = make_vec_env("gym_aloha/AlohaSimple", n_envs=4)

    #observation, info = env.reset()
    
    #TODO: the model cant use the saved Buffer if more than one env's are used

    #load the last saved model in models with the graetest amounts of steps
    if load_saved_model:
        try:
            #find the last saved model
            directory = "/media/local/fornepaetz/models/*"
            list_of_files = glob.glob(directory)
            list_of_files_zip = [file for file in list_of_files if file.endswith('.zip')]
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
            
        # Berechne Target Entropy
        action_space_dim = np.prod(env.action_space.shape)
        target_entropy = -action_space_dim
    
        model = stable_baselines3.SAC("MultiInputPolicy",
                                    env, 
                                    verbose=verbose, 
                                    buffer_size=buffer_size,
                                    batch_size=batch_size,
                                    device="cuda",
                                    ent_coef=ent_coef,
                                    gradient_steps=gradient_steps,
                                    train_freq=train_freq,
                                    learning_rate=learning_rate,
                                    tau=tau,
                                    gamma=gamma,
                                    target_entropy=target_entropy)
        
        
        # Setze die gewünschte Target Entropy
        model.target_entropy = float(target_entropy)

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

    new_logger = configure("/media/local/fornepaetz/models/logs", ["stdout", "csv", "tensorboard"])
    model.set_logger(new_logger)

    class WandbCallback(BaseCallback):
        def __init__(self, verbose=0):
            super(WandbCallback, self).__init__(verbose)

        def _on_step(self) -> bool:

            # Log training metrics to W&B
            wandb.log({
                "reward": self.locals["rewards"].mean(),
                "actor loss": self.model.logger.name_to_value["train/actor_loss"],
                "critic loss": self.model.logger.name_to_value["train/critic_loss"],
                "ent_coef": self.model.logger.name_to_value["train/ent_coef"],
                "ent_coef_loss": self.model.logger.name_to_value["train/ent_coef_loss"],
                "learning_rate": self.model.logger.name_to_value["train/learning_rate"],
            })
            return True
    
    # Custom callback to catch errors during learning
    class ErrorCatching_Wandb_Callback(CheckpointCallback):
        def __init__(self, save_freq: int, save_path: str,
                    name_prefix: str, save_replay_buffer=save_replay_buffer,
                    del_old_checkpoints=del_old_checkpoints):
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
    ecc = ErrorCatching_Wandb_Callback(save_freq=save_freq, save_path='/media/local/fornepaetz/models/',
                                                    name_prefix='sac_aloha_simple',
                                                    save_replay_buffer=save_replay_buffer,
                                                    del_old_checkpoints=del_old_checkpoints)
    wandb_callback = WandbCallback(verbose=1)
    
    print("starting to learn")
    model.learn(total_timesteps=total_learning_timesteps, 
                callback=[ecc, wandb_callback])
    # Save the model and log it to W&B as an artifact
    model_path = "/media/local/fornepaetz/models/sac_aloha_simple.zip"
    model.save(model_path)

    artifact = wandb.Artifact('trained-model', type='model')
    artifact.add_file(model_path)
    wandb.log_artifact(artifact)

    # End the W&B run at the end of the training
    wandb.finish()
    env.close()

finally:
    try:
        model_path = "/media/local/fornepaetz/models/sac_aloha_simple.zip"
        model.save(model_path)
        print('------- successfully saved Model -------')
        print("model saved to: ", model_path)
        artifact = wandb.Artifact('trained-model', type='model')
        artifact.add_file(model_path)
        wandb.log_artifact(artifact)
        wandb.finish()
        env.close()
   
    except:
        pass

    print("Ich stoppe nun. Ich wuensche Ihnen noch einen schoenendv<senv<!")
    vdisplay.stop()



'''
Epsiodische Tasks

Demos in Replaybuffer

(Live Corrections)
(On Policy Learning probieren)

Das Wackeln ist typisches verhalten für wenn das damping nicht richtig eingestellt ist 
und/oder Fehler in der Jakobimatrix anstehen
Mujoco müsste die Jakobimatrix aber richtig berechnen und zur Verfügung stellen
'''