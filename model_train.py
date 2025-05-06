import os
import time
#from xvfbwrapper import Xvfbp

# Start Xvfb
#vdisplay = Xvfb()
#vdisplay.start()

# Set the DISPLAY environment variable
#os.environ['DISPLAY'] = ':{}'.format(vdisplay.new_display)

# Set up EGL for headless rendering
#os.environ['MUJOCO_GL'] = 'egl'
import pickle
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
from stable_baselines3.common.logger import HParam
from stable_baselines3.common.utils import configure_logger
from stable_baselines3.common.buffers import ReplayBuffer, DictReplayBuffer # Import buffer classes
#from offline_trainer import train_offline
#from combined_trainer import train_combined
from combined_buffer import DictCombinedBuffer

# Parameters
batch_size = 256
learning_starts=1000#batch_size*2
verbose = 0
learning_rate = 0.002 #0.0006
tau = 0.005
gamma = 0.99
calc_tatget_entropy_from_action_space = False #overrights the followiung
target_entropy = "auto" #"auto" #-14
ent_coef ='auto'
buffer_size = 2**20
gradient_steps = 1
train_freq: Union[int, Tuple[int, str]] = (1, "step")
model_path_save = "models/sac_sparse.zip"
model_path_load = "models/sac_sparse.zip"
load_saved_model = False
load_generated_replay_buffer = True
total_learning_timesteps = 10**9
save_freq = 10000
save_replay_buffer=True
make_video_after_learning = True
video_length = 500 # number of frames in the video
# --- Demo Buffer Setup ---
demo_buffer_path = "models/buffers/generated_buffer.pkl" # Path to your expert demo buffer
demo_replay_buffer = None



# Initialize W&B project
wandb.init(
    project="aloha-simple-sparse-50_50_Buffer",  # Replace with your project name
    config={
        "algorithm": "SAC",
        "env": "SimpleAloha",
        "batch_size": batch_size,
        "buffer_size": buffer_size,
        "learning_timesteps": total_learning_timesteps,
        "gradient_steps_per_env_step": gradient_steps,
        "learning_rate": learning_rate,
    }
)

try:    
    env = gym.make("gym_aloha/AlohaSimple")
    env.reset()
    #env = make_vec_env("gym_aloha/AlohaSimple", n_envs=4)

    if calc_tatget_entropy_from_action_space:
        target_entropy = -env.action_space.shape[0]
        print(f"setting target_entropy: {target_entropy}")
    
    #load the last saved model in models with the graetest amounts of steps
    if load_saved_model:
        try:
            model = stable_baselines3.SAC.load(model_path_load, env=env, device="cuda")
            print(f'------- successfully loaded Model: {model_path_load} -------')
            # Ensure parameters match current config if needed
            model.batch_size = batch_size # SB3 uses this internally, but our function overrides
            model.learning_rate = learning_rate
            model.learning_starts = learning_starts
            model.gradient_steps = gradient_steps
            model.train_freq = train_freq
            # Re-setup schedule if LR is a schedule
            # model._setup_lr_schedule()
        except Exception as e:
            print(f'------- Could not load Model ({e}), creating new one -------')
            load_saved_model = False
    if not load_saved_model:
        print('------- creating new Model -------')
            
        model = stable_baselines3.SAC("MultiInputPolicy",
                                    env,
                                    verbose=verbose,
                                    buffer_size=buffer_size,
                                    learning_starts=learning_starts,
                                    batch_size=batch_size,
                                    train_freq=train_freq,
                                    learning_rate=learning_rate,
                                    gradient_steps=int(gradient_steps),
                                    tau=tau,
                                    gamma=gamma,
                                    target_entropy=target_entropy,
                                    ent_coef=ent_coef,
                                    device="cuda",
                                    )
        
    # --- Load Demonstration Buffer ---
    if load_generated_replay_buffer:
        if os.path.exists(demo_buffer_path):
            print(f"Loading demonstration buffer from: {demo_buffer_path}")
            try:
                with open(demo_buffer_path, "rb") as f:
                    demo_replay_buffer = pickle.load(f)
                # Set device again after loading
                demo_replay_buffer.device = model.device
                print(f'------- Successfully loaded demonstration buffer ({demo_replay_buffer.size()} transitions) -------')
            except Exception as e:
                 ValueError(f"ERROR loading demonstration buffer: {e}")
        else:
            ValueError(f"ERROR: Demonstration replay buffer file not found at {demo_buffer_path}.")
            
        # Create a CombinedReplayBuffer
        replay_buffer_kwargs = model.replay_buffer_kwargs.copy()
        combined_buffer = DictCombinedBuffer(
            model.buffer_size,
            model.observation_space,
            model.action_space,
            device=model.device,
            n_envs=model.n_envs,
            optimize_memory_usage=model.optimize_memory_usage,
            demonstration_buffer=demo_replay_buffer,
            sample_ratio=0.5,
            **replay_buffer_kwargs,
            )
    else:
        print("Skipping demonstration buffer loading.")

    new_logger = configure("models/logs", ["stdout", "csv", "tensorboard"])
    model.set_logger(new_logger)

    class WandbCallback(BaseCallback):
        def __init__(self, verbose=0):
            super(WandbCallback, self).__init__(verbose)

        def _on_step(self) -> bool:
            # Log training metrics to W&B
            wandb.log({
                "reward": self.locals["rewards"].mean(),
                #"episode_length": self.locals["episode_lengths"].mean(),
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
    # --- Final Cleanup & Save ---
    print("Executing finally block...")
    try:
        if 'model' in locals(): # TODO Check if training happened
            model.save(model_path_save)
            print(f"Final model saved to {model_path_save}")
            if wandb.run is not None:
                try:
                    artifact = wandb.Artifact('trained-combined-model-final', type='model')
                    artifact.add_file(model_path_save)
                    wandb.log_artifact(artifact)
                except Exception as e:
                    print(f"Error logging final artifact: {e}")
                wandb.finish()
                print("Wandb finished in finally block.")
        else:
            print("Model not saved in finally block (conditions not met).")
            if wandb.run is not None:
                 wandb.finish()
                 print("Wandb finished in finally block (no successful training).")

    except Exception as e:
        print(f"Error during final save/cleanup: {e}")
        if wandb.run is not None:
             wandb.finish()
             print("Wandb finished after error in finally block.")

    print("Weeeerbung Eeeende")