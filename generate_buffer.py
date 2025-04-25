import pickle
import imageio
import numpy as np
from stable_baselines3.common.buffers import ReplayBuffer, DictReplayBuffer
import gymnasium as gym # Or import gym if using older version
import gym_aloha
from gym_aloha.constants import DT
import torch

speed = 2.0 # speed of the agent

#init the env
env = gym.make("gym_aloha/AlohaSimple")
obs, info = env.reset()

env.use_speed_limit = False
env.speed_limit = speed
env.action_is_vel_not_pos = True
observation_space = env.observation_space
action_space = env.action_space

#ToDo: make shure thet the actions taken are clipped what the action space is

# init before filling with actual data from generation
N = 1000 # Number of demonstration steps
obs_shape = np.shape(obs["agent_pos"]) # Example observation shape
action_dim = env.action_space.shape[0]
all_obs = [] # Use lists to store dictionary observations
all_next_obs = []
all_actions = np.zeros((N, action_dim), dtype=np.float32)
all_rewards = np.zeros(N).astype(np.float32)
all_dones = np.zeros(N, dtype=np.int8) # 0 or 1
all_timeouts = np.zeros(N, dtype=np.int8) # Assume no timeouts for simplicity here
# Make sure the last step's done/timeout is handled correctly if it ends an episode

# Date generation in env
frames = []

obs, info = env.reset()
for i in range(N):
    # generate action towars position = 0
    goal_pos = obs["agent_pos"]*0
    current_pos = np.copy(obs["agent_pos"])
    delta = goal_pos - current_pos
    action = (goal_pos -current_pos)/DT
    # limit all actions to be in the range of -speed to speed
    action = np.clip(action, -speed, speed)
    action[6:] = 0 #move only left arm
    next_obs, reward, terminated, truncated, info = env.step(action)
    
    #append
    all_obs.append(obs)
    all_next_obs.append(next_obs)
    all_actions[i] = action
    all_rewards[i] = reward
    all_dones[i] = terminated
    all_timeouts[i] = truncated
    image = env.render()
    frames.append(image)
    
    if terminated or truncated:
        next_obs, info = env.reset()
        
    obs = next_obs

filename = "videos/generated_.mp4"
imageio.mimsave(filename, np.stack(frames), fps=25)


# --- Post-processing ---
# At this point, len(all_obs) might be N+1 if the loop didn't end on a reset.
# We need len(all_obs) == len(all_next_obs) == N
# Remove the last observation if it doesn't have a corresponding next_observation
if len(all_obs) > N:
    all_obs = all_obs[:N]
elif len(all_obs) < N:
     # This case might happen if the loop ended exactly after a reset
     # Need to ensure all arrays match the actual number of transitions recorded
     actual_N = len(all_obs)
     print(f"Adjusting N to actual transitions recorded: {actual_N}")
     all_actions = all_actions[:actual_N]
     all_rewards = all_rewards[:actual_N]
     all_dones = all_dones[:actual_N]
     all_timeouts = all_timeouts[:actual_N]
     # all_next_obs should already be the correct length
     
# tur all_obs and all_next_obs into dicts
for i in range(len(all_obs)):
    all_obs[i] = {
        "agent_pos": all_obs[i]["agent_pos"]
    }
    all_next_obs[i] = {
        "agent_pos": all_next_obs[i]["agent_pos"]
    }


print(f"Data generation complete. Number of transitions: {len(all_obs)}")

# --- Now you can proceed to create and save the DictReplayBuffer ---
# (Code from the previous answer, using DictReplayBuffer)

# Example: Define Buffer Parameters
buffer_size = N # Or larger if you plan to add more data later
n_envs = 1
device = "cuda" if torch.cuda.is_available() else "cpu"
optimize_memory_usage = False # Required False for DictReplayBuffer
handle_timeout_termination = True

# Create DictReplayBuffer
replay_buffer = DictReplayBuffer(
    buffer_size,
    observation_space,
    action_space,
    device=device,
    n_envs=n_envs,
    optimize_memory_usage=optimize_memory_usage,
    handle_timeout_termination=handle_timeout_termination,
)

# Populate the buffer
print("Populating DictReplayBuffer...")
actual_N = len(all_obs) # Use the actual number of transitions generated
for i in range(actual_N):
    infos = [{"TimeLimit.truncated": bool(all_timeouts[i])}]
    replay_buffer.add(
        all_obs[i],         # obs is a dict
        all_next_obs[i],    # next_obs is a dict
        all_actions[i],
        all_rewards[i],
        [all_dones[i]],     # done needs to be list/array
        infos
    )

print(f"Buffer populated. Size: {replay_buffer.size()}")

# Save the buffer
buffer_save_path = "models/buffers/generated_buffer.pkl"
print(f"Saving buffer to {buffer_save_path}...")
try:
    with open(buffer_save_path, "wb") as f:
        pickle.dump(replay_buffer, f)
    print("Buffer saved successfully.")
except Exception as e:
    print(f"Error saving buffer: {e}")

env.close()