import gym_aloha.constants
import imageio
import gymnasium as gym
import numpy as np
import gym_aloha
#from spinningup.spinup.algos.pytorch.sac import sac
observation, info = env.reset()
frames = []

start_pose = np.asarray(gym_aloha.constants.START_ARM_POSE.copy())
start_pose = np.delete(start_pose, [8, 15])

# Move the Grippers closer to each other
close_pose = start_pose.copy()
close_pose[1],close_pose[8] = -0.5,-0.5
close_pose[2],close_pose[9] = 0.9,0.9
close_pose[6],close_pose[13] = 0.5,0.5

# initiate a sac network to learn the task from spining up


# Bring grippers into position
for i in range(10):
    # interpolate between start_pose and close_pose
    action = start_pose + (close_pose - start_pose) * i / 10
    observation, reward, terminated, truncated, info = env.step(action)
    image = env.render()
    frames.append(image)

    if terminated or truncated:
        observation, info = env.reset()

# loop for learning
for i in range(100):
    action = close_pose
    observation, reward, terminated, truncated, info = env.step(action)
    image = env.render()
    frames.append(image)

    if terminated or truncated:
        observation, info = env.reset()

env.close()
imageio.mimsave("sac_test1.mp4", np.stack(frames), fps=25)


logger_kwargs = setup_logger_kwargs('sac_aloha', seed=0)

# Define the environment function
def env_fn():
    return gym.make("gym_aloha/AlohaInsertion-v0")

# Set up SAC training
sac(env_fn=env_fn, 
    actor_critic=core.MLPActorCritic, 
    ac_kwargs=dict(hidden_sizes=[256, 256]), 
    seed=0, 
    steps_per_epoch=4000, 
    epochs=100, 
    replay_size=int(1e6), 
    gamma=0.99, 
    polyak=0.995, 
    lr=1e-3, 
    alpha=0.2, 
    batch_size=100, 
    start_steps=10000, 
    update_after=1000, 
    update_every=50, 
    num_test_episodes=10, 
    max_ep_len=1000, 
    logger_kwargs=logger_kwargs, 
    save_freq=1)