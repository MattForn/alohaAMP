import imageio
import gymnasium as gym
import numpy as np
import gym_aloha
import gym_aloha.constants
import os
from xvfbwrapper import Xvfb

# Start Xvfb
vdisplay = Xvfb()
vdisplay.start()

# Set the DISPLAY environment variable
os.environ['DISPLAY'] = ':{}'.format(vdisplay.new_display)

# Set up EGL for headless rendering
os.environ['MUJOCO_GL'] = 'egl'

try:
    env = gym.make("gym_aloha/AlohaInsertion-v0")
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
    imageio.mimsave("videos/sac_test1.mp4", np.stack(frames), fps=25)

finally:
    # Stop Xvfb
    print("stopping Xvfb")
    vdisplay.stop()
