from simple_1d_gym import SimpleGymEnv
from stable_baselines3 import SAC
import numpy as np 
import matplotlib.pyplot as plt

# Create environment
env = SimpleGymEnv()

# Train SAC model
model = SAC("MlpPolicy", env, verbose=1)
# model = SAC.load("sac_simple_env.zip", env=env, verbose=1)
model.learn(total_timesteps=5000)

states = env.get_past_states()
# print(states)
# plot all states on a map with colors representing the light/dark state
# the coardiantes are [x,y,light or dark]
# the light state is represented by a red dot and the dark state is represented by a blue dot
light = states[:,2] == 1
dark = states[:,2] == 0
plt.scatter(states[:,0][light], states[:,1][light], c='r', label='light')
plt.scatter(states[:,0][dark], states[:,1][dark], c='b', label='dark')
plt.legend()
plt.show()

# Save the trained model
# model.save("sac_simple_env")