import gym_aloha.constants
import imageio
import gymnasium as gym
import numpy as np
import gym_aloha
import stable_baselines3
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import CheckpointCallback
from torchvision.transforms import Compose, Normalize, ToTensor, Resize
import torch
import torch.nn.functional as F
from torchvision.models import convnext_base

class ConvNeXtFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, output_dim=1024):  # ConvNeXt output size
        super().__init__(observation_space, features_dim=output_dim)
        # Load pretrained ConvNeXt model
        self.convnext = convnext_base(pretrained=True)
        self.convnext.classifier = torch.nn.Identity()  # Remove classification layer

        # Freeze ConvNeXt parameters = Dont train them
        for param in self.convnext.parameters():
            param.requires_grad = False
            
        # Define ImageNet mean and std for normalization
        self.mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)

    def forward(self, observations):
        try:
            observations = observations["top"]
        except:
            observations = observations
        # Step 1: Convert NumPy array to tensor (if necessary)
        if isinstance(observations, np.ndarray):
            observations = torch.tensor(observations, dtype=torch.float32)

        # Step 2: Permute to (N, C, H, W) format if needed
        if observations.ndim == 4 and observations.shape[1] != 3:
            observations = observations.permute(0, 3, 1, 2)

        # Step 3: Resize to (224, 224) using bilinear interpolation
        observations = F.interpolate(observations, size=(224, 224), mode="bilinear", align_corners=False)

        # Step 4: Normalize using ImageNet mean and std
        observations = (observations / 255.0 - self.mean.to(observations.device)) / self.std.to(observations.device)


        # Step 5: Pass through ConvNeXt
        return self.convnext(observations)
    
# Custom policy
from stable_baselines3.sac.policies import SACPolicy
from gymnasium import spaces
from stable_baselines3.common.type_aliases import Schedule
from typing import Any, Dict, List, Optional, Type, Union


class CustomSACPolicy(SACPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        lr_schedule: Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[torch.nn.Module] = torch.nn.ReLU,
        use_sde: bool = False,
        log_std_init: float = -3,
        use_expln: bool = False,
        clip_mean: float = 2.0,
        features_extractor_class: Type[BaseFeaturesExtractor] = ConvNeXtFeatureExtractor,
        features_extractor_kwargs={"output_dim": 1024},
        #features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = False,
        optimizer_class: Type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        n_critics: int = 2,
        share_features_extractor: bool = False,
    ):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            use_sde,
            log_std_init,
            use_expln,
            clip_mean,
            features_extractor_class,
            features_extractor_kwargs,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
            n_critics,
            share_features_extractor,
        )


# Load ConvNeXt as feature extractor
convnext = convnext_base(weights='IMAGENET1K_V1')  # You can use convnext_tiny, small, base, etc.

# Remove the final classification layer to use it as a feature extractor
convnext.classifier = torch.nn.Identity()

env = gym.make("gym_aloha/AlohaInsertion-v0", obs_type="features")


observation, info = env.reset()
frames = []
start_pose = np.asarray(gym_aloha.constants.START_ARM_POSE.copy())
start_pose = np.delete(start_pose, [8, 15])

# Move the Grippers closer to each other
close_pose = start_pose.copy()
close_pose[0:7] = [0-0, -0.4, 0.9, 0.0, 0.5, 0.0 , 1.5]
close_pose[7:14] = [0-0, -0.4, 0.9, 0.0, 0.5, 0.0 , 1.5]
# close_pose[1],close_pose[8] = -0.4,-0.3
# close_pose[2],close_pose[9] = 0.9,1.3
# close_pose[4] = 0.5
# close_pose[6],close_pose[7] = 1.5,0.0 #1.5

# Bring grippers into position
for i in range(10):
    # interpolate between start_pose and close_pose
    action = start_pose + (close_pose - start_pose) * i / 10
    observation, reward, terminated, truncated, info = env.step(action)
    image = env.render()
    frames.append(image)

    if terminated or truncated:
        observation, info = env.reset()
        


# loop for acting
for i in range(100):
    # get model predicted action
    action = close_pose
    
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

env.close()
filename = "get_close" ".mp4"
imageio.mimsave(filename, np.stack(frames), fps=25)


'''
# example.py
import imageio
import gymnasium as gym
import numpy as np
import gym_aloha
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch as th
from torchvision import models

env = gym.make("gym_aloha/AlohaInsertion-v0")
observation, info = env.reset()
frames = []

for _ in range(1000):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    image = env.render()
    frames.append(image)

    if terminated or truncated:
        observation, info = env.reset()

env.close()
imageio.mimsave("example.mp4", np.stack(frames), fps=25)
'''