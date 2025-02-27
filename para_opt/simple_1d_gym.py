import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import convnext_base
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class ConvNeXtFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, output_dim=1024):  # ConvNeXt output size
        super().__init__(observation_space, features_dim=output_dim)
        # Load pretrained ConvNeXt model
        self.convnext = convnext_base(pretrained=True)
        self.convnext.classifier = torch.nn.Identity()  # Remove classification layer

        # Freeze ConvNeXt parameters = Don't train them
        for param in self.convnext.parameters():
            param.requires_grad = False
            
        # Define ImageNet mean and std for normalization
        self.mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
        self.std  = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)
        
        
    def forward(self, observations):
        try:
            observations = observations["top"]
        except:
            observations = observations
        # Step 1: Convert NumPy array to tensor (if necessary)
        if isinstance(observations, np.ndarray):
            observations = torch.tensor(observations, dtype=torch.float32)

        # Step 2: Permute to (N, C, H, W) format if needed
        if observations.ndim == 3 and observations.shape[1] != 3:
            observations = observations.permute(2, 0, 1)
            observations = observations.unsqueeze(0)

        # Step 3: Resize to (224, 224) using bilinear interpolation
        observations = F.interpolate(observations, size=(224, 224), mode="bilinear", align_corners=False)

        # Step 4: Normalize using ImageNet mean and std
        observations = (observations / 255.0 - self.mean.to(observations.device)) / self.std.to(observations.device)
        
        # Step 4.5: Move from CPU to CUDA
        device = next(self.convnext.parameters()).device
        observations = observations.to(device)
        
        # Step 5: Pass through ConvNeXt
        out = self.convnext(observations)
        
        # Step 6: Squeeze and convert to NumPy
        feature_vector = out.squeeze().detach().cpu().numpy()  # Shape [1024]
        
        # Step 7: Wrap in a dictionary
        return feature_vector

class SimpleGymEnv(gym.Env):
    """A simple custom environment for SAC training."""
    
    def __init__(self):
        super(SimpleGymEnv, self).__init__()
        
        # Continuous action space between -1 and 1
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        
        # Observation space, representing the position of the agent
        self.observation_space = spaces.Box(low=-1, high=1, shape=(1026,), dtype=np.float32)
        
        # Feature space, representing the image of the agent
        self.feature_space = spaces.Box(low=0, high=255, shape=(100,100,3), dtype=np.float32)
        
        # Feature extractor for ConvNeXt
        self.f_extr = ConvNeXtFeatureExtractor(self.feature_space)
        
        # Initial state
        self.image = np.zeros((100,100,3), dtype=np.float32)
        self.state = np.array([0.0, 0.0], dtype=np.float32)
        self.target = np.array([4.0, 4.0], dtype=np.float32)  # Goal position
        self.obs = self.f_extr(self.image)
        self.light_dark = 1.0
        self.past_states = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if np.random.uniform() < 0.5:
            self.light_dark = 0.0
            self.target = np.array([4.0, 4.0], dtype=np.float32)
            self.image = np.zeros((100,100,3), dtype=np.float32)
        else:
            self.light_dark = 1.0
            self.target = np.array([-4.0, -4.0], dtype=np.float32)
            self.image = np.zeros((100,100,3), dtype=np.float32) + 200
        
        self.obs = self.f_extr(self.image)
        self.state = np.array([0.0, 0.0], dtype=np.float32)
        return np.concatenate((self.obs, self.state)), {}

    def step(self, action):
        self.state += action  # Move based on action
        
        # Reward is negative distance to target
        reward = -np.linalg.norm(self.state - self.target)
        
        # Check if done (close enough to target)
        done = -reward < 0.1
        
        out = np.concatenate((self.obs, self.state))
        a = np.zeros((3,))
        a[0:2] = self.state
        a[2] = self.light_dark
        self.past_states = np.append(self.past_states, values=a)
        #reshape the list to a 2D array with 3 columns
        self.past_states = np.reshape(self.past_states, (-1, 3))
        return out, reward, done, False, {}
    
    def render(self):
        print(f"State: {self.state}")
        
    def get_past_states(self):
        return self.past_states
    
    def close(self):
        pass
