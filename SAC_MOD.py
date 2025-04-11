import stable_baselines3.sac.sac as sac
import numpy as np

#modified from stable_baselines3.sac.sac.py
class SAC_MOD(sac.SAC):
    """
    Soft Actor-Critic (SAC) algorithm with modifications for the Aloha environment.
    """
    def __init__(self, *args, **kwargs):
        super(SAC_MOD, self).__init__(*args, **kwargs)
    
    # create costom logger for q_values
    def train(self, gradient_steps, batch_size = 64):
        out = super().train(gradient_steps, batch_size)
        self.logger.record("train/current_q_values", np.mean(current_q_values))
        return out