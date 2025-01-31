from ray.rllib.core.rl_module.rl_module import RLModule
import gymnasium as gym
import torch
import torch.nn as nn

class CustomRLModule(RLModule):
    def __init__(self, observation_space, action_space, model_config=None):
        super().__init__()

        if observation_space is None or action_space is None:
            raise ValueError("Observation space or action space is None. Ensure the environment correctly defines these.")

        self.obs_dim = observation_space.shape[0]
        self.action_dim = action_space.shape[0]

        # Define a simple feedforward network
        self.network = nn.Sequential(
            nn.Linear(self.obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, self.action_dim),
        )

    def forward(self, obs):
        return self.network(obs)
