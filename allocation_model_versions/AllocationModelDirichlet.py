import numpy as np
import torch
import torch.nn as nn
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.typing import Dict, TensorType, List, ModelConfigDict
from gymnasium import spaces

class AllocationModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        nn.Module.__init__(self)
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)

        obs_size = np.prod(obs_space.shape)

        # Define the network
        self.network = nn.Sequential(
            nn.Linear(obs_size, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
        )

        # Output head for Dirichlet parameters (alpha values)
        self.alpha_head = nn.Linear(512, num_outputs)  # Dirichlet parameter α

        # Value function head for PPO
        self.value_head = nn.Linear(512, 1)

    def forward(self, input_dict: Dict[str, TensorType], state: List[TensorType], seq_lens: TensorType):
        """
        Takes a batch of observations and outputs allocation decisions that sum to 1.
        """

        obs = input_dict["obs"]  # Extract observation dictionary

        specialist_actions = obs["specialist_actions"]
        open_positions = obs["open_positions"]
        cash_reserve = obs["cash_reserve"]
        action_mask = obs["action_mask"]

        # Concatenate all observation features into a single tensor
        obs_tensor = torch.cat([
            action_mask.float(),
            cash_reserve.float(),
            open_positions.float(),
            specialist_actions.float()
        ], dim=-1)

        # Process observation through the network
        x = self.network(obs_tensor)

        # Store last layer output for value function
        self._last_fc2_output = x.detach()

        # Compute Dirichlet distribution parameters (ensure positivity)
        alpha = torch.nn.functional.softplus(self.alpha_head(x)) + 1.0  # α > 1 for stability

        # Sample from Dirichlet distribution (allocations sum to 1)
        allocations = torch.distributions.Dirichlet(alpha).sample()

        # Debug prints
        print("\n--- DEBUG: Dirichlet Distributed Allocations ---")
        print("Alpha:", alpha)
        print("Allocations (sum=1):", allocations.sum(dim=-1))  # Should always be ~1.0

        return allocations, state  # Return final allocations directly

    def value_function(self):
        """
        Returns the value estimate for the given state.
        """
        return self.value_head(self._last_fc2_output).squeeze(-1)  # Shape: (batch_size,)
