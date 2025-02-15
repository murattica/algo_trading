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

        # Output heads for Beta distribution parameters
        self.alpha_head = nn.Linear(512, num_outputs)  # Alpha (shape parameter)
        self.beta_head = nn.Linear(512, num_outputs)  # Beta (shape parameter)

        # Value function head for PPO
        self.value_head = nn.Linear(512, 1)

    def forward(self, input_dict: Dict[str, TensorType], state: List[TensorType], seq_lens: TensorType):
        """
        Takes a batch of observations and outputs allocation decisions.
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
        self._last_fc2_output = x.detach()  # Prevents accidental backpropagation issues

        # Compute Beta distribution parameters
        alpha = torch.nn.functional.softplus(self.alpha_head(x)) + 1.0  # Ensure alpha > 1
        beta = torch.nn.functional.softplus(self.beta_head(x)) + 1.0  # Ensure beta > 1

        # Sample actions from Beta distribution
        allocations = torch.distributions.Beta(alpha, beta).sample()

        # Debug prints
        print("\n--- DEBUG: Beta Distributed Allocations ---")
        print("Alpha:", alpha)
        print("Beta:", beta)
        print("Allocations:", allocations)

        return torch.cat([alpha, beta], dim=-1), state  # Return both parameters

    def value_function(self):
        """
        Returns the value estimate for the given state.
        """
        return self.value_head(self._last_fc2_output).squeeze(-1)  # Shape: (batch_size,)
