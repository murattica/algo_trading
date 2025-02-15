import torch
import torch.nn as nn
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.typing import Dict, TensorType, List


class AllocationModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        nn.Module.__init__(self)
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)

        # Extract `num_pairs` from model_config (default to 4 if missing)
        self.num_pairs = model_config.get("custom_model_config", {}).get("num_pairs", 4)

        print(f"DEBUG: num_pairs={self.num_pairs}")  # Debugging
        print(f"DEBUG: num_outputs={num_outputs}")  # Debugging

        # Calculate total observation size for the meta-agent
        obs_size = (self.num_pairs) + (self.num_pairs * 2) + (self.num_pairs) + 1
        # (specialist_signals) + (position_pnl) + (position_size) + (cash_reserve)

        self.network = nn.Sequential(
            nn.Linear(obs_size, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
        )

        # Output layers for actions
        self.allocation_head = nn.Linear(512, self.num_pairs + 1)  # Allocation (pairs + cash)
        self.position_adjustment_head = nn.Linear(512, self.num_pairs)  # Position adjustments

        # Value function head for PPO
        self.value_head = nn.Linear(512, 1)

    def forward(self, input_dict: Dict[str, TensorType], state: List[TensorType], seq_lens: TensorType):
        """
        Takes a batch of observations and outputs mean and log_std for the Gaussian policy.
        """

        obs = input_dict["obs"]
        print("burdayım be")
        print(obs)

        # Extract meta-agent observations
        specialist_signals = obs["specialist_signals"]
        position_pnl = obs["position_pnl"].view(obs["position_pnl"].shape[0], -1)
        position_size = obs["position_size"]
        cash_reserve = obs["cash_reserve"]

        # Concatenate all features into a single observation tensor
        obs_tensor = torch.cat([
            specialist_signals.float(),
            position_pnl.float(),
            position_size.float(),
            cash_reserve.float(),
        ], dim=-1)

        # Process observation through the network
        x = self.network(obs_tensor)

        # Store last hidden layer output for value function
        self._last_fc2_output = x.detach()

        # Compute allocations and position adjustments
        raw_allocations = self.allocation_head(x)  # (batch_size, num_pairs + 1) = (batch_size, 5)
        raw_position_adjustments = self.position_adjustment_head(x)  # (batch_size, num_pairs) = (batch_size, 4)

        # Concatenate them into a single tensor
        raw_output = torch.cat([raw_allocations, raw_position_adjustments], dim=-1)  # Shape: (batch_size, 9)
        
        print("DEBUG: raw_output.shape =", raw_output.shape)  # Should be (batch_size, 9)

        # Ensure even split between mean and log_std
        half_dim = raw_output.shape[-1]  # Should be 9
        mean = raw_output  # First 9 values = means
        log_std = torch.zeros_like(mean)  # Initialize log_std as zeros (or learnable later)

        print("DEBUG: mean.shape =", mean.shape, "log_std.shape =", log_std.shape)  # Both should be (batch_size, 9)

        # Clamp log_std to avoid extreme values
        log_std = torch.clamp(log_std, min=-2, max=1)
        std = torch.exp(log_std) + 1e-3  # Convert log_std into standard deviations

        # Concatenate both for RLlib
        output_tensor = torch.cat([mean, log_std], dim=-1)  # Shape: (batch_size, 18)

        return output_tensor, state


    def value_function(self):
        """
        Returns the value estimate for the given state.
        """
        return self.value_head(self._last_fc2_output).squeeze(-1)  # Shape: (batch_size,)
