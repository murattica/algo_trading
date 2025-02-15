import torch
import torch.distributions as dist
from ray.rllib.models.torch.torch_action_dist import TorchDistributionWrapper
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import TensorType

class TorchBetaDistribution(TorchDistributionWrapper):
    def __init__(self, inputs: TensorType, model):
        super().__init__(inputs, model)
        # Split the inputs into alpha and beta parameters for the Beta distribution
        self.alpha = torch.exp(inputs[..., 0])  # Ensure alpha > 0
        self.beta = torch.exp(inputs[..., 1])   # Ensure beta > 0
        self.dist = dist.Beta(self.alpha, self.beta)

    @override(TorchDistributionWrapper)
    def sample(self) -> TensorType:
        return self.dist.sample()

    @override(TorchDistributionWrapper)
    def logp(self, actions: TensorType) -> TensorType:
        return self.dist.log_prob(actions)

    @override(TorchDistributionWrapper)
    def entropy(self) -> TensorType:
        return self.dist.entropy()

    @override(TorchDistributionWrapper)
    def kl(self, other: "TorchBetaDistribution") -> TensorType:
        return dist.kl_divergence(self.dist, other.dist)