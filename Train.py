import ray
from ray.rllib.env import PettingZooEnv
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.models import ModelCatalog
from AllocationModelGaussian import AllocationModel
from db import prepare_numpy_array, fetch_state_history
from SequentialTradingEnv import SequentialTradingEnv
import numpy as np
import os

# Initialize Ray
ray.init(include_dashboard=True, ignore_reinit_error=True)

# Fetch and prepare market data
trading_pairs = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'ADA/USDT']
raw_data = fetch_state_history(trading_pairs)
market_data, time_index_map, symbol_index_map = prepare_numpy_array(raw_data, num_pairs=len(trading_pairs))

# Environment creator function
def env_creator(config):
    return PettingZooEnv(SequentialTradingEnv(
        market_data=config.get("market_data", market_data),
        budget=config.get("budget", 100),
        risk_free_rate=config.get("risk_free_rate", 0.001),
        symbol_index_map=symbol_index_map
    ))

# Register the environment
register_env("SequentialTradingEnv", env_creator)

# Test environment
test_env = env_creator({})

# Register the custom model
ModelCatalog.register_custom_model("allocation_model", AllocationModel)

# Define policies
policies = {
    **{
        f"policy_{i}": (None, test_env.observation_space[f"pair_{i}"], test_env.action_space[f"pair_{i}"], {})
        for i in range(len(trading_pairs))
    },
    "meta": (
        None,
        test_env.observation_space["meta"],
        test_env.action_space["meta"],
        {
            "model": {
                "custom_model": "allocation_model",
                "custom_model_config": {
                    "num_pairs": len(trading_pairs),
                },
                "fcnet_hiddens": [512, 512],  # Ensure consistency
                "fcnet_activation": "relu",
                "free_log_std": False
            }
        }  # Assign custom model to meta agent
    )
}

# Policy mapping function
def policy_mapping_fn(agent_id, *args, **kwargs):
    if agent_id == "meta":
        return "meta"
    return f"policy_{agent_id.split('_')[1]}"

# Configure PPO training
config = (
    PPOConfig()
    .environment(env="SequentialTradingEnv")
    .framework("torch")
    .api_stack(
        enable_rl_module_and_learner=False,  # Disables the new RLModule API
        enable_env_runner_and_connector_v2=False  # Disables the new env runner API
    )
    .multi_agent(
        policies=policies,
        policy_mapping_fn=policy_mapping_fn,
    )
    .training(
        model={"fcnet_hiddens": [512, 512]},  # Default model for specialist agents
        lr=5e-5,  # Learning rate
        gamma=0.99  # Discount factor for future rewards
    )
    .env_runners(
        num_envs_per_env_runner=1,
        rollout_fragment_length='auto',
        sample_timeout_s=300
    )
    .resources(num_gpus=0)  # Use CPU for training
    .debugging(
        log_level="DEBUG",  # Enables detailed logging
        logger_config={"type": "ray.tune.logger.TBXLogger", "logdir": "./logs"}  # Saves logs for TensorBoard
    )
)

# Build the trainer
trainer = config.build_algo()

# Training loop
rolling_window = 50  # Evaluate performance over the last 50 iterations
best_return = -np.inf
patience = 20  # Stop if no improvement for 20 checks
stagnation_counter = 0

# Define save directory
save_dir = "./saved_models"
os.makedirs(save_dir, exist_ok=True)  # Create directory if it doesn't exist

for iteration in range(1, 50000):  # Max safe upper limit
    result = trainer.train()
    current_return = result['env_runners']['episode_return_mean']
    print(f"Iteration {iteration}, Return: {current_return}")

    # Early stopping
    if current_return > best_return:
        best_return = current_return
        stagnation_counter = 0
        model_path = os.path.join(save_dir, f"best_model_{iteration}")
        trainer.save(model_path)
    else:
        stagnation_counter += 1

    if stagnation_counter >= patience:
        print(f"Early stopping at iteration {iteration}")
        break

print("Training complete!")
ray.shutdown()


import torch
from collections import OrderedDict

num_samples = 32
action_mask_dim = 13
position_dim = 12
cash_dim = 1
action_mask = torch.zeros((num_samples, action_mask_dim))
cash_reserve = torch.zeros((num_samples, cash_dim))
open_positions = torch.zeros((num_samples, position_dim))
specialist_actions = torch.zeros((num_samples, position_dim))

data = OrderedDict([
    ('action_mask', action_mask),
    ('cash_reserve', cash_reserve),
    ('open_positions', open_positions),
    ('specialist_actions', specialist_actions),
])