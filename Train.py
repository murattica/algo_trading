import ray
from ray.rllib.env import PettingZooEnv
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPOConfig
from db import prepare_numpy_array, fetch_state_history
from SequentialTradingEnv import SequentialTradingEnv
import numpy as np
import os

ray.init(include_dashboard=True, ignore_reinit_error=True)

trading_pairs = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'ADA/USDT']
raw_data = fetch_state_history(trading_pairs)
market_data, time_index_map, symbol_index_map = prepare_numpy_array(raw_data, num_pairs=len(trading_pairs))

def env_creator(config):
    return PettingZooEnv(SequentialTradingEnv(
        market_data=config.get("market_data", market_data),
        budget=config.get("budget", 100),
        risk_free_rate=config.get("risk_free_rate", 0.001)
    ))

register_env("SequentialTradingEnv", env_creator)

test_env = env_creator({})

# **Define a unique policy for each agent**
policies = {
    **{
        f"policy_{i}": (None, test_env.observation_space[f"pair_{i}"], test_env.action_space[f"pair_{i}"], {})
        for i in range(len(trading_pairs))
    },
    "meta": (
        None,
        test_env.observation_space["meta"],
        test_env.action_space["meta"],
        {}
    )
}

def policy_mapping_fn(agent_id, *args, **kwargs):
    """
    Maps each agent to its corresponding policy.
    """
    if agent_id == "meta":
        return "meta"
    return f"policy_{agent_id.split('_')[1]}"

# **Configure PPO training**
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
        model={"fcnet_hiddens": [512, 512]},  # Neural network structure
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

trainer = config.build_algo()

# Training loop**
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
    print(current_return)
    
    # Early stopping
    if current_return > best_return:
        best_return = current_return
        stagnation_counter = 0

        trainer.save(model_path)
    else:
        stagnation_counter += 1
    
    if stagnation_counter >= patience:
        print(f"Early stopping at iteration {iteration}")
        break

print("Training complete!")
