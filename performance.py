import ray
import os
import numpy as np
import matplotlib.pyplot as plt

from ray.rllib.algorithms.ppo import PPO
from ray.tune.registry import register_env
from ray.rllib.env import ParallelPettingZooEnv

from db import prepare_numpy_array, fetch_state_history
from SequentialTradingEnv import SequentialTradingEnv

# Ensure Ray is properly initialized
ray.shutdown()
ray.init(ignore_reinit_error=True)

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

# Load Trained Model
model_path = os.path.abspath("mappo_trading_model")
trainer = PPO.from_checkpoint(model_path)

print("Model loaded successfully!")

# Initialize Environment for Evaluation
test_env = env_creator({})
obs, _ = test_env.reset()
active_agents = list(obs.keys())
done = {agent: False for agent in active_agents}  # Track which agents are active

# Store performance metrics
total_rewards = {agent: 0 for agent in active_agents}
portfolio_values = {agent: 25 for agent in active_agents if agent != "meta"}  # Exclude meta agent
portfolio_history = {agent: [] for agent in active_agents}

# Run Evaluation
current_step = 0
while not all(done.values()):
    print(f"\n🔹 Step {current_step}: Processing")

    # Compute actions for active agents
    actions = {}
    for agent in active_agents:
        if agent in obs and not done.get(agent, False):
            if agent == "meta":
                # Meta-agent: Allocates across pairs + cash
                actions[agent] = trainer.compute_single_action(observation=obs[agent], policy_id="meta")
            else:
                # Pair agents: Provide (recommendation, risk_score)
                actions[agent] = trainer.compute_single_action(observation=obs[agent], policy_id=f"policy_{agent.split('_')[1]}")

    # Step the environment
    obs, rewards, done, truncated, info = test_env.step(actions)

    # Track rewards & portfolio updates
    for agent in rewards:
        total_rewards[agent] += rewards[agent]
        portfolio_values[agent] *= (1 + rewards[agent])  # Growth based on reward
        portfolio_history[agent].append(portfolio_values[agent])

    # Render environment state
    test_env.render()

    print(f"🔹 Step {current_step} complete. Actions: {actions}")
    print(f"🔹 Rewards: {rewards}")
    current_step += 1

# Print Final Performance Metrics
print("\n--- 📈 Evaluation Summary ---")
for agent in test_env.par_env.agents:
    print(f"{agent}: Total Reward = {total_rewards[agent]:.4f}, Final Portfolio Value = ${portfolio_values[agent]:.2f}")

