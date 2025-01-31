from pettingzoo.utils.env import AECEnv
from pettingzoo.utils import agent_selector
import numpy as np
from gymnasium import spaces

class SequentialTradingEnv(AECEnv):
    metadata = {"render_modes": ["human"], "name": "sequential_trading_v1"}

    def __init__(self, market_data, budget=100.0, risk_free_rate=0.001, render_mode="human"):
        super().__init__()
        if not isinstance(market_data, np.ndarray):
            raise ValueError("market_data must be a NumPy array")

        # Environment parameters
        self.market_data = market_data
        self.state_history, self.num_pairs, self.num_features = market_data.shape
        self.risk_free_rate = risk_free_rate
        self.render_mode = render_mode
        self.budget = budget

        # Initialize agents (specialists first, then meta)
        self.possible_agents = [f"pair_{i}" for i in range(self.num_pairs)] + ["meta"]
        self.agents = self.possible_agents.copy()
        self._agent_selector = agent_selector(self.possible_agents)
        
        # Initialize state tracking with underscore prefixes
        self._rewards = {agent: 0 for agent in self.possible_agents}
        self._cumulative_rewards = {agent: 0.0 for agent in self.possible_agents}  #  Initialize
        self._terminations = {agent: False for agent in self.possible_agents}
        self._truncations = {agent: False for agent in self.possible_agents}
        self._infos = {agent: {} for agent in self.possible_agents}
        self._stored_actions = {}

        # Agent tracking
        self.agent_selection = None
        self.current_actions = {}

        # Normalization stats
        self.mean = np.mean(market_data, axis=(0, 1))
        self.std = np.std(market_data, axis=(0, 1)) + 1e-8

        # Observation spaces
        self.observation_spaces = {
            **{
                f"pair_{i}": spaces.Box(
                    low=-np.inf, high=np.inf,
                    shape=(self.num_features + 1,),
                    dtype=np.float32
                ) for i in range(self.num_pairs)
            },
            "meta": spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(self.num_pairs * 2 + 1,),
                dtype=np.float32
            )
        }

        # Action spaces
        self.action_spaces = {
            **{
                f"pair_{i}": spaces.Box(
                    low=0, high=1,
                    shape=(2,),
                    dtype=np.float32
                ) for i in range(self.num_pairs)
            },
            "meta": spaces.Box(
                low=0, high=1,
                shape=(self.num_pairs + 1,),
                dtype=np.float32
            )
        }

        # Portfolio state
        self.portfolio_value = budget
        self.current_step = 0
        self.last_price_changes = np.zeros(self.num_pairs)


    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def reset(self, seed=None, options=None):
        self.current_step = 0
        self.portfolio_value = self.budget
        self._agent_selector = agent_selector(self.possible_agents)
        self.agent_selection = self._agent_selector.reset()
        self.agents = self.possible_agents.copy()
        
        # Reset INTERNAL state tracking (use underscore variables)
        self._rewards = {agent: 0 for agent in self.possible_agents}
        self._terminations = {agent: False for agent in self.possible_agents}
        self._truncations = {agent: False for agent in self.possible_agents}
        self._stored_actions = {}

        return self._infos

    def observe(self, agent):
        if agent == "meta":
            # Meta agent sees the recommendations from the current step
            specialist_data = [
                self._stored_actions[f"pair_{i}"][0] if f"pair_{i}" in self._stored_actions else 0.0
                for i in range(self.num_pairs)
            ]

            return np.array(specialist_data + [self.portfolio_value / self.budget], dtype=np.float32)

        else:
            # Specialist agent observes current market data
            pair_idx = int(agent.split("_")[1])
            normalized_features = (self.market_data[self.current_step, pair_idx] - self.mean) / self.std
            return np.append(normalized_features, self.portfolio_value / self.budget).astype(np.float32)


    def step(self, action):
        current_agent = self.agent_selection
        
        # Validate action
        if not isinstance(action, np.ndarray):
            raise ValueError(f"Action for {current_agent} must be a NumPy array")
        
        self._stored_actions[current_agent] = action[current_agent]
        
        # Check if all specialists have acted
        if current_agent != "meta":
            # Move to next agent
            self.agent_selection = self._agent_selector.next()
        else:
            # All agents have acted - process the full step
            self._process_meta_step()
            
            # Prepare for next step
            self.current_step += 1
            if self.current_step >= self.state_history - 1:
                self._terminations = {agent: True for agent in self.possible_agents}
                self._truncations = {agent: True for agent in self.possible_agents}
            
            # Reset agent selection for next cycle
            self.agent_selection = self._agent_selector.reset()
            self._stored_actions = {}
            
        # Update the list of active agents (remove terminated agents)
        self.agents = [agent for agent in self.possible_agents if not self._terminations[agent]]

        # Calculate observations
        observations = {agent: self.observe(agent) for agent in self.possible_agents}

        # Calculate rewards
        rewards = self._rewards.copy()

        # Prepare infos
        infos = {
            agent: {
                "portfolio_value": float(self.portfolio_value),
                "price_changes": self.last_price_changes.tolist(),  # Convert array to list
                "allocation": np.array(self._stored_actions.get(agent, [0, 0])).tolist()  # Convert array to list
            }
            for agent in self.possible_agents
        }

        # Ensure terminations and truncations are dictionaries
        terminations = self._terminations.copy()
        truncations = self._truncations.copy()

        print(f"DEBUG: Rewards at step {self.current_step} = {self._rewards}")

        return observations, rewards, terminations, truncations, infos

    def _process_meta_step(self):
        # Validate meta_action
        meta_action = self._stored_actions["meta"]
        if not isinstance(meta_action, np.ndarray):
            raise ValueError("Meta action must be a NumPy array")
        
        # Process allocations
        allocs = np.clip(meta_action[:-1], 0, 1)
        cash = np.clip(meta_action[-1], 0, 1)
        
        # Normalize allocations
        total_alloc = allocs.sum() + cash
        if total_alloc <= 0:
            allocs = np.zeros_like(allocs)
            cash = 1.0
        else:
            allocs = allocs / total_alloc
            cash = cash / total_alloc

        # Calculate price changes
        current_prices = self.market_data[self.current_step, :, 1]
        next_prices = self.market_data[self.current_step + 1, :, 1]
        valid = current_prices > 1e-8
        price_changes = np.where(valid, (next_prices - current_prices) / current_prices, 0)
        
        # Validate price_changes
        if not isinstance(price_changes, np.ndarray):
            raise ValueError("Price changes must be a NumPy array")
        
        self.last_price_changes = price_changes

        # Calculate returns
        pair_returns = allocs * price_changes
        cash_return = cash * self.risk_free_rate
        total_return = pair_returns.sum() + cash_return

        # Update portfolio
        new_portfolio = max(self.portfolio_value * (1 + total_return), 1)
        self.portfolio_value = new_portfolio

        # Calculate rewards
        for i in range(self.num_pairs):
            agent = f"pair_{i}"
            alloc_impact = float(allocs[i] * price_changes[i])  # Ensure scalar
            realized_volatility = float(self._calculate_volatility(i))  # Ensure scalar
            risk_accuracy = float(1 - abs(self._stored_actions[f"pair_{i}"][1] - realized_volatility))  # Ensure scalar
            self._rewards[agent] = 0.7 * alloc_impact + 0.3 * risk_accuracy
        
        self._rewards["meta"] = float(total_return)  # Ensure scalar

    def _calculate_volatility(self, asset_index, window=10):
        if self.current_step < window:
            return 0.0
        
        close_prices = self.market_data[self.current_step - window:self.current_step, asset_index, 1]
        if len(close_prices) < 2:
            return 0.0
            
        log_returns = np.log(close_prices[1:] / close_prices[:-1])
        volatility = np.std(log_returns)
        return np.clip(volatility / 0.2, 0, 1)  # Normalize by max expected 20% volatility

    def render(self, mode="human"):
        if mode == "human":
            print(f"\nStep {self.current_step}")
            print(f"Portfolio Value: ${self.portfolio_value:.2f}")
            print("Current Allocations:")
            for i in range(self.num_pairs):
                action = self._stored_actions.get(f"pair_{i}", [0, 0])
                print(f"- Pair {i}: {action[0]:.2%} rec, {action[1]:.2f} risk")

    def close(self):
        pass

    # Required property definitions
    @property
    def terminations(self):
        return self._terminations

    @property
    def truncations(self):
        return self._truncations

    @property
    def rewards(self):
        return self._rewards

    @property
    def infos(self):
        return self._infos

    @rewards.setter
    def rewards(self, value):
        self._rewards = value

    @terminations.setter
    def terminations(self, value):
        self._terminations = value

    @truncations.setter
    def truncations(self, value):
        self._truncations = value

    @infos.setter
    def infos(self, value):
        self._infos = value

    @property
    def observations(self):
        """Dictionary containing current observations for all active agents."""
        return {agent: self.observe(agent) for agent in self.possible_agents}