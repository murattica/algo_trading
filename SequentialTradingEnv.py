from pettingzoo.utils.env import AECEnv
from pettingzoo.utils import agent_selector
import numpy as np
from gymnasium import spaces
from Portfolio import Portfolio

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
        self.portfolio = Portfolio(initial_cash=self.budget)
        

        # Initialize agents (specialists first, then meta)
        self.possible_agents = [f"pair_{i}" for i in range(self.num_pairs)] + ["meta"]
        self.agents = self.possible_agents.copy()
        self._agent_selector = agent_selector(self.possible_agents)

        # State tracking
        self._rewards = {agent: 0 for agent in self.possible_agents}
        self._cumulative_rewards = {agent: 0.0 for agent in self.possible_agents}
        self.terminations = {agent: False for agent in self.possible_agents} 
        self.truncations = {agent: False for agent in self.possible_agents} 
        self.infos = {agent: {} for agent in self.possible_agents}

        # Normalization stats
        self.mean = np.mean(market_data, axis=(0, 1))
        self.std = np.std(market_data, axis=(0, 1)) + 1e-8

        # Portfolio state
        self.current_step = 0
        self._stored_actions = {}

        # Define Observation Spaces
        self.observation_spaces = spaces.Dict({
            **{
                f"pair_{i}": spaces.Dict({
                    "market_data": spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_features,), dtype=np.float32),  # Market data
                    "own_position": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)  # (direction, size, return_rate)
                }) for i in range(self.num_pairs)
            },
            "meta": spaces.Dict({
                "specialist_actions": spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_pairs * 2,), dtype=np.float32),  # Specialists' recommendations
                "open_positions": spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_pairs * 3,), dtype=np.float32),  # All open positions
                "cash_reserve": spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32)
            })
        })

        # Define Action Spaces
        self.action_spaces = {
            **{
                f"pair_{i}": spaces.Tuple((
                    spaces.Discrete(5),  # {Hold, Buy, Sell, Increase, Decrease}
                    spaces.Box(low=0, high=1, shape=(), dtype=np.float32)  # Change shape from (1,) to ()
                )) for i in range(self.num_pairs)
            },
            "meta": spaces.Box(
                low=0, high=1,
                shape=(self.num_pairs + 1,),  # Adjust weight on pairs + cash allocation
                dtype=np.float32
            )
        }

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def _execute_trades(self, meta_action):
        """
        Executes trades based on the meta agent's allocation decision.
        Uses the Portfolio and Position classes for managing trades.
        """
        market_prices = {f"pair_{i}": self._get_market_data(f"pair_{i}")[0] for i in range(self.num_pairs)}

        for i in range(self.num_pairs):
            pair = f"pair_{i}"
            allocation = meta_action[i]  # Fraction of portfolio value to allocate
            new_size = allocation * self.portfolio.get_portfolio_value(market_prices)  # New position size

            current_position = self.portfolio.positions.get(pair, None)
            current_price = market_prices[pair]

            if current_position is None:
                # Open a new position if size > 0 and enough cash is available
                if new_size > 0 and self.portfolio.cash_reserve >= new_size:
                    self.portfolio.open_position(pair, entry_price=current_price, direction="long",
                                                size=new_size, timestamp=self.current_step, agent="meta")
            else:
                # If the position exists, check whether to increase, decrease, or close
                if new_size > current_position.size:
                    increase_amount = new_size - current_position.size
                    if self.portfolio.cash_reserve >= increase_amount:
                        self.portfolio.modify_position(pair, action="increase", size=increase_amount,
                                                    new_price=current_price, timestamp=self.current_step)
                elif new_size < current_position.size:
                    decrease_amount = current_position.size - new_size
                    if new_size == 0:
                        self.portfolio.modify_position(pair, action="close", size=0, new_price=current_price,
                                                    timestamp=self.current_step)
                    else:
                        self.portfolio.modify_position(pair, action="decrease", size=decrease_amount,
                                                    new_price=current_price, timestamp=self.current_step)

    def observe(self, agent):
        """
        Returns the observation for the given agent.
        """
        if agent == "meta":
            return {
                "specialist_actions": np.array([
                    [self._stored_actions[f"pair_{i}"][0], self._stored_actions[f"pair_{i}"][1]]  
                    for i in range(self.num_pairs)
                ], dtype=np.float32).flatten(),
                "open_positions": np.array([
                    (pos.direction, pos.size, pos.pnl) if pos else (0, 0.0, 0.0)
                    for pos in [self.portfolio.positions.get(f"pair_{i}", None) for i in range(self.num_pairs)]
                ], dtype=np.float32).flatten(),
                "cash_reserve": np.array([self.portfolio.cash_reserve], dtype=np.float32)
            }

        elif agent.startswith("pair_"):
            agent_id = int(agent.split("_")[1])  
            position = self.portfolio.positions.get(f"pair_{agent_id}", None)
            return {
                "market_data": self._get_market_data(agent),  
                "own_position": np.array(
                    (position.direction, position.size, position.pnl) if position else (0, 0.0, 0.0),
                    dtype=np.float32
                )
            }

        else:
            raise ValueError(f"Unknown agent: {agent}")

    def step(self, actions):
        current_agent = self.agent_selection
        self._stored_actions.setdefault(current_agent, None)  # Ensure key exists before logging

        if current_agent not in self._step_counts:
            self._step_counts[current_agent] = 0
        self._step_counts[current_agent] += 1

        # Convert actions if needed
        if not isinstance(actions, dict):
            if isinstance(actions, tuple) and current_agent.startswith("pair_"):
                actions = {current_agent: actions}
            elif isinstance(actions, np.ndarray) and current_agent == "meta":
                actions = {current_agent: actions}
            else:
                raise ValueError(f"Expected actions to be a dictionary, got {type(actions)}")

        if current_agent not in actions:
            raise ValueError(f"No action provided for agent {current_agent}")

        agent_action = actions[current_agent]

        # Store the action
        self._stored_actions[current_agent] = agent_action

        # Move to the next agent
        if current_agent.startswith("pair_"):
            self.agent_selection = self._agent_selector.next()
            if all(f"pair_{i}" in self._stored_actions for i in range(self.num_pairs)):
                self.agent_selection = "meta"
            return {}, {}, {}, {}, {}

        elif current_agent == "meta":
            return self._process_meta_step(current_agent)


    def _process_meta_step(self,current_agent):
        """
        Handles the meta-agent's decision-making and trade execution.
        This function is automatically called inside `step()` when the meta-agent acts.
        """
        meta_action = self._stored_actions["meta"]  # Store the meta-agent's action

        # Execute trades based on meta-agent's decision
        self._execute_trades(meta_action)

        # Update return rates based on market movement
        self._update_return_rates()

        # Calculate rewards
        self._calculate_rewards()

        # Move to next time step
        self.current_step += 1

        # Check if the episode should terminate
        if self.current_step >= self.state_history - 1:
            self.terminations = {agent: True for agent in self.possible_agents}
            self.truncations = {agent: True for agent in self.possible_agents}

        # Reset agent selection for the next cycle
        self.agent_selection = self._agent_selector.reset()
        
        # 🔹 Construct final return values
        observations = {agent: self.observe(agent) for agent in self.possible_agents}
        rewards = self._rewards.copy()
        terminations = self.terminations.copy()
        truncations = self.truncations.copy()
        infos = {}

        self._stored_actions = {}

        return observations, rewards, terminations, truncations, infos

    def render(self, mode="human"):
        if mode == "human":
            print(f"\nStep {self.current_step}")
            print(f"Portfolio Value: ${self.portfolio_value:.2f}")
            print("Current Allocations:")
            for i in range(self.num_pairs):
                allocation = self.current_allocations[i]
                print(f"- Pair {i}: {allocation:.2%} allocation")

    def reset(self, seed=None, options=None):
        print("reset initiated")
        self.current_step = 0
        self.portfolio_value = self.budget
        self.current_allocations = np.zeros(self.num_pairs + 1)
        self._stored_actions = {}  # Reset _stored_actions fully at the start of an episode

        self._step_counts = {}  # Track execution counts per agent
        self._rewards = {agent: 0.0 for agent in self.possible_agents}
        self._terminations = {agent: False for agent in self.possible_agents}
        self._truncations = {agent: False for agent in self.possible_agents}
        self._infos = {agent: {} for agent in self.possible_agents}

        self.agent_selection = self._agent_selector.reset()
        self.agents = self.possible_agents.copy()

        return self._infos


    def _update_return_rates(self):
        """
        Updates the return rate for all open positions based on market price movement.
        """
        for i in range(self.num_pairs):
            direction, size, _ = self.open_positions[f"pair_{i}"]

            if direction != 0:  # Only update if position is open
                price_change = self._get_market_movement(i)  # Fetch market movement
                new_return_rate = ((price_change / size) if size > 0 else 0.0) * direction  # Adjust by position type
                self.open_positions[f"pair_{i}"] = (direction, size, new_return_rate)

    def _get_market_data(self, agent_id):
        """
        Returns the latest market data snapshot for the current timestep for a specific trading pair.
        """
        if agent_id.startswith("pair_"):
            pair_index = int(agent_id.split("_")[1])  # Extract the pair index
            return self.market_data[self.current_step, pair_index, :]  # Fetch only relevant pair's data
        else:
            raise ValueError(f"Market data requested for unknown agent: {agent_id}")

    def _get_market_movement(self, pair_index):
        """
        Returns the percentage price movement for a given trading pair between the last and current step.

        - If it's the first timestep, assume no price change.
        - Otherwise, compute the relative change from the previous timestep.
        """
        if self.current_step == 0:
            return 0.0  # No movement at the first timestep

        # Fetch previous and current prices
        prev_price = self.market_data[self.current_step - 1, pair_index, 0]  # Price at t-1
        current_price = self.market_data[self.current_step, pair_index, 0]  # Price at t

        # Compute relative price change (percentage)
        price_change = ((current_price - prev_price) / prev_price) if prev_price > 0 else 0.0

        return price_change

    def _calculate_rewards(self):
        """
        Calculates and assigns rewards to all agents:
        - Specialists: Rewarded based on their recommendations' profitability.
        - Meta-Agent: Rewarded based on portfolio performance.
        """
        market_prices = {f"pair_{i}": self._get_market_data(f"pair_{i}")[0] for i in range(self.num_pairs)}
        portfolio_pnl = self.portfolio.calculate_total_pnl(market_prices)

        for i in range(self.num_pairs):
            position = self.portfolio.positions.get(f"pair_{i}", None)
            self._rewards[f"pair_{i}"] = position.pnl if position else -0.01  

        self._rewards["meta"] = portfolio_pnl / self.budget  

        print(f"DEBUG: Rewards at step {self.current_step} = {self._rewards}")


    def close(self):
        pass

    # Required properties
    @property
    def rewards(self):
        return self._rewards

    @property
    def observations(self):
        return {agent: self.observe(agent) for agent in self.possible_agents}
