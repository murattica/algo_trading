from pettingzoo.utils.env import AECEnv
from pettingzoo.utils import agent_selector
import numpy as np
from gymnasium import spaces
from Portfolio import Portfolio
import torch

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
        self.transaction_fee = 0.001

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
                    "market_data": spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_features,), dtype=np.float32),
                }) for i in range(self.num_pairs)
            },
            "meta": spaces.Dict({
                "specialist_signals": spaces.Box(low=-1.0, high=1.0, shape=(self.num_pairs,), dtype=np.float32),  # Processed Buy/Sell signals
                "position_pnl": spaces.Box(low=-1.0, high=1.0, shape=(self.num_pairs, 2), dtype=np.float32),  # (Direction, Normalized PnL)
                "position_size": spaces.Box(low=0, high=np.inf, shape=(self.num_pairs,), dtype=np.float32),  # Open position sizes
                "cash_reserve": spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32),  # Available capital in cash
            })
        })

        # Define Action Spaces
        self.action_spaces = {
            **{
                f"pair_{i}": spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)  # [-1] = Sell, [1] = Buy, [0] = Hold
                for i in range(self.num_pairs)
            },
            "meta": spaces.Dict({
                "allocation": spaces.Box(
                    low=0.0, high=1.0,
                    shape=(self.num_pairs + 1,),  # Allocations for each pair + cash reserve
                    dtype=np.float32
                ),
                "position_adjustment": spaces.Box(
                    low=-1.0, high=1.0,
                    shape=(self.num_pairs,),  # Adjust existing positions (-1 = Reduce, 1 = Increase)
                    dtype=np.float32
                )
            })
        }

    def _get_action_mask(self):
        """
        Generates action masks to prevent invalid allocations and position adjustments.
        """
        allocation_mask = np.ones(self.num_pairs + 1, dtype=np.int32)  # Allows cash + all pairs
        position_adjustment_mask = np.zeros(self.num_pairs, dtype=np.int32)  # Default: No adjustments allowed

        for i in range(self.num_pairs):
            position = self.portfolio.positions.get(f"pair_{i}", None)

            # Disable allocation if the pair is unavailable for trading
            if self._is_pair_disabled(f"pair_{i}"):  
                allocation_mask[i] = 0  # Prevent allocation to disabled pairs

            # Allow position adjustments only if a position is open
            if position:
                position_adjustment_mask[i] = 1  # Can adjust active positions
                
                # Prevent increasing position if out of cash
                if self.portfolio.cash_reserve <= 0:
                    position_adjustment_mask[i] = 0

        return {
            "allocation": allocation_mask,
            "position_adjustment": position_adjustment_mask,
        }

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def execute_trades(self, meta_action):
        """
        Executes trades based on the meta-agent's allocation and position adjustment decisions.
        Ensures at least 5% of total capital remains in cash reserves.
        Applies transaction fees to all executed trades.
        """
        allocation = np.array(meta_action["allocation"])  # Convert allocation to NumPy array
        position_adjustment = np.array(meta_action["position_adjustment"])  # Convert to NumPy array

        total_capital = self.portfolio.cash_reserve + sum(pos.size for pos in self.portfolio.positions.values())
        min_cash_reserve = total_capital * 0.05  # Ensure at least 5% cash is reserved

        # Step 1: Apply Softmax to allocation (ensures total sum = 1)
        allocation = allocation / allocation.sum()

        # Step 2: Adjust allocation to ensure 5% cash reserve
        allocation[:-1] *= (1 - 0.05)  # Reduce allocations by 5% to ensure cash reserve is kept
        allocation[-1] = max(0.05, allocation[-1])  # Ensure at least 5% remains in cash

        # Step 3: Get current market prices for all pairs
        market_prices = {f"pair_{i}": self._get_market_data(f"pair_{i}")[0] for i in range(self.num_pairs)}

        # Step 4: Process each trading pair
        for i in range(self.num_pairs):
            pair = f"pair_{i}"
            new_allocation = allocation[i] * total_capital  # Capital to allocate
            current_position = self.portfolio.positions.get(pair, None)
            direction = np.sign(position_adjustment[i])  # 1 = Long, -1 = Short, 0 = No change
            current_price = market_prices[pair]

            # Trading fee (applied to all executed trades)
            trading_fee = self.transaction_fee

            # Step 4a: Open a new position if none exists
            if current_position is None and new_allocation > 0:
                fee_cost = new_allocation * trading_fee
                if self.portfolio.cash_reserve >= (new_allocation + fee_cost + min_cash_reserve):
                    self.portfolio.open_position(
                        pair=pair,
                        entry_price=current_price,
                        direction=direction,
                        size=new_allocation
                    )
                    self.portfolio.cash_reserve -= fee_cost  # Deduct trading fee

            # Step 4b: Adjust existing positions
            elif current_position:
                adjustment_size = abs(position_adjustment[i]) * current_position.size  # Scale adjustment
                fee_cost = adjustment_size * trading_fee

                if position_adjustment[i] > 0 and self.portfolio.cash_reserve >= (adjustment_size + fee_cost + min_cash_reserve):
                    # Increase position
                    self.portfolio.modify_position(
                        pair, action="increase", size=adjustment_size, new_price=current_price
                    )
                    self.portfolio.cash_reserve -= fee_cost  # Deduct trading fee
                
                elif position_adjustment[i] < 0:
                    # Decrease position
                    self.portfolio.modify_position(
                        pair, action="decrease", size=adjustment_size, new_price=current_price
                    )
                    self.portfolio.cash_reserve -= fee_cost  # Deduct trading fee

                # Step 4c: Fully close position if `position_adjustment[i] == -1`
                if position_adjustment[i] == -1:
                    close_size = current_position.size
                    fee_cost = close_size * trading_fee

                    self.portfolio.modify_position(
                        pair, action="close", size=close_size, new_price=current_price
                    )
                    self.portfolio.cash_reserve -= fee_cost  # Deduct trading fee

        # Step 5: Update cash reserve for remaining unallocated capital
        self.portfolio.cash_reserve = max(allocation[-1] * total_capital, min_cash_reserve)  # Ensure at least 5% remains

    def observe(self, agent):
        """
        Returns the observation for the given agent, including action masks.
        """

        if agent == "meta":
            action_mask = self._get_action_mask()  # Get valid actions at this step

            return {
                # Specialist agents' signals (already summarized indicators)
                "specialist_signals": np.array([
                    self._stored_actions.get(f"pair_{i}", 0)  # Default to 0 if no action
                    for i in range(self.num_pairs)
                ], dtype=np.float32),

                # Open positions: (Direction, Size, PnL) per pair
                "open_positions": np.array([
                    (
                        self.portfolio.positions[f"pair_{i}"].direction,
                        self.portfolio.positions[f"pair_{i}"].size,
                        self.portfolio.positions[f"pair_{i}"].calculate_pnl(self._get_market_data(f"pair_{i}")[0])
                    ) if f"pair_{i}" in self.portfolio.positions else (0, 0.0, 0.0)
                    for i in range(self.num_pairs)
                ], dtype=np.float32).flatten(),

                # Cash reserves
                "cash_reserve": np.array([self.portfolio.cash_reserve], dtype=np.float32),

                # Action masks
                "action_mask": action_mask  # Ensures invalid actions are not chosen
            }

        elif agent.startswith("pair_"):
            pair_index = int(agent.split("_")[1])
            return {
                "market_data": self._get_market_data(agent),  # Market indicators for the pair
            }

        else:
            raise ValueError(f"Unknown agent: {agent}")


    def step(self, action):
        """
        Executes a step in the environment for the current agent.
        - Pair agents receive immediate hypothetical rewards.
        - Meta agent executes trades and gets realized rewards.
        """
        current_agent = self.agent_selection
        self._stored_actions[current_agent] = action  # Store action

        # Handle Pair Agents
        if current_agent.startswith("pair_"):
            self._calculate_pair_agent_rewards(current_agent)  # Assign reward

            # Move to next agent
            self.agent_selection = self._agent_selector.next()
            return self.observe(self.agent_selection), self._rewards[current_agent], self.terminations[current_agent], self.truncations[current_agent], {}

        # Handle Meta-Agent
        elif current_agent == "meta":
            self._execute_trades(action)  # Execute trades based on decision
            self._update_return_rates()  # Update PnL & market-based calculations
            self._calculate_meta_agent_rewards()  # Assign reward for realized trades

            # Move time forward
            self.current_step += 1

            # Check for episode termination
            if self.current_step >= self.state_history - 1:
                self.terminations = {agent: True for agent in self.possible_agents}
                self.truncations = {agent: True for agent in self.possible_agents}

            # Reset agent selection for the next cycle
            self.agent_selection = self._agent_selector.reset()

            return self.observe(self.agent_selection), self._rewards[current_agent], self.terminations[current_agent], self.truncations[current_agent], {}


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
        self.current_allocations[-1] = 1.0    # 100% allocation in cash
        
        self._stored_actions = {}  
        self.open_positions = np.zeros(self.num_pairs * 3, dtype=np.float32) 

        self.cash_reserve = np.array([self.budget], dtype=np.float32)
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
        Updates return rates and PnL for all open positions.
        Ensures that each position’s unrealized PnL is based on the latest market prices.
        """
        market_prices = {f"pair_{i}": self._get_market_data(f"pair_{i}")[0] for i in range(self.num_pairs)}

        for pair, position in self.portfolio.positions.items():
            if position.status == "open":
                current_price = market_prices[pair]
                position.pnl = position.calculate_pnl(current_price)  # Update only PnL

    def _get_market_data(self, agent_id):
        """
        Returns the latest market data snapshot for the current timestep for a specific trading pair.
        """
        if agent_id.startswith("pair_"):
            pair_index = int(agent_id.split("_")[1])  # Extract the pair index
            return self.market_data[self.current_step, pair_index, :]  # Fetch only relevant pair's data
        else:
            raise ValueError(f"Market data requested for unknown agent: {agent_id}")


    def _calculate_pair_agent_rewards(self):
        """
        Computes rewards for pair agents based on:
        - Immediate hypothetical profitability of the last signal.
        - Long-term hypothetical profitability (exponential decay over 10 steps).
        - Flip-flopping penalty if the agent switches direction too often.
        """

        for pair_id in range(self.num_pairs):
            agent_id = f"pair_{pair_id}"
            signal = self._stored_actions.get(agent_id, 0)  # Default to 0 if no action taken
            confidence = abs(signal)  # Confidence level (how strong the signal was)

            # Store signal in rolling buffer (last 10 signals)
            self.signal_history[agent_id].append({"timestamp": self.current_step, "signal": signal})

            # Ensure we don't exceed the last 10 steps
            if len(self.signal_history[agent_id]) > 10:
                old_signal = self.signal_history[agent_id].pop(0)  # Remove oldest signal

            # 1. Compute Immediate Hypothetical Reward ---
            past_price = self.market_data[self.current_step - 1, pair_id, 0] if self.current_step > 0 else None
            current_price = self.market_data[self.current_step, pair_id, 0]

            if past_price is not None:
                # Compute hypothetical return if the signal had been executed exactly
                expected_return = (current_price - past_price) * np.sign(signal)  # If Buy, profit from increase; if Sell, profit from decrease
                expected_pnl = expected_return - (2 * self.transaction_fee)  # Subtract open + close transaction fees

                # Apply confidence weighting
                short_term_reward = expected_pnl * confidence
            else:
                short_term_reward = 0  # No reward on first step

            # 2. Compute Long-Term Hypothetical Reward ---
            long_term_reward = 0
            if len(self.signal_history[agent_id]) == 10:
                old_signal = self.signal_history[agent_id][0]  # Signal from 10 steps ago
                old_price = self.market_data[self.current_step - 10, pair_id, 0]

                if old_price is not None:
                    # Compute the return if the agent had held that position for 10 steps
                    long_term_pnl = (current_price - old_price) * np.sign(old_signal["signal"])
                    long_term_pnl -= (2 * self.transaction_fee)  # Subtract transaction costs
                    long_term_reward = long_term_pnl * confidence * np.exp(-0.9 * 10)  # Apply exponential decay

            # 3. Compute Flip-Flopping Penalty ---
            flip_flop_penalty = 0
            flip_flop_count = sum(
                np.sign(self.signal_history[agent_id][i]["signal"]) != np.sign(self.signal_history[agent_id][i - 1]["signal"])
                for i in range(1, len(self.signal_history[agent_id]))
            )
            if flip_flop_count > 2:
                flip_flop_penalty = 2 * self.transaction_fee * confidence  # Higher penalty for confident bad signals

            # Final Reward Assignment ---
            self._rewards[agent_id] = (
                0.2 * short_term_reward +  # Short-term impact
                0.3 * long_term_reward -  # Long-term impact
                flip_flop_penalty  # Penalize erratic signals
            )

    def _calculate_meta_agent_reward(self):
        """
        Computes the reward for the meta-agent based on:
        - Realized PnL (3x reward multiplier).
        - Unrealized PnL change (continuous reward).
        - Position modification impact (PnL change due to increasing/decreasing size).
        - Avoiding excessive holding (penalty after 20 steps).
        """

        reward = 0
        market_prices = {pair: self._get_market_data(pair)[0] for pair in self.portfolio.positions.keys()}

        for pair, position in self.portfolio.positions.items():
            if position.status == "closed":
                # 1. Reward for Realized PnL (3x multiplier)
                realized_pnl = position.pnl
                reward += 3 * realized_pnl

                # 2. Evaluate Position Modifications Using History
                modification_rewards = []
                for i in range(1, len(position.history)):
                    action, size, price = position.history[i][1], position.history[i][2], position.history[i][3]
                    if action in ["increase", "decrease"]:
                        pnl_before_mod = position.calculate_pnl(position.history[i - 1][3])  # PnL before modification
                        pnl_after_mod = position.calculate_pnl(price)  # PnL after modification

                        mod_reward = pnl_after_mod - pnl_before_mod
                        modification_rewards.append(mod_reward)

                reward += sum(modification_rewards)  # Total modification-based reward

            else:
                # 3. Reward/Penalty for Unrealized PnL Change
                unrealized_pnl_now = position.calculate_pnl(market_prices[pair])
                unrealized_pnl_change = unrealized_pnl_now - position.pnl  # Change in unrealized profit
                reward += 0.5 * unrealized_pnl_change  # Reward for holding profitable trades

                # 4. Penalize Holding Losing Trades Beyond 20 Steps**
                if self.current_step - position.timestamp > 20:
                    reward -= 0.2 * abs(position.pnl)  # Holding penalty

        # Subtract Trading Fees
        reward -= sum(self.transaction_fee for _ in self.portfolio.positions)

        # Assign reward
        self._rewards["meta"] = reward


    def close(self):
        pass

    # Required properties
    @property
    def rewards(self):
        return self._rewards

    @property
    def observations(self):
        return {agent: self.observe(agent) for agent in self.possible_agents}


