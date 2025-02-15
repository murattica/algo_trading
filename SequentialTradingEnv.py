from pettingzoo.utils.env import AECEnv
from pettingzoo.utils import agent_selector
import numpy as np
from gymnasium import spaces
from Portfolio import Portfolio


class SequentialTradingEnv(AECEnv):
    metadata = {"render_modes": ["human"], "name": "sequential_trading_v1"}

    def __init__(self, market_data, symbol_index_map, budget=100.0, risk_free_rate=0.001, render_mode="human"):
        super().__init__()
        if not isinstance(market_data, np.ndarray):
            raise ValueError("market_data must be a NumPy array")

        # --- Environment Parameters ---
        self.market_data = market_data
        self.index_to_symbol = {value: key for key, value in symbol_index_map.items()}
        self.state_history, self.num_pairs, self.num_features = market_data.shape
        self.risk_free_rate = risk_free_rate
        self.render_mode = render_mode
        self.budget = budget
        self.portfolio = Portfolio(initial_cash=self.budget)
        self.transaction_fee = 0.001
        self.current_step = 0
        self._stored_actions = {}
        self.signal_history = {f"pair_{i}": [] for i in range(self.num_pairs)}
        self._is_pair_disabled = {f"pair_{i}": False for i in range(self.num_pairs)}

        # --- Agent Setup ---
        self.possible_agents = [f"pair_{i}" for i in range(self.num_pairs)] + ["meta"]
        self.agents = self.possible_agents.copy()
        self._agent_selector = agent_selector(self.possible_agents)

        # --- State Tracking ---
        self._rewards = {agent: 0 for agent in self.possible_agents}
        self._cumulative_rewards = {agent: 0.0 for agent in self.possible_agents}
        self.terminations = {agent: False for agent in self.possible_agents}
        self.truncations = {agent: False for agent in self.possible_agents}
        self.infos = {agent: {} for agent in self.possible_agents}

        # --- Observation Spaces ---
        self.observation_spaces = spaces.Dict({
            **{
                f"pair_{i}": spaces.Dict({
                    "market_data": spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_features,), dtype=np.float32),
                }) for i in range(self.num_pairs)
            },
            "meta": spaces.Dict({
                "specialist_signals": spaces.Box(low=-1.0, high=1.0, shape=(self.num_pairs,), dtype=np.float32),
                "position_pnl": spaces.Box(low=-1.0, high=1.0, shape=(self.num_pairs, 2), dtype=np.float32),
                "position_size": spaces.Box(low=0, high=np.inf, shape=(self.num_pairs,), dtype=np.float32),
                "cash_reserve": spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32),
                "action_mask": spaces.Box(low=0, high=1, shape=(self.num_pairs + 1 + self.num_pairs,), dtype=np.int32)
            })
        })

        # --- Action Spaces ---
        self.action_spaces = {
            **{
                f"pair_{i}": spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
                for i in range(self.num_pairs)
            },
            "meta": spaces.Box(
                low=np.concatenate([np.zeros(self.num_pairs + 1), -np.ones(self.num_pairs)]),  # [0 to 1 for allocation, -1 to 1 for position adj]
                high=np.concatenate([np.ones(self.num_pairs + 1), np.ones(self.num_pairs)]),   # [1 for allocation, 1 for position adj]
                shape=(self.num_pairs + 1 + self.num_pairs,),  # Total shape: (num_pairs + 1 + num_pairs,)
                dtype=np.float32
            )
        }   

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    # --- Core Functions ---

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

    def reset(self, seed=None, options=None):
            print("reset initiated")
            self.current_step = 0
            self.portfolio_value = self.budget
            self.current_allocations = np.zeros(self.num_pairs + 1)
            self.current_allocations[-1] = 1.0    # 100% allocation in cash
            
            self._stored_actions = {}  
            self.open_positions = np.zeros(self.num_pairs * 3, dtype=np.float32) 

            self._step_counts = {}  # Track execution counts per agent
            self._rewards = {agent: 0.0 for agent in self.possible_agents}
            self._terminations = {agent: False for agent in self.possible_agents}
            self._truncations = {agent: False for agent in self.possible_agents}
            self._infos = {agent: {} for agent in self.possible_agents}

            self.agent_selection = self._agent_selector.reset()
            self.agents = self.possible_agents.copy()

            return self._infos

    def observe(self, agent):
        """
        Returns the observation for the given agent, ensuring it follows the defined observation space.
        """

        if agent == "meta":
            action_mask = self._get_action_mask()  # Get valid actions at this step

            obs = {
                # Specialist agents' signals (already summarized indicators)
                "specialist_signals": np.array([
                    self._stored_actions.get(f"pair_{i}", 0)  # Default to 0 if no action
                    for i in range(self.num_pairs)
                ], dtype=np.float32).flatten(),  # Ensure correct shape (num_pairs,),

                # Position PnL: (entry_price, unrealized_pnl) per pair
                "position_pnl": np.array([
                    (
                        self.portfolio.positions[f"pair_{i}"].get_avg_entry_price(),
                        self.portfolio.positions[f"pair_{i}"].calculate_pnl(self._get_market_data(f"pair_{i}")[0])
                    ) if f"pair_{i}" in self.portfolio.positions else (0.0, 0.0)
                    for i in range(self.num_pairs)
                ], dtype=np.float32),

                # Position Size: Position sizes for each pair
                "position_size": np.array([
                    self.portfolio.positions[f"pair_{i}"].valuation
                    if f"pair_{i}" in self.portfolio.positions else 0.0
                    for i in range(self.num_pairs)
                ], dtype=np.float32),

                # Cash reserves
                "cash_reserve": np.array([self.portfolio.cash_reserve], dtype=np.float32),

                # Action masks
                "action_mask": action_mask  # Ensures invalid actions are not chosen
            }

            return obs

        elif agent.startswith("pair_"):
            pair_index = int(agent.split("_")[1])
            return {
                "market_data": self._get_market_data(agent),  # Market indicators for the pair
            }

        else:
            raise ValueError(f"Unknown agent: {agent}")

    def _execute_trades(self, meta_action):
        """
        Executes trades based on the meta-agent's allocation and position adjustment decisions.
        - Ensures at least 5% of total capital remains in cash reserves.
        - Applies transaction fees to all executed trades.
        """

        num_pairs = self.num_pairs
        allocations = np.array(meta_action[: num_pairs + 1])  # Allocations for assets + cash
        position_adjustment = np.array(meta_action[num_pairs + 1:])  # Position adjustments (-1 to 1)

        # Step 1: Get current total capital
        total_capital = self.portfolio.cash_reserve + sum(pos.valuation for pos in self.portfolio.positions.values())
        min_cash_reserve_ratio = 0.05  
        min_cash_reserve = total_capital * min_cash_reserve_ratio  # Ensure 5% minimum cash

        # Step 2: Normalize allocations to sum ≤ 1 using Softmax
        allocation_sum = allocations.sum()
        if allocation_sum > 0:
            allocations = allocations / allocation_sum
        else:
            allocations[-1] = 1.0  # If all allocations are zero, put everything in cash

        # Step 3: Ensure at least 5% cash reserve
        if allocations[-1] < min_cash_reserve_ratio:
            # Calculate the shortfall
            shortfall = min_cash_reserve_ratio - allocations[-1]

            # Find the index of the largest allocated position (excluding cash)
            largest_alloc_index = np.argmax(allocations[:-1])

            # Reduce the largest allocation to make room for cash reserve
            if allocations[largest_alloc_index] > shortfall:
                allocations[largest_alloc_index] -= shortfall
                allocations[-1] = min_cash_reserve  # Set cash reserve to 5%

        # Step 4: Get current market prices
        market_prices = {f"pair_{i}": self._get_market_data(f"pair_{i}")[0] for i in range(num_pairs)}
        
        # Step 5: Process each trading pair
        for i in range(num_pairs):
            pair = f"pair_{i}"
            new_allocation = allocations[i] * total_capital  # Capital allocated for this pair
            current_position = self.portfolio.positions.get(pair, None)
            direction = np.sign(position_adjustment[i])  # 1 = Long, -1 = Short, 0 = No change
            current_price = market_prices[pair]

            # Step 5a: Open a new position if none exists
            if current_position is None and new_allocation > 0:
                    self.portfolio.open_position(
                        pair=pair,
                        symbol=self.index_to_symbol[i],  # Map to symbol
                        entry_price=current_price,
                        direction=direction,
                        size=new_allocation,
                        timestamp=self.current_step,
                        agent="meta",
                    )

            # Step 5b: Adjust existing positions
            elif current_position:
                adjustment_amount = abs(position_adjustment[i]) * current_position.valuation  # Scale adjustment
                if position_adjustment[i] > 0:
                    # Increase position
                    self.portfolio.modify_position(
                        pair, action="increase", timestamp=self.current_step, size=adjustment_amount, new_price=current_price
                    )

                elif position_adjustment[i] < 0:
                    # Ensure size does not go negative
                    adjustment_amount = min(adjustment_amount, current_position.valuation)

                    # Decrease position
                    self.portfolio.modify_position(
                        pair, action="decrease", timestamp=self.current_step, size=adjustment_amount, new_price=current_price
                    )

                    # Step 5c: Fully close position if `position_adjustment[i] == -1`
                    if position_adjustment[i] == -1 and current_position.valuation > 0:
                        close_amount = current_position.valuation

                        self.portfolio.modify_position(
                            pair, action="close", timestamp=self.current_step, size=close_amount, new_price=current_price
                        )

    def _get_action_mask(self):
        """
        0 for forbidden, 1 for allowed

        Generates action masks for the meta-agent to prevent invalid allocations and position adjustments.
        Pair agents do not need action masks as they only generate signals.

        Enforces a rule where at least 10% of the total capital must remain in cash.
        If cash reserves drop below this level, the meta-agent must reduce at least one position.
        """

        # Initialize action masks
        allocation_mask = np.ones(self.num_pairs + 1, dtype=np.int32)  # Allows allocation to cash + all pairs
        position_adjustment_mask = np.zeros(self.num_pairs, dtype=np.int32)  # Default: All position adjustments allowed

        market_prices = {f"pair_{i}": self._get_market_data(f"pair_{i}")[0] for i in range(self.num_pairs)}

        total_portfolio_value = self.portfolio.get_portfolio_value(market_prices)  # Total available capital (cash + positions)
        cash_reserve = self.portfolio.cash_reserve
        min_cash_reserve = 0.1 * total_portfolio_value  # 10% cash requirement

        # Prevent reducing cash allocation below the minimum required level
        if cash_reserve <= min_cash_reserve:
            allocation_mask[-1] = 0  # Prevent further allocation away from cash

            # If cash is too low, enforce reducing at least one position
            must_reduce_position = True
        else:
            must_reduce_position = False

        reduction_possible = False  # Track if any position can be reduced

        for i in range(self.num_pairs):
            position = self.portfolio.positions.get(f"pair_{i}", None)

            # Ensure position adjustments only happen if a position exists
            if not position:
                position_adjustment_mask[i] = 0  # Cannot adjust positions that do not exist
            else:
                # Prevent increasing positions if cash reserves are too low
                if cash_reserve <= min_cash_reserve:
                    position_adjustment_mask[i] = -1  # Only reductions allowed
                    reduction_possible = True

        # If no positions can be reduced but cash is too low, allow at least one reduction
        if must_reduce_position and not reduction_possible:
            # Find the largest position and allow its reduction
            largest_position_id = max(
                self.portfolio.positions.keys(), key=lambda pid: self.portfolio.positions[pid]["size"], default=None
            )
            if largest_position_id:
                pair_index = int(largest_position_id.split("_")[1])
                position_adjustment_mask[pair_index] = -1  # Allow only reduction

        return np.concatenate([allocation_mask, position_adjustment_mask])


    # --- Reward Calculation ---
    def _calculate_pair_agent_rewards(self, agent_id):
        """
        Computes the reward for a single pair agent based on:
        - Immediate hypothetical profitability of the last signal.
        - Long-term hypothetical profitability (exponential decay over 10 steps).
        - Flip-flopping penalty if the agent switches direction too often.

        This function is called in `step()`, where `agent_id` is already selected.
        """

        # Extract pair index from agent_id
        pair_id = int(agent_id.split("_")[1])

        # Retrieve the last action taken by the agent (default to 0 if no action)
        signal = self._stored_actions.get(agent_id, 0)
        confidence = abs(signal)  # Confidence level (absolute value of signal)

        # Store signal in rolling buffer (last 10 signals)
        self.signal_history[agent_id].append({"timestamp": self.current_step, "signal": signal})

        # Ensure we don't exceed the last 10 steps
        if len(self.signal_history[agent_id]) > 10:
            self.signal_history[agent_id].pop(0)  # Remove oldest signal

        # 1. Compute Immediate Hypothetical Reward ---
        past_price = self.market_data[self.current_step - 1, pair_id, 0] if self.current_step > 0 else None
        current_price = self.market_data[self.current_step, pair_id, 0]

        if past_price is not None:
            # Compute hypothetical return if the signal had been executed exactly
            expected_return = (current_price - past_price) * np.sign(signal)  # Profit from Buy/Sell
            expected_pnl = expected_return - (2 * self.transaction_fee)  # Subtract transaction fees

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
                # Compute return if the agent had held the position for 10 steps
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


    def _calculate_meta_agent_rewards(self):
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

    # --- Market Data & Portfolio Helpers ---
    def _get_market_data(self, agent_id):
        """
        Returns the latest market data snapshot for the current timestep for a specific trading pair.
        """
        if agent_id.startswith("pair_"):
            pair_index = int(agent_id.split("_")[1])  # Extract the pair index
            return self.market_data[self.current_step, pair_index, :]  # Fetch only relevant pair's data
        else:
            raise ValueError(f"Market data requested for unknown agent: {agent_id}")

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

    # --- Rendering & Misc ---
    def render(self, mode="human"):
        if mode == "human":
            print(f"\nStep {self.current_step}")
            print(f"Portfolio Value: ${self.portfolio_value:.2f}")
            print("Current Allocations:")
            for i in range(self.num_pairs):
                allocation = self.current_allocations[i]
                print(f"- Pair {i}: {allocation:.2%} allocation")

    def close(self):
        pass

    @property
    def rewards(self):
        return self._rewards

    @property
    def observations(self):
        return {agent: self.observe(agent) for agent in self.possible_agents}