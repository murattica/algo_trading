from datetime import datetime
import json
from db import insert_query
from Position import Position 
import numpy as np

class Portfolio:
    def __init__(self, initial_cash, transaction_fee = 0.001):
        """
        Initialize a portfolio with a starting cash balance.
        """
        self.cash_reserve = initial_cash  # Start with all cash
        self.positions = {}  # Dictionary to store active `Position` objects
        self.transaction_fee = transaction_fee

    def open_position(self, pair: str, symbol: str, entry_price: float, direction: str, size: float, timestamp, agent: str) -> bool:
        """
        Open a new position if enough cash is available, accounting for transaction fees.
        """
        total_cost = size * (1 + self.transaction_fee)  # Include the transaction fee

        if self.cash_reserve < total_cost:
            print(f"Not enough cash to open position on {pair}")
            return False

        if pair in self.positions and self.positions[pair].status == "open":
            print(f"Position already open on {pair}, use modify_position() instead")
            return False

        # Deduct from cash reserve
        self.cash_reserve -= total_cost

        print(f"Opening position on {pair} with size {size}")
        # Create and store a Position object
        position = Position(pair, symbol, entry_price, direction, timestamp, size, agent, self.transaction_fee)
        self.positions[pair] = position

        return True


    def modify_position(self, pair, action, size, new_price, timestamp):
        """
        Modify an existing position by increasing, decreasing, or closing it.
        Logs every action in the database.
        """
        if pair not in self.positions or self.positions[pair].status == "closed":
            print(f"No open position on {pair} to modify")
            return False

        position = self.positions[pair]

        if action == "increase":
            fee = size * (1 + self.transaction_fee)
            if self.cash_reserve < fee:
                print(f"Not enough cash to increase position on {pair}")
                return False
            position.update_position("increase", size, new_price, timestamp)
            self.cash_reserve -= fee  # Deduct including fee


        elif action == "decrease":
            fee = size * (1 + self.transaction_fee)
            if self.cash_reserve < fee:
                print(f"Not enough cash to increase position on {pair}")
                return False
            position.update_position("decrease", size, new_price, timestamp)
            self.cash_reserve += position.amount - fee 

        elif action == "close":
            fee = size * (1 + self.transaction_fee) 
            if self.cash_reserve < fee:
                print(f"Not enough cash to increase position on {pair}")
                return False
            position.update_position("close", size, new_price, timestamp)
            self.cash_reserve -=  fee

        return True


    def calculate_total_pnl(self, market_prices):
        """
        Calculate the total portfolio value (cash + PnL of open positions).
        """
        total_pnl = sum(
            position.calculate_pnl(market_prices[pair])
            for pair, position in self.positions.items()
            if position.status == "open"
        )
        return total_pnl

    def get_portfolio_value(self, market_prices):
        """
        Return the total portfolio value (cash + unrealized PnL).
        """
        return self.cash_reserve + self.calculate_total_pnl(market_prices)

    def get_open_position_pnls(self, market_prices):
        """
        Returns a dictionary of open position PnLs.
        """
        return {
            pair: position.calculate_pnl(market_prices[pair])
            for pair, position in self.positions.items()
            if position.status == "open"
        }

    def print_portfolio(self, market_prices):
        """
        Print portfolio summary.
        """
        total_value = self.get_portfolio_value(market_prices)
        position_pnls = self.get_open_position_pnls(market_prices)

        print(f"Cash Reserve: {self.cash_reserve:.2f}")
        print(f"Portfolio PnL: {self.calculate_total_pnl(market_prices):.2f}")
        print(f"Portfolio Value: {total_value:.2f}")
        print("\nOpen Positions PnLs:")
        for pair, pnl in position_pnls.items():
            print(f"   {pair}: {pnl:.2f}")

        print("\nOpen Positions:")
        for pair, position in self.positions.items():
            if position.status == "open":
                print(f"   {pair} | Size: {position.amount:.2f} | PnL: {position.pnl:.2f}")
                print(f"   Entry Points: {position.entry_points}")


    def get_historical_pnl(self, pair, steps_ago=10):
        """
        Returns the PnL of a position from a given number of steps ago.
        If the position was not open back then, return 0.
        """
        position = self.positions.get(pair)
        if not position or len(position.history) < steps_ago:
            return 0.0  # No history available

        historical_price = position.history[-steps_ago][3]  # Get price at `steps_ago`
        return position.calculate_pnl(historical_price)


    def evaluate_hypothetical_trade(self, pair, current_price):
        """
        Estimates the profit/loss if the specialist's recommendation had been executed
        without the meta-agent overriding it.
        """
        position = self.positions.get(pair)  
        if not position:
            return 0.0  # No position to evaluate

        avg_entry_price = position.get_avg_entry_price()
        return (current_price - avg_entry_price) * position.amount * position.direction


    def evaluate_contribution(self, pair, position, meta_action):
        """
        Evaluates how well a specialist's recommendation aligns with the meta-agent's decision.
        - If the specialist recommended an action, but the meta ignored it, low reward.
        - If the specialist was aligned with the meta-agent and resulted in profit, high reward.
        - The closer the meta-agent follows the confidence level, the higher the reward.
        """

        if pair not in self.positions:
            return 0.0  # No stored action for this pair

        stored_action = self.positions[pair].stored_action  # Ensure `stored_action` exists
        if stored_action is None:
            return 0.0  # No recommendation was made

        action, direction, confidence = stored_action  # Unpacking correctly
        meta_allocation = meta_action[self.get_position_index(pair)]  # Meta-agent's allocation decision

        # Determine Sign Matching**  
        # Meta should follow buy/sell direction. If aligned, sign_matching = 1, else -1
        sign_matching = 1 if (
            (direction == 1 and meta_allocation > 0.5) or (direction == -1 and meta_allocation < 0.5)
        ) else -1

        # Compute Alignment Score**  
        # The closer the meta decision is to the specialist's confidence, the higher the alignment score.
        alignment_score = (1 - abs(meta_allocation - confidence)) * sign_matching

        # Compute Actual PnL of the Position**
        actual_pnl = position.calculate_pnl(meta_action[self.get_position_index(pair)]) if position else 0.0

        # Compute Final Contribution**
        return alignment_score * actual_pnl


    def get_position_index(self, pair):
        """
        Retrieves the index of a trading pair from the portfolio's stored positions.
        Useful for accessing corresponding meta-agent actions.
        """
        return list(self.positions.keys()).index(pair) if pair in self.positions else -1
