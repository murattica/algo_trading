import uuid
import json
from datetime import datetime
from db import insert_query
import numpy as np

class Position:
    """
    Represents a trading position with attributes such as valuation, amount, and transaction history.
    Supports increasing, decreasing, and closing positions while tracking trading fees and PnL.
    """

    def __init__(self, pair, symbol, entry_price, direction, timestamp, size, agent="meta", transaction_fee=0.001):
        """
        Initializes a new position.
        
        Args:
            pair (str): The trading pair identifier.
            symbol (str): Symbol representing the asset (e.g., BTCUSDT).
            entry_price (float): The price at which the position is opened.
            direction (int): 1 for "long", -1 for "short".
            timestamp (int): Time step when the position is opened.
            size (float): Monetary value allocated to the position.
            agent (str): The agent that opened the position.
            transaction_fee (float): Percentage fee for each trade.
        """
        self.id = str(uuid.uuid4())
        self.pair = pair
        self.symbol = symbol
        self.entry_points = [(entry_price, size)]
        self.direction = direction
        self.timestamp = timestamp
        self.valuation = size
        self.amount = size / entry_price
        self.status = "open"
        self.pnl = 0.0
        self.agent = agent
        self.transaction_fee = transaction_fee
        self.stored_action = ("open", direction, size)

        self.pnl_history = [(timestamp, self.pnl)]
        self.history = [(timestamp, "open", self.valuation, self.amount, entry_price, self.pnl)]

        self.trading_fees_paid = size * transaction_fee
        self.valuation -= self.trading_fees_paid

        self.log_action(
            "open",
            self.valuation,
            self.amount,
            float(entry_price),
            self.trading_fees_paid,
            {}
        )

    def update_position(self, action, size, new_price, timestamp):
        """
        Modifies an existing position.

        Args:
            action (str): Action type ("increase", "decrease", "close").
            size (float): Monetary value for the modification.
            new_price (float): Price at which the modification is applied.
            timestamp (int): Current time step.
        """
        if action in ["increase", "decrease", "close"]:
            self.stored_action = (action, self.direction, size)

        if action == "increase":
            buy_amount = size / new_price
            self.entry_points.append((new_price, size))
            self.valuation += size
            self.amount += buy_amount
            trading_fee = size * self.transaction_fee
            self.valuation -= trading_fee
            self.trading_fees_paid += trading_fee
            self.pnl = self.calculate_pnl(new_price)
            self.pnl_history.append((timestamp, self.pnl))

            self.history.append((timestamp, "increase", self.valuation, self.amount, new_price, self.pnl))
            self.log_action(
                "increase",
                self.valuation,
                self.amount,
                float(new_price),
                trading_fee,
                {"buy_amount": float(buy_amount), "avg_entry_point": self.get_avg_entry_price()}
            )

        elif action == "decrease":
            sell_amount = size / new_price
            self.entry_points.append((new_price, -size))
            self.valuation -= size
            self.amount -= sell_amount
            trading_fee = size * self.transaction_fee
            self.valuation -= trading_fee
            self.trading_fees_paid += trading_fee
            self.pnl = self.calculate_pnl(new_price)
            self.pnl_history.append((timestamp, self.pnl))

            self.history.append((timestamp, "decrease", self.valuation, self.amount, new_price, self.pnl))
            self.log_action(
                "decrease",
                self.valuation,
                self.amount,
                float(new_price),
                trading_fee,
                {"sell_amount": float(sell_amount), "avg_entry_point": self.get_avg_entry_price()}
            )

            if self.valuation <= 0:
                self.close_position(new_price, timestamp)

        elif action == "close":
            self.close_position(new_price, timestamp)

    def close_position(self, new_price, timestamp):
        """
        Closes the position and logs the final action.

        Args:
            new_price (float): The price at which the position is closed.
            timestamp (int): Current time step.
        """
        self.status = "closed"
        self.pnl = self.calculate_pnl(new_price)
        self.pnl_history.append((timestamp, self.pnl))
        self.history.append((timestamp, "closed", self.valuation, self.amount, new_price, self.pnl))

        self.log_action(
            "close",
            self.valuation,
            self.amount,
            self.get_avg_entry_price(),
            self.trading_fees_paid,
            {"final_exit_price": float(new_price), "total_pnl": float(self.pnl)}
        )

        self.valuation = 0
        self.amount = 0

    def calculate_pnl(self, current_price):
        """
        Computes profit or loss based on the latest market price.

        Args:
            current_price (float): The latest available price.

        Returns:
            float: The current PnL of the position.
        """
        avg_entry_price = self.get_avg_entry_price()
        if avg_entry_price == 0:
            return 0.0
        return ((current_price - avg_entry_price) / avg_entry_price) * self.valuation * self.direction

    def get_avg_entry_price(self):
        """
        Computes the weighted average entry price.

        Returns:
            float: The average entry price considering active positions.
        """
        total_cost = sum(price * abs(size) for price, size in self.entry_points if size > 0)
        total_size = sum(abs(size) for _, size in self.entry_points if size > 0)
        return total_cost / total_size if total_size > 0 else 0

    def log_action(self, action, valuation, amount, entry_price, trading_fee, properties):
        """
        Logs trading actions to the database.

        Args:
            action (str): Type of action performed.
            valuation (float): Total monetary value of the position.
            amount (float): Quantity of the underlying asset.
            avg_entry_price (float): Weighted average entry price.
            trading_fee (float): Transaction fee paid.
            properties (dict): Additional event-specific data.
        """
        query = """
        INSERT INTO order_history (position_id, timestamp, agent, pair, direction, action, valuation, amount, entry_point, trading_fee, properties)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
        """

        direction_mapping = {1: "long", -1: "short"}
        mapped_direction = direction_mapping.get(int(self.direction), "unknown")

        values = (
            str(self.id),
            datetime.utcnow() if not isinstance(self.timestamp, datetime) else self.timestamp,
            str(self.agent),
            str(self.pair),
            mapped_direction,
            str(action),
            float(valuation),
            float(amount),
            float(entry_price),
            float(trading_fee),
            json.dumps(properties)
        )

        success = insert_query(query, values)
        if not success:
            print(f"Failed to log action {action} for {self.pair}")
