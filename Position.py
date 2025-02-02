import uuid
import json
from datetime import datetime
from db import insert_query  # Use your existing insert function

class Position:
    def __init__(self, pair, entry_point, direction, timestamp, size, agent):
        self.id = str(uuid.uuid4())  # Unique Position ID
        self.pair = pair  # Trading pair (e.g., BTCUSDT)
        self.entry_points = [(entry_point, size)]  # Store entry prices & size
        self.direction = direction  # "long" or "short"
        self.timestamp = timestamp  # Step number when position was opened
        self.size = size  # Total position size
        self.status = "open"  # "open" or "closed"
        self.history = [(timestamp, "open", size, entry_point)]  # Log of all changes
        self.pnl = 0.0  # Profit & Loss
        self.agent = agent  # The agent who opened the position

        # Log position opening
        self.log_action("open", size, {"entry_point": entry_point})

    def update_position(self, action, size, new_price, timestamp):
        """
        Modify position: Increase, Decrease, or Close.
        """
        if action == "increase":
            old_size = self.size
            self.entry_points.append((new_price, size))  # Store new entry price
            self.size += size
            self.history.append((timestamp, "increase", size, new_price))

            self.log_action("increase", size, {
                "old_size": old_size, 
                "entry_point": new_price
            })

        elif action == "decrease":
            old_size = self.size
            self.size -= size
            self.history.append((timestamp, "decrease", size, new_price))
            self.log_action("decrease", size, {
                "old_size": old_size, 
                "entry_point": new_price
            })

            if self.size <= 0:  # If size becomes zero, close position
                self.status = "closed"
                self.history.append((timestamp, "closed", 0, new_price))
                self.log_action("close", 0, {
                    "avg_entry_point": self.get_avg_entry_price(),
                    "exit_point": new_price,
                    "pnl": self.pnl
                })

        elif action == "close":
            self.status = "closed"
            self.history.append((timestamp, "closed", self.size, new_price))

            self.log_action("close", self.size, {
                "avg_entry_point": self.get_avg_entry_price(),
                "exit_point": new_price,
                "pnl": self.pnl
            })

            self.size = 0  # Clear size

    def calculate_pnl(self, current_price):
        """
        Calculate Profit & Loss (PnL) based on all entry points.
        """
        if self.status == "closed":
            return self.pnl  # Frozen PnL for closed positions

        # Weighted average entry price
        total_cost = sum(price * size for price, size in self.entry_points)
        total_size = sum(size for _, size in self.entry_points)
        avg_entry_price = total_cost / total_size if total_size > 0 else 0

        # PnL Calculation
        price_change = current_price - avg_entry_price
        multiplier = 1 if self.direction == "long" else -1  # Long gains on increase, short gains on decrease
        self.pnl = multiplier * price_change * self.size
        return self.pnl

    def get_avg_entry_price(self):
        """
        Returns the weighted average entry price.
        """
        total_cost = sum(price * size for price, size in self.entry_points)
        total_size = sum(size for _, size in self.entry_points)
        return total_cost / total_size if total_size > 0 else 0

    def log_action(self, action, size, properties):
        """
        Logs the action to the `order_history` table using `insert_query()`.
        """
        query = """
        INSERT INTO order_history (position_id, step_number, timestamp, agent, pair, direction, action, size, properties)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s);
        """
        values = (
            self.id,  # Position ID
            self.timestamp,
            datetime.utcnow(),
            self.agent,
            self.pair,
            self.direction,
            action,
            size,
            json.dumps(properties)  # Convert dictionary to JSON
        )

        success = insert_query(query, values)
        if not success:
            print(f"⚠️ Failed to log action {action} for {self.pair}")
