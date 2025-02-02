from datetime import datetime
import json
from db import insert_query
from Position import Position 

class Portfolio:
    def __init__(self, initial_cash):
        """
        Initialize a portfolio with a starting cash balance.
        """
        self.cash_reserve = initial_cash  # Start with all cash
        self.positions = {}  # Dictionary to store active `Position` objects

    def open_position(self, pair, entry_price, direction, size, timestamp, agent):
        """
        Open a new position if enough cash is available.
        Logs action in the database.
        """
        if self.cash_reserve < size:
            print(f"Not enough cash to open position on {pair}")
            return False

        if pair in self.positions and self.positions[pair].status == "open":
            print(f"Position already open on {pair}, use modify_position() instead")
            return False

        # Deduct from cash reserve
        self.cash_reserve -= size

        # Create and store a Position object
        position = Position(pair, entry_price, direction, timestamp, size, agent)
        self.positions[pair] = position

        # Log the action in the database
        self.log_action(position.id, timestamp, agent, pair, direction, "open", size, {"entry_point": entry_price})

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
            if self.cash_reserve < size:
                print(f"Not enough cash to increase position on {pair}")
                return False
            position.update_position("increase", size, new_price, timestamp)
            self.cash_reserve -= size  # Deduct from cash
            self.log_action(position.id, timestamp, position.agent, pair, position.direction, "increase", size, {
                "old_size": position.size - size,
                "entry_point": new_price
            })

        elif action == "decrease":
            position.update_position("decrease", size, new_price, timestamp)
            self.cash_reserve += size  # Return cash from reduced position
            self.log_action(position.id, timestamp, position.agent, pair, position.direction, "decrease", size, {
                "old_size": position.size + size,
                "entry_point": new_price
            })

        elif action == "close":
            position.update_position("close", size, new_price, timestamp)
            self.cash_reserve += position.size  # Return full position size to cash

            # Log closing position
            self.log_action(position.id, timestamp, position.agent, pair, position.direction, "close", position.size, {
                "avg_entry_point": position.get_avg_entry_price(),
                "exit_point": new_price,
                "pnl": position.pnl
            })

            # Remove closed position from portfolio
            del self.positions[pair]

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
                print(f"   {pair} | Size: {position.size:.2f} | PnL: {position.pnl:.2f}")
                print(f"   Entry Points: {position.entry_points}")

    def log_action(self, position_id, step_number, agent, pair, direction, action, size, properties):
        """
        Logs the action to the `order_history` table using `insert_query()`.
        """
        query = """
        INSERT INTO order_history (position_id, step_number, timestamp, agent, pair, direction, action, size, properties)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s);
        """
        values = (
            position_id,
            step_number,
            datetime.utcnow(),
            agent,
            pair,
            direction,
            action,
            size,
            json.dumps(properties)  # Convert dictionary to JSON
        )

        success = insert_query(query, values)
        if not success:
            print(f"Failed to log action {action} for {pair}")
