class Portfolio:
    def __init__(self, initial_budget=100000):
        self.budget = initial_budget  # Cash available for trading
        self.positions = {}  # Open positions: {pair_name: {amount, direction, entry_price}}
        self.history = []  # Track portfolio value over time

    def reset(self):
        self.budget = 100000
        self.positions = {}
        self.history = []

    def open_position(self, pair_name, amount, direction, entry_price):
        if amount > self.budget:
            raise ValueError("Not enough budget to open the position")
        self.budget -= amount
        if pair_name not in self.positions:
            self.positions[pair_name] = {"amount": 0, "direction": direction, "entry_price": entry_price}
        self.positions[pair_name]["amount"] += amount

    def close_position(self, pair_name, exit_price, adjustment=None):
        if pair_name not in self.positions:
            raise ValueError(f"No open position for {pair_name}")

        position = self.positions[pair_name]
        close_amount = adjustment if adjustment else position["amount"]
        profit = 0

        if position["direction"] == "long":
            profit = close_amount * (exit_price / position["entry_price"] - 1)
        elif position["direction"] == "short":
            profit = close_amount * (position["entry_price"] / exit_price - 1)

        self.budget += close_amount + profit
        position["amount"] -= close_amount

        if position["amount"] <= 0:
            del self.positions[pair_name]

        return profit

    def get_allocation(self, pair_name):
        """Calculate the current allocation percentage for a pair."""
        if pair_name in self.positions:
            total_allocated = sum(pos["amount"] for pos in self.positions.values())
            return self.positions[pair_name]["amount"] / total_allocated
        return 0.0

    def total_value(self):
        total = self.budget
        for pair_name, position in self.positions.items():
            current_price = trading_pairs[pair_name].price
            if position["direction"] == "long":
                total += position["amount"] * (current_price / position["entry_price"])
            elif position["direction"] == "short":
                total += position["amount"] * (position["entry_price"] / current_price)
        return total

# The following block will only execute when this file is run directly
if __name__ == "__main__":
    # Test the Portfolio class
    portfolio = Portfolio(initial_budget=50000)
    print(f"Testing Portfolio: {portfolio}")