from fetch_data import backfill_for_symbol
from db import init_db
import time

# Top trading pairs
top_pairs = [
    'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'XRP/USDT', 'ADA/USDT',
    'SOL/USDT', 'DOGE/USDT', 'DOT/USDT', 'MATIC/USDT', 'LTC/USDT',
    'SHIB/USDT', 'AVAX/USDT', 'LINK/USDT', 'UNI/USDT', 'TON/USDT',
    'XLM/USDT'
]

def run_service():
    conn = init_db()
    end_time = int(time.time() * 1000)  # Current time in milliseconds

    for symbol in top_pairs:
        print(f"Running service for {symbol}...")
        # Let backfill_for_symbol determine the correct start_time
        backfill_for_symbol(symbol, conn, end_time=end_time)
        print(f"Completed backfill for {symbol}.")
    conn.close()

def manual_backfill(symbol, start_time, end_time):
    conn = init_db()
    backfill_for_symbol(symbol, conn, start_time, end_time)
    conn.close()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Binance Data Backfill Script")
    parser.add_argument("--service", action="store_true", help="Run the backfill service.")
    parser.add_argument("--manual", action="store_true", help="Run manual backfill.")
    parser.add_argument("--symbol", type=str, help="Trading pair for manual backfill.")
    parser.add_argument("--start_time", type=int, help="Start time in milliseconds for manual backfill.")
    parser.add_argument("--end_time", type=int, help="End time in milliseconds for manual backfill.")
    args = parser.parse_args()

    if args.service:
        run_service()
    elif args.manual:
        if not (args.symbol and args.start_time and args.end_time):
            print("For manual backfill, provide --symbol, --start_time, and --end_time.")
        else:
            manual_backfill(args.symbol, args.start_time, args.end_time)
    else:
        print("Please specify --service or --manual.")
