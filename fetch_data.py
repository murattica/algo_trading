import time
from datetime import datetime, timedelta
from db import get_latest_timestamp
import ccxt
import keyring

# Initialize Binance API
api_key = keyring.get_password('binance', 'api_key')
secret_key = keyring.get_password('binance', 'secret_key')
binance = ccxt.binance({'apiKey': api_key, 'secret': secret_key})

# Fetch data in batches
def fetch_and_insert(symbol, start_time, end_time, conn):
    cursor = conn.cursor()
    current_time = start_time

    while current_time < end_time:
        ohlcv = binance.fetch_ohlcv(symbol, timeframe='1m', since=current_time, limit=1000)

        if not ohlcv:
            print(f"No more data to fetch for {symbol}.")
            break

        for entry in ohlcv:
            timestamp = entry[0]  # Epoch time in milliseconds
            dt = datetime.fromtimestamp(timestamp / 1000)  # Convert to datetime
            date = dt.date()  # Extract date
            hour = dt.hour  # Extract hour
            minute = dt.minute  # Extract minute
            open_price = entry[1]
            high_price = entry[2]
            low_price = entry[3]
            close_price = entry[4]
            volume_crypto = entry[5]
            volume_usd = entry[5] * entry[1]

            # Insert the transformed data into the database
            cursor.execute('''
                INSERT INTO ticker_data (symbol, timestamp, date, hour, min, open, high, low, close, volume_crypto, volume_usd)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ''', (symbol, timestamp, date, hour, minute, open_price, high_price, low_price, close_price, volume_crypto, volume_usd))

        conn.commit()
        current_time = ohlcv[-1][0] + 60000  # Move to the next batch
        print(f"Fetched up to {datetime.fromtimestamp(current_time / 1000)} for {symbol}")


def backfill_for_symbol(symbol, conn, start_time=None, end_time=None):
    if start_time is None:
        # Check the latest timestamp in the database
        latest_timestamp = get_latest_timestamp(symbol, conn)
        print(f"DEBUG: Latest timestamp for {symbol} is {latest_timestamp}")  # Debugging

        if latest_timestamp:
            # Start backfill from the next minute after the latest timestamp
            start_time = latest_timestamp + 60000
        else:
            # If no data exists, backfill for the last 2 weeks by default
            start_time = int((datetime.now() - timedelta(weeks=2)).timestamp() * 1000)

    # Default to the current time if no end_time is provided
    if end_time is None:
        end_time = int(time.time() * 1000)

    # Ensure start_time is less than end_time
    if start_time >= end_time:
        print(f"No backfill needed for {symbol}. Data is already up to date.")
        return

    print(f"Backfilling {symbol} from {datetime.fromtimestamp(start_time / 1000)} to {datetime.fromtimestamp(end_time / 1000)}")
    fetch_and_insert(symbol, start_time, end_time, conn)


