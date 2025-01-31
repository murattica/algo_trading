import psycopg2
import keyring
import numpy as np

def init_db():

    username = keyring.get_password('postgres', 'username')
    password = keyring.get_password('postgres', 'password')

    conn = psycopg2.connect(
        dbname="binance",
        user=username,  # Replace with your PostgreSQL username
        password=password,  # Replace with your PostgreSQL password
        host="localhost",
        port="5432"
    )
    cursor = conn.cursor()

    # Create table if it doesn't exist
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS ticker_data (
    symbol TEXT,
    timestamp BIGINT,
    date DATE,
    hour INT,
    min INT,
    open REAL,
    high REAL,
    low REAL,
    close REAL,
    volume_crypto REAL,
    volume_usd REAL,
    PRIMARY KEY (symbol, timestamp)
        )
    ''')
    conn.commit()
    return conn

def get_query(query):
    username = keyring.get_password('postgres', 'username')
    password = keyring.get_password('postgres', 'password')

    try:
        with psycopg2.connect(
            dbname="binance",
            user=username,
            password=password,
            host="localhost",
            port="5432"
        ) as conn:
            with conn.cursor() as cursor:
                cursor.execute(query)
                if query.strip().lower().startswith("select"):
                    results = cursor.fetchall()  # Fetch all rows
                    if not results:
                        print("No data returned from the query.")
                        return []  # Return an empty list if no rows
                    return list(results)  # Convert to list explicitly
                else:
                    conn.commit()  # Commit for non-select queries
                    return None
    except Exception as e:
        print(f"Error executing query: {e}")
        return None


def get_latest_timestamp(symbol, conn):
    cursor = conn.cursor()
    cursor.execute('''
        SELECT MAX(timestamp) FROM ticker_data WHERE symbol = %s
    ''', (symbol,))
    result = cursor.fetchone()[0]
    print(f"Latest timestamp for {symbol}: {result}")  # Debugging statement
    return result


def fetch_state_history(symbols):
    """Fetches data for the specified symbols and formats it for the environment."""
    symbol_list = ", ".join([f"'{symbol}'" for symbol in symbols])
    query = f'''
        SELECT symbol, timestamp, open, close, low, high, volume_usd, rsi_7, rsi_14, rsi_30, ichimoku, bolinger
        FROM state_history
        WHERE symbol IN ({symbol_list})
        ORDER BY timestamp ASC
    '''
    return list(get_query(query))

def prepare_numpy_array(data, num_pairs):
    """Converts raw SQL data into a NumPy array for the trading environment."""
    if not data:
        raise ValueError("Input data is empty. Ensure the database query is correct.")
    
    # Map symbols to indices
    symbol_index_map = {symbol: idx for idx, symbol in enumerate(sorted(set(row[0] for row in data)))}

    
    # Initialize NumPy array
    timestamps = sorted(list(set(row[1] for row in data)))
    if not timestamps:
        raise ValueError("No timestamps found in data.")
    
    time_index_map = {timestamp: i for i, timestamp in enumerate(timestamps)}
    
    market_data = np.zeros((len(timestamps), num_pairs, len(data[0])-2))
    
    # Populate array
    for row in data:
        try:
            symbol_, timestamp_, open_, close_, low_, high_, volume_usd_, rsi_7_, rsi_14_, rsi_30_, ichimoku_, bolinger_ = row
            t_idx = time_index_map[timestamp_]
            s_idx = symbol_index_map[symbol_]
            market_data[t_idx, s_idx, :] = [open_, close_, low_, high_, volume_usd_, rsi_7_, rsi_14_, rsi_30_, ichimoku_, bolinger_] 
        except KeyError as e:
            print(f"Skipping row due to missing mapping: {row}, Error: {e}")
    
    return market_data, time_index_map, symbol_index_map


# The following block will only execute when this file is run directly
if __name__ == "__main__":
    print("db.py is being run directly")
else:
    print("db.py is being imported into another module")