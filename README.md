# Algo Trading Framework

This is an **algorithmic trading framework** that integrates **Reinforcement Learning (RL)** for algorithmic trading strategies, along with **Airflow for automation** and **dbt for data modeling**. This project is currently **under development**.

---

## Core Components

### 1. **SequentialTradingEnv** (Custom RL Environment)
The `SequentialTradingEnv` is a custom multi-agent trading environment built using `PettingZoo`. It provides:

- **Agents**: Each trading pair is represented as a specialist agent, with a meta-agent for overall portfolio management.
- **Features**: Includes normalized historical market data and portfolio state.
- **Actions**: Agents make allocation and risk-management decisions.
- **Rewards**: Combines returns and risk-adjusted metrics for each agent.

The environment enables training reinforcement learning models for multi-asset portfolio optimization and risk management.

### 2. **Reinforcement Learning Models**
- **Custom RL Module**: Implements a feedforward neural network using PyTorch for policy generation.
- **Training Framework**: Uses `Ray RLlib` for scalable multi-agent training.
- **Policy Mapping**: Unique policies for each agent and the meta-agent to specialize their decision-making.
- **Training Script** (`Train.py`): Trains the RL agents using PPO (Proximal Policy Optimization).

### 3. **Airflow Automation**
Airflow is used to automate data ingestion and preprocessing tasks:
- **Data Fetching**: Fetches OHLCV market data for selected trading pairs.
- **DAGs**: Orchestrates periodic data updates and backfills.

### 4. **Data Layer**
- **Database**: PostgreSQL database to store historical market data.
- **Data Fetching** (`fetch_data.py`): Uses `ccxt` to pull OHLCV data from Binance.
- **dbt**: Supports data transformations and feature engineering (e.g., RSI, Bollinger Bands).

---

## Setup Instructions

### 1. Install Dependencies
Ensure you have Python installed. Install the required dependencies:

```bash
pip install -r requirements.txt
```

### 2. Configure Database
Set up a PostgreSQL database and ensure the credentials are stored in a secure location (e.g., using `keyring`).

- Example database initialization script:
```sql
CREATE DATABASE binance;
CREATE TABLE ticker_data (
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
);
```

### 3. Setting Up Airflow

#### Initialize Airflow
```bash
export AIRFLOW_HOME=$(pwd)/airflow_home
airflow db init
```

#### Start Airflow
```bash
airflow webserver -p 8080 &
airflow scheduler &
```
Visit `http://localhost:8080` to access the Airflow UI.

### 4. Training the RL Model
Run the training script:

```bash
python Train.py
```

This will initialize the custom environment and start training the RL agents. Training configurations (e.g., learning rate, policy networks) can be modified in the script.

---

## Project Structure

```
.
├── airflow_home/              # Airflow DAGs and configuration
├── db.py                      # Database connection and query management
├── fetch_data.py              # Fetches historical market data from Binance
├── data_fetch_job.py          # Airflow job for fetching market data
├── portfolio.py               # Portfolio management logic
├── SequentialTradingEnv.py    # Custom RL trading environment
├── Train.py                   # RL training script using Ray RLlib
├── custom_rl_module.py        # Custom reinforcement learning model
├── performance.py             # Evaluation of trained RL models
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
```

---

## Key Features

- **Multi-Agent RL**: Train and evaluate specialized agents for each trading pair and a meta-agent for portfolio management.
- **Customizable Environment**: Fully parameterized for different market conditions and trading strategies.
- **Airflow Automation**: Automates data ingestion and preprocessing tasks.
- **Modular Design**: Easily extendable for new trading pairs, RL models, and data pipelines.

---

## Notes
- The project is **under development**.
- Ensure database credentials are securely managed.
- Contributions and feedback are welcome.

---

### Contributors
- **Murat** (Project Owner)

---
