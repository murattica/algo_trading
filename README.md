# Algorithmic Trading Project

## Overview
This project is focused on building an advanced **algorithmic trading framework** leveraging **Reinforcement Learning (RL)**. The aim is to develop a robust trading agent that can manage portfolios efficiently, utilizing multi-dimensional reward functions and advanced RL techniques like Multi-Agent RL. The project is currently under development, and components are being iteratively refined.

---

## Repository Structure

```
|-- airflow_home/           # Airflow DAGs and pipelines (no recent updates)
|-- binance/                # Binance API-related utilities
|-- logs/                   # Logs generated during training and evaluation
|-- saved_models/           # Directory for storing trained models
|-- .env                    # Environment variables (e.g., API keys)
|-- .gitignore              # Git ignore file
|-- data_fetch_job.py       # Script to manage data-fetching tasks
|-- db.py                   # Database interaction utilities
|-- fetch_data.py           # Fetch market data from external sources
|-- performance.py          # Evaluate portfolio performance metrics
|-- Portfolio.py            # Portfolio management class
|-- Position.py             # Position management class
|-- README.md               # This file
|-- requirements.txt        # Python dependencies
|-- SequentialTradingEnv.py # Custom RL environment based on PettingZoo
|-- Train.py                # Training pipeline for RL models
```

---

## Key Components

### Data Layer
- **Database**: PostgreSQL database to store historical market data.
- **Data Fetching** (`fetch_data.py`): Uses `ccxt` to pull OHLCV data from Binance.
- **dbt**: Supports data transformations and feature engineering (e.g., RSI, Bollinger Bands).

### Airflow Automation
Airflow is used to automate data ingestion and preprocessing tasks:
- **Data Fetching**: Fetches binance OHLCV market data for selected trading pairs.
- **DAGs**: Orchestrates periodic data updates and backfills.

- **`data_fetch_job.py`**: Handles the periodic scheduling and execution of data-fetching tasks.
- **`fetch_data.py`**: Fetches market data from Binance and other APIs.
- **`db.py`**: Interfaces with the database for storing and retrieving market data.

### Trading Logic
- **`Portfolio.py`**:
  - Manages portfolio allocations, including cash reserves and position rebalancing.
  - Ensures that total allocations (across pairs and cash) sum to **1**.

- **`Position.py`**:
  - Handles individual trading positions.
  - Tracks open and closed positions, as well as profit and loss calculations.

### Performance Evaluation
- **`performance.py`**:
  - Computes key performance metrics such as Sharpe ratio, drawdowns, and portfolio returns.
  - Supports backtesting and benchmarking.

### RL Environment
- **`SequentialTradingEnv.py`**:
  - Implements the custom RL environment using PettingZoo.
  - Features:
    - **Specialist Agents**: Provide trade recommendations based on predefined strategies or signals.
    - **Meta-Agent**: Allocates capital based on specialist input.
    - **Reward Mechanism**: Multi-dimensional rewards per pair.
    - **Cash Reserve Management**: Allows holding unallocated funds for future opportunities.

### Training Pipeline
- **`Train.py`**:
  - Orchestrates the training process using RLlib.
  - Handles hyperparameter tuning and model checkpoints.

---

## Modeling Overview

The project uses a **multi-agent reinforcement learning (MARL)** framework:
- **Specialist Agents**:
  - Focus on individual trading pairs.
  - Recommend buy/sell/hold actions based on specific metrics.
- **Meta-Agent**:
  - Allocates funds across pairs and cash.
  - Ensures total allocations remain balanced.

The system integrates a **multi-dimensional reward function** to evaluate trades across pairs, promoting long-term profitability and robust portfolio management.
