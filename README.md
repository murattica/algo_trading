# Binance dbt Library Documentation

This document provides an overview and instructions for setting up and utilizing the **Binance dbt Library** for your project.

---

## Overview

The Binance dbt library contains SQL models, macros, and configurations to enable data transformations and feature engineering for algorithmic trading. The models support creating technical indicators like RSI, Bollinger Bands, and Ichimoku Clouds, along with preparing historical state data for backtesting and machine learning workflows.

### Structure:
```
.
├── models/
│   ├── signals/
│   │   ├── bolinger.sql
│   │   ├── ichimoku.sql
│   │   ├── rsi.sql
│   │   └── state_history.sql
│   └── sources.yml
├── seeds/
├── snapshots/
├── target/
├── tests/
└── dbt_project.yml
```

---

## Setup Instructions

### 1. Install dbt
Ensure dbt and the required adapter for your database are installed:

```bash
pip install dbt-core dbt-postgres
```

### 2. Configure dbt Profiles
Create a `profiles.yml` file under `~/.dbt/` with the following configuration:

```yaml
default:
  outputs:
    dev:
      type: postgres
      host: localhost
      user: your_db_user
      password: your_db_password
      port: 5432
      dbname: binance
      schema: public
      threads: 4
  target: dev
```

### 3. Initialize the dbt Project
Navigate to the `binance` directory and run:

```bash
dbt init binance
```

This will set up the project structure and configuration.

### 4. Run Models
Run the models to generate outputs in your database:

```bash
dbt run
```

### 5. Test and Debug
Test the integrity of your models:

```bash
dbt test
```

If you encounter issues, use:

```bash
dbt debug
```

---

## Key Components

### Models

1. **`bolinger.sql`**
   - Generates Bollinger Bands for specified trading pairs.

2. **`ichimoku.sql`**
   - Computes Ichimoku Cloud indicators for trend analysis.

3. **`rsi.sql`**
   - Calculates the Relative Strength Index (RSI) to identify overbought/oversold conditions.

4. **`state_history.sql`**
   - Prepares historical state data for analysis and modeling.

### Sources
Defined in `sources.yml` to map raw data inputs for transformations.

### Snapshots
Maintain historical records of critical data transformations over time.

---

## Notes

- Ensure your database credentials are correctly configured in the `profiles.yml` file.
- For additional customization, modify the SQL models under the `models/signals` directory.
- Contributions and suggestions for improvement are welcome.

---

### Contributors
- **Murat** (Project Owner)

---
