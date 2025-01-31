from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

# Define the default arguments
default_args = {
    'owner': 'murat',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Define the DAG
with DAG(
    'data_fetch_job_dag',
    default_args=default_args,
    description='Run the data_fetch_job.py script every minute',
    schedule_interval='* * * * *',  # Every minute
    start_date=datetime(2025, 1, 27),  # Updated to a past date
    catchup=False,
) as dag:
    
    # BashOperator to run the script
    run_data_fetch_job = BashOperator(
        task_id='run_data_fetch_job',
        bash_command='/Users/murat/.pyenv/versions/3.9.6/envs/algotrading/bin/python /Users/murat/Desktop/algo_trading/data_fetch_job.py --service',
    )