from __future__ import annotations

from datetime import datetime

from airflow import DAG
from airflow.providers.standard.operators.bash import (
    BashOperator,
)

PROJECT_DIR = "/opt/airflow/project"

with DAG(
    dag_id="ml_training_pipeline",
    start_date=datetime(
        2026,
        1,
        1,
    ),
    schedule="@weekly",
    catchup=False,
    tags=[
        "mlops",
        "training",
        "retraining",
    ],
) as dag:
    generate_data = BashOperator(
        task_id="generate_data",
        bash_command=(f"cd {PROJECT_DIR} && python -m src.data.generate_data"),
    )

    validate_data = BashOperator(
        task_id="validate_data",
        bash_command=(f"cd {PROJECT_DIR} && python -m src.data.validation"),
    )

    build_features = BashOperator(
        task_id="build_features",
        bash_command=(f"cd {PROJECT_DIR} && python -m src.features.build_features"),
    )

    train_and_promote = BashOperator(
        task_id=("train_evaluate_register_promote"),
        bash_command=(f"cd {PROJECT_DIR} && python -m src.models.train"),
    )

    evaluation_report = BashOperator(
        task_id="evaluation_report",
        bash_command=(f"cd {PROJECT_DIR} && python -m src.models.evaluate"),
    )

    (generate_data >> validate_data >> build_features >> train_and_promote >> evaluation_report)
