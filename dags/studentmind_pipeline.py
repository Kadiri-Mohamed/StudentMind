from datetime import datetime
from airflow.decorators import dag, task


@dag(
    dag_id="studentmind_ml_pipeline",
    start_date=datetime(2026, 11, 12),
    schedule_interval=None,
    catchup=False,
    tags=["StudentMind", "ML"]
)
def studentmind_pipeline():

    @task
    def check_dataset():
        from pathlib import Path

        path = Path("/opt/airflow/data/student_mental_health.csv")

        if not path.exists():
            raise FileNotFoundError("Dataset not found")

        print("Dataset loaded successfully")
        return str(path)

    @task
    def preprocess_data(dataset_path):
        from src.preprocessing import prepare_data

        return prepare_data(
            dataset_path,
            "/opt/airflow/data/processed"
        )

    @task
    def train_models(processed_path):
        from src.training import train_models

        return train_models(
            processed_path,
            "/opt/airflow/models",
            "/opt/airflow/reports"
        )

    @task
    def evaluate_models(processed_path, model_path):
        from src.evaluation import evaluate_model

        return evaluate_model(
            processed_path,
            model_path,
            "/opt/airflow/reports"
        )

    dataset = check_dataset()
    processed = preprocess_data(dataset)
    model = train_models(processed)
    evaluate_models(processed, model)


studentmind_pipeline()