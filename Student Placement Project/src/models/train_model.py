import pandas as pd
import joblib
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient


def train(n_estimators=100, random_state=42):
    project_root = Path(__file__).resolve().parents[2]

    df = pd.read_csv(
        project_root / "data" / "raw" / "student_placement_data.csv"
    )

    X = df.drop("Placement", axis=1)
    y = df["Placement"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    mlflow_db_path = str(project_root / "mlflow.db").replace("\\", "/")
    mlflow.set_tracking_uri(f"sqlite:///{mlflow_db_path}")
    mlflow.set_experiment("Student Placement Prediction")

    with mlflow.start_run():
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
        )
        model.fit(X_train, y_train)

        accuracy = accuracy_score(y_test, model.predict(X_test))

        mlflow.log_param("model_type", "RandomForestClassifier")
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("random_state", random_state)
        mlflow.log_metric("accuracy", accuracy)

        mlflow.sklearn.log_model(
            model,
            name="model",
            skops_trusted_types=["sklearn.tree._tree.Tree"],
        )

        print(f"Model accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]

    train(n_estimators=100, random_state=42)
    train(n_estimators=200, random_state=42)
    train(n_estimators=150, random_state=0)

    client = MlflowClient()
    experiment = client.get_experiment_by_name("Student Placement Prediction")

    best_run = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.accuracy DESC"],
        max_results=1,
    )[0]

    print(
        f"\nBest run: {best_run.info.run_name} | "
        f"Accuracy: {best_run.data.metrics['accuracy']:.4f}"
    )

    models_dir = project_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    best_model_uri = f"runs:/{best_run.info.run_id}/model"
    best_model = mlflow.sklearn.load_model(best_model_uri)

    joblib.dump(best_model, models_dir / "model.pkl")
    print("Best model saved to models/model.pkl")