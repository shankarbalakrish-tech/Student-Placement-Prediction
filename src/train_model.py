from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import mlflow
import mlflow.sklearn


def main():
    project_root = Path(__file__).resolve().parents[1]

    df = pd.read_csv(
        project_root / "data" / "student_placement_data.csv"
    )

    X = df.drop("Placement", axis=1)
    y = df["Placement"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        random_state=42
    )

    mlflow_db_path = str(
        project_root / "mlflow.db"
    ).replace("\\", "/")

    mlflow.set_tracking_uri(
        f"sqlite:///{mlflow_db_path}"
    )

    mlflow.set_experiment(
        "Student Placement Prediction"
    )

    with mlflow.start_run():
        model = RandomForestClassifier(n_estimators=100)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)

        mlflow.log_param("n_estimators", 100)
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(
        sk_model=model,
        name="model",
        serialization_format="skops",
        skops_trusted_types=[
        "sklearn.tree._tree.Tree"
        ],
        )
        print("Accuracy:", acc)


if __name__ == "__main__":
    main()