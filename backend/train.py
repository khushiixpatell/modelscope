from pathlib import Path

import joblib
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from preprocessing import load_data


BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = (
    BASE_DIR
    / "data"
    / "WA_Fn-UseC_-HR-Employee-Attrition.csv"
)

MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)


def evaluate_model(name, model, X_test, y_test):
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(
            y_test,
            predictions,
            zero_division=0,
        ),
        "recall": recall_score(
            y_test,
            predictions,
            zero_division=0,
        ),
        "f1": f1_score(
            y_test,
            predictions,
            zero_division=0,
        ),
        "roc_auc": roc_auc_score(
            y_test,
            probabilities,
        ),
    }

    print(f"\n{name}")
    print("-" * len(name))

    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    return metrics


def main():
    X, y, preprocessor = load_data(DATA_PATH)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    models = {
        "logistic_regression": LogisticRegression(
            max_iter=1000
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            random_state=42,
        ),
        "gradient_boosting": GradientBoostingClassifier(
            random_state=42
        ),
    }

    for name, classifier in models.items():

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", classifier),
            ]
        )

        pipeline.fit(X_train, y_train)

        evaluate_model(
            name,
            pipeline,
            X_test,
            y_test,
        )

        model_path = MODEL_DIR / f"{name}.joblib"

        joblib.dump(
            pipeline,
            model_path,
        )

        print(f"Saved: {model_path}")


if __name__ == "__main__":
    main()