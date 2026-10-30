from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"

TRAIN_FILE = DATA_DIR / "train_window_features.csv"
VALIDATION_FILE = DATA_DIR / "validation_window_features.csv"

MODEL_FILE = MODEL_DIR / "hist_gradient_boosting_window.joblib"
REPORT_FILE = REPORT_DIR / "window_model_validation_metrics.txt"


TARGET_COLUMN = "attack"

# These columns are labels, identifiers, or metadata.
# They must never be supplied to the classifier.
EXCLUDED_COLUMNS = {
    "window",
    "attack",
    "attack_type",
    "file",
    "file_class",
}


def load_dataset(path):
    """Load a window-level dataset."""

    df = pd.read_csv(path)

    feature_columns = [
        column
        for column in df.columns
        if column not in EXCLUDED_COLUMNS
    ]

    X = df[feature_columns].copy()
    y = df[TARGET_COLUMN].astype(int)

    # HistGradientBoostingClassifier can handle NaN values,
    # so we intentionally preserve the small amount of missingness.
    X = X.apply(
        pd.to_numeric,
        errors="coerce",
    )

    return X, y, feature_columns


def main():

    print("=" * 70)
    print("WIDS WINDOW-LEVEL MACHINE LEARNING")
    print("=" * 70)

    # ---------------------------------------------------------------
    # LOAD DATA
    # ---------------------------------------------------------------

    print("\nLoading training data...")

    X_train, y_train, feature_columns = load_dataset(
        TRAIN_FILE
    )

    print(
        f"Training windows: "
        f"{len(X_train):,}"
    )

    print(
        f"Training features: "
        f"{len(feature_columns)}"
    )

    print("\nLoading validation data...")

    X_validation, y_validation, _ = load_dataset(
        VALIDATION_FILE
    )

    print(
        f"Validation windows: "
        f"{len(X_validation):,}"
    )

    # ---------------------------------------------------------------
    # TARGET DISTRIBUTION
    # ---------------------------------------------------------------

    print("\n" + "-" * 70)
    print("TARGET DISTRIBUTION")
    print("-" * 70)

    print("\nTraining:")
    print(
        y_train.value_counts()
        .sort_index()
        .to_string()
    )

    print("\nValidation:")
    print(
        y_validation.value_counts()
        .sort_index()
        .to_string()
    )

    # ---------------------------------------------------------------
    # MODEL
    # ---------------------------------------------------------------

    print("\n" + "-" * 70)
    print("MODEL CONFIGURATION")
    print("-" * 70)

    model = HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=0.08,
        max_iter=200,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        l2_regularization=1.0,
        class_weight="balanced",
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=15,
        tol=1e-4,
        random_state=42,
        verbose=1,
    )

    print(
        "\nHistGradientBoostingClassifier"
    )

    print(
        f"Learning rate: "
        f"{model.learning_rate}"
    )

    print(
        f"Maximum iterations: "
        f"{model.max_iter}"
    )

    print(
        f"Maximum leaf nodes: "
        f"{model.max_leaf_nodes}"
    )

    print(
        f"Minimum samples per leaf: "
        f"{model.min_samples_leaf}"
    )

    # ---------------------------------------------------------------
    # TRAIN
    # ---------------------------------------------------------------

    print("\n" + "-" * 70)
    print("TRAINING")
    print("-" * 70)

    model.fit(
        X_train,
        y_train,
    )

    print("\nTraining complete.")

    if hasattr(model, "n_iter_"):
        print(
            f"Boosting iterations used: "
            f"{model.n_iter_}"
        )

    # ---------------------------------------------------------------
    # VALIDATION PREDICTION
    # ---------------------------------------------------------------

    print("\n" + "-" * 70)
    print("VALIDATION")
    print("-" * 70)

    predictions = model.predict(
        X_validation
    )

    # ---------------------------------------------------------------
    # METRICS
    # ---------------------------------------------------------------

    accuracy = accuracy_score(
        y_validation,
        predictions,
    )

    precision = precision_score(
        y_validation,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_validation,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_validation,
        predictions,
        zero_division=0,
    )

    matrix = confusion_matrix(
        y_validation,
        predictions,
    )

    report = classification_report(
        y_validation,
        predictions,
        target_names=[
            "Normal",
            "Attack",
        ],
        zero_division=0,
    )

    # ---------------------------------------------------------------
    # DISPLAY RESULTS
    # ---------------------------------------------------------------

    print(
        f"\nAccuracy : {accuracy:.4f}"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall   : {recall:.4f}"
    )

    print(
        f"F1-score : {f1:.4f}"
    )

    print("\nConfusion matrix:")

    print(matrix)

    print(
        "\nClassification report:"
    )

    print(report)

    # ---------------------------------------------------------------
    # SAVE MODEL
    # ---------------------------------------------------------------

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_package = {
        "model": model,
        "features": feature_columns,
        "target": TARGET_COLUMN,
        "window_seconds": 1.0,
    }

    joblib.dump(
        model_package,
        MODEL_FILE,
    )

    print(
        f"Model saved: "
        f"{MODEL_FILE}"
    )

    # ---------------------------------------------------------------
    # SAVE REPORT
    # ---------------------------------------------------------------

    with open(
        REPORT_FILE,
        "w",
        encoding="utf-8",
    ) as report_file:

        report_file.write(
            "WIDS WINDOW-LEVEL MODEL VALIDATION\n"
        )

        report_file.write(
            "=" * 60 + "\n\n"
        )

        report_file.write(
            f"Training windows: "
            f"{len(X_train):,}\n"
        )

        report_file.write(
            f"Validation windows: "
            f"{len(X_validation):,}\n"
        )

        report_file.write(
            f"Number of features: "
            f"{len(feature_columns)}\n\n"
        )

        report_file.write(
            "Features:\n"
        )

        for feature in feature_columns:
            report_file.write(
                f"- {feature}\n"
            )

        report_file.write(
            "\nMetrics\n"
        )

        report_file.write(
            "-" * 60 + "\n"
        )

        report_file.write(
            f"Accuracy : {accuracy:.4f}\n"
        )

        report_file.write(
            f"Precision: {precision:.4f}\n"
        )

        report_file.write(
            f"Recall   : {recall:.4f}\n"
        )

        report_file.write(
            f"F1-score : {f1:.4f}\n"
        )

        report_file.write(
            "\nConfusion Matrix\n"
        )

        report_file.write(
            str(matrix)
        )

        report_file.write(
            "\n\nClassification Report\n"
        )

        report_file.write(
            report
        )

    print(
        f"Metrics saved: "
        f"{REPORT_FILE}"
    )

    print("\n" + "=" * 70)
    print("WINDOW-LEVEL MODEL TRAINING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
