"""
WIDS Window-Level Attack-Type Evaluation

Evaluates the trained window-level model on the validation dataset
and reports detection performance for each attack type.

The test dataset is intentionally NOT used here.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)


# ---------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

VALIDATION_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "validation_window_features.csv"
)

MODEL_FILE = (
    PROJECT_ROOT
    / "models"
    / "hist_gradient_boosting_window.joblib"
)

REPORT_FILE = (
    PROJECT_ROOT
    / "reports"
    / "window_attack_type_evaluation.txt"
)


# ---------------------------------------------------------------------
# COLUMNS THAT MUST NOT BE USED AS MODEL FEATURES
# ---------------------------------------------------------------------

EXCLUDED_COLUMNS = {
    "window",
    "attack",
    "attack_type",
    "file",
    "file_class",
}


# ---------------------------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------------------------

def load_model():

    print("\nLoading trained model...")

    saved_object = joblib.load(MODEL_FILE)

    # The training script saved a dictionary containing
    # the classifier and the feature list.
    if isinstance(saved_object, dict):

        print("Saved model format: dictionary")

        # Try the expected model keys.
        if "model" in saved_object:
            model = saved_object["model"]

        elif "classifier" in saved_object:
            model = saved_object["classifier"]

        elif "estimator" in saved_object:
            model = saved_object["estimator"]

        else:
            raise ValueError(
                "The saved model is a dictionary, but no model "
                "object was found.\n"
                f"Available keys: {list(saved_object.keys())}"
            )

    else:

        print("Saved model format: estimator")

        model = saved_object

    print(f"Model type: {type(model).__name__}")

    return model, saved_object


# ---------------------------------------------------------------------
# DETERMINE FEATURES
# ---------------------------------------------------------------------

def get_feature_columns(df, saved_object):

    # If the training script stored the exact feature list,
    # use that list. This is safer than reconstructing it.
    if isinstance(saved_object, dict):

        for key in [
            "features",
            "feature_columns",
            "feature_names",
            "columns",
        ]:

            if key in saved_object:

                stored_features = saved_object[key]

                if isinstance(stored_features, (list, tuple)):

                    feature_columns = list(stored_features)

                    missing = [
                        column
                        for column in feature_columns
                        if column not in df.columns
                    ]

                    if missing:
                        raise ValueError(
                            "The saved model references features "
                            "that are missing from the validation "
                            f"dataset: {missing}"
                        )

                    print(
                        f"Using saved feature list: "
                        f"{len(feature_columns)} features"
                    )

                    return feature_columns

    # Fallback: reconstruct numeric feature list.
    feature_columns = [
        column
        for column in df.columns
        if column not in EXCLUDED_COLUMNS
        and pd.api.types.is_numeric_dtype(df[column])
    ]

    print(
        f"Using reconstructed feature list: "
        f"{len(feature_columns)} features"
    )

    return feature_columns


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():

    print("=" * 70)
    print("WIDS WINDOW-LEVEL ATTACK-TYPE EVALUATION")
    print("=" * 70)

    # ---------------------------------------------------------------
    # Load model
    # ---------------------------------------------------------------

    model, saved_object = load_model()

    print(f"Model loaded: {MODEL_FILE}")

    # ---------------------------------------------------------------
    # Load validation data
    # ---------------------------------------------------------------

    print("\nLoading validation data...")

    df = pd.read_csv(VALIDATION_FILE)

    print(f"Validation windows: {len(df):,}")

    # ---------------------------------------------------------------
    # Check required columns
    # ---------------------------------------------------------------

    required_columns = {
        "attack",
        "attack_type",
        "file",
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:

        raise ValueError(
            f"Missing required columns: "
            f"{sorted(missing_columns)}"
        )

    # ---------------------------------------------------------------
    # Determine model features
    # ---------------------------------------------------------------

    feature_columns = get_feature_columns(
        df,
        saved_object,
    )

    print(
        f"Model features found: "
        f"{len(feature_columns)}"
    )

    # ---------------------------------------------------------------
    # Verify model feature count
    # ---------------------------------------------------------------

    if hasattr(model, "n_features_in_"):

        expected_features = model.n_features_in_

        if len(feature_columns) != expected_features:

            raise ValueError(
                "\nFeature mismatch!\n"
                f"Model expects:       {expected_features}\n"
                f"Dataset provides:    {len(feature_columns)}\n\n"
                f"Features selected:\n{feature_columns}"
            )

    # ---------------------------------------------------------------
    # Prepare data
    # ---------------------------------------------------------------

    X = df[feature_columns]

    y_true = df["attack"].astype(int)

    # ---------------------------------------------------------------
    # Predict
    # ---------------------------------------------------------------

    print("\nGenerating predictions...")

    y_pred = model.predict(X)

    y_pred = np.asarray(y_pred).astype(int)

    print("Predictions complete.")

    # ---------------------------------------------------------------
    # Overall performance
    # ---------------------------------------------------------------

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
    )

    classification_report_text = classification_report(
        y_true,
        y_pred,
        target_names=[
            "Normal",
            "Attack",
        ],
        zero_division=0,
    )

    print("\n" + "-" * 70)
    print("OVERALL VALIDATION PERFORMANCE")
    print("-" * 70)

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

    print(cm)

    print(
        "\nClassification report:"
    )

    print(
        classification_report_text
    )

    # ---------------------------------------------------------------
    # Per-attack-type analysis
    # ---------------------------------------------------------------

    print("-" * 70)
    print("PER-ATTACK-TYPE DETECTION")
    print("-" * 70)

    attack_types = sorted(
        df.loc[
            df["attack"] == 1,
            "attack_type",
        ]
        .dropna()
        .astype(str)
        .unique()
    )

    results = []

    for attack_type in attack_types:

        mask = (
            (df["attack_type"].astype(str) == attack_type)
            & (df["attack"] == 1)
        )

        total = int(
            mask.sum()
        )

        detected = int(
            (y_pred[mask] == 1).sum()
        )

        missed = (
            total - detected
        )

        detection_rate = (
            detected / total
            if total > 0
            else 0.0
        )

        results.append(
            {
                "attack_type": attack_type,
                "total_windows": total,
                "detected": detected,
                "missed": missed,
                "detection_rate": detection_rate,
            }
        )

        print(
            f"\n{attack_type}"
        )

        print(
            f"  Total windows : {total:,}"
        )

        print(
            f"  Detected      : {detected:,}"
        )

        print(
            f"  Missed        : {missed:,}"
        )

        print(
            f"  Detection rate: "
            f"{detection_rate:.2%}"
        )

    # ---------------------------------------------------------------
    # Normal false-positive analysis
    # ---------------------------------------------------------------

    normal_mask = (
        df["attack"] == 0
    )

    normal_total = int(
        normal_mask.sum()
    )

    normal_false_positives = int(
        (y_pred[normal_mask] == 1).sum()
    )

    normal_correct = (
        normal_total
        - normal_false_positives
    )

    false_positive_rate = (
        normal_false_positives / normal_total
        if normal_total > 0
        else 0.0
    )

    print("\n" + "-" * 70)
    print("NORMAL-TRAFFIC ANALYSIS")
    print("-" * 70)

    print(
        f"\nNormal windows       : "
        f"{normal_total:,}"
    )

    print(
        f"Correctly classified : "
        f"{normal_correct:,}"
    )

    print(
        f"False positives      : "
        f"{normal_false_positives:,}"
    )

    print(
        f"False-positive rate  : "
        f"{false_positive_rate:.2%}"
    )

    # ---------------------------------------------------------------
    # Deauthentication-specific analysis
    # ---------------------------------------------------------------

    deauth_mask = (
        df["attack_type"]
        .astype(str)
        .str.contains(
            "Deauth",
            case=False,
            na=False,
        )
        & (df["attack"] == 1)
    )

    deauth_total = int(
        deauth_mask.sum()
    )

    deauth_detected = 0
    deauth_missed = 0
    deauth_detection_rate = 0.0

    if deauth_total > 0:

        deauth_detected = int(
            (y_pred[deauth_mask] == 1).sum()
        )

        deauth_missed = (
            deauth_total
            - deauth_detected
        )

        deauth_detection_rate = (
            deauth_detected
            / deauth_total
        )

        print("\n" + "-" * 70)
        print("DEAUTHENTICATION DETECTION")
        print("-" * 70)

        print(
            f"\nDeauth windows : "
            f"{deauth_total:,}"
        )

        print(
            f"Detected       : "
            f"{deauth_detected:,}"
        )

        print(
            f"Missed         : "
            f"{deauth_missed:,}"
        )

        print(
            f"Detection rate : "
            f"{deauth_detection_rate:.2%}"
        )

    # ---------------------------------------------------------------
    # Save report
    # ---------------------------------------------------------------

    REPORT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        REPORT_FILE,
        "w",
        encoding="utf-8",
    ) as report_file:

        report_file.write(
            "WIDS WINDOW-LEVEL ATTACK-TYPE EVALUATION\n"
        )

        report_file.write(
            "=" * 70 + "\n\n"
        )

        report_file.write(
            "Dataset: validation_window_features.csv\n"
        )

        report_file.write(
            "Test set was not used.\n\n"
        )

        report_file.write(
            "OVERALL VALIDATION PERFORMANCE\n"
        )

        report_file.write(
            "-" * 70 + "\n"
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
            f"F1-score : {f1:.4f}\n\n"
        )

        report_file.write(
            "Confusion matrix:\n"
        )

        report_file.write(
            f"{cm}\n\n"
        )

        report_file.write(
            "Classification report:\n"
        )

        report_file.write(
            classification_report_text
            + "\n"
        )

        report_file.write(
            "\nPER-ATTACK-TYPE DETECTION\n"
        )

        report_file.write(
            "-" * 70 + "\n"
        )

        for result in results:

            report_file.write(
                f"\n{result['attack_type']}\n"
            )

            report_file.write(
                f"Total windows : "
                f"{result['total_windows']:,}\n"
            )

            report_file.write(
                f"Detected      : "
                f"{result['detected']:,}\n"
            )

            report_file.write(
                f"Missed        : "
                f"{result['missed']:,}\n"
            )

            report_file.write(
                f"Detection rate: "
                f"{result['detection_rate']:.2%}\n"
            )

        report_file.write(
            "\nNORMAL TRAFFIC\n"
        )

        report_file.write(
            "-" * 70 + "\n"
        )

        report_file.write(
            f"Normal windows       : "
            f"{normal_total:,}\n"
        )

        report_file.write(
            f"Correctly classified : "
            f"{normal_correct:,}\n"
        )

        report_file.write(
            f"False positives      : "
            f"{normal_false_positives:,}\n"
        )

        report_file.write(
            f"False-positive rate  : "
            f"{false_positive_rate:.2%}\n"
        )

        if deauth_total > 0:

            report_file.write(
                "\nDEAUTHENTICATION DETECTION\n"
            )

            report_file.write(
                "-" * 70 + "\n"
            )

            report_file.write(
                f"Deauth windows : "
                f"{deauth_total:,}\n"
            )

            report_file.write(
                f"Detected       : "
                f"{deauth_detected:,}\n"
            )

            report_file.write(
                f"Missed         : "
                f"{deauth_missed:,}\n"
            )

            report_file.write(
                f"Detection rate : "
                f"{deauth_detection_rate:.2%}\n"
            )

    print(
        f"\nReport saved: {REPORT_FILE}"
    )

    print("\n" + "=" * 70)
    print(
        "WINDOW ATTACK-TYPE EVALUATION COMPLETE"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()
