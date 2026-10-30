from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"


FILES = {
    "train": DATA_DIR / "train_window_features.csv",
    "validation": DATA_DIR / "validation_window_features.csv",
    "test": DATA_DIR / "test_window_features.csv",
}


# Columns that are targets or metadata and must NOT be used
# as ML input features.
NON_FEATURE_COLUMNS = {
    "window",
    "attack",
    "attack_type",
    "file",
    "file_class",
}


def inspect_split(name, path):
    print("\n" + "=" * 70)
    print(f"{name.upper()} WINDOW DATASET")
    print("=" * 70)

    if not path.exists():
        print(f"ERROR: File not found: {path}")
        return None

    df = pd.read_csv(path)

    print(f"\nFile: {path}")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    print("\nColumn list:")
    for column in df.columns:
        print(f"  - {column}")

    # ---------------------------------------------------------------
    # TARGET DISTRIBUTION
    # ---------------------------------------------------------------

    print("\n" + "-" * 70)
    print("TARGET DISTRIBUTION")
    print("-" * 70)

    print("\nBinary target:")
    print(
        df["attack"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print("\nBinary target percentages:")
    print(
        (
            df["attack"]
            .value_counts(normalize=True)
            .sort_index()
            * 100
        ).round(2).to_string()
    )

    print("\nAttack type:")
    print(
        df["attack_type"]
        .value_counts()
        .to_string()
    )

    # ---------------------------------------------------------------
    # DATA TYPES
    # ---------------------------------------------------------------

    print("\n" + "-" * 70)
    print("DATA TYPES")
    print("-" * 70)

    print(
        df.dtypes.to_string()
    )

    # ---------------------------------------------------------------
    # MISSING VALUES
    # ---------------------------------------------------------------

    print("\n" + "-" * 70)
    print("MISSING VALUES")
    print("-" * 70)

    missing = df.isna().sum()
    missing_pct = (
        missing / len(df) * 100
    ).round(2)

    missing_table = pd.DataFrame({
        "missing": missing,
        "missing_percent": missing_pct,
    })

    missing_table = missing_table[
        missing_table["missing"] > 0
    ].sort_values(
        "missing_percent",
        ascending=False,
    )

    if missing_table.empty:
        print("No missing values.")
    else:
        print(
            missing_table.to_string()
        )

    # ---------------------------------------------------------------
    # INFINITE VALUES
    # ---------------------------------------------------------------

    print("\n" + "-" * 70)
    print("INFINITE VALUES")
    print("-" * 70)

    numeric_df = df.select_dtypes(
        include="number"
    )

    infinite_counts = (
        numeric_df
        .isin([float("inf"), float("-inf")])
        .sum()
    )

    infinite_counts = infinite_counts[
        infinite_counts > 0
    ]

    if infinite_counts.empty:
        print("No infinite values.")
    else:
        print(
            infinite_counts.to_string()
        )

    # ---------------------------------------------------------------
    # NUMERIC FEATURE SUMMARY
    # ---------------------------------------------------------------

    feature_columns = [
        column
        for column in df.columns
        if column not in NON_FEATURE_COLUMNS
    ]

    feature_df = df[
        feature_columns
    ].select_dtypes(
        include="number"
    )

    print("\n" + "-" * 70)
    print("MODEL FEATURE SUMMARY")
    print("-" * 70)

    print(
        f"Candidate numeric features: "
        f"{len(feature_df.columns)}"
    )

    print("\nFeature statistics:")

    summary = feature_df.describe().T[
        [
            "count",
            "mean",
            "std",
            "min",
            "max",
        ]
    ]

    print(
        summary.round(4).to_string()
    )

    # ---------------------------------------------------------------
    # DEAUTH BEHAVIOR
    # ---------------------------------------------------------------

    print("\n" + "-" * 70)
    print("DEAUTHENTICATION BEHAVIOR")
    print("-" * 70)

    if "deauth_frames_per_sec" in df.columns:

        print(
            "\nDeauthentication frames/sec:"
        )

        print(
            df[
                "deauth_frames_per_sec"
            ].describe().round(4).to_string()
        )

        print(
            "\nWindows containing deauthentication:"
        )

        print(
            (
                df[
                    "deauth_frames_per_sec"
                ] > 0
            ).sum()
        )

        print(
            "\nAttack windows containing "
            "deauthentication:"
        )

        print(
            (
                (
                    df["deauth_frames_per_sec"] > 0
                )
                & (df["attack"] == 1)
            ).sum()
        )

        print(
            "\nNormal windows containing "
            "deauthentication:"
        )

        print(
            (
                (
                    df["deauth_frames_per_sec"] > 0
                )
                & (df["attack"] == 0)
            ).sum()
        )

    # ---------------------------------------------------------------
    # POSSIBLE LABEL LEAKAGE CHECK
    # ---------------------------------------------------------------

    print("\n" + "-" * 70)
    print("LABEL / METADATA EXCLUSION CHECK")
    print("-" * 70)

    print(
        "The following columns are excluded from "
        "model features:"
    )

    for column in sorted(
        NON_FEATURE_COLUMNS
    ):
        if column in df.columns:
            print(f"  EXCLUDED: {column}")

    print("\nCandidate model features:")

    for column in feature_columns:
        print(f"  INCLUDED: {column}")

    return df


def main():

    print("=" * 70)
    print("WIDS WINDOW FEATURE VALIDATION")
    print("=" * 70)

    datasets = {}

    for name, path in FILES.items():

        df = inspect_split(
            name,
            path,
        )

        if df is not None:
            datasets[name] = df

    # ---------------------------------------------------------------
    # SPLIT CONSISTENCY
    # ---------------------------------------------------------------

    print("\n" + "=" * 70)
    print("SPLIT CONSISTENCY")
    print("=" * 70)

    if len(datasets) == 3:

        train_files = set(
            datasets["train"]["file"]
        )

        validation_files = set(
            datasets["validation"]["file"]
        )

        test_files = set(
            datasets["test"]["file"]
        )

        train_validation_overlap = (
            train_files & validation_files
        )

        train_test_overlap = (
            train_files & test_files
        )

        validation_test_overlap = (
            validation_files & test_files
        )

        print(
            f"\nTrain files: "
            f"{len(train_files)}"
        )

        print(
            f"Validation files: "
            f"{len(validation_files)}"
        )

        print(
            f"Test files: "
            f"{len(test_files)}"
        )

        print(
            f"\nTrain/Validation overlap: "
            f"{len(train_validation_overlap)}"
        )

        print(
            f"Train/Test overlap: "
            f"{len(train_test_overlap)}"
        )

        print(
            f"Validation/Test overlap: "
            f"{len(validation_test_overlap)}"
        )

        if (
            not train_validation_overlap
            and not train_test_overlap
            and not validation_test_overlap
        ):
            print(
                "\nPASS: No capture files overlap "
                "between the three splits."
            )
        else:
            print(
                "\nWARNING: Capture-file overlap detected!"
            )

    print("\n" + "=" * 70)
    print("WINDOW FEATURE VALIDATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
