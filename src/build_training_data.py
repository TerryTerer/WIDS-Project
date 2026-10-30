from pathlib import Path

import pandas as pd

from src.feature_config import FEATURES, TARGET_COLUMN


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MANIFEST = PROJECT_ROOT / "reports" / "split_manifest.csv"

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_split(split_name):
    """
    Load one dataset split using the capture-file manifest.

    The manifest already stores paths relative to the project root,
    so we resolve each path from PROJECT_ROOT.
    """

    manifest = pd.read_csv(MANIFEST)

    files = manifest.loc[
        manifest["split"] == split_name,
        "file"
    ].tolist()

    print(f"\n=== {split_name.upper()} SPLIT ===")
    print(f"Capture files: {len(files)}")

    frames = []

    columns = FEATURES + [TARGET_COLUMN]

    for index, relative_path in enumerate(files, start=1):

        # The manifest paths already begin with:
        # data/raw/WPA3-dataset/...
        path = PROJECT_ROOT / relative_path

        if not path.exists():
            raise FileNotFoundError(
                f"\nCapture file not found:\n{path}\n"
                f"\nManifest path:\n{relative_path}"
            )

        df = pd.read_csv(
            path,
            usecols=columns
        )

        # Preserve the original multiclass attack label.
        df["attack_type"] = df[TARGET_COLUMN]

        # Binary target:
        # Normal = 0
        # Any attack = 1
        df["attack"] = (
            df[TARGET_COLUMN] != "Normal"
        ).astype("int8")

        frames.append(df)

        if index % 50 == 0 or index == len(files):
            rows_loaded = sum(len(frame) for frame in frames)

            print(
                f"Loaded {index}/{len(files)} files "
                f"({rows_loaded:,} rows)"
            )

    result = pd.concat(
        frames,
        ignore_index=True
    )

    return result


def main():

    # ---------------------------------------------------------
    # Load all three leakage-safe splits
    # ---------------------------------------------------------

    train = load_split("train")
    validation = load_split("validation")
    test = load_split("test")

    # ---------------------------------------------------------
    # Dataset sizes
    # ---------------------------------------------------------

    print("\n=== DATASET SIZES ===")

    print(f"Train:      {len(train):,}")
    print(f"Validation: {len(validation):,}")
    print(f"Test:       {len(test):,}")

    # ---------------------------------------------------------
    # Original attack labels
    # ---------------------------------------------------------

    print("\n=== TRAIN LABELS ===")
    print(train["attack_type"].value_counts())

    print("\n=== VALIDATION LABELS ===")
    print(validation["attack_type"].value_counts())

    print("\n=== TEST LABELS ===")
    print(test["attack_type"].value_counts())

    # ---------------------------------------------------------
    # Binary target distribution
    # ---------------------------------------------------------

    print("\n=== TRAIN BINARY TARGET ===")
    print(
        train["attack"]
        .value_counts()
        .sort_index()
    )

    print("\n=== VALIDATION BINARY TARGET ===")
    print(
        validation["attack"]
        .value_counts()
        .sort_index()
    )

    print("\n=== TEST BINARY TARGET ===")
    print(
        test["attack"]
        .value_counts()
        .sort_index()
    )

    # ---------------------------------------------------------
    # Verify selected features
    # ---------------------------------------------------------

    print("\n=== FEATURE CHECK ===")
    print(f"Selected features: {len(FEATURES)}")

    missing_train_features = [
        feature
        for feature in FEATURES
        if feature not in train.columns
    ]

    if missing_train_features:
        print("\nMISSING FEATURES:")
        for feature in missing_train_features:
            print(f"  {feature}")

        raise SystemExit(
            "\nERROR: One or more selected features are missing."
        )

    print("PASS: All selected features are present.")

    # ---------------------------------------------------------
    # Save processed datasets
    # ---------------------------------------------------------

    train_path = OUTPUT_DIR / "train_binary.csv"
    validation_path = OUTPUT_DIR / "validation_binary.csv"
    test_path = OUTPUT_DIR / "test_binary.csv"

    print("\n=== SAVING DATASETS ===")

    train.to_csv(
        train_path,
        index=False
    )

    print(f"Saved: {train_path}")

    validation.to_csv(
        validation_path,
        index=False
    )

    print(f"Saved: {validation_path}")

    test.to_csv(
        test_path,
        index=False
    )

    print(f"Saved: {test_path}")

    print("\n=== COMPLETE ===")


if __name__ == "__main__":
    main()
