from pathlib import Path
import numpy as np
import pandas as pd

from src.feature_config import FEATURES, TARGET_COLUMN


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
SPLIT_MANIFEST = PROJECT_ROOT / "reports" / "split_manifest.csv"

WINDOW_SIZE = 1.0
CONTEXT_WINDOWS = 5


# ============================================================
# FEATURE DEFINITIONS
# ============================================================

AGGREGATE_COLUMNS = [
    "frame.len",
    "frame.time_delta",
    "wlan.duration",
    "wlan_radio.data_rate",
    "wlan_radio.duration",
    "radiotap.dbm_antsignal",
    "radiotap.datarate",
]


# ============================================================
# HELPERS
# ============================================================

def safe_numeric(series):
    """Convert a Series to numeric values."""
    return pd.to_numeric(series, errors="coerce")


def load_capture(csv_path):
    """Load only columns required for feature engineering."""

    required_columns = list(
        dict.fromkeys(
            FEATURES
            + [
                TARGET_COLUMN,
                "frame.time_epoch",
            ]
        )
    )

    df = pd.read_csv(
        csv_path,
        usecols=lambda column: column in required_columns,
        low_memory=False,
    )

    for column in required_columns:
        if column not in df.columns:
            df[column] = np.nan

    for column in FEATURES:
        if column in df.columns:
            df[column] = safe_numeric(df[column])

    df["frame.time_epoch"] = safe_numeric(
        df["frame.time_epoch"]
    )

    return df


def identify_deauthentication(df):
    """
    Identify observable IEEE 802.11 deauthentication frames.

    IEEE 802.11 management subtype 12 = Deauthentication.
    """

    subtype = safe_numeric(
        df["wlan.fc.subtype"]
    )

    return subtype.eq(12).fillna(False)


def get_binary_label(label):
    """
    Normal -> 0
    Anything else -> 1
    """

    if pd.isna(label):
        return 0

    return int(
        str(label).strip().lower() != "normal"
    )


def get_attack_type(label):
    """Normalize an individual packet label."""

    if pd.isna(label):
        return "Normal"

    label = str(label).strip()

    return label if label else "Normal"


def determine_window_attack_label(labels):
    """
    Determine the label of a window from ALL packets in that window.

    Important:
    We do NOT use the first packet's label.

    If any attack traffic is present, the window is considered
    an attack window.

    If multiple attack labels occur in the same window, the
    attack label with the highest packet count is selected.

    This prevents normal packets appearing before attack packets
    from incorrectly making the entire window Normal.
    """

    labels = (
        labels
        .map(get_attack_type)
    )

    attack_labels = labels[
        labels != "Normal"
    ]

    if attack_labels.empty:
        return "Normal"

    counts = (
        attack_labels
        .value_counts()
    )

    return counts.index[0]


# ============================================================
# WINDOW FEATURE ENGINEERING
# ============================================================

def build_window_features(csv_path):
    """
    Convert packet-level records from one capture into
    one-second windows plus five-second behavioral context.
    """

    df = load_capture(csv_path)

    if df.empty:
        return pd.DataFrame()

    # --------------------------------------------------------
    # Timestamp handling
    # --------------------------------------------------------

    df = df.dropna(
        subset=["frame.time_epoch"]
    ).copy()

    if df.empty:
        return pd.DataFrame()

    df = df.sort_values(
        "frame.time_epoch"
    ).reset_index(drop=True)

    start_time = df["frame.time_epoch"].iloc[0]

    df["relative_time"] = (
        df["frame.time_epoch"] - start_time
    )

    df["window"] = (
        df["relative_time"] // WINDOW_SIZE
    ).astype(int)

    # --------------------------------------------------------
    # Deauthentication identification
    # --------------------------------------------------------

    df["is_deauth"] = identify_deauthentication(df)

    # --------------------------------------------------------
    # Frame indicators
    # --------------------------------------------------------

    df["is_retry"] = (
        safe_numeric(
            df["wlan.fc.retry"]
        )
        .fillna(0)
        .eq(1)
        .astype(int)
    )

    df["is_protected"] = (
        safe_numeric(
            df["wlan.fc.protected"]
        )
        .fillna(0)
        .eq(1)
        .astype(int)
    )

    df["is_power_management"] = (
        safe_numeric(
            df["wlan.fc.pwrmgt"]
        )
        .fillna(0)
        .eq(1)
        .astype(int)
    )

    df["is_fragment"] = (
        safe_numeric(
            df["wlan.fc.frag"]
        )
        .fillna(0)
        .eq(1)
        .astype(int)
    )

    # --------------------------------------------------------
    # 1-second aggregation
    # --------------------------------------------------------

    grouped = df.groupby(
        "window",
        sort=True,
    )

    result = grouped.size().to_frame(
        "frames_per_sec"
    )

    # --------------------------------------------------------
    # Statistical features
    # --------------------------------------------------------

    for column in AGGREGATE_COLUMNS:

        if column not in df.columns:
            continue

        aggregation = grouped[column].agg(
            [
                "mean",
                "std",
                "min",
                "max",
            ]
        )

        aggregation.columns = [
            f"{column}_mean",
            f"{column}_std",
            f"{column}_min",
            f"{column}_max",
        ]

        result = result.join(
            aggregation
        )

    # --------------------------------------------------------
    # Rate features
    # --------------------------------------------------------

    result["retry_rate"] = grouped[
        "is_retry"
    ].mean()

    result["protected_rate"] = grouped[
        "is_protected"
    ].mean()

    result["power_management_rate"] = grouped[
        "is_power_management"
    ].mean()

    result["fragment_rate"] = grouped[
        "is_fragment"
    ].mean()

    # --------------------------------------------------------
    # Frame diversity
    # --------------------------------------------------------

    if "wlan.fc.type" in df.columns:
        result["unique_frame_types"] = grouped[
            "wlan.fc.type"
        ].nunique()
    else:
        result["unique_frame_types"] = 0

    if "wlan.fc.subtype" in df.columns:
        result["unique_frame_subtypes"] = grouped[
            "wlan.fc.subtype"
        ].nunique()
    else:
        result["unique_frame_subtypes"] = 0

    # --------------------------------------------------------
    # Deauthentication behavior
    # --------------------------------------------------------

    deauth_counts = grouped[
        "is_deauth"
    ].sum()

    result["deauth_frames_per_sec"] = (
        deauth_counts
    )

    result["deauth_ratio"] = (
        deauth_counts
        / result["frames_per_sec"].replace(
            0,
            np.nan,
        )
    )

    # --------------------------------------------------------
    # Five-second behavioral context
    # --------------------------------------------------------

    result["frames_per_sec_5sec"] = (
        result["frames_per_sec"]
        .rolling(
            window=CONTEXT_WINDOWS,
            min_periods=1,
        )
        .sum()
    )

    result["deauth_frames_per_sec_5sec"] = (
        result["deauth_frames_per_sec"]
        .rolling(
            window=CONTEXT_WINDOWS,
            min_periods=1,
        )
        .sum()
    )

    result["deauth_ratio_5sec"] = (
        result["deauth_frames_per_sec_5sec"]
        / result["frames_per_sec_5sec"]
        .replace(0, np.nan)
    )

    result["retry_rate_5sec"] = (
        result["retry_rate"]
        .rolling(
            window=CONTEXT_WINDOWS,
            min_periods=1,
        )
        .mean()
    )

    result["protected_rate_5sec"] = (
        result["protected_rate"]
        .rolling(
            window=CONTEXT_WINDOWS,
            min_periods=1,
        )
        .mean()
    )

    result["power_management_rate_5sec"] = (
        result["power_management_rate"]
        .rolling(
            window=CONTEXT_WINDOWS,
            min_periods=1,
        )
        .mean()
    )

    result["fragment_rate_5sec"] = (
        result["fragment_rate"]
        .rolling(
            window=CONTEXT_WINDOWS,
            min_periods=1,
        )
        .mean()
    )

    result["unique_frame_types_5sec"] = (
        result["unique_frame_types"]
        .rolling(
            window=CONTEXT_WINDOWS,
            min_periods=1,
        )
        .mean()
    )

    result["unique_frame_subtypes_5sec"] = (
        result["unique_frame_subtypes"]
        .rolling(
            window=CONTEXT_WINDOWS,
            min_periods=1,
        )
        .mean()
    )

    # --------------------------------------------------------
    # CORRECT WINDOW LABELING
    # --------------------------------------------------------
    #
    # The previous implementation used the FIRST packet's
    # label in a window.
    #
    # That caused windows such as:
    #
    # Normal
    # Normal
    # Deauth
    # Deauth
    #
    # to become Normal.
    #
    # We now examine ALL labels in every window.
    # --------------------------------------------------------

    window_labels = grouped[
        TARGET_COLUMN
    ].agg(
        determine_window_attack_label
    )

    result["attack_type"] = (
        window_labels
        .map(get_attack_type)
        .values
    )

    result["attack"] = (
        window_labels
        .map(get_binary_label)
        .values
    )

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    result["file"] = csv_path.name

    attack_labels = set(
        window_labels[
            window_labels != "Normal"
        ].unique()
    )

    if attack_labels:
        result["file_class"] = sorted(
            attack_labels
        )[0]
    else:
        result["file_class"] = "Normal"

    # --------------------------------------------------------
    # Final cleanup
    # --------------------------------------------------------

    result = result.reset_index()

    return result


# ============================================================
# SPLIT PROCESSING
# ============================================================

def process_split(
    manifest,
    split_name,
):
    """Process every capture belonging to one split."""

    files = manifest[
        manifest["split"] == split_name
    ]["file"].tolist()

    split_features = []

    print(
        f"\nProcessing {split_name.upper()} split..."
    )

    for index, file_path in enumerate(
        files,
        start=1,
    ):

        csv_path = Path(file_path)

        if not csv_path.is_absolute():
            csv_path = (
                PROJECT_ROOT
                / csv_path
            )

        if not csv_path.exists():
            print(
                f"WARNING: file not found: "
                f"{csv_path}"
            )
            continue

        features = build_window_features(
            csv_path
        )

        if not features.empty:
            split_features.append(
                features
            )

        if index % 25 == 0:
            print(
                f"  Processed "
                f"{index:,}/{len(files):,} files"
            )

    if not split_features:
        raise RuntimeError(
            f"No features generated for "
            f"{split_name}."
        )

    return pd.concat(
        split_features,
        ignore_index=True,
    )


# ============================================================
# REPORTING
# ============================================================

def print_split_summary(
    split_name,
    df,
):
    """Print summary of generated windows."""

    print(
        f"\n{split_name.upper()} SUMMARY"
    )

    print(
        f"Windows: {len(df):,}"
    )

    print(
        f"Attack windows: "
        f"{int(df['attack'].sum()):,}"
    )

    print(
        f"Normal windows: "
        f"{int((df['attack'] == 0).sum()):,}"
    )

    print(
        "Windows containing "
        "deauthentication frames: "
        f"{int((df['deauth_frames_per_sec'] > 0).sum()):,}"
    )

    print(
        "\nAttack types:"
    )

    print(
        df["attack_type"]
        .value_counts()
        .to_string()
    )

    print(
        "\nFive-second features:"
    )

    for column in df.columns:
        if "_5sec" in column:
            print(
                f"  {column}"
            )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print(
        "WIDS WINDOW-LEVEL FEATURE ENGINEERING"
    )
    print("=" * 70)

    print(
        f"\nWindow size: "
        f"{WINDOW_SIZE} second"
    )

    print(
        f"Behavioral context: "
        f"{CONTEXT_WINDOWS} seconds"
    )

    if not SPLIT_MANIFEST.exists():
        raise FileNotFoundError(
            f"Split manifest not found: "
            f"{SPLIT_MANIFEST}"
        )

    manifest = pd.read_csv(
        SPLIT_MANIFEST
    )

    print(
        f"Capture files: "
        f"{len(manifest):,}"
    )

    split_outputs = {
        "train": (
            PROCESSED_DIR
            / "train_window_features.csv"
        ),
        "validation": (
            PROCESSED_DIR
            / "validation_window_features.csv"
        ),
        "test": (
            PROCESSED_DIR
            / "test_window_features.csv"
        ),
    }

    for split_name, output_path in (
        split_outputs.items()
    ):

        features = process_split(
            manifest,
            split_name,
        )

        print_split_summary(
            split_name,
            features,
        )

        features.to_csv(
            output_path,
            index=False,
        )

        print(
            f"\nSaved: {output_path}"
        )

    print("\n" + "=" * 70)
    print(
        "WINDOW FEATURE ENGINEERING COMPLETE"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()
