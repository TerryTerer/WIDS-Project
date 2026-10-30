import argparse
import csv
import math
import shutil
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "hist_gradient_boosting_window.joblib"
)

ALERT_LOG = PROJECT_ROOT / "reports" / "alerts.csv"


# ============================================================
# DETECTION SETTINGS
# ============================================================

WINDOW_SIZE = 1.0
CONTEXT_WINDOWS = 5

# Deauthentication subtype in IEEE 802.11.
DEAUTH_SUBTYPE = 12

# Minimum ML probability before generating an alert.
ATTACK_THRESHOLD = 0.50

# Prevent printing the same alert every second indefinitely.
ALERT_COOLDOWN = 5.0


# ============================================================
# FEATURES USED BY THE WINDOW MODEL
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


BEHAVIORAL_COLUMNS = [
    "frames_per_sec",
    "retry_rate",
    "protected_rate",
    "power_management_rate",
    "fragment_rate",
    "unique_frame_types",
    "unique_frame_subtypes",
    "deauth_frames_per_sec",
    "deauth_ratio",
]


CONTEXT_COLUMNS = [
    "frames_per_sec_5sec",
    "deauth_frames_per_sec_5sec",
    "deauth_ratio_5sec",
    "retry_rate_5sec",
    "protected_rate_5sec",
    "power_management_rate_5sec",
    "fragment_rate_5sec",
    "unique_frame_types_5sec",
    "unique_frame_subtypes_5sec",
]


# ============================================================
# TSHARK FIELDS
# ============================================================

TSHARK_FIELDS = [
    "frame.time_epoch",
    "frame.len",
    "frame.time_delta",
    "wlan.duration",
    "wlan_radio.data_rate",
    "wlan_radio.duration",
    "radiotap.dbm_antsignal",
    "radiotap.datarate",
    "wlan.fc.subtype",
    "wlan.fc.retry",
    "wlan.fc.protected",
    "wlan.fc.pwrmgt",
    "wlan.fc.frag",
    "wlan.fc.type",
]


# ============================================================
# HELPERS
# ============================================================

def safe_float(value):
    """
    Convert a TShark field into float.

    Handles empty fields, malformed values, and common
    Wireshark representations.
    """
    if value is None:
        return np.nan

    value = str(value).strip()

    if not value:
        return np.nan

    try:
        return float(value)
    except ValueError:
        return np.nan


def safe_int(value):
    """
    Convert a TShark field into integer.
    """
    number = safe_float(value)

    if np.isnan(number):
        return 0

    return int(number)


def binary_flag(value):
    """
    Convert a Wireshark boolean-ish field to 0/1.
    """
    if value is None:
        return 0

    value = str(value).strip().lower()

    if value in {
        "1",
        "true",
        "yes",
        "set",
        "present",
    }:
        return 1

    try:
        return int(float(value) != 0)
    except ValueError:
        return 0


def normalize_frame_type(value):
    """
    Convert frame type into a stable string representation.
    """
    if value is None or str(value).strip() == "":
        return "unknown"

    return str(value).strip()


def normalize_subtype(value):
    """
    Convert frame subtype into a stable integer representation.
    """
    return safe_int(value)


def mean_or_zero(series):
    if len(series) == 0:
        return 0.0

    value = pd.to_numeric(series, errors="coerce").mean()

    if pd.isna(value):
        return 0.0

    return float(value)


def std_or_zero(series):
    if len(series) <= 1:
        return 0.0

    value = pd.to_numeric(series, errors="coerce").std()

    if pd.isna(value):
        return 0.0

    return float(value)


def min_or_zero(series):
    if len(series) == 0:
        return 0.0

    value = pd.to_numeric(series, errors="coerce").min()

    if pd.isna(value):
        return 0.0

    return float(value)


def max_or_zero(series):
    if len(series) == 0:
        return 0.0

    value = pd.to_numeric(series, errors="coerce").max()

    if pd.isna(value):
        return 0.0

    return float(value)


# ============================================================
# PACKET PROCESSING
# ============================================================

def convert_packet(row):
    """
    Convert one TShark CSV row into the internal packet format.
    """

    return {
        "timestamp": safe_float(row.get("frame.time_epoch")),

        "frame.len": safe_float(row.get("frame.len")),

        "frame.time_delta": safe_float(
            row.get("frame.time_delta")
        ),

        "wlan.duration": safe_float(
            row.get("wlan.duration")
        ),

        "wlan_radio.data_rate": safe_float(
            row.get("wlan_radio.data_rate")
        ),

        "wlan_radio.duration": safe_float(
            row.get("wlan_radio.duration")
        ),

        "radiotap.dbm_antsignal": safe_float(
            row.get("radiotap.dbm_antsignal")
        ),

        "radiotap.datarate": safe_float(
            row.get("radiotap.datarate")
        ),

        "wlan.fc.subtype": normalize_subtype(
            row.get("wlan.fc.subtype")
        ),

        "wlan.fc.retry": binary_flag(
            row.get("wlan.fc.retry")
        ),

        "wlan.fc.protected": binary_flag(
            row.get("wlan.fc.protected")
        ),

        "wlan.fc.pwrmgt": binary_flag(
            row.get("wlan.fc.pwrmgt")
        ),

        "wlan.fc.frag": binary_flag(
            row.get("wlan.fc.frag")
        ),

        "wlan.fc.type": normalize_frame_type(
            row.get("wlan.fc.type")
        ),
    }


def create_window_features(packets):
    """
    Convert packets belonging to one 1-second window into
    the same 46-feature representation used by the trained
    window model.
    """

    if not packets:
        return None

    df = pd.DataFrame(packets)

    result = {}

    # --------------------------------------------------------
    # Statistical features
    # --------------------------------------------------------

    for column in AGGREGATE_COLUMNS:
        result[f"{column}_mean"] = mean_or_zero(df[column])
        result[f"{column}_std"] = std_or_zero(df[column])
        result[f"{column}_min"] = min_or_zero(df[column])
        result[f"{column}_max"] = max_or_zero(df[column])

    # --------------------------------------------------------
    # Behavioral features
    # --------------------------------------------------------

    packet_count = len(df)

    result["frames_per_sec"] = float(packet_count)

    result["retry_rate"] = float(
        df["wlan.fc.retry"].mean()
    )

    result["protected_rate"] = float(
        df["wlan.fc.protected"].mean()
    )

    result["power_management_rate"] = float(
        df["wlan.fc.pwrmgt"].mean()
    )

    result["fragment_rate"] = float(
        df["wlan.fc.frag"].mean()
    )

    result["unique_frame_types"] = int(
        df["wlan.fc.type"].nunique()
    )

    result["unique_frame_subtypes"] = int(
        df["wlan.fc.subtype"].nunique()
    )

    deauth_mask = (
        df["wlan.fc.subtype"] == DEAUTH_SUBTYPE
    )

    deauth_count = int(deauth_mask.sum())

    result["deauth_frames_per_sec"] = float(
        deauth_count
    )

    result["deauth_ratio"] = float(
        deauth_count / packet_count
    )

    return result


def add_context_features(current_features, history):
    """
    Add 5-second rolling behavioral context.

    history contains the most recent completed 1-second
    feature windows.
    """

    windows = list(history)

    if not windows:
        windows = [current_features]

    frames = [
        window["frames_per_sec"]
        for window in windows
    ]

    deauth_rates = [
        window["deauth_frames_per_sec"]
        for window in windows
    ]

    retry_rates = [
        window["retry_rate"]
        for window in windows
    ]

    protected_rates = [
        window["protected_rate"]
        for window in windows
    ]

    power_rates = [
        window["power_management_rate"]
        for window in windows
    ]

    fragment_rates = [
        window["fragment_rate"]
        for window in windows
    ]

    result = dict(current_features)

    result["frames_per_sec_5sec"] = float(
        np.mean(frames)
    )

    result["deauth_frames_per_sec_5sec"] = float(
        np.mean(deauth_rates)
    )

    total_frames = sum(frames)

    total_deauth = sum(
        window["deauth_frames_per_sec"]
        for window in windows
    )

    if total_frames > 0:
        result["deauth_ratio_5sec"] = float(
            total_deauth / total_frames
        )
    else:
        result["deauth_ratio_5sec"] = 0.0

    result["retry_rate_5sec"] = float(
        np.mean(retry_rates)
    )

    result["protected_rate_5sec"] = float(
        np.mean(protected_rates)
    )

    result["power_management_rate_5sec"] = float(
        np.mean(power_rates)
    )

    result["fragment_rate_5sec"] = float(
        np.mean(fragment_rates)
    )

    result["unique_frame_types_5sec"] = int(
        sum(
            window["unique_frame_types"]
            for window in windows
        )
    )

    result["unique_frame_subtypes_5sec"] = int(
        sum(
            window["unique_frame_subtypes"]
            for window in windows
        )
    )

    return result


# ============================================================
# MODEL
# ============================================================

def load_model():
    if not MODEL_PATH.exists():
        print(
            f"ERROR: Model not found:\n{MODEL_PATH}"
        )
        sys.exit(1)

    saved = joblib.load(MODEL_PATH)

    if isinstance(saved, dict):
        model = saved["model"]
        feature_names = saved["features"]
    else:
        model = saved
        feature_names = None

    if feature_names is None:
        print(
            "ERROR: Saved model does not contain "
            "feature names."
        )
        sys.exit(1)

    print(
        f"Loaded model: {type(model).__name__}"
    )

    print(
        f"Expected features: {len(feature_names)}"
    )

    return model, feature_names


# ============================================================
# ALERT LOGGING
# ============================================================

def initialize_alert_log():
    ALERT_LOG.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if ALERT_LOG.exists():
        return

    with open(
        ALERT_LOG,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.writer(file)

        writer.writerow(
            [
                "timestamp",
                "window_timestamp",
                "prediction",
                "attack_probability",
                "attack_type",
                "frames_per_sec",
                "deauth_frames_per_sec",
                "deauth_ratio",
                "deauth_frames_per_sec_5sec",
                "deauth_ratio_5sec",
            ]
        )


def log_alert(
    window_timestamp,
    prediction,
    attack_probability,
    attack_type,
    features,
):
    initialize_alert_log()

    with open(
        ALERT_LOG,
        "a",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.writer(file)

        writer.writerow(
            [
                time.strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
                f"{window_timestamp:.3f}",
                prediction,
                f"{attack_probability:.6f}",
                attack_type,
                f"{features['frames_per_sec']:.2f}",
                f"{features['deauth_frames_per_sec']:.2f}",
                f"{features['deauth_ratio']:.6f}",
                f"{features['deauth_frames_per_sec_5sec']:.2f}",
                f"{features['deauth_ratio_5sec']:.6f}",
            ]
        )


# ============================================================
# ATTACK TYPE
# ============================================================

def determine_attack_type(features):
    """
    Identify the attack type from the current detection window.

    The trained model is binary, so a specific attack type is
    identified only when the current window contains evidence
    of deauthentication traffic.
    """

    deauth_count = features.get(
        "deauth_frames_per_sec",
        0.0,
    )

    deauth_ratio = features.get(
        "deauth_ratio",
        0.0,
    )

    # Identify Deauthentication only when the CURRENT
    # 1-second window contains deauthentication traffic.
    if deauth_count > 0 or deauth_ratio > 0.10:
        return "Deauthentication"

    # The ML model detected an attack, but the current
    # window does not provide enough evidence to identify it.
    return "Unknown attack"


# ============================================================
# MODEL PREDICTION
# ============================================================

def predict_window(
    model,
    feature_names,
    features,
):
    """
    Run the trained model against one window.
    """

    row = {
        feature: features.get(feature, 0.0)
        for feature in feature_names
    }

    X = pd.DataFrame(
        [row],
        columns=feature_names,
    )

    prediction = int(
        model.predict(X)[0]
    )

    probability = float(
        model.predict_proba(X)[0][1]
    )

    return prediction, probability


# ============================================================
# OUTPUT
# ============================================================

def print_status(
    window_timestamp,
    features,
    prediction,
    probability,
):
    timestamp = time.strftime(
        "%H:%M:%S"
    )

    if prediction == 1:
        attack_type = determine_attack_type(
            features
        )

        print(
            f"\n[{timestamp}] "
            f"🚨 ALERT | "
            f"{attack_type} | "
            f"probability={probability:.2%} | "
            f"fps={features['frames_per_sec']:.1f} | "
            f"deauth/s={features['deauth_frames_per_sec']:.1f} | "
            f"deauth_ratio={features['deauth_ratio']:.2%}"
        )

    else:
        print(
            f"[{timestamp}] "
            f"NORMAL | "
            f"probability={probability:.2%} | "
            f"fps={features['frames_per_sec']:.1f} | "
            f"deauth/s={features['deauth_frames_per_sec']:.1f}"
        )


# ============================================================
# TSHARK
# ============================================================

def check_tshark():
    executable = shutil.which("tshark")

    if executable is None:
        print(
            "ERROR: tshark was not found."
        )

        print(
            "\nInstall it with:"
        )

        print(
            "sudo apt update && sudo apt install tshark"
        )

        sys.exit(1)

    return executable


def tshark_command(
    tshark,
    interface=None,
    pcap=None,
):
    command = [
        tshark,
        "-n",
        "-T",
        "fields",
        "-E",
        "header=y",
        "-E",
        "separator=,",
        "-E",
        "quote=d",
        "-E",
        "occurrence=f",
    ]

    for field in TSHARK_FIELDS:
        command.extend(
            [
                "-e",
                field,
            ]
        )

    if pcap:
        command.extend(
            [
                "-r",
                str(pcap),
            ]
        )

    elif interface:
        command.extend(
            [
                "-i",
                interface,
            ]
        )

    else:
        raise ValueError(
            "Either interface or pcap is required."
        )

    return command


# ============================================================
# DETECTOR
# ============================================================

def run_detector(
    model,
    feature_names,
    interface=None,
    pcap=None,
):
    tshark = check_tshark()

    command = tshark_command(
        tshark,
        interface=interface,
        pcap=pcap,
    )

    print("\nStarting TShark:")
    print(" ".join(command))

    print(
        "\nStarting WIDS detection..."
    )

    print(
        "Press Ctrl+C to stop.\n"
    )

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )

    reader = csv.DictReader(
        process.stdout
    )

    current_window = None
    current_packets = []

    history = deque(
        maxlen=CONTEXT_WINDOWS - 1
    )

    last_alert_time = 0.0

    packet_count = 0
    window_count = 0

    try:

        for row in reader:

            packet = convert_packet(row)

            timestamp = packet["timestamp"]

            if (
                timestamp is None
                or math.isnan(timestamp)
            ):
                continue

            packet_count += 1

            window_id = int(
                timestamp / WINDOW_SIZE
            )

            if current_window is None:
                current_window = window_id

            if window_id != current_window:

                if current_packets:

                    features = (
                        create_window_features(
                            current_packets
                        )
                    )

                    if features is not None:

                        context_features = (
                            add_context_features(
                                features,
                                history,
                            )
                        )

                        prediction, probability = (
                            predict_window(
                                model,
                                feature_names,
                                context_features,
                            )
                        )

                        print_status(
                            current_window
                            * WINDOW_SIZE,
                            context_features,
                            prediction,
                            probability,
                        )

                        window_count += 1

                        if (
                            prediction == 1
                            and probability
                            >= ATTACK_THRESHOLD
                        ):

                            now = time.time()

                            if (
                                now
                                - last_alert_time
                                >= ALERT_COOLDOWN
                            ):

                                attack_type = (
                                    determine_attack_type(
                                        context_features
                                    )
                                )

                                log_alert(
                                    current_window
                                    * WINDOW_SIZE,
                                    prediction,
                                    probability,
                                    attack_type,
                                    context_features,
                                )

                                last_alert_time = now

                        history.append(
                            features
                        )

                current_packets = []
                current_window = window_id

            current_packets.append(packet)

    except KeyboardInterrupt:

        print(
            "\n\nStopping WIDS detector..."
        )

    finally:

        process.terminate()

        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()

    print(
        "\nDetection stopped."
    )

    print(
        f"Packets processed: {packet_count:,}"
    )

    print(
        f"Windows processed: {window_count:,}"
    )

    print(
        f"Alert log: {ALERT_LOG}"
    )


# ============================================================
# ARGUMENTS
# ============================================================

def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "ML-based Wireless Intrusion Detection System"
        )
    )

    source = parser.add_mutually_exclusive_group(
        required=True
    )

    source.add_argument(
        "--interface",
        "-i",
        help=(
            "Wireless interface to monitor, "
            "for example wlan0mon"
        ),
    )

    source.add_argument(
        "--pcap",
        "-r",
        help=(
            "Read traffic from a PCAP file "
            "instead of a live interface"
        ),
    )

    return parser.parse_args()


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("MACHINE-LEARNING WIRELESS INTRUSION DETECTION SYSTEM")
    print("=" * 70)

    args = parse_arguments()

    model, feature_names = load_model()

    if args.pcap:

        pcap_path = Path(args.pcap)

        if not pcap_path.exists():
            print(
                f"ERROR: PCAP does not exist:\n{pcap_path}"
            )
            sys.exit(1)

        print(
            "\nInput: PCAP replay"
        )

        print(
            f"PCAP: {pcap_path}"
        )

    else:

        print(
            "\nInput: live interface"
        )

        print(
            f"Interface: {args.interface}"
        )

    initialize_alert_log()

    run_detector(
        model,
        feature_names,
        interface=args.interface,
        pcap=args.pcap,
    )


if __name__ == "__main__":
    main()
