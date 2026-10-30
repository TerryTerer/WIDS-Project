import pandas as pd
from sklearn.model_selection import train_test_split

PROFILE = "reports/file_label_profile.csv"
OUTPUT = "reports/split_manifest.csv"

RANDOM_STATE = 42

df = pd.read_csv(PROFILE)

# Actual attack-count columns in file_label_profile.csv
attack_columns = {
    "deauth": "Deauth",
    "beacon_flood": "BeaconFlood",
    "dragon_rain": "DragonRain",
    "downgrade": "Downgrade",
    "unsupported_group": "UnspportGroup",
    "sae_commit_value": "SAECommitValue",
    "dragon_time": "DragonTime",
}


def get_file_class(row):
    present = [
        label
        for column, label in attack_columns.items()
        if row[column] > 0
    ]

    if len(present) == 0:
        return "Normal"

    if len(present) == 1:
        return present[0]

    raise ValueError(
        f"Multiple attack labels found in {row['file']}: {present}"
    )


# Assign exactly one class to every capture file.
df["file_class"] = df.apply(get_file_class, axis=1)

print("=== FILE CLASS DISTRIBUTION ===")
print(df["file_class"].value_counts().to_string())

# ---------------------------------------------------------
# 70% TRAIN / 15% VALIDATION / 15% TEST
# ---------------------------------------------------------

train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    random_state=RANDOM_STATE,
    stratify=df["file_class"],
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=RANDOM_STATE,
    stratify=temp_df["file_class"],
)

train_df = train_df.copy()
val_df = val_df.copy()
test_df = test_df.copy()

train_df["split"] = "train"
val_df["split"] = "validation"
test_df["split"] = "test"

manifest = pd.concat(
    [train_df, val_df, test_df],
    ignore_index=True,
)

manifest = manifest.sort_values(
    ["split", "file"]
).reset_index(drop=True)

manifest.to_csv(OUTPUT, index=False)

# ---------------------------------------------------------
# RESULTS
# ---------------------------------------------------------

print("\n=== SPLIT CREATED ===")
print(f"Total files:       {len(manifest)}")
print(f"Training files:    {len(train_df)}")
print(f"Validation files:  {len(val_df)}")
print(f"Test files:        {len(test_df)}")

print("\n=== FILES BY CLASS AND SPLIT ===")

cross = pd.crosstab(
    manifest["file_class"],
    manifest["split"],
)

# Ensure consistent column order
for column in ["train", "validation", "test"]:
    if column not in cross.columns:
        cross[column] = 0

cross = cross[["train", "validation", "test"]]

print(cross.to_string())

# ---------------------------------------------------------
# CHECK FOR FILE LEAKAGE
# ---------------------------------------------------------

train_files = set(train_df["file"])
val_files = set(val_df["file"])
test_files = set(test_df["file"])

train_val = train_files & val_files
train_test = train_files & test_files
val_test = val_files & test_files

print("\n=== LEAKAGE CHECK ===")
print(f"Train ∩ Validation: {len(train_val)}")
print(f"Train ∩ Test:       {len(train_test)}")
print(f"Validation ∩ Test:  {len(val_test)}")

if not train_val and not train_test and not val_test:
    print("\nPASS: No capture file appears in more than one split.")
else:
    raise RuntimeError(
        "FAIL: Capture-file leakage detected!"
    )

print(f"\nSaved split manifest to:")
print(OUTPUT)
