"""
Split SV.csv and MMS.csv into per-attack-type subfolders (time-series safe).

For each dataset and each attack class the output tree is:

    data/SV_MMS_Data/<dataset>/<attack_class>/
        test/
            attack_and_normal.csv      # Normal + this attack type  (attack=0/1)
        train/
            normal_only.csv            # Normal rows only           (attack=0)
            attack_and_normal.csv      # Normal + this attack type  (attack=0/1)

Strategy (preserves temporal order, per attack type):
    1. Filter the full CSV to only (Normal + one attack class) rows.
    2. Within each of those two groups, take the last TEST_FRAC as test.
    3. Concatenate per-group test slices   → test/attack_and_normal.csv
    4. From per-group train slices:
       a) Keep only Normal rows            → train/normal_only.csv
       b) Keep all rows                    → train/attack_and_normal.csv
    5. Rename 'class' → 'attack', map Normal→0, attack→1.
"""

import os
import pandas as pd

# ── config ──────────────────────────────────────────────────────────────────
TEST_FRAC = 0.20          # 20 % of each class goes to test
DATA_DIR  = os.path.dirname(os.path.abspath(__file__))
SV_MMS_DIR = os.path.join(DATA_DIR, "SV_MMS_Data")

DATASETS = {
    "SV":  os.path.join(SV_MMS_DIR, "SV.csv"),
    "MMS": os.path.join(SV_MMS_DIR, "MMS.csv"),
}

NORMAL_LABEL = "Normal"
CLASS_COL    = "class"


def _binarize(df: pd.DataFrame, attack_class: str) -> pd.DataFrame:
    """Rename 'class' → 'attack' and map Normal→0, attack_class→1."""
    df = df.copy()
    df["attack"] = (df[CLASS_COL] != NORMAL_LABEL).astype(int)
    df.drop(columns=[CLASS_COL], inplace=True)
    return df


def split_one_attack(name: str, df_full: pd.DataFrame, attack_class: str):
    """Create train/test split for one (dataset, attack_class) pair."""

    # Filter to Normal + this specific attack class
    mask = df_full[CLASS_COL].isin([NORMAL_LABEL, attack_class])
    df = df_full[mask].copy()

    print(f"\n  Attack type: {attack_class}")
    print(f"    Filtered rows : {len(df)}")
    print(f"    Class dist    : {df[CLASS_COL].value_counts().to_dict()}")

    train_parts = []
    test_parts  = []

    # For each class (Normal / attack_class), do a sequential split
    for cls in [NORMAL_LABEL, attack_class]:
        cls_df = df[df[CLASS_COL] == cls].copy()
        n      = len(cls_df)
        n_test = max(1, int(n * TEST_FRAC))
        n_train = n - n_test

        train_parts.append(cls_df.iloc[:n_train])
        test_parts.append(cls_df.iloc[n_train:])

        print(f"      {cls:20s}  train={n_train:>6,}  test={n_test:>6,}")

    # Assemble splits
    df_train_all    = pd.concat(train_parts, ignore_index=True)
    df_test         = pd.concat(test_parts,  ignore_index=True)
    df_train_normal = df_train_all[df_train_all[CLASS_COL] == NORMAL_LABEL].copy()

    # Binarize: class → attack (0/1)
    df_train_all    = _binarize(df_train_all, attack_class)
    df_test         = _binarize(df_test, attack_class)
    df_train_normal = _binarize(df_train_normal, attack_class)

    # ── write out ────────────────────────────────────────────────────────
    out_root  = os.path.join(SV_MMS_DIR, name, attack_class)
    test_dir  = os.path.join(out_root, "test")
    train_dir = os.path.join(out_root, "train")
    os.makedirs(test_dir,  exist_ok=True)
    os.makedirs(train_dir, exist_ok=True)

    test_path         = os.path.join(test_dir,  "attack_and_normal.csv")
    train_normal_path = os.path.join(train_dir, "normal_only.csv")
    train_mixed_path  = os.path.join(train_dir, "attack_and_normal.csv")

    df_test.to_csv(test_path,         index=False)
    df_train_normal.to_csv(train_normal_path, index=False)
    df_train_all.to_csv(train_mixed_path,     index=False)

    print(f"    Written to: {out_root}")
    print(f"      test/attack_and_normal.csv   → {len(df_test):>6,} rows")
    print(f"      train/normal_only.csv        → {len(df_train_normal):>6,} rows")
    print(f"      train/attack_and_normal.csv  → {len(df_train_all):>6,} rows")


def split_dataset(name: str, csv_path: str):
    print(f"\n{'='*60}")
    print(f"  Processing: {name}")
    print(f"{'='*60}")

    df = pd.read_csv(csv_path)
    print(f"  Total rows : {len(df)}")
    print(f"  Class dist : {df[CLASS_COL].value_counts().to_dict()}")

    # Get all non-Normal attack classes
    attack_classes = [c for c in df[CLASS_COL].unique() if c != NORMAL_LABEL]
    print(f"  Attack types: {attack_classes}")

    for attack_class in sorted(attack_classes):
        split_one_attack(name, df, attack_class)


if __name__ == "__main__":
    for name, path in DATASETS.items():
        split_dataset(name, path)
    print("\n✅ Done.")
