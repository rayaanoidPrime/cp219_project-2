"""
Split SV.csv and MMS.csv into train/test (time-series safe, binary labels).

All attack scenario labels are merged into a single binary label:
  - Normal + Fault → attack=0
  - All other classes → attack=1

Strategy (preserves temporal order):
    1. Binarize: Fault+Normal → 0, everything else → 1.  Drop 'class' column.
    2. Separate normal (attack==0) and attack (attack==1) segments.  No shuffle.
    3. Take last NORMAL_TEST_FRAC from normal segment and
       last ATTACK_TEST_FRAC from attack segment → test/attack_and_normal.csv
    4. Remove those test rows from the original df.
    5. From remaining:
       a) Normal rows only            → train/normal_only.csv
       b) All remaining rows          → train/attack_and_normal.csv

Output tree:
    data/SV_MMS_Data/<dataset>/
        test/
            attack_and_normal.csv
        train/
            normal_only.csv
            attack_and_normal.csv
"""

import os
import pandas as pd

# ── config ──────────────────────────────────────────────────────────────────
NORMAL_TEST_FRAC = 0.20       # last 20 % of normal segment → test
ATTACK_TEST_FRAC = 0.30       # last 30 % of attack segment → test
DATA_DIR   = os.path.dirname(os.path.abspath(__file__))
SV_MMS_DIR = os.path.join(DATA_DIR, "SV_MMS_Data")

DATASETS = {
    "SV":  os.path.join(SV_MMS_DIR, "SV.csv"),
    "MMS": os.path.join(SV_MMS_DIR, "MMS.csv"),
}

NORMAL_LABEL = "Normal"
CLASS_COL    = "class"


def _binarize(df: pd.DataFrame) -> pd.DataFrame:
    """Rename 'class' → 'attack': Normal/Fault → 0, everything else → 1."""
    df = df.copy()
    # Merge Fault into Normal first
    if "Fault" in df[CLASS_COL].unique():
        df.loc[df[CLASS_COL] == "Fault", CLASS_COL] = NORMAL_LABEL
    # Binary mapping
    df["attack"] = (df[CLASS_COL] != NORMAL_LABEL).astype(int)
    df.drop(columns=[CLASS_COL], inplace=True)
    return df


def split_dataset(name: str, csv_path: str):
    print(f"\n{'='*60}")
    print(f"  Processing: {name}")
    print(f"{'='*60}")

    df = pd.read_csv(csv_path)
    print(f"  Total rows : {len(df)}")
    print(f"  Class dist : {df[CLASS_COL].value_counts().to_dict()}")

    # Step 1 – binarize (Fault+Normal → 0, attacks → 1)
    df = _binarize(df)
    n_normal = int((df["attack"] == 0).sum())
    n_attack = int((df["attack"] == 1).sum())
    print(f"  After binarize: Normal(0)={n_normal:,}  Attack(1)={n_attack:,}")

    # Step 2 – separate segments (preserve original order)
    normal_segment = df[df["attack"] == 0].copy()
    attack_segment = df[df["attack"] == 1].copy()

    # Step 3 – sample test from the END of each segment
    n_normal_test = max(1, int(len(normal_segment) * NORMAL_TEST_FRAC))
    n_attack_test = max(1, int(len(attack_segment) * ATTACK_TEST_FRAC))

    normal_test  = normal_segment.iloc[-n_normal_test:]
    normal_train = normal_segment.iloc[:-n_normal_test]

    attack_test  = attack_segment.iloc[-n_attack_test:]
    attack_train = attack_segment.iloc[:-n_attack_test]

    print(f"  Normal split: train={len(normal_train):,}  test={n_normal_test:,}")
    print(f"  Attack split: train={len(attack_train):,}  test={n_attack_test:,}")

    # Step 4 – assemble splits
    df_test         = pd.concat([normal_test, attack_test], ignore_index=True)
    df_train_all    = pd.concat([normal_train, attack_train], ignore_index=True)
    df_train_normal = normal_train.copy().reset_index(drop=True)

    # Step 5 – write out (flat structure, no attack-type subfolders)
    out_root  = os.path.join(SV_MMS_DIR, name)
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

    print(f"  Written to: {out_root}")
    print(f"    test/attack_and_normal.csv   → {len(df_test):>6,} rows")
    print(f"    train/normal_only.csv        → {len(df_train_normal):>6,} rows")
    print(f"    train/attack_and_normal.csv  → {len(df_train_all):>6,} rows")


if __name__ == "__main__":
    for name, path in DATASETS.items():
        split_dataset(name, path)
    print("\n✅ Done.")
