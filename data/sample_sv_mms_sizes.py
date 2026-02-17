"""
Create sampled versions of normal_only.csv at different size fractions.

For each dataset (SV, MMS) × each attack type × each fraction:
  - Sample the FIRST N*frac rows from train/normal_only.csv (sequential, preserves time order)
  - Copy test/attack_and_normal.csv and train/attack_and_normal.csv unchanged

Output structure:
  data/SV_MMS_Data_sized/<frac_label>/<dataset>/<attack>/train/normal_only.csv
  data/SV_MMS_Data_sized/<frac_label>/<dataset>/<attack>/train/attack_and_normal.csv
  data/SV_MMS_Data_sized/<frac_label>/<dataset>/<attack>/test/attack_and_normal.csv
"""

import os
import shutil
import pandas as pd

# ── config ──────────────────────────────────────────────────────────────────
DATA_DIR   = os.path.dirname(os.path.abspath(__file__))
SRC_DIR    = os.path.join(DATA_DIR, "SV_MMS_Data")
DST_DIR    = os.path.join(DATA_DIR, "SV_MMS_Data_sized")

DATASETS = {
    "SV":  ["Fault", "Injection", "Replay"],
    "MMS": ["Delay", "Fault", "Modification"],
}

FRACTIONS = [
    (0.06,   "6pct"),
    (0.125,  "12pct"),
    (0.25,   "25pct"),
    (0.50,   "50pct"),
    (0.75,   "75pct"),
    (1.00,   "100pct"),
]


def copy_file(src, dst):
    """Copy a file, creating parent dirs as needed."""
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)


def main():
    for dataset, attack_types in DATASETS.items():
        for attack in attack_types:
            src_root = os.path.join(SRC_DIR, dataset, attack)

            # Paths to the source files
            normal_only_src   = os.path.join(src_root, "train", "normal_only.csv")
            train_mixed_src   = os.path.join(src_root, "train", "attack_and_normal.csv")
            test_src          = os.path.join(src_root, "test",  "attack_and_normal.csv")

            if not os.path.exists(normal_only_src):
                print(f"  ⚠ Missing: {normal_only_src}, skipping")
                continue

            df_normal = pd.read_csv(normal_only_src)
            total_rows = len(df_normal)
            print(f"\n{dataset}/{attack}: {total_rows} normal-only rows")

            for frac, label in FRACTIONS:
                dst_root  = os.path.join(DST_DIR, label, dataset, attack)
                train_dir = os.path.join(dst_root, "train")
                test_dir  = os.path.join(dst_root, "test")
                os.makedirs(train_dir, exist_ok=True)
                os.makedirs(test_dir,  exist_ok=True)

                # Sample normal_only: first N*frac rows (sequential)
                n_sample = max(1, int(total_rows * frac))
                df_sampled = df_normal.iloc[:n_sample]
                dst_normal = os.path.join(train_dir, "normal_only.csv")
                df_sampled.to_csv(dst_normal, index=False)

                # Copy train/attack_and_normal.csv unchanged
                copy_file(train_mixed_src, os.path.join(train_dir, "attack_and_normal.csv"))

                # Copy test/attack_and_normal.csv unchanged
                copy_file(test_src, os.path.join(test_dir, "attack_and_normal.csv"))

                print(f"  {label:>7s}: {n_sample:>6,} / {total_rows:>6,} normal rows")

    print("\n✅ All sampled datasets created.")


if __name__ == "__main__":
    main()
