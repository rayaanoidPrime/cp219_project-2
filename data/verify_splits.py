import pandas as pd
import os

base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "SV_MMS_Data")
for ds in ["SV", "MMS"]:
    print(f"\n=== {ds} ===")
    for split in ["test/attack_and_normal.csv", "train/normal_only.csv", "train/attack_and_normal.csv"]:
        path = os.path.join(base, ds, split)
        df = pd.read_csv(path)
        dist = df["class"].value_counts().to_dict()
        print(f"  {split:40s}  rows={len(df):>6,}  classes={dist}")
