import pandas as pd
import os

sv_path = r"c:\Users\Rayaan_Ghosh\Desktop\OSS\cp219_project-2\data\SV_MMS_Data\SV.csv"
mms_path = r"c:\Users\Rayaan_Ghosh\Desktop\OSS\cp219_project-2\data\SV_MMS_Data\MMS.csv"

def print_cols(name, path):
    if os.path.exists(path):
        df = pd.read_csv(path, nrows=1)
        print(f"--- {name} Columns ({len(df.columns)}) ---")
        for c in df.columns:
            print(c)
        print("\n")
    else:
        print(f"{name} not found at {path}")

print_cols("SV", sv_path)
print_cols("MMS", mms_path)
