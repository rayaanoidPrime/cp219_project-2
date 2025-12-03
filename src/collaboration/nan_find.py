import os
import pandas as pd
from pathlib import Path

def discover_csv_files(root_folder: str):
    """
    Recursively find all CSV files in the given directory.
    Returns a list of absolute file paths.
    """
    root = Path(root_folder)
    return list(root.rglob("*.csv"))


def analyze_nan_counts(file_path: Path):
    """
    Load a dataset and return NaN counts per column.
    Returns a dict: {column_name: nan_count}
    """
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"[ERROR] Could not read {file_path}: {e}")
        return None

    nan_counts = df.isna().sum()
    nan_counts = nan_counts[nan_counts > 0]  # Keep only columns with NaNs

    return nan_counts.to_dict()


def analyze_all_datasets(root_folder: str):
    """
    Main function:
    - Discover all CSV files
    - Compute NaN statistics
    - Print clean report
    """
    csv_files = discover_csv_files(root_folder)

    if not csv_files:
        print("No CSV files found.")
        return

    print(f"\n🔍 Found {len(csv_files)} datasets in '{root_folder}'")
    print("=" * 80)

    results = {}

    for file_path in csv_files:
        nan_info = analyze_nan_counts(file_path)

        print(f"\n📌 Dataset: {file_path}")
        print("-" * 80)

        if nan_info is None:
            print("  ❌ Could not analyze (read error)")
            continue

        if len(nan_info) == 0:
            print("  ✅ No NaN values found.")
        else:
            for col, count in nan_info.items():
                print(f"  • {col}: {count} NaNs")

        results[str(file_path)] = nan_info

    print("\n" + "=" * 80)
    print("✔ Completed scanning all datasets.\n")

    return results


ROOT_DIR = r"C:\Users\sengu\Documents\cp219_project-2\data\Final_Datasets\Final_Datasets\preprocessed_new\preprocessed_new"
results = analyze_all_datasets(f"{ROOT_DIR}/")
