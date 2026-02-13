import pandas as pd
from pathlib import Path
from typing import Dict, Set, List

# Configuration
ROOT_DIR = r"C:\Users\sengu\Documents\cp219_project-2\data\Final_Datasets\Final_Datasets\preprocessed_new\preprocessed_new"

# Target columns to search for
TARGET_COLUMNS = [
    'integer_7',
    'integer_8',
    'time_diff',
    'timestamp_diff',
    'integer_6',
    'stNum',
    'integer_5',
    'Length',
    'sqNum_diff',
    'sqNum',
    'floatvalue_3',
    'integer_3',
    'floatvalue_1',
    'stNum_diff',
    'time_from_start'
]


def check_columns_in_dataset(root_dir: str, target_cols: List[str]) -> Dict:
    """
    Check which target columns exist in each dataset.
    Returns dict: {dataset_name: {device_name: set(found_columns)}}
    """
    root = Path(root_dir)
    results = {}
    
    # Level 1: Protocols/Datasets
    for protocol_dir in sorted(root.iterdir()):
        if not protocol_dir.is_dir():
            continue
        
        protocol_name = protocol_dir.name
        results[protocol_name] = {}
        
        print(f"\n{'='*80}")
        print(f"Checking Dataset: {protocol_name}")
        print(f"{'='*80}")
        
        # Level 2: Devices
        for device_dir in sorted(protocol_dir.iterdir()):
            if not device_dir.is_dir():
                continue
            
            device_name = device_dir.name
            found_columns = set()
            
            # Level 3: Attack Scenarios
            for scenario_dir in sorted(device_dir.iterdir()):
                if not scenario_dir.is_dir():
                    continue
                
                # Check train directory
                train_csv = scenario_dir / 'train' / 'attack_and_normal.csv'
                if train_csv.exists():
                    try:
                        df = pd.read_csv(train_csv, nrows=0)  # Read only headers
                        csv_columns = set(df.columns)
                        
                        # Find intersection with target columns
                        found = csv_columns.intersection(set(target_cols))
                        found_columns.update(found)
                    except Exception as e:
                        print(f"  ⚠️  Error reading {train_csv}: {e}")
            
            if found_columns:
                results[protocol_name][device_name] = found_columns
                print(f"  ✓ {device_name}: Found {len(found_columns)} columns: {sorted(found_columns)}")
            else:
                results[protocol_name][device_name] = set()
                print(f"  ✗ {device_name}: NONE of the target columns found")
    
    return results

def analyze_results(results: Dict, target_cols: List[str]):
    """Analyze and print summary statistics."""
    print(f"\n{'='*80}")
    print("SUMMARY ANALYSIS")
    print(f"{'='*80}\n")
    
    # Datasets with NO target columns at all
    datasets_with_no_columns = []
    
    for dataset, devices in results.items():
        all_found = set()
        for device, cols in devices.items():
            all_found.update(cols)
        
        if len(all_found) == 0:
            datasets_with_no_columns.append(dataset)
    
    if datasets_with_no_columns:
        print("❌ Datasets with ZERO target columns found:")
        for ds in datasets_with_no_columns:
            print(f"   - {ds}")
    else:
        print("✅ All datasets contain at least one target column")
    
    print(f"\n{'-'*80}\n")
    
    # Column frequency across all datasets
    print("Column Frequency Across All Datasets:")
    print(f"{'-'*80}")
    
    column_frequency = {col: 0 for col in target_cols}
    column_in_datasets = {col: [] for col in target_cols}
    
    for dataset, devices in results.items():
        dataset_cols = set()
        for device, cols in devices.items():
            dataset_cols.update(cols)
        
        for col in dataset_cols:
            column_frequency[col] += 1
            column_in_datasets[col].append(dataset)
    
    # Sort by frequency
    sorted_cols = sorted(column_frequency.items(), key=lambda x: x[1], reverse=True)
    
    for col, freq in sorted_cols:
        if freq > 0:
            datasets = ', '.join(column_in_datasets[col])
            print(f"  {col:20s} → Found in {freq} dataset(s): [{datasets}]")
        else:
            print(f"  {col:20s} → NOT FOUND in any dataset")

def main():
    print("\n" + "="*80)
    print("COLUMN EXISTENCE CHECK")
    print("="*80)
    print(f"Root Directory: {ROOT_DIR}")
    print(f"Target Columns: {len(TARGET_COLUMNS)}")
    print("="*80)
    
    results = check_columns_in_dataset(ROOT_DIR, TARGET_COLUMNS)
    analyze_results(results, TARGET_COLUMNS)
    
    print(f"\n{'='*80}")
    print("✅ Analysis Complete!")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()