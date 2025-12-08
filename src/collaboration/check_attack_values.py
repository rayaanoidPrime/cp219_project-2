"""
Check which datasets have non-binary values in the 'attack' column.
"""
import os
import pandas as pd
from pathlib import Path

# Configuration
BASE_DIR = Path(r"C:\Users\sengu\Documents\cp219_project-2\data\Final_Datasets\Final_Datasets\preprocessed_new\preprocessed_new")
TARGET_COL = "attack"
TRAIN_FILE = "attack_and_normal.csv"
TEST_FILE = "attack_and_normal.csv"

print("=" * 80)
print("ATTACK COLUMN VALUE CHECK")
print("=" * 80)
print(f"Root Directory: {BASE_DIR}")
print(f"Target Column: {TARGET_COL}")
print("=" * 80)

# Find all datasets
datasets_with_issues = []
datasets_checked = 0

for train_dir in BASE_DIR.rglob("train"):
    test_dir = train_dir.parent / "test"
    train_file = train_dir / TRAIN_FILE
    test_file = test_dir / TEST_FILE
    
    if not train_file.exists() or not test_file.exists():
        continue
    
    dataset_name = str(train_dir.parent.relative_to(BASE_DIR))
    datasets_checked += 1
    
    try:
        # Load train data
        train_df = pd.read_csv(train_file)
        test_df = pd.read_csv(test_file)
        
        issues = []
        
        # Check train data
        if TARGET_COL in train_df.columns:
            train_values = train_df[TARGET_COL].unique()
            train_dtype = train_df[TARGET_COL].dtype
            train_min = train_df[TARGET_COL].min()
            train_max = train_df[TARGET_COL].max()
            
            # Check if values are outside 0-1 range
            if train_min < 0 or train_max > 1:
                issues.append(f"TRAIN: Values out of 0-1 range (min={train_min}, max={train_max})")
            
            # Check if there are more than 2 unique values
            if len(train_values) > 2:
                issues.append(f"TRAIN: More than 2 unique values: {sorted(train_values)[:10]}...")
            
            # Check for NaN
            if train_df[TARGET_COL].isna().any():
                issues.append(f"TRAIN: Contains NaN values ({train_df[TARGET_COL].isna().sum()} NaNs)")
            
            # Check dtype
            if train_dtype not in ['int64', 'int32', 'int', 'float64', 'float32']:
                issues.append(f"TRAIN: Unusual dtype: {train_dtype}")
        else:
            issues.append(f"TRAIN: Missing '{TARGET_COL}' column!")
        
        # Check test data
        if TARGET_COL in test_df.columns:
            test_values = test_df[TARGET_COL].unique()
            test_min = test_df[TARGET_COL].min()
            test_max = test_df[TARGET_COL].max()
            
            if test_min < 0 or test_max > 1:
                issues.append(f"TEST: Values out of 0-1 range (min={test_min}, max={test_max})")
            
            if len(test_values) > 2:
                issues.append(f"TEST: More than 2 unique values: {sorted(test_values)[:10]}...")
            
            if test_df[TARGET_COL].isna().any():
                issues.append(f"TEST: Contains NaN values ({test_df[TARGET_COL].isna().sum()} NaNs)")
        else:
            issues.append(f"TEST: Missing '{TARGET_COL}' column!")
        
        if issues:
            datasets_with_issues.append({
                'name': dataset_name,
                'issues': issues,
                'train_unique': list(train_values) if TARGET_COL in train_df.columns else None,
                'test_unique': list(test_values) if TARGET_COL in test_df.columns else None
            })
            print(f"\n⚠️  {dataset_name}")
            for issue in issues:
                print(f"    → {issue}")
        else:
            # Show info for datasets with proper binary values
            print(f"✓ {dataset_name}: Train={list(train_values)}, Test={list(test_values)}")
            
    except Exception as e:
        print(f"❌ Error checking {dataset_name}: {str(e)}")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"Total datasets checked: {datasets_checked}")
print(f"Datasets with issues: {len(datasets_with_issues)}")

if datasets_with_issues:
    print("\n📋 DATASETS WITH NON-STANDARD ATTACK VALUES:")
    for ds in datasets_with_issues:
        print(f"\n  {ds['name']}:")
        if ds['train_unique']:
            print(f"    Train unique values: {ds['train_unique']}")
        if ds['test_unique']:
            print(f"    Test unique values: {ds['test_unique']}")
        for issue in ds['issues']:
            print(f"    • {issue}")
else:
    print("\n✅ All datasets have proper binary (0/1) attack values!")
