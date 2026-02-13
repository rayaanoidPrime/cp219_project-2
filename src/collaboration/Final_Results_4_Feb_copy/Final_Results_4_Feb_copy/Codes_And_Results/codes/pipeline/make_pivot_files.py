#!/usr/bin/env python3

import argparse
import sys
import pandas as pd

sys.path.append('c:/Users/Rayaan_Ghosh/Desktop/OSS/cp219_project-2/src/collaboration/Final_Results_4_Feb_copy/Final_Results_4_Feb_copy/Codes_And_Results/codes')
from  utility import resource_usage as ru
from  utility import unsupervised_helper as uh
from  utility import plot_helper as ph
from utility import unsupervised_helper as uh
col=uh.col_name

def make_pivot(df,
               col_dataset='Dataset',
               col_goid='GoID',
               col_attack='Attack_Scn',
               col_algo='Algorithm',
               col_value=col,
               aggfunc='mean'):
    # Check required columns
    missing = [c for c in (col_dataset, col_goid, col_attack, col_algo, col_value) if c not in df.columns]
    if missing:
        raise KeyError(f"Input file is missing required columns: {missing}")

    # Create a combined identifier (optional) and ensure types are strings for consistent indexing
    df[col_dataset] = df[col_dataset].astype(str)
    df[col_goid] = df[col_goid].astype(str)
    df[col_attack] = df[col_attack].astype(str)
    df[col_algo] = df[col_algo].astype(str)

    # Aggregate duplicates (same datafile + algorithm)
    grouped = df.groupby([col_dataset, col_goid, col_attack, col_algo], as_index=False)[col_value].agg(aggfunc)

    # Pivot: index = Dataset / GoID / Attack_Scn, columns = Algorithm, values = PR-AUC
    pivot = grouped.pivot_table(index=[col_dataset, col_goid, col_attack],
                                columns=col_algo,
                                values=col_value,
                                aggfunc='first')  # grouped already aggregated, so first is fine

    # Flatten column names (Algorithm names) if needed
    pivot.columns.name = None
    pivot = pivot.reset_index()
    pivot = pivot.fillna(0)

    return pivot

def main(input_csv,out_csv):
    p = argparse.ArgumentParser(description="Make pivot table by datafile (Dataset/GoID/Attack_Scn) x Algorithm.")

    p.add_argument("--col_dataset", default="Dataset", help="Column name for Dataset (default: Dataset)")
    p.add_argument("--col_goid", default="GoID", help="Column name for GoID (default: GoID)")
    p.add_argument("--col_attack", default="Attack_Scn", help="Column name for Attack scenario (default: Attack_Scn)")
    p.add_argument("--col_algo", default="Algorithm", help="Column name for Algorithm name (default: Algorithm)")
    p.add_argument("--col_value", default=col, help=f"Column name for {col} value (default: {col})")
    p.add_argument("--agg", default="mean", choices=["mean","median","max","min","std"],
                   help="Aggregation function when multiple rows exist for same datafile+algorithm (default: mean)")
    args = p.parse_args()

    try:
        df = pd.read_csv(input_csv)
    except Exception as e:
        print(f"Error reading input file '{input_csv}': {e}", file=sys.stderr)
        sys.exit(2)

    try:
        pivot = make_pivot(df,
                           col_dataset=args.col_dataset,
                           col_goid=args.col_goid,
                           col_attack=args.col_attack,
                           col_algo=args.col_algo,
                           col_value=args.col_value,
                           aggfunc=args.agg)
    except KeyError as ke:
        print(f"Column error: {ke}", file=sys.stderr)
        print("Available columns in input file:", list(df.columns), file=sys.stderr)
        sys.exit(3)
    except Exception as e:
        print(f"Failed to create pivot: {e}", file=sys.stderr)
        sys.exit(4)

    try:
        pivot.to_csv(out_csv, index=False)
        print(f"Pivot table written to: {out_csv}")
    except Exception as e:
        print(f"Error writing output file '{out_csv}': {e}", file=sys.stderr)
        sys.exit(5)

if __name__ == "__main__":

    # Unsupervised Algorithms
    # input_csv = "c:/Users/Rayaan_Ghosh/Desktop/OSS/cp219_project-2/src/collaboration/Final_Results_4_Feb_copy/Final_Results_4_Feb_copy/Codes_And_Results/all_plots_data/combined_results_data/combined_averages.csv" 
    # out_csv = f"c:/Users/Rayaan_Ghosh/Desktop/OSS/cp219_project-2/src/collaboration/Final_Results_4_Feb_copy/Final_Results_4_Feb_copy/Codes_And_Results/all_plots_data/combined_results_data/pivot_unsupervised_{col}.csv"
    
    # Supervised Algorithms
    input_csv = 'c:/Users/Rayaan_Ghosh/Desktop/OSS/cp219_project-2/src/collaboration/Final_Results_4_Feb_copy/Final_Results_4_Feb_copy/Codes_And_Results/all_plots_data/supervised_results/combined_averages.csv' #SUpervised
    out_csv = f"c:/Users/Rayaan_Ghosh/Desktop/OSS/cp219_project-2/src/collaboration/Final_Results_4_Feb_copy/Final_Results_4_Feb_copy/Codes_And_Results/all_plots_data/supervised_results/pivot_supervised_{col}.csv"

    main(input_csv,out_csv)
