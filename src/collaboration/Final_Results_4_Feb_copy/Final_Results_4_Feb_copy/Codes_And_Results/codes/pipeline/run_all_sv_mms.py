import os
import csv
import sys

import warnings
import subprocess

# Headless plotting backend (no figures here; just data dumping)
import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd

sys.path.append('c:/Users/Rayaan_Ghosh/Desktop/OSS/cp219_project-2/src/collaboration/Final_Results_4_Feb_copy/Final_Results_4_Feb_copy/Codes_And_Results/codes')
from  utility import resource_usage as ru
from  utility import unsupervised_helper as uh
from  utility import plot_helper as ph

from  plot_bar import make_bar_plots as plot_bars
import AggreateResults as aggregateResults
import make_pivot_files as makePivotFiles
import plot_tables as plotTables
import run_friedman2 as ftest2

# ============== ALGORITHMS ==============

ALGORITHMS_DIR = 'c:/Users/Rayaan_Ghosh/Desktop/OSS/cp219_project-2/src/collaboration/Final_Results_4_Feb_copy/Final_Results_4_Feb_copy/Codes_And_Results/codes/algorithms'


# -----------------------------------------
#   Dataset discovery  (SV / MMS layout)
# -----------------------------------------
#   root_dir/
#     SV/
#       Replay/
#         train/   normal_only.csv, attack_and_normal.csv
#         test/    attack_and_normal.csv
#       Injection/
#         ...
#     MMS/
#       Delay/
#         ...
#
#   Hierarchy:  dataset = SV|MMS
#               goid    = 'all'           (no GoID for SV/MMS)
#               attack_type = Replay|Injection|Fault|...
# -----------------------------------------
def find_dataset_paths(root_dir, validation=False):
    all_paths = []
    if not os.path.isdir(root_dir):
        print(f"Error: Root directory not found at '{root_dir}'")
        return all_paths
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if 'train' in dirnames and 'test' in dirnames:
            train_dir = os.path.join(dirpath, 'train')
            test_dir  = os.path.join(dirpath, 'test')
            if validation:
                validation_input_path = os.path.join(train_dir, 'attack_and_normal.csv')
                train_input_path      = os.path.join(train_dir, 'normal_only.csv')
            else:
                validation_input_path = None
                train_input_path      = os.path.join(train_dir, 'attack_and_normal.csv')
            test_input_path = os.path.join(test_dir, 'attack_and_normal.csv')

            if os.path.exists(train_input_path) and os.path.exists(test_input_path) and \
               (os.path.exists(validation_input_path) if validation else True):
                relative_path = os.path.relpath(dirpath, root_dir)
                parts = relative_path.split(os.sep)

                # SV/MMS layout: parts[0] = dataset (SV|MMS),
                #                parts[1] = attack_type
                if len(parts) >= 2:
                    pg = {
                        'dataset': parts[0],
                        'goid': 'all',
                        'attack_type': parts[-1],
                        'train_input_path': train_input_path,
                        'test_input_path': test_input_path
                    }
                    if validation:
                        pg['validation_input_path'] = validation_input_path
                    all_paths.append(pg)
    return all_paths

# -----------------------------------------
#   MAIN
# -----------------------------------------
def main(algorithm=None,
        root_directory=None,
        validation=True,
        use_freq=True,
        use_features='original',
        scaled_input=True
    ):
    warnings.filterwarnings('ignore')

    datasets = find_dataset_paths(root_directory, validation=validation)
    if not datasets:
        print("No datasets found. Check directory structure.")
        return

    total_datasets = len(datasets)

    for i, ds in enumerate(datasets):
        try:
            progress = (i + 1) / total_datasets
            bar_len = 40
            bar = '#' * int(round(bar_len * progress)) + '-' * (bar_len - int(round(bar_len * progress)))

            line = (
                f"progress: [{bar}] {progress:.1%} ({i+1}/{total_datasets})"
                f" | Processing: {ds['dataset']} > {ds['attack_type']}"
            )
            # Clear previous line completely before writing a new one
            try:
                sys.stdout.write('\r' + ' ' * (os.get_terminal_size().columns - 1) + '\r')
                sys.stdout.write(line)
                sys.stdout.flush()
            except Exception:
                pass

            output_dir = os.path.join('results_dir_sv_mms', ds['dataset'], ds['goid'], ds['attack_type'])
            os.makedirs(output_dir, exist_ok=True)

            script_name = f"{ALGORITHMS_DIR}/{algorithm}.py"

            cmd = [
                sys.executable,                 # Use the same Python as the parent process
                script_name,                    # The script to execute
                '--train_input_path', ds['train_input_path'],
                '--test_input_path', ds['test_input_path'],
                '--validation_input_path',ds.get('validation_input_path'),
                '--output_dir', output_dir,
                '--dataset',ds['dataset'],
                '--goid',ds['goid'],
                '--attack_type',ds['attack_type'],
            ]

            original_stdout = sys.stdout
            # sys.stdout = open(os.devnull, 'w')
            
            subprocess.run(cmd, check=True)
            
            # Restore stdout
            # sys.stdout.close()
            # sys.stdout = original_stdout

        
        except Exception as e:
            print(f"Encountered error {e}, skipping dataset : {ds['dataset']} > {ds['attack_type']}")




if __name__ == "__main__":

    algorithms = [
        # ---IQR-------
        # 'iqr_mom',
        # 'iqr',
        # 'z_score',
        # # ---Clustering-------
        # 'pam',
        # 'clara',
        'optics',
        'singleLink',
        'ward',
        'spectral_clustering',
        'kmeans',
        'gmm',
        'hbos',
        # ---Reconstruction Err-------
        'pca',
        'vae',
        'ae',
        # ---AD-------
        'lof',
        'If',
        'ocsvm',
        'svdd',
        # ---supervised-------
        # 'LR',
        # 'NB',
        # 'DT',
        # 'RF',
        # 'XGB',
        # 'NN',
        # 'KNN',
        # 'SVM',
        # ---supervised Deep-Learning-------
        # 'RNN',
        # 'CNN',
        # 'RNN_LSTM'
    ]

    root_input_directory = 'c:/Users/Rayaan_Ghosh/Desktop/OSS/cp219_project-2/data/SV_MMS_Data'


    for algorithm in algorithms:
        validation=True
        print()
        print("="*100)
        print(f"Running Algorithm : {algorithm}")

        main(algorithm=algorithm,
            root_directory=root_input_directory,
            validation=True,
            use_freq=False,
            use_features='original',
            scaled_input=True
            )
        print()
