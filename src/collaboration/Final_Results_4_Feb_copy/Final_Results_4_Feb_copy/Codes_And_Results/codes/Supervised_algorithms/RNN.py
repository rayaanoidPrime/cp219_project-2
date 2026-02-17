import os
import sys
import argparse
import numpy as np
import pandas as pd
import random
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import (
    confusion_matrix, classification_report,
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    balanced_accuracy_score, matthews_corrcoef
)

import torch
import torch.nn as nn
import torch.optim as optim
# --- Project utilities ---
sys.path.append('c:/Users/Rayaan_Ghosh/Desktop/OSS/cp219_project-2/src/collaboration/Final_Results_4_Feb_copy/Final_Results_4_Feb_copy/Codes_And_Results/codes')
import utility.resource_usage as ru
import utility.unsupervised_helper as uh
import utility.plot_helper as ph

LABEL_COL   = 'attack'
ALGO_NAME   = 'RNN'
IF_DATA_DIR = 'c:/Users/Rayaan_Ghosh/Desktop/OSS/cp219_project-2/src/collaboration/Final_Results_4_Feb_copy/Final_Results_4_Feb_copy/OCT7Files/unsupervised_codes_Shailja/LR_data'
RESULTS_DIR = 'c:/Users/Rayaan_Ghosh/Desktop/OSS/cp219_project-2/src/collaboration/Final_Results_4_Feb_copy/Final_Results_4_Feb_copy/Codes_And_Results/results/Individual_algorithms'
TRIPLET_DIR = 'c:/Users/Rayaan_Ghosh/Desktop/OSS/cp219_project-2/src/collaboration/Final_Results_4_Feb_copy/Final_Results_4_Feb_copy/Codes_And_Results/triplet_plot_data'


def _ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p

# =============================================================
# Set Up Logging
# =============================================================
import logging
logging.basicConfig(filename =uh.LOG_FILE,level = logging.INFO)
logger = logging.getLogger(ALGO_NAME)
def log_run_status(ds, run_key, success=True, e=None):
    # 1. Define the icons
    # ✅ = U+2705, ❌ = U+274C
    status_icon = "✅ Success" if success else "❌ Failed "

    # 2. Extract variables (as per your request)
    dataset = ds['dataset']
    attack = ds['attack_type']
    goid = ds['goid']
    split = "Test" # Hardcoded as requested
    run = run_key

    # 3. Create the formatted string with alignment
    # :<15 means "align left, occupy 15 spaces"
    log_msg = (
        f"{dataset:<15} | "
        f"{attack:<15} | "
        f"{goid:<10} | "
        f"{split:<6} | "
        f"Run {run:<4} | "
        f"{status_icon}"
        f"Error {e}"
    )

    # 4. Log it
    if success:
        logger.info(log_msg)
    else:
        # Use logger.error if it failed, so it highlights in log viewers
        logger.error(log_msg)

     
# ==============================================================================
# Core primitives (capture cpu/mem/time)
# ==============================================================================

def run_rnn(X_train, y_train,
            hidden_size=64,
            num_layers=1,
            lr=1e-3,
            num_epochs=50):

    n_samples, n_features = X_train.shape
    seq_len = n_features
    input_size = 1

    class SimpleRNN(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers):
            super().__init__()
            self.rnn = nn.RNN(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                nonlinearity='tanh',
                batch_first=True
            )
            self.fc = nn.Linear(hidden_size, 1)

        def forward(self, x):
            # x: (B, seq_len, input_size)
            out, _ = self.rnn(x)
            last_out = out[:, -1, :]
            logits = self.fc(last_out)
            return logits

    model = SimpleRNN(input_size, hidden_size, num_layers)

    X_tensor = torch.tensor(
        X_train, dtype=torch.float32
    ).view(n_samples, seq_len, 1)

    y_tensor = torch.tensor(
        y_train, dtype=torch.float32
    ).view(-1, 1)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    for _ in range(num_epochs):
        optimizer.zero_grad()
        logits = model(X_tensor)
        loss = criterion(logits, y_tensor)
        loss.backward()
        optimizer.step()

    return model



def predict(model, X, threshold=0.5, batch_size=1024):
    """
    Returns:
      y_pred  : binary predictions
      scores  : sigmoid probabilities
    """

    model.eval()
    n_samples, n_features = X.shape
    seq_len = n_features

    all_probs = []

    for i in range(0, n_samples, batch_size):
        xb = X[i:i+batch_size]
        xb_tensor = torch.tensor(
            xb, dtype=torch.float32
        ).view(-1, seq_len, 1)

        with torch.no_grad():
            logits = model(xb_tensor)
            probs = torch.sigmoid(logits).view(-1).cpu().numpy()

        all_probs.append(probs)

    scores = np.concatenate(all_probs, axis=0)
    y_pred = (scores >= threshold).astype(int)

    return y_pred, scores




# ==============================================================================
# Main pipeline (same outputs; adds resources into Misc)
# ==============================================================================
def main(train_input_path=None,
         test_input_path=None,
         validation_input_path=None,
         output_dir=None,
         scaled_input=False,
         use_freq=False,
         use_features='all',
         contamination=0.05,
         n_estimators=400,
         max_samples='auto',
                  ds=None):

    # ---------------- load data ----------------
    df_train = pd.read_csv(train_input_path)
    df_val   = pd.read_csv(validation_input_path)
    df_test  = pd.read_csv(test_input_path)

    # Extract features/labels via project helper (unchanged core flow)
    X_train, X_val, X_test, \
    y_train, y_val, y_test, \
    df_train, df_val, df_test, \
    cols = uh.get_trainable_data(
        df_train=df_train,
        df_val=df_val,
        df_test=df_test,
        scaled_input=scaled_input,
        use_freq=use_freq,
        use_features=use_features
    )

    n_train_attack, n_train_normal = uh.count_stat(y_train).get(1, 0), uh.count_stat(y_train).get(0, 0)
    n_test_attack,  n_test_normal  = uh.count_stat(y_test).get(1, 0),  uh.count_stat(y_test).get(0, 0)
    n_val_attack,   n_val_normal   = uh.count_stat(y_val).get(1, 0),   uh.count_stat(y_val).get(0, 0)

    # Keep features in the frames (for convenient saving)
    df_train[cols] = X_train; df_val[cols] = X_val; df_test[cols] = X_test

    # Hierarchy & output dirs
    dataset, goid, attack = ph.infer_hierarchy_from_output_dir(output_dir or "")
    vis_base = os.path.join(IF_DATA_DIR, dataset, goid, attack)
    _ensure_dir(vis_base)
    _ensure_dir(RESULTS_DIR)
    triplet_root = os.path.join(TRIPLET_DIR, ALGO_NAME, dataset, goid, attack)

    all_runs_results = {}
    N_LOOPS = 3
    num_runs = 1
    # ==============================================================
    #                         MAIN LOOP
    # ==============================================================
    for i in range(1, num_runs + 1):
        run_key = f"Run_{i}"
        run_dir = os.path.join(vis_base, run_key)
        _ensure_dir(run_dir)
        try:

            # ---------------- Fit IF (capturing usage) ----------------
            with ru.ResourceProfiler() as profiler_train:
                # Loop the training task N_LOOPS times
                for _ in range(N_LOOPS):
                    model= run_rnn(
                        X_train=X_val,
                        y_train=y_val
                    )
            avg_train_wall_ns = profiler_train.wall_nanoseconds / N_LOOPS
            avg_time_pkt_tr = avg_train_wall_ns / len(y_train) if len(y_train) else None


            # ---------------- TEST predict (capturing usage) -----------------

            with ru.ResourceProfiler() as profiler:
                # Loop the training task N_LOOPS times
                for _ in range(N_LOOPS):        
                        y_test_pred, scores = predict(
                    model, X_test
                )
            avg_test_wall_ns = profiler.wall_nanoseconds / N_LOOPS
            avg_time_pkt_te = avg_test_wall_ns / len(y_test) if len(y_test) else None

            cm_te= confusion_matrix(y_test, y_test_pred, labels=[0,1])
            tn_te, fp_te, fn_te, tp_te = cm_te.ravel()
            rpt_te = classification_report(y_test, y_test_pred, labels=[0,1], target_names=['normal','attack'],
                                        zero_division=0, output_dict=True)
            precision_anom_te = precision_score(y_test, y_test_pred, pos_label=1, zero_division=0)
            recall_anom_te    = recall_score(y_test, y_test_pred, pos_label=1, zero_division=0)
            f1_anom_te        = f1_score(y_test, y_test_pred, pos_label=1, zero_division=0)
            balanced_acc_te   = balanced_accuracy_score(y_test, y_test_pred)
            mcc_te            = matthews_corrcoef(y_test, y_test_pred)

            pr_auc = average_precision_score(y_test, scores)
            roc_auc = roc_auc_score(y_test, scores)
            test_json = {
                "Normal count"          : int(n_test_normal),
                "Attack count"          : int(n_test_attack),
                "Total"                 : int(n_test_normal + n_test_attack),
                "tp"                    : int(tp_te),
                "tn"                    : int(tn_te),
                "fp"                    : int(fp_te),
                "fn"                    : int(fn_te),
                "Accuracy %"            : uh.r2(rpt_te['accuracy']*100),
                "Precision_anom %"           : uh.r2(precision_anom_te*100),
                "Precision %"           : uh.r2(rpt_te["macro avg"]["precision"]*100),
                "Recall_anom %"              : uh.r2(recall_anom_te*100),
                "Recall %"           : uh.r2(rpt_te["macro avg"]["recall"]*100),
                "F1-Score_anom %"       : uh.r2(f1_anom_te*100),
                "F1-Score %"            : uh.r2(rpt_te["macro avg"]["f1-score"]*100),
                "BalancedAcc %"         : uh.r2(balanced_acc_te*100),
                "MCC"                   : uh.r3(mcc_te),
                "PR-AUC"                : uh.r3(pr_auc*100),
                "ROC-AUC"               : uh.r3(roc_auc*100),
                "TotalTime (ms)"        : uh.r3(avg_test_wall_ns / 1_000_000),
                "AvgTimePerPacket(ns)"  : uh.r3(avg_time_pkt_te),
                "Ram_usage"             : uh.r3(profiler.peak_ram_mb),
                "CPU_avg%"              : uh.r3(profiler.cpu_avg_machine_pct),
                "CPU_peak%"             : uh.r3(profiler.cpu_peak_machine_pct),
                "training_time_ms"     : uh.r3(avg_train_wall_ns / 1_000_000),
                "training_avg_time_per_packet_ns": uh.r3(avg_time_pkt_tr),
                "training_peak_ram_mb"  : uh.r3(profiler_train.peak_ram_mb),
                "training_cpu_avg_pct"  : uh.r3(profiler_train.cpu_avg_machine_pct),
                "training_cpu_peak_pct" : uh.r3(profiler_train.cpu_peak_machine_pct),           
                "n_train_attack"        : int(n_val_attack),
                "n_train_normal"        : int(n_val_normal)               
            }
            log_run_status(ds, run_key, success=True)
        except Exception as e:
            print(f"Error during run {i} of {ALGO_NAME}: {e}", file=sys.stderr)
            # continue
            test_json = {
                "Normal count"          : int(n_train_normal), 
                "Attack count"          : int(n_train_attack),
                "Total"                 : int(n_train_normal + n_train_attack),
                "tp"                    : 0,
                "tn"                    : 0, 
                "fp"                    : 0,
                "fn"                    : 0,
                "Accuracy %"            : 0,
                "Precision %"           : 0,
                "Recall %"              : 0,
                "F1-Score %"            : 0,
                "Precision_anom %"      : 0,
                "Recall_anom %"         : 0,
                "F1-Score_anom %"       : 0,
                "PR-AUC"                : 0,
                "ROC-AUC"               : 0,
                "TotalTime (ms)"        : 0, # Convert avg ns to ms
                "AvgTimePerPacket(ns)"  : 0,
                "Ram_usage"             : 0, # This is TOTAL RAM
                "CPU_avg%"              : 0,
                "CPU_peak%"             : 0,
                "training_time_ms"      : 0,
                "training_avg_time_per_packet_ns": 0,
                "training_peak_ram_mb"  : 0,
                "training_cpu_avg_pct"  : 0,
                "training_cpu_peak_pct" : 0,
                "n_train_attack"        : int(n_val_attack),
                "n_train_normal"        : int(n_val_normal)    
            }
            log_run_status(ds, run_key, success=False, e=e)

        all_runs_results[run_key] = {
            "Test":       test_json,
        }



    return all_runs_results


# -------------------- local test --------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=f"Run {ALGO_NAME} analysis.")
    
    # --- Required Paths ---
    parser.add_argument('--train_input_path', type=str, required=True, help='Path to the training CSV (normal_only.csv)')
    parser.add_argument('--test_input_path', type=str, required=True, help='Path to the testing CSV (attack_and_normal.csv)')
    parser.add_argument('--validation_input_path', type=str, required=True, help='Path to the validation CSV (attack_and_normal.csv)')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save results and plots')



    parser.add_argument('--dataset', type=str, default='all', help='Dataset')
    parser.add_argument('--goid', type=str, default='all', help='GoID')
    parser.add_argument('--attack_type', type=str, default='all', help='Attack_Scn')
    
    args = parser.parse_args()

    # Create the output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # print(f"Running {ALGO_NAME} on {args.train_input_path}")    

    get_bool_from_str =lambda x : False if x == 'False' else True
    dataset=args.dataset
    goid=args.goid
    attack_type=args.attack_type
    ds={'dataset':dataset,
        'goid':goid,
        'attack_type':attack_type
    }
    
    # Call your main function with the parsed arguments
    results = main(
        train_input_path=args.train_input_path,
        test_input_path=args.test_input_path,
        validation_input_path=args.validation_input_path,
        output_dir=args.output_dir,
        scaled_input=True,
        use_freq=False,
        use_features='original',
        ds=ds
    )

    output_csv_path = f'{uh.ROOT_OUTPUT_DIR}/aggregated_{ALGO_NAME}_results.csv'


    uh.append_results_to_csv(output_csv_path, results, {
            'dataset': dataset, 'goid': goid, 'attack_type': attack_type
        })


    long_csv_path = os.path.join(uh.PLOT_DATA_DIR, f"{ALGO_NAME}_metrics_long.csv")
    # if os.path.exists(long_csv_path):
    #     os.remove(long_csv_path)    

    # metrics_long rows
    rows = uh.extract_plot_rows(results, ALGO_NAME, ds) + uh.extract_average_rows_over_runs(results, ALGO_NAME, ds)
    uh.append_rows_to_long_csv(long_csv_path, rows)

