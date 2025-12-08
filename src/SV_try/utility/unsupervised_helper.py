import os, csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from typing import Literal, List, Optional
from sklearn.preprocessing import MinMaxScaler
LABEL_COL = 'attack'

PLOT_FIELDS_ORDER = [
    "Algorithm", "Dataset", "GoID", "Attack_Scn", "Split", "Run",
    "Accuracy %", "Precision %", "Recall %", "F1-Score %","Precision_anom %","Recall_anom %","F1-Score_anom %",
    "BalancedAcc %", "MCC",
    "tp", "tn", "fp", "fn",
    "Normal count", "Attack count", "Total",
    "TotalTime (ms)", "AvgTimePerPacket(ns)", "CPU_avg%", "CPU_peak%", "Ram_usage",
    # training-specific fields
    "training_time_ms", "training_avg_time_per_packet_ns",
    "training_peak_ram_mb", "training_cpu_avg_pct", "training_cpu_peak_pct", "n_train_attack", "n_train_normal"
]

ROOT_OUTPUT_DIR     ='/home/mahalakshmi/Journal_May2025/SV_dec/SV Dataset/NEW_results/results_aggregate'

# (Images are created by plot_bar.py)
PLOT_DATA_DIR      = "'/home/mahalakshmi/Journal_May2025/SV_dec/SV Dataset/NEW_results/all_plots_data/Algorithm_wise_plots_data"
PLOTS_DIR          = "'/home/mahalakshmi/Journal_May2025/SV_dec/SV Dataset/NEW_results/all_plots/Algorithm_wise_plots"         # bar plots data (metrics_long)



os.makedirs(PLOT_DATA_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)


def _sanitize_to_numeric(df: pd.DataFrame, label_col: str = LABEL_COL) -> pd.DataFrame:
    """Coerce booleans/boolean-like strings/numeric-as-strings to real numerics.
    Leaves non-convertible values as NaN (filtered later)."""

    df=df.copy()
    # 1) Real booleans → int
    bool_cols = df.select_dtypes(include=['bool']).columns
    if len(bool_cols) > 0:
        df[bool_cols] = df[bool_cols].astype(int)

    # 2) Object columns: boolean-like or numeric-like
    for col in df.select_dtypes(include=['object']).columns:
        s = df[col].astype(str).str.strip().str.lower()
        s.replace({'': 0, 'nan': 0, 'none': 0}, inplace=True)

        if s.isin(['True', 'False', '1', '0']).all():
            df[col] = s.map({'True': 1, 'False': 0, '1': 1, '0': 0}).astype(float)
        else:
            df[col] = pd.to_numeric(s, errors='coerce')

    # 3) Ensure label column is numeric 0/1
    if label_col in df.columns:
        lab = df[label_col]
        if lab.dtype == 'bool':
            df[label_col] = lab.astype(int)
        elif lab.dtype == object:
            s = lab.astype(str).str.strip().str.lower()
            s.replace({'': 0, 'nan': 0, 'none': 0}, inplace=True)
            if s.isin(['true', 'false', '1', '0']).all():
                df[label_col] = s.map({'true': 1, 'false': 0, '1': 1, '0': 0}).astype('Int64')
            else:
                df[label_col] = pd.to_numeric(s, errors='coerce').astype('Int64')
        # leave Int64 (nullable) if there are NaNs; cast later when filtering rows

    return df

def _flatten(d, parent=""):
    out = {}
    for k, v in d.items():
        nk = f"{parent}.{k}" if parent else k
        if isinstance(v, dict):
            out.update(_flatten(v, nk))
        else:
            out[nk] = v
    return out

def dump_csv(results_data, output_dir, filename="gmm_results.csv"):
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, filename)
    
    # Exit if there's no data to write
    if not results_data:
        # print(" Warning: No data available to dump.")
        return

    # Get the keys from the first run to establish the headers for each section
    first_run_key = next(iter(results_data))
    first_run_data = results_data[first_run_key]

    try:
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)

            if 'Train' in first_run_data:
                # --- Validation Section ---
                val_headers = list(first_run_data["Train"].keys())
                writer.writerow(["Train:"] + val_headers) # Write section header
                # Write data rows for each run in this section
                for run_key, run_data in results_data.items():
                    writer.writerow([run_key] + list(run_data["Train"].values()))

            if 'Validation' in first_run_data:
                writer.writerow([]) # Add a blank line for separation
                # --- Validation Section ---
                val_headers = list(first_run_data["Validation"].keys())
                writer.writerow(["Validation:"] + val_headers) # Write section header
                # Write data rows for each run in this section
                for run_key, run_data in results_data.items():
                    writer.writerow([run_key] + list(run_data["Validation"].values()))
    
            # --- Test Section ---
            writer.writerow([]) # Add a blank line for separation
            test_headers = list(first_run_data["Test"].keys())
            writer.writerow(["Test:"] + test_headers) # Write section header
            for run_key, run_data in results_data.items():
                writer.writerow([run_key] + list(run_data["Test"].values()))

            # --- Misc Section ---
            if 'Misc' in first_run_data:
                writer.writerow([]) # Add a blank line for separation
                misc_headers = list(first_run_data["Misc"].keys())
                writer.writerow(["Misc:"] + misc_headers) # Write section header
                for run_key, run_data in results_data.items():
                    writer.writerow([run_key] + list(run_data["Misc"].values()))

        # print(f"Successfully saved all run results to {csv_path}")

    except Exception as e:
        print(f"Error saving CSV file: {e}")

def r3(x):
    try:
        return round(float(x), 3)
    except Exception as e:
        return e
    
def r2(x):
    try:
        return round(float(x), 2)
    except Exception as e:
        return e


def get_features(df: pd.DataFrame):
    """Return numeric feature column names (after sanitization). 
        Convert bool -> int and string bool (True/False) to int(0/1)"""
    df = _sanitize_to_numeric(df)
    cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if LABEL_COL in cols:
        cols.remove(LABEL_COL)
    if 'index' in cols:
        cols.remove('index')


    return cols

def count_stat(vector):
    # Because it is '0' and '1', we can run a count statistic.
    unique, counts = np.unique(vector, return_counts=True)
    return dict(zip(unique, counts))

def get_trainable_data(
    df_train: Optional[pd.DataFrame] = None,
    df_val: Optional[pd.DataFrame] = None,
    df_test: Optional[pd.DataFrame] = None,
    scaled_input: bool = True,
    use_freq: bool = False,
    use_features: Literal["original", "derived", "selected", "all"] = "original",
    scaler: Optional[MinMaxScaler] = None
):
    X_train, y_train, df_clean_train = None, None, None
    X_val, y_val, df_clean_val = None, None, None
    X_test, y_test, df_clean_test = None, None, None
    feat_cols = None
    if(df_train is not None):
        X_train, y_train ,feat_cols, df_clean_train=get_clean_data_from_dataframe(df_train, use_features=use_features, use_freq=use_freq)
    if(df_val is not None):
        X_val, y_val ,feat_cols, df_clean_val= get_clean_data_from_dataframe(df_val,use_features=use_features, use_freq=use_freq)
    if(df_test is not None):
        X_test, y_test ,feat_cols, df_clean_test= get_clean_data_from_dataframe(df_test,use_features=use_features, use_freq=use_freq)
    
    # Scaling
    if scaler is not None:
        X_train = scaler.transform(X_train) if X_train is not None else None
        X_val   = scaler.transform(X_val) if X_val is not None else None
        X_test  = scaler.transform(X_test) if X_test is not None else None
    elif scaled_input:
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train) if X_train is not None else None
        X_val   = scaler.transform(X_val) if X_val is not None else None
        X_test  = scaler.transform(X_test) if X_test is not None else None

    return X_train, X_val, X_test, y_train, y_val, y_test, df_clean_train, df_clean_val, df_clean_test, feat_cols

def get_clean_data_from_dataframe(df: pd.DataFrame, use_freq=False, use_features='all'):
    """Return (X, y) with X as float64 and y as int, dropping any rows with NaNs."""
    
    
    # df = df.drop(columns='time_from_start')

    # Get only numeric columns
    df_clean = _sanitize_to_numeric(df)

    # numeric feature columns
    feat_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
    if LABEL_COL in feat_cols:
        feat_cols.remove(LABEL_COL)
    
    if 'index' in feat_cols:
        feat_cols.remove('index')

    if use_features=='derived':
        feat_cols=get_only_derived_features(feat_cols)

    elif use_features=='selected_1':
        feat_cols=get_selected_features(feat_cols)

    elif use_features=='selected_2':
        feat_cols=get_selected_small(feat_cols)

    elif use_features=='selected_3':
        feat_cols=get_selected_3(feat_cols)
    
    elif use_features=='selected_4':
        feat_cols=get_selected_4(feat_cols)

    elif use_features=='original':
        feat_cols=get_original_features(feat_cols)

    if use_freq:
        feat_cols.append('freq')

    # print('Using cols', feat_cols)
    

    # Else Feature columns will include all feature columns
    
    # Build X and y

    X = df_clean[feat_cols].astype(float).to_numpy(copy=False)
    y_series = df_clean[LABEL_COL] if LABEL_COL in df_clean.columns else None


    # Row mask: no NaNs in X and valid y
    mask = ~np.isnan(X).any(axis=1) # What value is NaN here?
    df_clean = df_clean[mask].reset_index(drop=True)
    if y_series is not None:
        mask &= y_series.notna().to_numpy()

    X = X[mask].astype(np.float64, copy=False)
    y = y_series[mask].astype(int).to_numpy() if y_series is not None else None

    return X, y ,feat_cols, df_clean

def plot_cm(cm_test,title='Confusion Matrix (Counts & %)',outdir=None):
    # Compute percentages
    cm_percent = cm_test / np.sum(cm_test) * 100

    # Combine count + percentage text
    labels = np.array([
        [f"{cm_test[i, j]}\n({cm_percent[i, j]:.1f}%)" for j in range(cm_test.shape[1])]
        for i in range(cm_test.shape[0])
    ])

    # Plot
    plt.figure(figsize=(5,4))
    sns.heatmap(cm_test, annot=labels, fmt='', cmap='Blues',
                xticklabels=['Normal (Pred)', 'Anomaly (Pred)'],
                yticklabels=['Normal (True)', 'Anomaly (True)'])

    plt.title(title)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    # plt.show()
    if(outdir is not None):
        plt.savefig(os.path.join(outdir,f'{title}.png'))
    else:
        plt.show()

def get_selected_small(feat_cols):
    selected_columns = ['Length', 'stNum', 'sqNum', 'timeAllowedtoLive',
       'numDatSetEntries', 'time_diff', 'stNum_diff','sqNum_diff']
    
    for col in feat_cols:
        if 'boolean' in col or 'bitstring' in col:
            selected_columns.append(col)

    return selected_columns

def get_selected_4(feat_cols):
    selected_columns = ['Length', 'stNum', 'sqNum', 'timeAllowedtoLive',
       'numDatSetEntries', 'time_diff', 'stNum_diff','sqNum_diff']
    
    for col in feat_cols:
        if 'bitstring_diff' in col or 'boolean_diff' in col:
            selected_columns.append(col)

    return selected_columns

def get_selected_features(feat_columns):
    '''Include : 'Length', 'stNum', 'sqNum', 'timeAllowedtoLive', 'numDatSetEntries', 'time_diff', 'stNum_diff', 'sqNum_diff'
        Also all original int|float|boolean|bitstring'''

    # selected_columns = ['Length', 'stNum', 'sqNum', 'timeAllowedtoLive',
    #    'numDatSetEntries', 'time_diff', 'stNum_diff','sqNum_diff']
    
    pattern_selected = r'^(?:(?:Length|stNum|sqNum|timeAllowedtoLive|numDatSetEntries|time_diff|stNum_diff|sqNum_diff|bitstring_\d+|floatvalue_\d+|bool.*|int.*))$'

    selected_cols = [c for c in feat_columns if re.match(pattern_selected, c)]
    return selected_cols

def get_only_derived_features(feat_columns):
    '''' Include: time_diff', 'sqNum_diff', 'stNum_diff', 'timestamp_diff', 'Length_diff', 'attack', 'index'
                Any _diff of bitstring/float/bool/int features
        Exclude: all non-diff versions.
    '''
    pattern_derived = r'^(?:attack|index|time_diff|timestamp_diff|Length_diff|stNum_diff|sqNum_diff|.*_diff)$'

    
    selected_cols = [c for c in feat_columns if re.match(pattern_derived, c)]
    return selected_cols

def get_selected_3(feat_columns):
    selected_columns = ['Length', 'stNum', 'sqNum', 'timeAllowedtoLive',
    'numDatSetEntries', 'time_diff', 'stNum_diff','sqNum_diff']
    return selected_columns

def get_original_features(feat_columns):
    # ''' Include: 'Length', 'stNum', 'sqNum', 'timeAllowedtoLive', 'numDatSetEntries', 'time_from_start'
    #                 Any bitstring_i+, floatvalue_i+, or boolean/int-like features
    #     Exclude: all _diff or timestamp_diff, time_diff, etc.
    # '''
    
    # pattern_original = r'^(?!(?:.*_diff$|timestamp_diff$|time_diff$))(?:(?:Length|stNum|sqNum|timeAllowedtoLive|numDatSetEntries|bitstring_\d+|floatvalue_\d+|bool.*|int.*))$'
    
    
    # selected_cols = [c for c in feat_columns if re.match(pattern_original, c)]
    # selected_cols.append('time_diff')
    # selected_cols.append('timestamp_diff')
    # selected_cols.append('stNum_diff')
    # selected_cols.append('sqNum_diff')

    # THESE FEATURES ALONG WITH FREQ=FALSE + SCALE=FALSE/TRUE is giving the best output
    selected_cols = ['Length', 'stNum', 'sqNum', 'timeAllowedtoLive',
       'numDatSetEntries', 'time_diff', 'stNum_diff','sqNum_diff' ]
    
    for col in feat_columns:
        if 'int' in col or 'float' in col:
            selected_cols.append(col)
    
    return selected_cols


# -----------------------------------------
#   Append raw sectioned results (for audit)
# -----------------------------------------
def append_results_to_csv(csv_path, results_data, hierarchy_info):
    if not results_data:
        return
    with open(csv_path, 'a', newline='') as f:
        w = csv.writer(f)
        w.writerow([])
        w.writerow(['Dataset:', hierarchy_info['dataset']])
        w.writerow(['GoID:', hierarchy_info['goid']])
        w.writerow(['Attack_Type:', hierarchy_info['attack_type']])
        w.writerow([])
        first_run_data = next(iter(results_data.values()))
        for section in ["Test"]:
            if section in first_run_data:
                headers = list(first_run_data[section].keys())
                w.writerow([f"{section}:"] + headers)
                for run_key, run_data in results_data.items():
                    if section in run_data:
                        w.writerow([run_key] + list(run_data[section].values()))
                w.writerow([])


# ================================
# Helpers for metrics_long.csv
# ================================
def extract_plot_rows(all_runs_results, algorithm, dataset_info):
    rows = []
    for run_key, sections in all_runs_results.items():
        # for split in ("Train", "Validation", "Test"):
        split="Test"
        s = sections[split]
        # Note: some fields may not exist for every split/run; use _as_float to coerce None safely
        row = {
            "Algorithm"             : algorithm,
            "Dataset"               : dataset_info["dataset"],
            "GoID"                  : dataset_info["goid"],
            "Attack_Scn"            : dataset_info["attack_type"],
            "Split"                 : split,
            "Run"                   : run_key,
            "Accuracy %"            : _as_float(s.get("Accuracy %")),
            "Precision %"           : _as_float(s.get("Precision %")),
            "Recall %"              : _as_float(s.get("Recall %")),
            "F1-Score %"            : _as_float(s.get("F1-Score %")),
            "Precision_anom %"      : _as_float(s.get("Precision_anom %")),
            "Recall_anom %"         : _as_float(s.get("Recall_anom %")),
            "F1-Score_anom %"       : _as_float(s.get("F1-Score_anom %")),
            "BalancedAcc %"         : _as_float(s.get("BalancedAcc %")),
            "MCC"                   : _as_float(s.get("MCC")),
            "tp"                    : _as_float(s.get("tp")),
            "tn"                    : _as_float(s.get("tn")),
            "fp"                    : _as_float(s.get("fp")),
            "fn"                    : _as_float(s.get("fn")),
            "Normal count"          : _as_float(s.get("Normal count")),
            "Attack count"          : _as_float(s.get("Attack count")),
            "Total"                 : _as_float(s.get("Total")),
            "TotalTime (ms)"        : r3(_as_float(s.get("TotalTime (ms)"))),
            "AvgTimePerPacket(ns)"  : r3(_as_float(s.get("AvgTimePerPacket(ns)"))),
            "CPU_avg%"              : r3(_as_float(s.get("CPU_avg%"))),
            "CPU_peak%"             : r3(_as_float(s.get("CPU_peak%"))),
            "Ram_usage"             : r3(_as_float(s.get("Ram_usage"))),
            # training-specific fields (may only be present in Test JSON as provided)
            "training_time_ms"                   : r3(_as_float(s.get("training_time_ms"))),
            "training_avg_time_per_packet_ns"    : r3(_as_float(s.get("training_avg_time_per_packet_ns"))),
            "training_peak_ram_mb"               : r3(_as_float(s.get("training_peak_ram_mb"))),
            "training_cpu_avg_pct"               : r3(_as_float(s.get("training_cpu_avg_pct"))),
            "training_cpu_peak_pct"              : r3(_as_float(s.get("training_cpu_peak_pct"))),
            "n_train_attack"        : r3(_as_float(s.get("n_train_attack"))),
            "n_train_normal"        : r3(_as_float(s.get("n_train_normal"))),
        }
        rows.append(row)
    return rows

def extract_average_rows_over_runs(all_runs_results, algorithm, dataset_info):
    rows = []
    split ="Test"
    # lists to accumulate per-run values
    acc = []
    prec = []
    rec = []
    f1 = []
    prec_anom =[]
    rec_anom =[]
    f1_anom =[]
    balanced_acc = []
    mcc = []
    ttime = []
    avgpkt = []
    tp = []
    tn = []
    fp = []
    fn = []
    n_norm = []
    n_att = []
    n_train_attack=[]
    n_train_normal=[]
    n_tot = []
    cpu_usg = []
    cpu_peak = []
    ram_usage = []
    # training-specific accumulators
    training_time = []
    training_avgpkt = []
    training_peak_ram = []
    training_cpu_avg = []
    training_cpu_peak = []

    any_split = False

    for _, sections in all_runs_results.items():
        if split not in sections:
            continue
        s = sections[split]
        any_split = True

        def add(lst, key):
            v = _as_float(s.get(key))
            if v is not None:
                lst.append(v)

        add(acc, "Accuracy %")
        add(prec, "Precision %")
        add(rec, "Recall %")
        add(f1, "F1-Score %")
        add(prec_anom, "Precision_anom %")
        add(rec_anom, "Recall_anom %")
        add(f1_anom, "F1-Score_anom %")
        add(balanced_acc, "BalancedAcc %")
        add(mcc, "MCC")
        add(ttime, "TotalTime (ms)")

        v_avgpkt = _as_float(s.get("AvgTimePerPacket(ns)"))
        if v_avgpkt is not None:
            avgpkt.append(v_avgpkt)

        add(tp, "tp"); add(tn, "tn")
        add(fp, "fp"); add(fn, "fn")
        add(n_norm, "Normal count"); add(n_att, "Attack count"); add(n_tot, "Total")
        add(cpu_usg, "CPU_avg%")
        add(cpu_peak, "CPU_peak%")
        add(ram_usage, "Ram_usage")

        # training-specific keys could be missing for Train/Val splits; only some runs may include them
        add(training_time, "training_time_ms")
        add(training_avgpkt, "training_avg_time_per_packet_ns")
        add(training_peak_ram, "training_peak_ram_mb")
        add(training_cpu_avg, "training_cpu_avg_pct")
        add(training_cpu_peak, "training_cpu_peak_pct")

        add(n_train_attack, "n_train_attack")
        add(n_train_normal, "n_train_normal")

    def mean_or_none(lst):
        return sum(lst) / len(lst) if lst else None

    def sum_or_none(lst):
        return sum(lst) if lst else None

    def max_or_none(lst):
        return max(lst) if lst else None

    rows.append({
        "Algorithm"             : algorithm,
        "Dataset"               : dataset_info["dataset"],
        "GoID"                  : dataset_info["goid"],
        "Attack_Scn"            : dataset_info["attack_type"],
        "Split"                 : split,
        "Run"                   : "Average",
        "Accuracy %"            : r2(mean_or_none(acc)),
        "Precision %"           : r2(mean_or_none(prec)),
        "Recall %"              : r2(mean_or_none(rec)),
        "F1-Score %"            : r2(mean_or_none(f1)),
        "Precision_anom %"      : r2(mean_or_none(prec_anom)),
        "Recall_anom %"         : r2(mean_or_none(rec_anom)),
        "F1-Score_anom %"       : r2(mean_or_none(f1_anom)),
        "BalancedAcc %"         : r2(mean_or_none(balanced_acc)),
        "MCC"                   : r3(mean_or_none(mcc)),
        "tp"                    : r3(sum_or_none(tp)),
        "tn"                    : r3(sum_or_none(tn)),
        "fp"                    : r3(sum_or_none(fp)),
        "fn"                    : r3(sum_or_none(fn)),
        "Normal count"          : r3(sum_or_none(n_norm)),
        "Attack count"          : r3(sum_or_none(n_att)),
        "Total"                 : r3(sum_or_none(n_tot)),
        "TotalTime (ms)"        : r3(mean_or_none(ttime)),
        "AvgTimePerPacket(ns)"  : r3(mean_or_none(avgpkt)),
        "CPU_avg%"              : r3(mean_or_none(cpu_usg)),
        "CPU_peak%"             : r3(mean_or_none(cpu_peak)),
        "Ram_usage"             : r3(mean_or_none(ram_usage)),
        # training-specific aggregated fields:
        # - training_time_ms & training_avg_time_per_packet_ns: mean over runs (if present)
        # - training_peak_ram_mb & training_cpu_peak_pct: max over runs (per your request)
        # - training_cpu_avg_pct: mean over runs
        "training_time_ms"                      : r3(mean_or_none(training_time)),
        "training_avg_time_per_packet_ns"       : r3(mean_or_none(training_avgpkt)),
        "training_peak_ram_mb"                  : r3(max_or_none(training_peak_ram)),
        "training_cpu_avg_pct"                  : r3(mean_or_none(training_cpu_avg)),
        "training_cpu_peak_pct"                 : r3(max_or_none(training_cpu_peak)),
        "n_train_attack"        : r3(sum_or_none(n_train_attack)),
        "n_train_normal"        : r3(sum_or_none(n_train_normal)),
    })
    return rows

def _as_float(x):
    if x is None:
        return None
    s = str(x).replace(" MB", "").replace(",", "").strip()
    try:
        return float(s)
    except Exception:
        return None
    

def append_rows_to_long_csv(long_csv_path, rows):
    os.makedirs(os.path.dirname(long_csv_path), exist_ok=True)
    file_exists = os.path.exists(long_csv_path)
    # print(f'Appending to {long_csv_path}, {len(rows)} rows')
    with open(long_csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=PLOT_FIELDS_ORDER)
        if not file_exists:
            w.writeheader()
        for r in rows:
            cleaned = {k: r.get(k, None) for k in PLOT_FIELDS_ORDER}
            w.writerow(cleaned)