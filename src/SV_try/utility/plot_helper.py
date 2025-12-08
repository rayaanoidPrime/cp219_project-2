import numpy as np
import pandas as pd
import os
from utility import unsupervised_helper as uh


def _ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p

def infer_hierarchy_from_output_dir(output_dir: str):
    """
    run_all.py passes output_dir like results_dir/<dataset>/<goId>/<attack>
    """
    parts = os.path.normpath(output_dir or "").split(os.sep)
    if len(parts) >= 3:
        return parts[-3], parts[-2], parts[-1]
    return "unknown_dataset", "unknown_goId", "unknown_attack"

def write_plot_data_csv(df: pd.DataFrame, y_true, y_pred, out_csv: str):
    feat_cols = uh.get_features(df.copy())

    try:
        len(df)==len(y_true) and len(y_true)==len(y_pred)
    except:
        print(f"Length Mismatch: Data Length {len(df)}, True_labels {len(y_true)}, Predicted_Labels {len(y_pred)}")
    
    n = min(len(df), len(y_true), len(y_pred))
    out = pd.DataFrame(df[feat_cols].iloc[:n]).copy()
    out["y_true"] = np.asarray(y_true[:n]).astype(int)
    out["y_pred"] = np.asarray(y_pred[:n]).astype(int)
    _ensure_dir(os.path.dirname(out_csv))
    out.to_csv(out_csv, index=False)