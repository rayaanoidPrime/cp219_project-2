"""
Feature Importance Analysis for SV & MMS Datasets
==================================================
Two techniques from one XGBoost model per dataset:
  1. XGBoost built-in  feature_importances_  (gain-based)
  2. SHAP               mean(|shap_values|) via TreeExplainer

Aggregation is per-protocol only (SV features ≠ MMS features).
"""

import os
import sys
import json
import traceback
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

# Add parent directory to sys.path to allow importing from utility
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from utility.unsupervised_helper import get_original_features

# ML
import xgboost as xgb
import shap

# Visualisation
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# LaTeX-style fonts
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Computer Modern Roman', 'CMU Serif', 'Times New Roman'],
    'mathtext.fontset': 'cm',
    'axes.unicode_minus': False,
})

# ======================================================================
# CONFIGURATION
# ======================================================================
class Config:
    BASE_DIR = r"c:\Users\Rayaan_Ghosh\Desktop\OSS\cp219_project-2\data\SV_MMS_Data"
    OUTPUT_DIR = r"c:\Users\Rayaan_Ghosh\Desktop\OSS\cp219_project-2\data\results_feature_importance_sv_mms"
    TOP_K = 15
    TARGET_COL = "attack"
    RANDOM_STATE = 42
    TRAIN_FILE = "attack_and_normal.csv"

    XGB_PARAMS = {
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_estimators': 100,
        'objective': 'binary:logistic',
        'base_score': 0.5,
        'random_state': 42,
        'n_jobs': -1,
        'verbosity': 0,
    }


# ======================================================================
# PHASE 1 — DATA DISCOVERY
# ======================================================================
def discover_datasets(base_dir):
    """Walk SV_MMS_Data/ and return list of dataset dicts.
    Structure: base_dir/{SV,MMS}/train|test/  (flat, no attack-type subfolders)
    """
    datasets = []
    base = Path(base_dir)
    for train_dir in base.rglob("train"):
        test_dir = train_dir.parent / "test"
        train_file = train_dir / Config.TRAIN_FILE
        test_file  = test_dir  / Config.TRAIN_FILE
        if train_file.exists() and test_file.exists():
            rel = train_dir.parent.relative_to(base)
            parts = rel.parts                       # e.g. ('SV',)
            if len(parts) >= 1:
                protocol    = parts[0]               # SV | MMS
                attack_type = 'all'
                name = f"{protocol}_all"
                datasets.append({
                    'name':        name,
                    'protocol':    protocol,
                    'attack_type': attack_type,
                    'train_path':  train_file,
                    'test_path':   test_file,
                })
                print(f"  Found: {name}  (train={train_file})")
    return datasets


# ======================================================================
# PHASE 2 — DATA PREPARATION
# ======================================================================
def load_and_prepare(train_path, target_col):
    """Load train CSV → (X, y) with only numeric features."""
    df = pd.read_csv(train_path)
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not in columns")

    y = df[target_col].astype(int)
    X = df.drop(columns=[target_col])

    # Drop columns that are clearly not features
    drop_if_present = ['index', 'freq', 'class']
    X = X.drop(columns=[c for c in drop_if_present if c in X.columns])

    # Drop non-numeric (object) columns
    non_num = X.select_dtypes(include=['object']).columns
    if len(non_num):
        print(f"    Dropping {len(non_num)} non-numeric cols: {list(non_num[:5])}{'...' if len(non_num) > 5 else ''}")
        X = X.drop(columns=non_num)

    # Coerce booleans to int
    bool_cols = X.select_dtypes(include=['bool']).columns
    if len(bool_cols):
        X[bool_cols] = X[bool_cols].astype(int)

    # Handle inf → NaN, then fill NaN with column mean
    X = X.replace([np.inf, -np.inf], np.nan)
    if X.isnull().any().any():
        X = X.fillna(X.mean())

    # Drop any columns that are still entirely NaN (constant columns after fill)
    all_nan = X.columns[X.isnull().all()]
    if len(all_nan):
        X = X.drop(columns=all_nan)

    # Use centralized feature selection logic (handles IDs and zero-variance)
    selected_feats = get_original_features(X.columns.tolist(), df=X)
    X = X[selected_feats]

    print(f"    Final: {X.shape[0]} rows × {X.shape[1]} features, {int(y.sum())} attacks / {int((y==0).sum())} normal")
    return X, y


# ======================================================================
# PHASE 3 — MODEL + IMPORTANCE
# ======================================================================
def compute_importance(X, y):
    """Train XGBoost, return (xgb_importance_dict, shap_importance_dict, model, shap_values)."""
    # Train
    model = xgb.XGBClassifier(**Config.XGB_PARAMS)
    model.fit(X, y)

    # Technique 1: built-in gain importance
    xgb_imp = dict(zip(X.columns, model.feature_importances_))

    # Technique 2: SHAP
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    shap_imp    = dict(zip(X.columns, np.abs(shap_values).mean(axis=0)))

    return xgb_imp, shap_imp, model, shap_values


def normalize(scores):
    """Min-max normalise a dict of scores to [0, 1]."""
    if not scores:
        return {}
    vals = np.array(list(scores.values()))
    lo, hi = vals.min(), vals.max()
    if hi == lo:
        return {k: 1.0 for k in scores}
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}


# ======================================================================
# PHASE 4 — PER-DATASET OUTPUTS
# ======================================================================
def save_importance_csv(scores, out_path):
    """Save {feature: score} dict as sorted CSV."""
    df = pd.DataFrame(list(scores.items()), columns=['feature', 'importance'])
    df = df.sort_values('importance', ascending=False).reset_index(drop=True)
    df.to_csv(out_path, index=False)
    return df


def plot_topk_bar(scores, title, out_path, k=None):
    """Horizontal bar chart of top-K features."""
    k = k or Config.TOP_K
    df = pd.DataFrame(list(scores.items()), columns=['feature', 'importance'])
    df = df.sort_values('importance', ascending=False).head(k)

    fig, ax = plt.subplots(figsize=(12, max(6, k * 0.5)))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(df)))
    ax.barh(range(len(df)), df['importance'].values, color=colors)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df['feature'].values, fontsize=14)
    ax.invert_yaxis()
    ax.set_xlabel('Importance Score', fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.grid(axis='x', alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def plot_shap_beeswarm(model, shap_values, X, title, out_path):
    """SHAP beeswarm (summary) plot."""
    fig, ax = plt.subplots(figsize=(12, max(6, len(X.columns) * 0.35)))
    shap.summary_plot(shap_values, X, show=False, max_display=Config.TOP_K)
    plt.title(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close('all')


def process_dataset(ds_info):
    """Process one sub-dataset: train, compute both importances, save outputs."""
    name = ds_info['name']
    protocol = ds_info['protocol']
    attack   = ds_info['attack_type']
    out_dir  = os.path.join(Config.OUTPUT_DIR, protocol, attack)
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  Processing: {name}")
    print(f"{'='*70}")

    X, y = load_and_prepare(ds_info['train_path'], Config.TARGET_COL)
    if X.shape[1] == 0:
        print("    No features remaining — skipping")
        return None

    xgb_imp, shap_imp, model, shap_vals = compute_importance(X, y)

    # Normalise
    xgb_norm  = normalize(xgb_imp)
    shap_norm = normalize(shap_imp)

    # Save CSVs
    save_importance_csv(xgb_norm,  os.path.join(out_dir, 'xgb_importance.csv'))
    save_importance_csv(shap_norm, os.path.join(out_dir, 'shap_importance.csv'))

    # Save plots
    plot_topk_bar(xgb_norm,
                  f'XGBoost Feature Importance — {protocol}/{attack}',
                  os.path.join(out_dir, 'xgb_top_features.png'))
    plot_topk_bar(shap_norm,
                  f'SHAP Feature Importance — {protocol}/{attack}',
                  os.path.join(out_dir, 'shap_top_features.png'))
    plot_shap_beeswarm(model, shap_vals, X,
                       f'SHAP Summary — {protocol}/{attack}',
                       os.path.join(out_dir, 'shap_beeswarm.png'))

    print(f"    Saved to {out_dir}")
    return {
        'name': name, 'protocol': protocol, 'attack_type': attack,
        'n_features': X.shape[1], 'n_samples': len(X),
        'features': list(X.columns),
        'xgb_scores': xgb_norm,
        'shap_scores': shap_norm,
    }


# ======================================================================
# PHASE 5 — PER-PROTOCOL AGGREGATION
# ======================================================================
def aggregate_protocol(results_list, protocol):
    """Average importance across attack types for one protocol."""
    xgb_agg  = defaultdict(list)
    shap_agg = defaultdict(list)

    for r in results_list:
        for feat in r['features']:
            xgb_agg[feat].append(r['xgb_scores'].get(feat, 0))
            shap_agg[feat].append(r['shap_scores'].get(feat, 0))

    rows = []
    for feat in xgb_agg:
        rows.append({
            'feature':            feat,
            'avg_xgb_importance': np.mean(xgb_agg[feat]),
            'std_xgb_importance': np.std(xgb_agg[feat]),
            'avg_shap_importance': np.mean(shap_agg[feat]),
            'std_shap_importance': np.std(shap_agg[feat]),
            'appearances':        len(xgb_agg[feat]),
        })

    df = pd.DataFrame(rows).sort_values('avg_shap_importance', ascending=False)
    return df


def make_protocol_plots(df, protocol, out_dir):
    """Create aggregated bar charts for a protocol."""
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f'{protocol.lower()}_aggregated_ranking.csv')
    df.to_csv(csv_path, index=False)

    # XGBoost aggregated
    plot_topk_bar(
        dict(zip(df['feature'], df['avg_xgb_importance'])),
        f'Aggregated XGBoost Importance — {protocol}',
        os.path.join(out_dir, f'{protocol.lower()}_xgb_aggregated.png')
    )

    # SHAP aggregated
    plot_topk_bar(
        dict(zip(df['feature'], df['avg_shap_importance'])),
        f'Aggregated SHAP Importance — {protocol}',
        os.path.join(out_dir, f'{protocol.lower()}_shap_aggregated.png')
    )
    print(f"  Aggregated results for {protocol} saved to {out_dir}")


# ======================================================================
# MAIN
# ======================================================================
def main():
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

    # Phase 1: discover
    print("Phase 1: Discovering datasets...")
    datasets = discover_datasets(Config.BASE_DIR)
    if not datasets:
        print("No datasets found — check BASE_DIR.")
        return
    print(f"Found {len(datasets)} datasets\n")

    # Phase 2-4: process each dataset
    all_results = []
    errors = []
    for ds in datasets:
        try:
            result = process_dataset(ds)
            if result:
                all_results.append(result)
        except Exception as e:
            print(f"  ERROR on {ds['name']}: {e}")
            traceback.print_exc()
            errors.append({'dataset': ds['name'], 'error': str(e)})

    print(f"\nProcessed {len(all_results)}/{len(datasets)} datasets")
    if errors:
        pd.DataFrame(errors).to_csv(
            os.path.join(Config.OUTPUT_DIR, 'error_log.csv'), index=False)

    # Phase 5: per-protocol aggregation
    print("\nPhase 5: Per-protocol aggregation...")
    for protocol in ('SV', 'MMS'):
        proto_results = [r for r in all_results if r['protocol'] == protocol]
        if not proto_results:
            print(f"  No results for {protocol} — skipping")
            continue
        agg_df = aggregate_protocol(proto_results, protocol)
        make_protocol_plots(agg_df, protocol,
                            os.path.join(Config.OUTPUT_DIR, protocol))

    # Final metadata
    metadata = {
        'total_datasets': len(datasets),
        'processed': len(all_results),
        'failed': len(errors),
        'protocols': list({r['protocol'] for r in all_results}),
    }
    with open(os.path.join(Config.OUTPUT_DIR, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

    print("\nDone!")
    print(f"Results in: {Config.OUTPUT_DIR}")


if __name__ == "__main__":
    main()
