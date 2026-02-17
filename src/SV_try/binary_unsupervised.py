"""
Binary Anomaly Detection Script for SV Dataset

Detectors included:
- Isolation Forest (trained on normal data only)
- Autoencoder (reconstruction error threshold)
- GMM (likelihood-based anomaly scoring)
- KMeans + distance threshold
- Hierarchical Clustering + distance threshold
"""

import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Sklearn imports (only metrics needed here)
from sklearn.metrics import (
    confusion_matrix, classification_report,
    precision_score, recall_score, f1_score,
    accuracy_score, roc_auc_score, average_precision_score,
    balanced_accuracy_score, matthews_corrcoef
)

# Import ResourceProfiler for resource tracking
sys.path.insert(0, str(Path(__file__).parent / 'utility'))
try:
    from resource_usage import ResourceProfiler
    RESOURCE_PROFILER_AVAILABLE = True
except Exception as e:
    RESOURCE_PROFILER_AVAILABLE = False
    print(f"Warning: ResourceProfiler not available. Error: {e}")

# Import helper functions
try:
    from utility import unsupervised_helper as uh
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent))
    import utility.unsupervised_helper as uh

# Import data preprocessor and override paths
import preprocessed
# Override the BASE path to use local preprocessed folder
preprocessed.BASE = str(Path(__file__).parent / 'preprocessed')

# Override helper output directory
uh.ROOT_OUTPUT_DIR = str(Path(__file__).parent)

warnings.filterwarnings('ignore')


# =============================================================================
# CONFIGURATION
# =============================================================================
DATASET_NAME = 'SV_Dataset'
GOID = 'NA'
ATTACK_LIST = ["replay", "injection"]
num_runs = 1
CONTAMINATION = 0.2  # Expected proportion of anomalies in training data
PREPROCESSED_DIR = str(Path(__file__).parent / 'preprocessed')


# =============================================================================
# IMPORT DETECTOR MODELS FROM UNSUP_ALGOS
# =============================================================================

from unsup_algos import (
    BinaryIsolationForestDetector,
    BinaryAutoencoderDetector,
    BinaryGMMDetector,
    BinaryKMeansDetector,
    BinaryHierarchicalDetector,
    get_all_detectors
)


# =============================================================================
# DETECTOR REGISTRY
# =============================================================================

def get_detectors() -> Dict:
    """Return dictionary of detector factories."""
    return get_all_detectors(contamination=CONTAMINATION)


# =============================================================================
# METRICS COMPUTATION
# =============================================================================

def compute_metrics(y_true, y_pred, y_scores=None):
    """Compute all binary classification metrics."""
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    
    # Classification report
    rpt = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    
    # Anomaly-specific metrics (class 1)
    precision_anom = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    recall_anom = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    f1_anom = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    
    # Other metrics
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    
    # AUC metrics
    if y_scores is not None:
        # Normalize scores if needed
        if np.min(y_scores) < 0:
            y_scores = (y_scores - np.min(y_scores)) / (np.max(y_scores) - np.min(y_scores) + 1e-9)
        try:
            roc_auc = roc_auc_score(y_true, y_scores)
            pr_auc = average_precision_score(y_true, y_scores)
        except:
            roc_auc = 0.5
            pr_auc = 0.5
    else:
        roc_auc = 0.5
        pr_auc = 0.5
    
    return {
        'tp': int(tp),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'accuracy': rpt['accuracy'] * 100,
        'precision_anom': precision_anom * 100,
        'precision_macro': rpt['macro avg']['precision'] * 100,
        'recall_anom': recall_anom * 100,
        'recall_macro': rpt['macro avg']['recall'] * 100,
        'f1_anom': f1_anom * 100,
        'f1_macro': rpt['macro avg']['f1-score'] * 100,
        'balanced_acc': balanced_acc * 100,
        'mcc': mcc,
        'roc_auc': roc_auc * 100,
        'pr_auc': pr_auc * 100
    }


def tune_threshold_on_validation(detector, X_val, y_val):
    """
    Tune the detector's threshold using the validation set to maximize F1-score.
    Args:
        detector: A fitted detector with predict_scores method
        X_val: Validation features
        y_val: Validation labels (0=normal, 1=attack)
    
    Returns:
        optimal_threshold: The threshold that maximizes F1-score on validation,
                          or None if detector doesn't use thresholds
    """
    # Check if detector uses threshold-based prediction
    if not hasattr(detector, 'threshold'):
        print(f"    Detector doesn't use threshold, skipping tuning")
        return None
    
    if not hasattr(detector, 'predict_scores'):
        return detector.threshold  # Can't tune, return existing
    
    # Get anomaly scores on validation set
    scores = detector.predict_scores(X_val)
    
    # Try different percentiles to find optimal threshold
    best_f1 = 0
    best_threshold = detector.threshold
    
    # Test thresholds from 50th to 99th percentile of validation scores
    for percentile in range(50, 100, 2):
        threshold = np.percentile(scores, percentile)
        y_pred = (scores > threshold).astype(int)
        
        # Compute F1-score for anomaly class
        f1 = f1_score(y_val, y_pred, pos_label=1, zero_division=0)
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    print(f"    Tuned threshold on validation: F1={best_f1:.4f}")
    return best_threshold


# =============================================================================
# RESULTS FORMATTING
# =============================================================================

def build_results_json(metrics, test_resources, train_resources, n_test_normal, 
                      n_test_attack, n_train_normal, n_train_attack):
    """Build results JSON matching aggregated_dt_results.csv format."""
    return {
        "Normal count": int(n_test_normal),
        "Attack count": int(n_test_attack),
        "Total": int(n_test_normal + n_test_attack),
        "tp": metrics['tp'],
        "tn": metrics['tn'],
        "fp": metrics['fp'],
        "fn": metrics['fn'],
        "Accuracy %": uh.r2(metrics['accuracy']),
        "Precision_anom %": uh.r2(metrics['precision_anom']),
        "Precision %": uh.r2(metrics['precision_macro']),
        "Recall_anom %": uh.r2(metrics['recall_anom']),
        "Recall %": uh.r2(metrics['recall_macro']),
        "F1-Score_anom %": uh.r2(metrics['f1_anom']),
        "F1-Score %": uh.r2(metrics['f1_macro']),
        "BalancedAcc %": uh.r2(metrics['balanced_acc']),
        "MCC": uh.r3(metrics['mcc']),
        "PR-AUC": uh.r3(metrics['pr_auc']),
        "ROC-AUC": uh.r3(metrics['roc_auc']),
        
        # Test resource metrics
        "TotalTime (ms)": uh.r3(test_resources['wall_ns'] / 1_000_000),
        "AvgTimePerPacket(ns)": uh.r3(test_resources['avg_time_per_pkt_ns']),
        "Ram_usage": uh.r3(test_resources['peak_ram_mb']),
        "CPU_avg%": uh.r3(test_resources['cpu_avg_pct']),
        "CPU_peak%": uh.r3(test_resources['cpu_peak_pct']),
        
        # Training resource metrics
        "training_time_ms": uh.r3(train_resources['wall_ns'] / 1_000_000),
        "training_avg_time_per_packet_ns": uh.r3(train_resources['avg_time_per_pkt_ns']),
        "training_peak_ram_mb": uh.r3(train_resources['peak_ram_mb']),
        "training_cpu_avg_pct": uh.r3(train_resources['cpu_avg_pct']),
        "training_cpu_peak_pct": uh.r3(train_resources['cpu_peak_pct']),
        "n_train_attack": int(n_train_attack),
        "n_train_normal": int(n_train_normal)
    }


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def run_detector(detector_name, detector_factory, X_train, y_train, X_val, y_val, X_test, y_test,
                 n_train_normal, n_train_attack, n_test_normal, n_test_attack):
    """Run a single detector and return results.
    
    Workflow:
    1. Train detector on training data
    2. Tune threshold on validation data (proper hyperparameter tuning)
    3. Evaluate on test data
    """
    print(f"  Running {detector_name}...")
    
    detector = detector_factory()
    
    # Training with resource profiling
    if RESOURCE_PROFILER_AVAILABLE:
        with ResourceProfiler() as profiler_train:
            detector.fit(X_train, y_train)
        train_resources = {
            'wall_ns': profiler_train.wall_nanoseconds,
            'avg_time_per_pkt_ns': profiler_train.wall_nanoseconds / len(y_train) if len(y_train) else 0,
            'peak_ram_mb': profiler_train.peak_ram_mb,
            'cpu_avg_pct': profiler_train.cpu_avg_machine_pct,
            'cpu_peak_pct': profiler_train.cpu_peak_machine_pct
        }
    else:
        start = time.time()
        detector.fit(X_train, y_train)
        elapsed = time.time() - start
        train_resources = {
            'wall_ns': elapsed * 1e9,
            'avg_time_per_pkt_ns': (elapsed * 1e9) / len(y_train) if len(y_train) else 0,
            'peak_ram_mb': 0,
            'cpu_avg_pct': 0,
            'cpu_peak_pct': 0
        }
    
    # Tune threshold on validation set (proper hyperparameter tuning)
    if X_val is not None and y_val is not None and len(X_val) > 0:
        tuned_threshold = tune_threshold_on_validation(detector, X_val, y_val)
        if tuned_threshold is not None:
            detector.threshold = tuned_threshold
    
    # Testing with resource profiling
    if RESOURCE_PROFILER_AVAILABLE:
        with ResourceProfiler() as profiler_test:
            y_pred = detector.predict(X_test)
            y_scores = detector.predict_scores(X_test) if hasattr(detector, 'predict_scores') else None
        test_resources = {
            'wall_ns': profiler_test.wall_nanoseconds,
            'avg_time_per_pkt_ns': profiler_test.wall_nanoseconds / len(y_test) if len(y_test) else 0,
            'peak_ram_mb': profiler_test.peak_ram_mb,
            'cpu_avg_pct': profiler_test.cpu_avg_machine_pct,
            'cpu_peak_pct': profiler_test.cpu_peak_machine_pct
        }
    else:
        start = time.time()
        y_pred = detector.predict(X_test)
        y_scores = detector.predict_scores(X_test) if hasattr(detector, 'predict_scores') else None
        elapsed = time.time() - start
        test_resources = {
            'wall_ns': elapsed * 1e9,
            'avg_time_per_pkt_ns': (elapsed * 1e9) / len(y_test) if len(y_test) else 0,
            'peak_ram_mb': 0,
            'cpu_avg_pct': 0,
            'cpu_peak_pct': 0
        }
    
    # Ensure predictions are binary (0 or 1)
    if set(np.unique(y_pred)) == {-1, 1}:
        y_pred = np.where(y_pred == -1, 1, 0)
    
    # Compute metrics
    metrics = compute_metrics(y_test, y_pred, y_scores)
    
    # Build results JSON
    results_json = build_results_json(
        metrics, test_resources, train_resources,
        n_test_normal, n_test_attack, n_train_normal, n_train_attack
    )
    
    print(f"    Accuracy: {metrics['accuracy']:.2f}%, F1-Score: {metrics['f1_macro']:.2f}%")
    
    return results_json


def main():
    """Main execution function."""
    print("="*60)
    print("BINARY ANOMALY DETECTION - SV Dataset")
    print("Using Unsupervised Algorithms")
    print("="*60)
    print(f"ResourceProfiler Available: {RESOURCE_PROFILER_AVAILABLE}")
    print(f"Attack Types: {ATTACK_LIST}")
    print(f"Number of Runs: {NUM_RUNS}")
    print()
    
    # Setup output directory
    script_dir = Path(__file__).parent
    output_dir = script_dir
    
    # Get detectors
    detectors = get_detectors()
    
    # Process each attack type
    for attack_name in ATTACK_LIST:
        print(f"\n{'='*60}")
        print(f"Processing Attack Type: {attack_name.upper()}")
        print(f"{'='*60}")
        
        # Load data
        try:
            X_train, y_train, X_val, y_val, X_test, y_test, feats, y_orig = \
                preprocessed.load_preprocessed_for_attack(attack_name, base_dir=PREPROCESSED_DIR)
        except Exception as e:
            print(f"Error loading data for {attack_name}: {e}")
            continue
        
        # Convert to numpy if needed
        if hasattr(X_train, 'values'):
            X_train = X_train.values
        if hasattr(X_val, 'values'):
            X_val = X_val.values
        if hasattr(X_test, 'values'):
            X_test = X_test.values
        if hasattr(y_train, 'values'):
            y_train = y_train.values
        if hasattr(y_val, 'values'):
            y_val = y_val.values
        if hasattr(y_test, 'values'):
            y_test = y_test.values
        
        # Get counts
        n_test_attack = int(np.sum(y_test == 1))
        n_test_normal = int(np.sum(y_test == 0))
        n_train_attack = int(np.sum(y_train == 1))
        n_train_normal = int(np.sum(y_train == 0))
        
        # Dataset info for CSV
        ds_info = {
            'dataset': DATASET_NAME,
            'goid': GOID,
            'attack_type': attack_name
        }
        
        # Run each detector
        for detector_name, detector_factory in detectors.items():
            print(f"\n--- {detector_name} ---")
            
            # Output CSV path for this detector
            agg_csv_path = output_dir / f"aggregated_{detector_name.lower()}_results.csv"
            
            all_runs_results = {}
            
            for i in range(1, NUM_RUNS + 1):
                run_key = f"Run_{i}"
                print(f"  Executing {run_key}...")
                
                try:
                    results_json = run_detector(
                        detector_name, detector_factory,
                        X_train, y_train, X_val, y_val, X_test, y_test,
                        n_train_normal, n_train_attack,
                        n_test_normal, n_test_attack
                    )
                    
                    all_runs_results[run_key] = {"Test": results_json}
                    
                except Exception as e:
                    print(f"  Error in {run_key}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            # Save results to CSV
            if all_runs_results:
                uh.append_results_to_csv(str(agg_csv_path), all_runs_results, ds_info)
                print(f"  Results saved to: {agg_csv_path}")
    
    print("\n" + "="*60)
    print("COMPLETE!")
    print("="*60)


if __name__ == "__main__":
    main()
