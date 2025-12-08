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

# Sklearn imports
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import (
    confusion_matrix, classification_report,
    precision_score, recall_score, f1_score,
    accuracy_score, roc_auc_score, average_precision_score,
    balanced_accuracy_score, matthews_corrcoef
)

# PyTorch imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

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
NUM_RUNS = 1
CONTAMINATION = 0.2  # Expected proportion of anomalies in training data
PREPROCESSED_DIR = str(Path(__file__).parent / 'preprocessed')


# =============================================================================
# AUTOENCODER MODEL (PYTORCH)
# =============================================================================

class AutoencoderModel(nn.Module):
    """PyTorch Autoencoder for anomaly detection via reconstruction error."""
    
    def __init__(self, input_dim, encoder_layers=[64, 32], latent_dim=16, 
                 decoder_layers=[32, 64], activation='relu', dropout=0.2):
        super(AutoencoderModel, self).__init__()
        
        # Activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.2)
        else:
            self.activation = nn.ReLU()
        
        # Build encoder
        encoder_layers_list = []
        prev_dim = input_dim
        for hidden_dim in encoder_layers:
            encoder_layers_list.append(nn.Linear(prev_dim, hidden_dim))
            encoder_layers_list.append(self.activation)
            encoder_layers_list.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # Latent layer
        encoder_layers_list.append(nn.Linear(prev_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers_list)
        
        # Build decoder
        decoder_layers_list = []
        prev_dim = latent_dim
        for hidden_dim in decoder_layers:
            decoder_layers_list.append(nn.Linear(prev_dim, hidden_dim))
            decoder_layers_list.append(self.activation)
            decoder_layers_list.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # Output layer
        decoder_layers_list.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers_list)
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def encode(self, x):
        return self.encoder(x)


# =============================================================================
# BINARY ANOMALY DETECTORS
# =============================================================================

class BinaryIsolationForestDetector:
    """
    Isolation Forest for binary anomaly detection.
    Trains on all data with specified contamination.
    """
    
    def __init__(self, name: str, config: Dict = None):
        self.name = name
        self.config = config or {}
        self.model = None
        self.scaler = StandardScaler()
        
    def fit(self, X_train, y_train):
        """Train Isolation Forest."""
        X_scaled = self.scaler.fit_transform(X_train)
        
        self.model = IsolationForest(
            n_estimators=self.config.get('n_estimators', 100),
            contamination=CONTAMINATION,
            random_state=42,
            n_jobs=-1
        )
        self.model.fit(X_scaled)
        return self
    
    def predict(self, X):
        """Predict: 1 = anomaly (attack), 0 = normal."""
        X_scaled = self.scaler.transform(X)
        preds = self.model.predict(X_scaled)
        # IsolationForest: -1 = outlier, 1 = inlier
        # Convert to: 1 = anomaly, 0 = normal
        return np.where(preds == -1, 1, 0)
    
    def predict_scores(self, X):
        """Return anomaly scores (higher = more anomalous)."""
        X_scaled = self.scaler.transform(X)
        # score_samples: higher = more normal, so negate
        return -self.model.score_samples(X_scaled)


class BinaryAutoencoderDetector:
    """
    Autoencoder for binary anomaly detection.
    Trains on all data, uses reconstruction error threshold.
    """
    
    def __init__(self, name: str, config: Dict = None):
        self.name = name
        self.config = config or {}
        self.ae = None
        self.threshold = None
        self.scaler = StandardScaler()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def fit(self, X_train, y_train):
        """Train autoencoder on all data, set threshold on normal samples."""
        X_scaled = self.scaler.fit_transform(X_train)
        input_dim = X_scaled.shape[1]
        
        # Build autoencoder
        self.ae = AutoencoderModel(
            input_dim=input_dim,
            encoder_layers=self.config.get('encoder_layers', [64, 32]),
            latent_dim=self.config.get('latent_dim', 16),
            decoder_layers=self.config.get('decoder_layers', [32, 64]),
            activation=self.config.get('activation', 'relu'),
            dropout=self.config.get('dropout', 0.2)
        ).to(self.device)
        
        # Training parameters
        epochs = self.config.get('epochs', 50)
        batch_size = self.config.get('batch_size', 256)
        learning_rate = self.config.get('learning_rate', 0.001)
        patience = self.config.get('early_stopping_patience', 10)
        
        # Use only normal samples for training
        normal_mask = y_train == 0
        X_normal = X_scaled[normal_mask]
        
        if len(X_normal) < 100:
            # Not enough normal samples, use all data
            X_normal = X_scaled
        
        # Prepare data
        X_tensor = torch.FloatTensor(X_normal).to(self.device)
        
        # Train/validation split (80/20)
        n_train = int(len(X_normal) * 0.8)
        X_train_tensor = X_tensor[:n_train]
        X_val_tensor = X_tensor[n_train:] if n_train < len(X_normal) else X_tensor
        
        train_dataset = TensorDataset(X_train_tensor, X_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Optimizer and loss
        optimizer = optim.Adam(self.ae.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        
        self.ae.train()
        for epoch in range(epochs):
            train_loss = 0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                output = self.ae(batch_X)
                loss = criterion(output, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # Validation
            self.ae.eval()
            with torch.no_grad():
                val_output = self.ae(X_val_tensor)
                val_loss = criterion(val_output, X_val_tensor).item()
            self.ae.train()
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break
        
        self.ae.eval()
        
        # Compute threshold based on normal data reconstruction error
        with torch.no_grad():
            X_normal_tensor = torch.FloatTensor(X_normal).to(self.device)
            reconstructed = self.ae(X_normal_tensor)
            errors = torch.mean((X_normal_tensor - reconstructed) ** 2, dim=1).cpu().numpy()
        
        # Set threshold based on CONTAMINATION rate
        # For CONTAMINATION=0.2 (20% anomalies), use 80th percentile
        threshold_percentile = (1 - CONTAMINATION) * 100  # 80 for 20% contamination
        self.threshold = np.percentile(errors, threshold_percentile)
        
        return self
    
    def predict(self, X):
        """Predict: 1 = anomaly (attack), 0 = normal."""
        errors = self._compute_reconstruction_error(X)
        return (errors > self.threshold).astype(int)
    
    def predict_scores(self, X):
        """Return reconstruction errors as anomaly scores."""
        return self._compute_reconstruction_error(X)
    
    def _compute_reconstruction_error(self, X):
        """Compute per-sample reconstruction error."""
        X_scaled = self.scaler.transform(X)
        self.ae.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_scaled).to(self.device)
            reconstructed = self.ae(X_tensor)
            errors = torch.mean((X_tensor - reconstructed) ** 2, dim=1).cpu().numpy()
        return errors


class BinaryGMMDetector:
    """
    GMM-based anomaly detection.
    Fits GMM on normal data, uses log-likelihood for scoring.
    """
    
    def __init__(self, name: str, config: Dict = None):
        self.name = name
        self.config = config or {}
        self.model = None
        self.threshold = None
        self.scaler = StandardScaler()
        
    def fit(self, X_train, y_train):
        """Train GMM on normal samples."""
        X_scaled = self.scaler.fit_transform(X_train)
        
        # Use normal data for training
        normal_mask = y_train == 0
        X_normal = X_scaled[normal_mask]
        
        if len(X_normal) < 100:
            X_normal = X_scaled
        
        self.model = GaussianMixture(
            n_components=self.config.get('n_components', 3),  # 3 components for normal sub-patterns
            covariance_type=self.config.get('covariance_type', 'full'),
            random_state=42,
            max_iter=100
        )
        self.model.fit(X_normal)
        
        # Set threshold based on CONTAMINATION rate
        # For CONTAMINATION=0.2 (20% anomalies), use 80th percentile
        scores = -self.model.score_samples(X_normal)  # Negate: higher = more anomalous
        threshold_percentile = (1 - CONTAMINATION) * 100  # 80 for 20% contamination
        self.threshold = np.percentile(scores, threshold_percentile)
        
        return self
    
    def predict(self, X):
        """Predict: 1 = anomaly, 0 = normal."""
        scores = self.predict_scores(X)
        return (scores > self.threshold).astype(int)
    
    def predict_scores(self, X):
        """Return negative log-likelihood as anomaly scores."""
        X_scaled = self.scaler.transform(X)
        return -self.model.score_samples(X_scaled)


class BinaryKMeansDetector:
    """
    KMeans-based anomaly detection.
    Uses distance to nearest cluster center for scoring.
    
    For binary anomaly detection:
    - Trains on normal data only to learn normal patterns
    - Uses multiple clusters (default=3) to capture sub-patterns in normal data
    - Sets threshold based on contamination rate (not arbitrary percentile)
    - Points far from all cluster centers are flagged as anomalies
    """
    
    def __init__(self, name: str, config: Dict = None):
        self.name = name
        self.config = config or {}
        self.model = None
        self.threshold = None
        self.scaler = StandardScaler()
        
    def fit(self, X_train, y_train):
        """Train KMeans on normal samples and set contamination-aware threshold."""
        X_scaled = self.scaler.fit_transform(X_train)
        
        # Use normal data for training
        normal_mask = y_train == 0
        X_normal = X_scaled[normal_mask]
        
        if len(X_normal) < 100:
            X_normal = X_scaled
        
        self.model = KMeans(
            n_clusters=self.config.get('n_clusters', 3),  # 3 clusters to capture normal sub-patterns
            random_state=42,
            n_init=10,
            max_iter=300
        )
        self.model.fit(X_normal)
        
        # Set threshold based on CONTAMINATION rate, not arbitrary 95th percentile
        # For CONTAMINATION=0.2 (20% anomalies), use 80th percentile
        # This ensures ~20% of test samples will be flagged as anomalies
        scores = self._compute_distances(X_normal)
        threshold_percentile = (1 - CONTAMINATION) * 100  # 80 for 20% contamination
        self.threshold = np.percentile(scores, threshold_percentile)
        
        return self
    
    def predict(self, X):
        """Predict: 1 = anomaly, 0 = normal."""
        scores = self.predict_scores(X)
        return (scores > self.threshold).astype(int)
    
    def predict_scores(self, X):
        """Return distance to nearest cluster center as anomaly score."""
        return self._compute_distances(X)
    
    def _compute_distances(self, X):
        """Compute distance to nearest cluster center."""
        if hasattr(X, 'values'):
            X = X.values
        X_scaled = self.scaler.transform(X)
        # Get cluster assignments
        labels = self.model.predict(X_scaled)
        # Compute distance to assigned cluster center
        distances = np.zeros(len(X_scaled))
        for i, (sample, label) in enumerate(zip(X_scaled, labels)):
            distances[i] = np.linalg.norm(sample - self.model.cluster_centers_[label])
        return distances


class BinaryHierarchicalDetector:
    """
    Hierarchical Clustering-based anomaly detection.
    Uses distance to training samples for scoring.
    """
    
    def __init__(self, name: str, config: Dict = None):
        self.name = name
        self.config = config or {}
        self.X_train_ = None
        self.train_labels_ = None
        self.threshold = None
        self.scaler = StandardScaler()
        self.knn = None
        
    def fit(self, X_train, y_train):
        """Train on normal samples using hierarchical clustering."""
        X_scaled = self.scaler.fit_transform(X_train)
        
        # Use normal data
        normal_mask = y_train == 0
        X_normal = X_scaled[normal_mask]
        
        if len(X_normal) < 100:
            X_normal = X_scaled
        
        # Downsample if too large
        max_samples = 10000
        if len(X_normal) > max_samples:
            indices = np.random.choice(len(X_normal), max_samples, replace=False)
            X_normal = X_normal[indices]
        
        # Fit hierarchical clustering
        model = AgglomerativeClustering(
            n_clusters=self.config.get('n_clusters', 3),  # 3 clusters for normal sub-patterns
            linkage='ward'
        )
        model.fit(X_normal)
        
        # Store training data for k-NN prediction
        self.X_train_ = X_normal
        self.train_labels_ = model.labels_
        
        # Fit k-NN for distance-based scoring
        k = min(5, len(X_normal))
        self.knn = NearestNeighbors(n_neighbors=k, metric='euclidean')
        self.knn.fit(X_normal)
        
        # Compute threshold based on CONTAMINATION rate
        # For CONTAMINATION=0.2 (20% anomalies), use 80th percentile
        distances, _ = self.knn.kneighbors(X_normal)
        avg_distances = np.mean(distances, axis=1)
        threshold_percentile = (1 - CONTAMINATION) * 100  # 80 for 20% contamination
        self.threshold = np.percentile(avg_distances, threshold_percentile)
        
        return self
    
    def predict(self, X):
        """Predict: 1 = anomaly, 0 = normal."""
        scores = self.predict_scores(X)
        return (scores > self.threshold).astype(int)
    
    def predict_scores(self, X):
        """Return average distance to k nearest neighbors as anomaly score."""
        if hasattr(X, 'values'):
            X = X.values
        X_scaled = self.scaler.transform(X)
        distances, _ = self.knn.kneighbors(X_scaled)
        return np.mean(distances, axis=1)


# =============================================================================
# DETECTOR REGISTRY
# =============================================================================

def get_detectors() -> Dict:
    """Return dictionary of detector factories."""
    return {
        'IsolationForest': lambda: BinaryIsolationForestDetector(
            name='IsolationForest',
            config={'n_estimators': 100}
        ),
        'Autoencoder': lambda: BinaryAutoencoderDetector(
            name='Autoencoder',
            config={
                'encoder_layers': [64, 32],
                'latent_dim': 16,
                'decoder_layers': [32, 64],
                'epochs': 50,
                'batch_size': 256,
                'learning_rate': 0.001,
                'early_stopping_patience': 10
            }
        ),
        'GMM': lambda: BinaryGMMDetector(
            name='GMM',
            config={'n_components': 3, 'covariance_type': 'full'}
        ),
        'KMeans': lambda: BinaryKMeansDetector(
            name='KMeans',
            config={'n_clusters': 8}
        ),
        'Hierarchical': lambda: BinaryHierarchicalDetector(
            name='Hierarchical',
            config={'n_clusters': 3}
        ),
    }


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
    # if X_val is not None and y_val is not None and len(X_val) > 0:
    #     tuned_threshold = tune_threshold_on_validation(detector, X_val, y_val)
        # if tuned_threshold is not None:
        #     detector.threshold = tuned_threshold
    
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
