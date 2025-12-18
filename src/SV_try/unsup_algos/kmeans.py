"""
KMeans Binary Anomaly Detector

Unsupervised anomaly detection using KMeans clustering.
Uses distance to nearest cluster center for scoring.
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from typing import Dict


class BinaryKMeansDetector:
    """
    KMeans-based anomaly detection.
    Uses distance to nearest cluster center for scoring.
    
    For binary anomaly detection:
    - Trains on normal data only to learn normal patterns
    - Uses multiple clusters (default=3) to capture sub-patterns in normal data
    - Sets threshold based on contamination rate
    - Points far from all cluster centers are flagged as anomalies
    """
    
    def __init__(self, name: str = 'KMeans', config: Dict = None, contamination: float = 0.2):
        self.name = name
        self.config = config or {}
        self.model = None
        self.threshold = None
        self.scaler = StandardScaler()
        self.contamination = contamination
        
    def fit(self, X_train, y_train=None):
        """Train KMeans on normal samples and set contamination-aware threshold."""
        X_scaled = self.scaler.fit_transform(X_train)
        
        # Use normal data for training if labels provided
        if y_train is not None:
            normal_mask = y_train == 0
            X_normal = X_scaled[normal_mask]
            if len(X_normal) < 100:
                X_normal = X_scaled
        else:
            X_normal = X_scaled
        
        self.model = KMeans(
            n_clusters=self.config.get('n_clusters', 3),
            random_state=42,
            n_init=10,
            max_iter=300
        )
        self.model.fit(X_normal)
        
        # Set threshold based on contamination rate
        scores = self._compute_distances(X_normal, already_scaled=True)
        threshold_percentile = (1 - self.contamination) * 100
        self.threshold = np.percentile(scores, threshold_percentile)
        
        return self
    
    def predict(self, X):
        """Predict: 1 = anomaly, 0 = normal."""
        scores = self.predict_scores(X)
        return (scores > self.threshold).astype(int)
    
    def predict_scores(self, X):
        """Return distance to nearest cluster center as anomaly score."""
        return self._compute_distances(X)
    
    def _compute_distances(self, X, already_scaled=False):
        """Compute distance to nearest cluster center."""
        if hasattr(X, 'values'):
            X = X.values
        if not already_scaled:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X
        # Get cluster assignments
        labels = self.model.predict(X_scaled)
        # Compute distance to assigned cluster center
        distances = np.zeros(len(X_scaled))
        for i, (sample, label) in enumerate(zip(X_scaled, labels)):
            distances[i] = np.linalg.norm(sample - self.model.cluster_centers_[label])
        return distances


def model(X_train, y_train, X_val=None, y_val=None, contamination=0.2):
    """
    Create and train a KMeans detector.
    
    Args:
        X_train: Training features
        y_train: Training labels (0=normal, 1=attack)
        X_val: Validation features (optional)
        y_val: Validation labels (optional)
        contamination: Expected proportion of anomalies
    
    Returns:
        Trained BinaryKMeansDetector
    """
    print("Training KMeans...")
    
    config = {'n_clusters': 8}
    
    detector = BinaryKMeansDetector(
        name='KMeans',
        config=config,
        contamination=contamination
    )
    
    detector.fit(X_train, y_train)
    
    if X_val is not None and y_val is not None:
        from sklearn.metrics import accuracy_score, f1_score
        y_pred = detector.predict(X_val)
        acc = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred, zero_division=0)
        print(f"KMeans Val Accuracy: {acc:.4f}, F1: {f1:.4f}")
    
    return detector
