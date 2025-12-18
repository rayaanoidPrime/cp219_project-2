"""
GMM (Gaussian Mixture Model) Binary Anomaly Detector

Unsupervised anomaly detection using GMM.
Fits GMM on normal data, uses log-likelihood for scoring.
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from typing import Dict


class BinaryGMMDetector:
    """
    GMM-based anomaly detection.
    Fits GMM on normal data, uses log-likelihood for scoring.
    """
    
    def __init__(self, name: str = 'GMM', config: Dict = None, contamination: float = 0.2):
        self.name = name
        self.config = config or {}
        self.model = None
        self.threshold = None
        self.scaler = StandardScaler()
        self.contamination = contamination
        
    def fit(self, X_train, y_train=None):
        """Train GMM on normal samples."""
        X_scaled = self.scaler.fit_transform(X_train)
        
        # Use normal data for training if labels provided
        if y_train is not None:
            normal_mask = y_train == 0
            X_normal = X_scaled[normal_mask]
            if len(X_normal) < 100:
                X_normal = X_scaled
        else:
            X_normal = X_scaled
        
        self.model = GaussianMixture(
            n_components=self.config.get('n_components', 3),
            covariance_type=self.config.get('covariance_type', 'full'),
            random_state=42,
            max_iter=100
        )
        self.model.fit(X_normal)
        
        # Set threshold based on contamination rate
        scores = -self.model.score_samples(X_normal)  # Negate: higher = more anomalous
        threshold_percentile = (1 - self.contamination) * 100
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


def model(X_train, y_train, X_val=None, y_val=None, contamination=0.2):
    """
    Create and train a GMM detector.
    
    Args:
        X_train: Training features
        y_train: Training labels (0=normal, 1=attack)
        X_val: Validation features (optional)
        y_val: Validation labels (optional)
        contamination: Expected proportion of anomalies
    
    Returns:
        Trained BinaryGMMDetector
    """
    print("Training GMM...")
    
    config = {
        'n_components': 3,
        'covariance_type': 'full'
    }
    
    detector = BinaryGMMDetector(
        name='GMM',
        config=config,
        contamination=contamination
    )
    
    detector.fit(X_train, y_train)
    
    if X_val is not None and y_val is not None:
        from sklearn.metrics import accuracy_score, f1_score
        y_pred = detector.predict(X_val)
        acc = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred, zero_division=0)
        print(f"GMM Val Accuracy: {acc:.4f}, F1: {f1:.4f}")
    
    return detector
