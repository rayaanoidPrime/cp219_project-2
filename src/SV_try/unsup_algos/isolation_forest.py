"""
Isolation Forest Binary Anomaly Detector

Unsupervised anomaly detection using Isolation Forest.
Trains on all data with specified contamination.
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from typing import Dict


class BinaryIsolationForestDetector:
    """
    Isolation Forest for binary anomaly detection.
    Trains on all data with specified contamination.
    """
    
    def __init__(self, name: str = 'IsolationForest', config: Dict = None, contamination: float = 0.2):
        self.name = name
        self.config = config or {}
        self.model = None
        self.scaler = StandardScaler()
        self.contamination = contamination
        
    def fit(self, X_train, y_train=None):
        """Train Isolation Forest."""
        X_scaled = self.scaler.fit_transform(X_train)
        
        self.model = IsolationForest(
            n_estimators=self.config.get('n_estimators', 100),
            contamination=self.contamination,
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


def model(X_train, y_train, X_val=None, y_val=None, contamination=0.2):
    """
    Create and train an Isolation Forest detector.
    
    Args:
        X_train: Training features
        y_train: Training labels (0=normal, 1=attack)
        X_val: Validation features (optional)
        y_val: Validation labels (optional)
        contamination: Expected proportion of anomalies
    
    Returns:
        Trained BinaryIsolationForestDetector
    """
    print("Training Isolation Forest...")
    
    config = {'n_estimators': 100}
    detector = BinaryIsolationForestDetector(
        name='IsolationForest',
        config=config,
        contamination=contamination
    )
    
    detector.fit(X_train, y_train)
    
    if X_val is not None and y_val is not None:
        from sklearn.metrics import accuracy_score, f1_score
        y_pred = detector.predict(X_val)
        acc = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred, zero_division=0)
        print(f"Isolation Forest Val Accuracy: {acc:.4f}, F1: {f1:.4f}")
    
    return detector
