"""
Hierarchical Clustering Binary Anomaly Detector

Unsupervised anomaly detection using Hierarchical Clustering.
Uses distance to training samples for scoring via k-NN.
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import NearestNeighbors
from typing import Dict


class BinaryHierarchicalDetector:
    """
    Hierarchical Clustering-based anomaly detection.
    Uses distance to training samples for scoring.
    """
    
    def __init__(self, name: str = 'Hierarchical', config: Dict = None, contamination: float = 0.2):
        self.name = name
        self.config = config or {}
        self.X_train_ = None
        self.train_labels_ = None
        self.threshold = None
        self.scaler = StandardScaler()
        self.knn = None
        self.contamination = contamination
        
    def fit(self, X_train, y_train=None):
        """Train on normal samples using hierarchical clustering."""
        X_scaled = self.scaler.fit_transform(X_train)
        
        # Use normal data if labels provided
        if y_train is not None:
            normal_mask = y_train == 0
            X_normal = X_scaled[normal_mask]
            if len(X_normal) < 100:
                X_normal = X_scaled
        else:
            X_normal = X_scaled
        
        # Downsample if too large
        max_samples = 10000
        if len(X_normal) > max_samples:
            indices = np.random.choice(len(X_normal), max_samples, replace=False)
            X_normal = X_normal[indices]
        
        # Fit hierarchical clustering
        model = AgglomerativeClustering(
            n_clusters=self.config.get('n_clusters', 3),
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
        
        # Compute threshold based on contamination rate
        distances, _ = self.knn.kneighbors(X_normal)
        avg_distances = np.mean(distances, axis=1)
        threshold_percentile = (1 - self.contamination) * 100
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


def model(X_train, y_train, X_val=None, y_val=None, contamination=0.2):
    """
    Create and train a Hierarchical Clustering detector.
    
    Args:
        X_train: Training features
        y_train: Training labels (0=normal, 1=attack)
        X_val: Validation features (optional)
        y_val: Validation labels (optional)
        contamination: Expected proportion of anomalies
    
    Returns:
        Trained BinaryHierarchicalDetector
    """
    print("Training Hierarchical Clustering...")
    
    config = {'n_clusters': 3}
    
    detector = BinaryHierarchicalDetector(
        name='Hierarchical',
        config=config,
        contamination=contamination
    )
    
    detector.fit(X_train, y_train)
    
    if X_val is not None and y_val is not None:
        from sklearn.metrics import accuracy_score, f1_score
        y_pred = detector.predict(X_val)
        acc = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred, zero_division=0)
        print(f"Hierarchical Val Accuracy: {acc:.4f}, F1: {f1:.4f}")
    
    return detector
