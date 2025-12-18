"""
Unsupervised Binary Anomaly Detection Algorithms

This package contains modular implementations of unsupervised 
anomaly detection algorithms for binary classification.

Available modules:
- isolation_forest: Isolation Forest detector
- autoencoder: PyTorch Autoencoder detector
- gmm: Gaussian Mixture Model detector
- kmeans: KMeans clustering-based detector
- hierarchical: Hierarchical Clustering detector

Each module provides:
- A detector class (e.g., BinaryIsolationForestDetector)
- A model() function for easy training
"""

from .isolation_forest import BinaryIsolationForestDetector, model as isolation_forest_model
from .autoencoder import BinaryAutoencoderDetector, model as autoencoder_model
from .gmm import BinaryGMMDetector, model as gmm_model
from .kmeans import BinaryKMeansDetector, model as kmeans_model
from .hierarchical import BinaryHierarchicalDetector, model as hierarchical_model

__all__ = [
    'BinaryIsolationForestDetector',
    'BinaryAutoencoderDetector', 
    'BinaryGMMDetector',
    'BinaryKMeansDetector',
    'BinaryHierarchicalDetector',
    'isolation_forest_model',
    'autoencoder_model',
    'gmm_model',
    'kmeans_model',
    'hierarchical_model',
]


def get_all_detectors(contamination=0.2):
    """
    Return a dictionary of all detector factories.
    
    Args:
        contamination: Expected proportion of anomalies (default 0.2)
    
    Returns:
        Dict of detector name -> factory function
    """
    return {
        'IsolationForest': lambda: BinaryIsolationForestDetector(
            name='IsolationForest',
            config={'n_estimators': 100},
            contamination=contamination
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
            },
            contamination=contamination
        ),
        'GMM': lambda: BinaryGMMDetector(
            name='GMM',
            config={'n_components': 3, 'covariance_type': 'full'},
            contamination=contamination
        ),
        'KMeans': lambda: BinaryKMeansDetector(
            name='KMeans',
            config={'n_clusters': 8},
            contamination=contamination
        ),
        'Hierarchical': lambda: BinaryHierarchicalDetector(
            name='Hierarchical',
            config={'n_clusters': 3},
            contamination=contamination
        ),
    }
