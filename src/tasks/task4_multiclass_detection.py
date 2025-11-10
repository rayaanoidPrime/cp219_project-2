"""
Task 4: Unsupervised Multi-Class Attack Detection
Perform unsupervised multi-class classification among Normal traffic and four attack types.
Compare multiple unsupervised algorithms using macro-averaged metrics and confusion matrices.
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any, List, Tuple
import time
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.manifold import TSNE
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import umap
from sklearn.cluster import DBSCAN, HDBSCAN
from scipy.optimize import linear_sum_assignment

from src.preprocessing import (
    load_combined_datasets_multiclass,
    preprocess_dataframe,
    load_combined_datasets,
    standardize_schema,
    engineer_features,
    get_numeric_features,
    CORE_FIELDS,
    ALLOWED_FIELDS
)



class MultiClassDetector:
    """
    Updated MultiClassDetector with proper Hierarchical Clustering support.
    """
    
    def __init__(self, name: str, model, config: Dict = None):
        self.name = name
        self.model = model
        self.config = config or {}
        self.train_time = 0
        self.inference_time = 0
        self.cluster_to_class_map_ = None
        self.n_clusters = 5
        
        # For Hierarchical Clustering: store training data and labels
        self.X_train_ = None
        self.train_cluster_labels_ = None
        
    def fit(self, X):
        """Train the clustering model."""
        start = time.time()
        self.model.fit(X)
        self.train_time = time.time() - start
        
        # For Hierarchical Clustering, store training data and cluster assignments
        if isinstance(self.model, AgglomerativeClustering):
            self.X_train_ = X.copy()
            self.train_cluster_labels_ = self.model.labels_.copy()
            print(f"      Stored {len(self.X_train_)} training samples for Hierarchical prediction")
        
        return self
    
    def predict_clusters(self, X):
        """Get cluster assignments."""
        start = time.time()
        
        if hasattr(self.model, 'predict'):
            # Standard prediction for K-Means, GMM
            clusters = self.model.predict(X)
        elif isinstance(self.model, AgglomerativeClustering):
            # Special handling for Hierarchical Clustering
            # Use k-NN approach to assign test samples to training clusters
            clusters = self._predict_hierarchical(X)
        else:
            # Fallback: use labels from training (only works if X is training data)
            if hasattr(self.model, 'labels_'):
                clusters = self.model.labels_
            else:
                raise ValueError(f"Model {self.name} has no predict method or labels_")
        
        self.inference_time = (time.time() - start) / len(X)
        return clusters
    
    def _predict_hierarchical(self, X_test):
        """
        Predict cluster assignments for Hierarchical Clustering using k-NN.
        
        Strategy: For each test sample, find k nearest training samples
        and assign to the majority cluster among those neighbors.
        """
        if self.X_train_ is None or self.train_cluster_labels_ is None:
            raise ValueError("Hierarchical model must be fit with training data first")
        
        from sklearn.neighbors import NearestNeighbors
        
        # Use k=5 neighbors for voting
        k = min(5, len(self.X_train_))
        
        # Fit k-NN on training data
        nbrs = NearestNeighbors(n_neighbors=k, metric='euclidean')
        nbrs.fit(self.X_train_)
        
        # Find nearest neighbors for each test sample
        distances, indices = nbrs.kneighbors(X_test)
        
        # Assign cluster based on majority vote among neighbors
        test_clusters = np.zeros(len(X_test), dtype=int)
        for i, neighbor_indices in enumerate(indices):
            neighbor_clusters = self.train_cluster_labels_[neighbor_indices]
            # Majority vote
            test_clusters[i] = np.bincount(neighbor_clusters).argmax()
        
        return test_clusters
    
    def fit_and_map(self, X_train, y_train_true):
        """
        Fit model and learn cluster-to-class mapping using training set.
        """
        # Fit model
        self.fit(X_train)
        
        # Get cluster predictions on training set
        if hasattr(self.model, 'predict'):
            y_train_clusters = self.model.predict(X_train)
        else:
            y_train_clusters = self.model.labels_
        
        # Learn optimal mapping
        self.cluster_to_class_map_ = self._find_optimal_mapping(
            y_train_true, y_train_clusters
        )
        
        return self
    
    def predict(self, X):
        """Predict class labels using learned mapping."""
        # Get cluster assignments
        clusters = self.predict_clusters(X)
        
        # Map to class labels
        if self.cluster_to_class_map_ is None:
            raise ValueError("Model must be fit with fit_and_map() first")
        
        predictions = np.array([
            self.cluster_to_class_map_.get(c, -1) for c in clusters
        ])
        
        return predictions
    
    def _find_optimal_mapping(self, y_true, y_clusters):
        """
        Find optimal cluster-to-class mapping using Hungarian algorithm.
        """
        from scipy.optimize import linear_sum_assignment
        
        unique_clusters = np.unique(y_clusters)
        unique_classes = np.unique(y_true)
        
        n_clusters = len(unique_clusters)
        n_classes = len(unique_classes)
        
        # Create cost matrix (negative because we want to maximize)
        cost_matrix = np.zeros((n_clusters, n_classes))
        
        for i, cluster in enumerate(unique_clusters):
            for j, cls in enumerate(unique_classes):
                # Count how many samples in this cluster belong to this class
                mask = (y_clusters == cluster) & (y_true == cls)
                cost_matrix[i, j] = -np.sum(mask)  # Negative for maximization
        
        # Solve assignment problem
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        # Create mapping
        mapping = {}
        for cluster_idx, class_idx in zip(row_ind, col_ind):
            cluster_id = unique_clusters[cluster_idx]
            class_id = unique_classes[class_idx]
            mapping[cluster_id] = class_id
        
        # Handle any unmapped clusters
        for cluster in unique_clusters:
            if cluster not in mapping:
                # Find most common class in this cluster
                mask = y_clusters == cluster
                if np.sum(mask) > 0:
                    most_common = np.bincount(y_true[mask]).argmax()
                    mapping[cluster] = most_common
                else:
                    mapping[cluster] = 0  # Default to Normal
        
        return mapping
    
    def get_memory_mb(self):
        """
        Estimate memory footprint more accurately.
        
        This accounts for:
        - Model parameters (numpy arrays)
        - Training data cache (for Hierarchical)
        - Cluster assignments
        """
        total_bytes = 0
        
        # 1. Base model object
        total_bytes += sys.getsizeof(self.model)
        
        # 2. Model-specific memory (numpy arrays in sklearn models)
        try:
            if hasattr(self.model, 'cluster_centers_'):
                # K-Means: cluster centers
                total_bytes += self.model.cluster_centers_.nbytes
            
            if hasattr(self.model, 'labels_'):
                # All clustering models: labels
                total_bytes += self.model.labels_.nbytes
            
            if hasattr(self.model, 'means_'):
                # GMM: means
                total_bytes += self.model.means_.nbytes
            
            if hasattr(self.model, 'covariances_'):
                # GMM: covariances
                total_bytes += self.model.covariances_.nbytes
            
            if hasattr(self.model, 'weights_'):
                # GMM: mixture weights
                total_bytes += self.model.weights_.nbytes
            
            if hasattr(self.model, 'precisions_'):
                # GMM: precisions
                total_bytes += self.model.precisions_.nbytes
            
            if hasattr(self.model, 'precisions_cholesky_'):
                # GMM: Cholesky decomposition
                total_bytes += self.model.precisions_cholesky_.nbytes
            
        except Exception as e:
            # If any attribute access fails, continue
            pass
        
        # 3. Cached training data (for Hierarchical Clustering)
        if self.X_train_ is not None:
            total_bytes += self.X_train_.nbytes
        
        if self.train_cluster_labels_ is not None:
            total_bytes += self.train_cluster_labels_.nbytes
        
        # 4. Cluster-to-class mapping
        if self.cluster_to_class_map_ is not None:
            total_bytes += sys.getsizeof(self.cluster_to_class_map_)
            # Add size of keys and values
            for k, v in self.cluster_to_class_map_.items():
                total_bytes += sys.getsizeof(k) + sys.getsizeof(v)
        
        # Convert to MB
        return total_bytes / (1024 * 1024)
    
    def get_detailed_memory_info(self):
        """
        Get detailed breakdown of memory usage for debugging/analysis.
        
        Returns:
            Dict with memory breakdown by component
        """
        memory_info = {
            'total_mb': 0,
            'model_base_mb': 0,
            'cluster_centers_mb': 0,
            'training_cache_mb': 0,
            'labels_mb': 0,
            'mapping_mb': 0,
            'gmm_params_mb': 0
        }
        
        # Model base
        memory_info['model_base_mb'] = sys.getsizeof(self.model) / (1024 * 1024)
        
        # Cluster centers (K-Means)
        if hasattr(self.model, 'cluster_centers_'):
            memory_info['cluster_centers_mb'] = self.model.cluster_centers_.nbytes / (1024 * 1024)
        
        # Labels
        if hasattr(self.model, 'labels_'):
            memory_info['labels_mb'] = self.model.labels_.nbytes / (1024 * 1024)
        
        # Training cache (Hierarchical)
        if self.X_train_ is not None:
            cache_bytes = self.X_train_.nbytes
            if self.train_cluster_labels_ is not None:
                cache_bytes += self.train_cluster_labels_.nbytes
            memory_info['training_cache_mb'] = cache_bytes / (1024 * 1024)
        
        # GMM parameters
        if hasattr(self.model, 'means_'):
            gmm_bytes = self.model.means_.nbytes
            if hasattr(self.model, 'covariances_'):
                gmm_bytes += self.model.covariances_.nbytes
            if hasattr(self.model, 'weights_'):
                gmm_bytes += self.model.weights_.nbytes
            memory_info['gmm_params_mb'] = gmm_bytes / (1024 * 1024)
        
        # Mapping
        if self.cluster_to_class_map_ is not None:
            map_bytes = sys.getsizeof(self.cluster_to_class_map_)
            for k, v in self.cluster_to_class_map_.items():
                map_bytes += sys.getsizeof(k) + sys.getsizeof(v)
            memory_info['mapping_mb'] = map_bytes / (1024 * 1024)
        
        # Total
        memory_info['total_mb'] = sum(v for k, v in memory_info.items() if k != 'total_mb')
        
        return memory_info
        

class AutoencoderModel(nn.Module):
    """PyTorch Autoencoder for dimensionality reduction."""
    
    def __init__(self, input_dim, encoder_layers, latent_dim, decoder_layers, 
                 activation='relu', dropout=0.2):
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


class AutoencoderDetector(MultiClassDetector):
    """Autoencoder-based detector that clusters in latent space."""
    
    def __init__(self, name: str, config: Dict):
        super().__init__(name, None, config)
        
        self.autoencoder = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.clustering_model = None
        
        # Extract config
        self.ae_config = config.get('autoencoder', {})
        self.n_clusters = 5
        
    def fit(self, X):
        """Train autoencoder, then cluster in latent space."""
        start = time.time()
        
        # Step 1: Train Autoencoder
        print(f"      Training Autoencoder...")
        input_dim = X.shape[1]
        
        # Build autoencoder
        self.autoencoder = AutoencoderModel(
            input_dim=input_dim,
            encoder_layers=self.ae_config.get('architecture', {}).get('encoder_layers', [128, 64, 32]),
            latent_dim=self.ae_config.get('architecture', {}).get('latent_dim', 16),
            decoder_layers=self.ae_config.get('architecture', {}).get('decoder_layers', [32, 64, 128]),
            activation=self.ae_config.get('activation', 'relu'),
            dropout=self.ae_config.get('dropout', 0.2)
        ).to(self.device)
        
        # Training parameters
        train_config = self.ae_config.get('training', {})
        epochs = train_config.get('epochs', 100)
        batch_size = train_config.get('batch_size', 256)
        learning_rate = train_config.get('learning_rate', 0.001)
        validation_split = train_config.get('validation_split', 0.2)
        patience = train_config.get('early_stopping_patience', 10)
        
        # Prepare data
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        # Train/validation split
        n_train = int(len(X) * (1 - validation_split))
        X_train_tensor = X_tensor[:n_train]
        X_val_tensor = X_tensor[n_train:]
        
        train_dataset = TensorDataset(X_train_tensor, X_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Optimizer and loss
        optimizer = optim.Adam(self.autoencoder.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        
        self.autoencoder.train()
        for epoch in range(epochs):
            train_loss = 0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                output = self.autoencoder(batch_X)
                loss = criterion(output, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # Validation
            self.autoencoder.eval()
            with torch.no_grad():
                val_output = self.autoencoder(X_val_tensor)
                val_loss = criterion(val_output, X_val_tensor).item()
            self.autoencoder.train()
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"        Early stopping at epoch {epoch+1}")
                    break
            
            if (epoch + 1) % 10 == 0:
                print(f"        Epoch {epoch+1}/{epochs}: Train Loss={train_loss:.6f}, Val Loss={val_loss:.6f}")
        
        # Step 2: Extract latent representations
        print(f"      Extracting latent representations...")
        self.autoencoder.eval()
        with torch.no_grad():
            X_latent = self.autoencoder.encode(X_tensor).cpu().numpy()
        
        # Step 3: Cluster in latent space
        print(f"      Clustering in latent space...")
        from sklearn.cluster import KMeans
        self.clustering_model = KMeans(
            n_clusters=self.n_clusters,
            random_state=42,
            n_init=10,
            max_iter=300
        )
        self.clustering_model.fit(X_latent)
        
        # Set self.model for parent class compatibility
        self.model = self.clustering_model
        
        # Store for prediction
        self.train_cluster_labels_ = self.clustering_model.labels_
        
        self.train_time = time.time() - start
        print(f"      Total training time: {self.train_time:.2f}s")
        
        return self
    
    def fit_and_map(self, X_train, y_train_true):
        """FIX: Override to handle latent space transformation."""
        # Fit model
        self.fit(X_train)
        
        # Get cluster predictions on training set (in latent space)
        y_train_clusters = self.predict_clusters(X_train)
        
        # Learn optimal mapping
        self.cluster_to_class_map_ = self._find_optimal_mapping(
            y_train_true, y_train_clusters
        )
        
        return self
    
    def predict_clusters(self, X):
        """Predict clusters for new data."""
        start = time.time()
        
        # Step 1: Encode with autoencoder
        self.autoencoder.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            X_latent = self.autoencoder.encode(X_tensor).cpu().numpy()
        
        # Step 2: Predict clusters
        clusters = self.clustering_model.predict(X_latent)
        
        self.inference_time = (time.time() - start) / len(X)
        return clusters
    
    def get_memory_mb(self):
        """Estimate memory usage."""
        total_bytes = 0
        
        # Autoencoder parameters
        if self.autoencoder is not None:
            for param in self.autoencoder.parameters():
                total_bytes += param.data.nelement() * param.data.element_size()
        
        # Clustering model
        if self.clustering_model is not None:
            total_bytes += sys.getsizeof(self.clustering_model)
            if hasattr(self.clustering_model, 'cluster_centers_'):
                total_bytes += self.clustering_model.cluster_centers_.nbytes
        
        # Labels
        if self.train_cluster_labels_ is not None:
            total_bytes += self.train_cluster_labels_.nbytes
        
        # Mapping
        if self.cluster_to_class_map_ is not None:
            total_bytes += sys.getsizeof(self.cluster_to_class_map_)
            for k, v in self.cluster_to_class_map_.items():
                total_bytes += sys.getsizeof(k) + sys.getsizeof(v)
        
        return total_bytes / (1024 * 1024)


class UMAPDetector(MultiClassDetector):
    """UMAP-based detector that clusters in UMAP projection space."""
    
    def __init__(self, name: str, config: Dict):
        super().__init__(name, None, config)
        
        self.umap_model = None
        self.clustering_model = None
        
        # Extract config
        self.umap_config = config.get('umap', {})
        self.n_clusters = 5
        
    def fit(self, X):
        """Apply UMAP projection, then cluster."""
        start = time.time()
        
        # Step 1: Apply UMAP
        print(f"      Applying UMAP projection...")
        self.umap_model = umap.UMAP(
            n_components=self.umap_config.get('n_components', 10),  # Higher dim for clustering
            n_neighbors=self.umap_config.get('n_neighbors', 15),
            min_dist=self.umap_config.get('min_dist', 0.1),
            metric=self.umap_config.get('metric', 'euclidean'),
            random_state=self.umap_config.get('random_state', 42)
        )
        X_projected = self.umap_model.fit_transform(X)
        print(f"        Projected to {X_projected.shape[1]} dimensions")
        
        # Step 2: Cluster in UMAP space
        print(f"      Clustering in UMAP space...")
        from sklearn.cluster import KMeans
        self.clustering_model = KMeans(
            n_clusters=self.n_clusters,
            random_state=42,
            n_init=10,
            max_iter=300
        )
        self.clustering_model.fit(X_projected)
        
        # Set self.model for parent class compatibility
        self.model = self.clustering_model
        
        # Store for prediction
        self.train_cluster_labels_ = self.clustering_model.labels_
        
        self.train_time = time.time() - start
        print(f"      Total training time: {self.train_time:.2f}s")
        
        return self
    
    def fit_and_map(self, X_train, y_train_true):
        """FIX: Override to handle UMAP transformation."""
        # Fit model
        self.fit(X_train)
        
        # Get cluster predictions on training set (in UMAP space)
        y_train_clusters = self.predict_clusters(X_train)
        
        # Learn optimal mapping
        self.cluster_to_class_map_ = self._find_optimal_mapping(
            y_train_true, y_train_clusters
        )
        
        return self
    
    def predict_clusters(self, X):
        """Predict clusters for new data."""
        start = time.time()
        
        # Step 1: Project with UMAP
        X_projected = self.umap_model.transform(X)
        
        # Step 2: Predict clusters
        clusters = self.clustering_model.predict(X_projected)
        
        self.inference_time = (time.time() - start) / len(X)
        return clusters
    
    def get_memory_mb(self):
        """Estimate memory usage."""
        total_bytes = 0
        
        # UMAP model
        if self.umap_model is not None:
            total_bytes += sys.getsizeof(self.umap_model)
            # UMAP stores embedding
            if hasattr(self.umap_model, 'embedding_'):
                total_bytes += self.umap_model.embedding_.nbytes
        
        # Clustering model
        if self.clustering_model is not None:
            total_bytes += sys.getsizeof(self.clustering_model)
            if hasattr(self.clustering_model, 'cluster_centers_'):
                total_bytes += self.clustering_model.cluster_centers_.nbytes
        
        # Labels
        if self.train_cluster_labels_ is not None:
            total_bytes += self.train_cluster_labels_.nbytes
        
        # Mapping
        if self.cluster_to_class_map_ is not None:
            total_bytes += sys.getsizeof(self.cluster_to_class_map_)
            for k, v in self.cluster_to_class_map_.items():
                total_bytes += sys.getsizeof(k) + sys.getsizeof(v)
        
        return total_bytes / (1024 * 1024)


class Task4MultiClassDetection:
    """Task 4: Multi-Class Intrusion Detection."""
    
    def __init__(self, config: Dict[str, Any], logger=None, mode: str = 'core'):
        """
        Args:
            mode: 'core', 'full', 'new', or 'core_new'
        """
        self.config = config
        self.logger = logger
        self.mode = mode
        
        # Create output directories
        self.fig_dir = Path(f'outputs/figures/task4_{mode}')
        self.table_dir = Path(f'outputs/tables/task4_{mode}')
        self.model_dir = Path(f'outputs/models/task4_{mode}')
        self.fig_dir.mkdir(parents=True, exist_ok=True)
        self.table_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Class labels
        self.class_names = ['normal', 'injection', 'masquerade', 'poisoning', 'replay']
        #{'normal': 0, 'injection': 1, 'masquerade': 2, 'poisoning': 3, 'replay': 4}        
        self.label_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
         
        # Reverse mapping for display
        self.idx_to_label = {idx: name for name, idx in self.label_to_idx.items()}
        
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.dpi'] = 300
        
        print(f"\n{'='*70}")
        print(f"RUNNING TASK 4 IN '{mode.upper()}' MODE (MULTI-CLASS DETECTION)")
        if mode == 'core':
            print("Using ONLY the 10 CORE_FIELDS")
        elif mode == 'full':
            print("Using CORE + ALLOWED + ENGINEERED features")
        elif mode == 'new':
            print("Using ONLY NEW engineered features")
        elif mode == 'core_new':
            print("Using CORE fields + NEW engineered features")
        else:
            raise ValueError(f"Unknown mode: {mode}. Must be 'core', 'full', 'new', or 'core_new'")
        print(f"Classes: {', '.join(self.class_names)}")
        print(f"{'='*70}\n")
    
    def load_data(self):
        """Load combined dataset with all attack types."""
        print("\nLoading combined dataset...")
        
    
        print(f"\n{'='*70}")
        print(f"LOADING DATA FOR TASK 4 (MULTI-CLASS DETECTION)")
        print(f"{'='*70}")
        config = self.config
        train_df, test_df = load_combined_datasets_multiclass(
            Path(config['data']['raw_dir']),
            config['data']['train_files'],
            config['data']['test_files'],
            mode=self.mode
        )
        
        return train_df, test_df


    def filter_features_by_importance(self, features: List[str], mode: str) -> List[str]:
        """
        Filter features based on importance threshold from Task 1 results.
        Uses combined dataset feature importance for multi-class detection.
        
        Args:
            features: List of all available features
            mode: Current mode ('core', 'full', 'new', 'core_new')
        
        Returns:
            Filtered list of features to keep
        """
        # Path to feature importance file from Task 1 (use combined dataset)
        importance_file = Path(f'outputs/tables/task2_{mode}/feature_importance_scores.csv')
        
        if not importance_file.exists():
            print(f"    WARNING: Feature importance file not found: {importance_file}")
            print(f"    Using all {len(features)} features (no filtering)")
            return features
        
        # Load feature importance
        try:
            importance_df = pd.read_csv(importance_file)
            
            # Get threshold from config
            threshold = self.config.get('feature_engineering', {}).get('importance_threshold', 0.0)
            print(f"    Applying feature importance threshold: {threshold}")
            
            # Filter features above threshold
            important_features = importance_df[importance_df['PCA_Importance'] >= threshold]['Feature'].tolist()
            
            # Keep only features that are in our current feature list
            filtered_features = [f for f in features if f in important_features]
            
            if len(filtered_features) == 0:
                print(f"    WARNING: No features passed threshold {threshold}!")
                print(f"    Falling back to top 10 features")
                top_features = importance_df.nlargest(10, 'Importance')['Feature'].tolist()
                filtered_features = [f for f in features if f in top_features]
            
            print(f"    Features: {len(features)} -> {len(filtered_features)} (kept {len(filtered_features)/len(features)*100:.1f}%)")
            print(f"    Kept features: {filtered_features}")
            
            return filtered_features
            
        except Exception as e:
            print(f"    ERROR loading feature importance: {e}")
            print(f"    Using all {len(features)} features (no filtering)")
            return features
    
    def preprocess_data(self, train_df, test_df):
        """Preprocess data for multi-class detection."""
        print("\nPreprocessing data...")
        
        # Get numeric features (excluding 'attack' and 'attack_type')
        numeric_cols = get_numeric_features(train_df, mode=self.mode)
        
        print(f"  Using {len(numeric_cols)} numeric features from {self.mode.upper()} mode")
        
        # Extract features
        X_train = train_df[numeric_cols].copy()
        X_test = test_df[numeric_cols].copy()
        
        # Extract multi-class labels from attack_type column
        # Map based on class_names order: normal->0, injection->1, masquerade->2, poisoning->3, replay->4
        y_train_str = train_df['attack_type'].str.lower().values
        y_test_str = test_df['attack_type'].str.lower().values

        # Encode labels using our explicit mapping
        y_train = np.array([self.label_to_idx[label] for label in y_train_str])
        y_test = np.array([self.label_to_idx[label] for label in y_test_str])
        
        # Print class distribution
        print("\n  Training set distribution:")
        for i, class_name in enumerate(self.class_names):
            count = np.sum(y_train == i)
            print(f"    {class_name}: {count} ({count/len(y_train)*100:.1f}%)")
        
        print("\n  Test set distribution:")
        for i, class_name in enumerate(self.class_names):
            count = np.sum(y_test == i)
            print(f"    {class_name}: {count} ({count/len(y_test)*100:.1f}%)")
        
        # Clean data
        print("\n  Cleaning data...")
        X_train = X_train.replace([np.inf, -np.inf], np.nan)
        X_test = X_test.replace([np.inf, -np.inf], np.nan)
        
        # Drop columns with too many NaN values (>50%)
        threshold = len(X_train) * 0.5
        cols_passing_threshold = X_train.columns[X_train.notna().sum() > threshold]
        cols_to_drop = set(X_train.columns) - set(cols_passing_threshold)
        
        if 'boolean_1' in cols_to_drop:
            cols_to_drop.remove('boolean_1')
        
        final_valid_cols = [col for col in X_train.columns if col not in cols_to_drop]
        X_train = X_train[final_valid_cols]
        X_test = X_test[final_valid_cols]
        
        print(f"  Features after dropping high-NaN columns: {len(X_train.columns)}")
        
        # Fill remaining NaN with median
        print("  Filling NaN values with column median...")
        for col in X_train.columns:
            median_val = X_train[col].median()
            if pd.isna(median_val):
                median_val = 0
            X_train[col].fillna(median_val, inplace=True)
            X_test[col].fillna(median_val, inplace=True)
        
        # Drop zero variance columns
        variances = X_train.var()
        valid_cols_variance = variances[variances > 0].index
        final_valid_cols = [col for col in X_train.columns if col in valid_cols_variance]
        X_train = X_train[final_valid_cols]
        X_test = X_test[final_valid_cols]
        
        print(f"  Features after cleaning: {len(X_train.columns)}")
        
        # ===== NEW: Filter features by importance =====
        print("  Filtering features by importance...")
        feature_names = list(X_train.columns)
        filtered_features = self.filter_features_by_importance(feature_names, self.mode)
        
        # Apply filtering
        X_train = X_train[filtered_features]
        X_test = X_test[filtered_features]
        print(f"  Features after importance filtering: {len(X_train.columns)}")
        # ===============================================
        
        # Store feature names
        feature_names = list(X_train.columns)
        
        # Standardize
        print("  Standardizing features...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Handle any remaining NaN
        X_train_scaled = np.nan_to_num(X_train_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        X_test_scaled = np.nan_to_num(X_test_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        
        print(f"\n  Final shapes: Train={X_train_scaled.shape}, Test={X_test_scaled.shape}")
        
        return X_train_scaled, y_train, X_test_scaled, y_test, feature_names, scaler
    
    # Update the create_models method
    def create_models(self):
        """Create all clustering models to compare."""
        
        # Load config
        dim_red_config = self.config.get('dimensionality_reduction', {})
        
        models = {
            'K-Means': MultiClassDetector(
                'K-Means',
                KMeans(
                    n_clusters=5,
                    random_state=42,
                    n_init=20,
                    max_iter=300
                )
            ),
            'GMM': MultiClassDetector(
                'GMM',
                GaussianMixture(
                    n_components=5,
                    random_state=42,
                    covariance_type='full',
                    max_iter=200
                )
            ),
            'Hierarchical': MultiClassDetector(
                'Hierarchical',
                AgglomerativeClustering(
                    n_clusters=5,
                    linkage='ward'
                )
            ),
        }
        
        # Add Autoencoder model if enabled
        if dim_red_config.get('enabled', False):
            if dim_red_config.get('autoencoder', {}).get('enabled', True):
                ae_config = {'autoencoder': dim_red_config.get('autoencoder', {})}
                models['Autoencoder'] = AutoencoderDetector('Autoencoder', ae_config)
            
            # Add UMAP model if enabled
            if dim_red_config.get('umap', {}).get('enabled', True):
                umap_config = {'umap': dim_red_config.get('umap', {})}
                models['UMAP'] = UMAPDetector('UMAP', umap_config)
        
        return models

    
    def evaluate_model(self, model: MultiClassDetector, X_test, y_test):
        """Evaluate multi-class model using macro-averaged metrics."""
        y_pred = model.predict(X_test)
        
        # Calculate macro-averaged metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            y_test, y_pred, average='macro', zero_division=0
        )
        
        # Calculate per-class metrics
        precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
            y_test, y_pred, average=None, zero_division=0, labels=range(5)
        )
        
        # Accuracy
        accuracy = accuracy_score(y_test, y_pred)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred, labels=range(5))
        
        return {
            'accuracy': accuracy,
            'precision_macro': precision,
            'recall_macro': recall,
            'f1_macro': f1,
            'precision_per_class': precision_per_class,
            'recall_per_class': recall_per_class,
            'f1_per_class': f1_per_class,
            'confusion_matrix': cm,
            'train_time': model.train_time,
            'inference_time_ms': model.inference_time * 1000,
            'memory_mb': model.get_memory_mb()
        }
    
    def train_and_evaluate(self, X_train, y_train, X_test, y_test):
        """
        Train and evaluate all models - IMPROVED WITH BETTER ERROR HANDLING.
        """
        print(f"\n{'='*70}")
        print(f"TRAINING AND EVALUATING MODELS")
        print(f"{'='*70}")
        
        models = self.create_models()
        results = []
        all_predictions = {}
        all_mappings = {}
        
        # Track failures
        failed_models = []
        
        for model_name, model in models.items():
            print(f"\n{'-'*60}")
            print(f"Model: {model_name}")
            print(f"{'-'*60}")
            
            try:
                # Train and learn cluster mapping
                print(f"  Training and learning cluster mapping...")
                model.fit_and_map(X_train, y_train)
                print(f"  Training time: {model.train_time:.2f}s")
                
                # Print learned mapping
                print(f"\n  Learned cluster-to-class mapping:")
                mapping_str = []
                for cluster_id in sorted(model.cluster_to_class_map_.keys()):
                    class_idx = model.cluster_to_class_map_[cluster_id]
                    class_name = self.idx_to_label[class_idx]
                    mapping_str.append(f"    Cluster {cluster_id} → {class_name} (idx={class_idx})")
                print("\n".join(mapping_str))
                all_mappings[model_name] = model.cluster_to_class_map_
                
                # Evaluate
                print(f"\n  Evaluating on test set...")
                metrics = self.evaluate_model(model, X_test, y_test)
                all_predictions[model_name] = model.predict(X_test)
                
                # Validate predictions
                pred_unique = np.unique(all_predictions[model_name])
                if len(pred_unique) < 2:
                    print(f"  ⚠️  WARNING: Model predicts only {len(pred_unique)} unique classes!")
                
                # Store results
                result = {
                    'Model': model_name,
                    'Accuracy': metrics['accuracy'],
                    'Precision (Macro)': metrics['precision_macro'],
                    'Recall (Macro)': metrics['recall_macro'],
                    'F1-Score (Macro)': metrics['f1_macro'],
                    'Train Time (s)': metrics['train_time'],
                    'Inference Time (ms)': metrics['inference_time_ms'],
                    'Memory (MB)': metrics['memory_mb']
                }
                
                # Add per-class F1 scores
                for i, class_name in enumerate(self.class_names):
                    result[f'F1_{class_name}'] = metrics['f1_per_class'][i]
                
                results.append(result)
                
                # Print macro metrics
                print(f"\n  Macro-averaged Results:")
                print(f"    Accuracy:  {metrics['accuracy']:.4f}")
                print(f"    Precision: {metrics['precision_macro']:.4f}")
                print(f"    Recall:    {metrics['recall_macro']:.4f}")
                print(f"    F1-Score:  {metrics['f1_macro']:.4f}")
                
                # Print per-class F1
                print(f"\n  Per-class F1-Scores:")
                for i, class_name in enumerate(self.class_names):
                    f1_val = metrics['f1_per_class'][i]
                    # Highlight poor performance
                    warning = " ⚠️" if f1_val < 0.1 else ""
                    print(f"    {class_name:12s}: {f1_val:.4f}{warning}")
                
                # Print memory info
                print(f"\n  Resource Usage:")
                print(f"    Memory: {metrics['memory_mb']:.2f} MB")
                print(f"    Inference: {metrics['inference_time_ms']:.4f} ms/sample")
                
                # Save model
                import joblib
                model_path = self.model_dir / f"{model_name.lower().replace(' ', '_')}.pkl"
                joblib.dump(model.model, model_path)
                print(f"    Saved model: {model_path.name}")
                
                # Save mapping
                mapping_path = self.model_dir / f"{model_name.lower().replace(' ', '_')}_mapping.pkl"
                joblib.dump(model.cluster_to_class_map_, mapping_path)
                print(f"    Saved mapping: {mapping_path.name}")
                
                print(f"\n  ✓ {model_name} completed successfully")
                
            except Exception as e:
                print(f"\n  ✗ ERROR in {model_name}: {str(e)}")
                print(f"\n  Stack trace:")
                # traceback.print_exc()
                failed_models.append(model_name)
                continue
        
        # Summary of failures
        if failed_models:
            print(f"\n{'='*70}")
            print(f"⚠️  WARNING: {len(failed_models)} model(s) failed:")
            for model_name in failed_models:
                print(f"    - {model_name}")
            print(f"{'='*70}")
        
        if len(results) == 0:
            raise RuntimeError("All models failed! Check error messages above.")
        
        return pd.DataFrame(results), all_predictions, all_mappings
    
    def visualize_results(self, results_df: pd.DataFrame, X_test, y_test, 
                     all_predictions, all_mappings):
        """
        Create comprehensive visualizations - IMPROVED VERSION.
        
        Now with clear separation of concerns and no duplicates:
        - Confusion matrices = prediction performance
        - Cluster mappings = interpretability/explainability
        """
        print("\n  Generating visualizations...")
        
        # 1. Performance Overview: Macro metrics comparison
        print("    → Macro metrics comparison...")
        self.plot_macro_metrics(results_df)
        
        # 2. Performance Detail: Per-class F1 scores heatmap
        print("    → Per-class F1 heatmap...")
        self.plot_per_class_heatmap(results_df)
        
        # 3. Performance Analysis: Confusion matrices (shows prediction accuracy)
        print("    → Confusion matrices (prediction performance)...")
        self.plot_confusion_matrices(y_test, all_predictions, all_mappings)
        
        # 4. Interpretability: Cluster-to-class mappings (shows how clusters map to classes)
        print("    → Cluster mappings (interpretability)...")
        self.plot_cluster_mappings(all_mappings)
        
        # 5. Dimensionality Reduction: t-SNE projections
        print("    → t-SNE projections...")
        self.plot_tsne_projections(X_test, y_test, all_predictions, results_df)
        
        print("    ✓ All visualizations complete")
    
    def plot_macro_metrics(self, results_df: pd.DataFrame):
        """Plot macro-averaged metrics comparison."""
        print("    Generating macro metrics comparison...")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Multi-Class Detection - Model Performance ({self.mode.upper()} Mode)', 
                     fontsize=14, fontweight='bold')
        
        metrics = ['Accuracy', 'Precision (Macro)', 'Recall (Macro)', 'F1-Score (Macro)']
        for idx, metric in enumerate(metrics):
            ax = axes[idx // 2, idx % 2]
            data = results_df.sort_values(metric, ascending=False)
            
            bars = ax.barh(data['Model'], data[metric], color='steelblue', alpha=0.7)
            ax.set_xlabel(metric)
            ax.set_title(f'{metric} by Model', fontweight='bold')
            ax.set_xlim([0, 1])
            ax.grid(True, alpha=0.3, axis='x')
            
            for i, (bar, val) in enumerate(zip(bars, data[metric])):
                ax.text(val, i, f' {val:.3f}', va='center')
        
        plt.tight_layout()
        fig_path = self.fig_dir / 'macro_metrics_comparison.png'
        plt.savefig(fig_path, bbox_inches='tight')
        print(f"      Saved: {fig_path}")
        
        if self.logger:
            self.logger.log_figure(fig, f"task4_{self.mode}/macro_metrics")
        plt.close(fig)
    
    def plot_per_class_heatmap(self, results_df: pd.DataFrame):
        """Plot per-class F1 scores as heatmap."""
        print("    Generating per-class F1 heatmap...")
        
        # Extract per-class F1 scores
        per_class_cols = [f'F1_{name}' for name in self.class_names]
        heatmap_data = results_df[['Model'] + per_class_cols].set_index('Model')
        heatmap_data.columns = self.class_names
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='YlOrRd',
                   cbar_kws={'label': 'F1-Score'}, ax=ax, vmin=0, vmax=1)
        ax.set_title(f'Per-Class F1-Scores ({self.mode.upper()} Mode)', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Attack Type', fontweight='bold')
        ax.set_ylabel('Model', fontweight='bold')
        
        plt.tight_layout()
        fig_path = self.fig_dir / 'per_class_f1_heatmap.png'
        plt.savefig(fig_path, bbox_inches='tight')
        print(f"      Saved: {fig_path}")
        
        if self.logger:
            self.logger.log_figure(fig, f"task4_{self.mode}/per_class_f1")
        plt.close(fig)
    
    def plot_confusion_matrices(self, y_test, all_predictions, all_mappings):
        """
        Plot ACTUAL confusion matrices showing prediction accuracy.
        
        This replaces the buggy version that was plotting cluster mappings.
        """
        print("    Generating confusion matrices...")
        
        n_models = len(all_predictions)
        n_rows = (n_models + 1) // 2
        n_cols = min(2, n_models)
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 7 * n_rows))
        fig.suptitle(f'Confusion Matrices - Multi-Class Detection ({self.mode.upper()} Mode)',
                    fontsize=14, fontweight='bold')
        
        if n_models == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        for idx, (model_name, y_pred) in enumerate(all_predictions.items()):
            if idx >= len(axes):
                break
            
            ax = axes[idx]
            
            # Compute actual confusion matrix
            cm = confusion_matrix(y_test, y_pred, labels=range(5))
            
            # Normalize for percentage display
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            cm_normalized = np.nan_to_num(cm_normalized)  # Handle division by zero
            
            # Create heatmap with both percentage and counts
            sns.heatmap(cm_normalized, annot=False, fmt='.2f', cmap='Blues',
                    xticklabels=self.class_names,
                    yticklabels=self.class_names,
                    cbar_kws={'label': 'Proportion'}, ax=ax, vmin=0, vmax=1)
            
            # Add text annotations with both percentage and count
            for i in range(5):
                for j in range(5):
                    count = cm[i, j]
                    percentage = cm_normalized[i, j] * 100
                    text = f'{count}\n({percentage:.1f}%)'
                    color = 'white' if cm_normalized[i, j] > 0.5 else 'black'
                    ax.text(j + 0.5, i + 0.5, text, 
                        ha='center', va='center', color=color, fontsize=9)
            
            # Calculate accuracy for this matrix
            accuracy = np.trace(cm) / np.sum(cm)
            
            ax.set_title(f'{model_name}\nAccuracy: {accuracy:.3f}', fontweight='bold')
            ax.set_xlabel('Predicted Class', fontweight='bold')
            ax.set_ylabel('True Class', fontweight='bold')
        
        # Hide unused subplots
        for idx in range(n_models, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        fig_path = self.fig_dir / 'confusion_matrices.png'
        plt.savefig(fig_path, bbox_inches='tight', dpi=300)
        print(f"      Saved: {fig_path}")
        
        if self.logger:
            self.logger.log_figure(fig, f"task4_{self.mode}/confusion_matrices")
        plt.close(fig)
    
    def plot_cluster_mappings(self, all_mappings):
        """
        Visualize cluster-to-class mappings for interpretability.
        
        This is the RENAMED version of what was incorrectly called 
        plot_confusion_matrices() in the original code.
        """
        print("    Generating cluster mapping visualization...")
        
        n_models = len(all_mappings)
        n_rows = (n_models + 1) // 2
        n_cols = min(2, n_models)
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 7 * n_rows))
        fig.suptitle(f'Cluster-to-Class Mappings ({self.mode.upper()} Mode)',
                    fontsize=14, fontweight='bold')
        
        if n_models == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        for idx, (model_name, mapping) in enumerate(all_mappings.items()):
            if idx >= len(axes):
                break
            
            ax = axes[idx]
            
            # Create mapping matrix (5x5)
            # Rows = Cluster IDs, Columns = Class Labels
            mapping_matrix = np.zeros((5, 5))
            for cluster_id, class_id in mapping.items():
                if cluster_id < 5 and class_id < 5:
                    mapping_matrix[cluster_id, class_id] = 1
            
            # Plot heatmap
            sns.heatmap(mapping_matrix, annot=True, fmt='.0f', cmap='RdYlGn',
                    xticklabels=self.class_names,
                    yticklabels=[f'Cluster {i}' for i in range(5)],
                    cbar=False, ax=ax, vmin=0, vmax=1, linewidths=1, linecolor='gray')
            
            ax.set_title(f'{model_name}', fontweight='bold')
            ax.set_xlabel('Mapped Class', fontweight='bold')
            ax.set_ylabel('Cluster ID', fontweight='bold')
        
        # Hide unused subplots
        for idx in range(n_models, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        fig_path = self.fig_dir / 'cluster_mappings.png'
        plt.savefig(fig_path, bbox_inches='tight', dpi=300)
        print(f"      Saved: {fig_path}")
        
        if self.logger:
            self.logger.log_figure(fig, f"task4_{self.mode}/cluster_mappings")
        plt.close(fig)

    def generate_summary(self, results_df: pd.DataFrame):
        """Generate comprehensive summary - FIXED for lowercase class names."""
        print(f"\n{'='*70}")
        print(f"MULTI-CLASS DETECTION SUMMARY ({self.mode.upper()} MODE)")
        print(f"{'='*70}")
        
        # Sort by F1-Score (Macro)
        results_df = results_df.sort_values('F1-Score (Macro)', ascending=False)
        
        # Display main results
        print("\nModel Comparison (Macro-Averaged Metrics):")
        display_cols = ['Model', 'Accuracy', 'Precision (Macro)', 
                    'Recall (Macro)', 'F1-Score (Macro)']
        print(results_df[display_cols].to_string(index=False))
        
        # Display per-class F1 scores
        print("\nPer-Class F1-Scores:")
        per_class_cols = ['Model'] + [f'F1_{name}' for name in self.class_names]
        print(results_df[per_class_cols].to_string(index=False))
        
        # Display computational metrics
        print("\nComputational Metrics:")
        comp_cols = ['Model', 'Train Time (s)', 'Inference Time (ms)', 'Memory (MB)']
        print(results_df[comp_cols].to_string(index=False))
        
        # Save to CSV
        results_df.to_csv(self.table_dir / 'model_comparison.csv', index=False)
        print(f"\nSaved: {self.table_dir / 'model_comparison.csv'}")
        
        # Best model analysis
        best_model = results_df.iloc[0]
        print(f"\n{'='*70}")
        print(f"BEST MODEL: {best_model['Model']}")
        print(f"{'='*70}")
        print(f"Macro F1-Score: {best_model['F1-Score (Macro)']:.4f}")
        print(f"Accuracy:       {best_model['Accuracy']:.4f}")
        print(f"Precision:      {best_model['Precision (Macro)']:.4f}")
        print(f"Recall:         {best_model['Recall (Macro)']:.4f}")
        print(f"\nPer-Class F1-Scores:")
        for class_name in self.class_names:
            # FIXED: Use lowercase class_name directly (no capitalization)
            print(f"  {class_name:12s}: {best_model[f'F1_{class_name}']:.4f}")
        
        # Identify which classes are hardest to detect
        print(f"\n{'='*70}")
        print("ATTACK TYPE DETECTABILITY ANALYSIS")
        print(f"{'='*70}")
        
        for class_name in self.class_names:
            # FIXED: Use lowercase class_name directly
            col = f'F1_{class_name}'
            avg_f1 = results_df[col].mean()
            std_f1 = results_df[col].std()
            best_f1 = results_df[col].max()
            worst_f1 = results_df[col].min()
            best_model_for_class = results_df.loc[results_df[col].idxmax(), 'Model']
            
            print(f"\n{class_name}:")
            print(f"  Average F1:    {avg_f1:.4f} ± {std_f1:.4f}")
            print(f"  Best F1:       {best_f1:.4f} ({best_model_for_class})")
            print(f"  Worst F1:      {worst_f1:.4f}")
        
        # Model comparison insights
        print(f"\n{'='*70}")
        print("KEY INSIGHTS")
        print(f"{'='*70}")
        
        # Find most consistent model (lowest std across classes)
        consistency_scores = {}
        for _, row in results_df.iterrows():
            # FIXED: Use lowercase class names
            per_class_f1 = [row[f'F1_{name}'] for name in self.class_names]
            consistency_scores[row['Model']] = np.std(per_class_f1)
        
        most_consistent = min(consistency_scores, key=consistency_scores.get)
        print(f"\nMost Consistent Model: {most_consistent}")
        print(f"  (Lowest std dev across classes: {consistency_scores[most_consistent]:.4f})")
        
        # Find fastest model
        fastest = results_df.loc[results_df['Train Time (s)'].idxmin()]
        print(f"\nFastest Training: {fastest['Model']}")
        print(f"  Train Time: {fastest['Train Time (s)']:.2f}s")
        
        # Find most efficient inference
        most_efficient = results_df.loc[results_df['Inference Time (ms)'].idxmin()]
        print(f"\nMost Efficient Inference: {most_efficient['Model']}")
        print(f"  Inference Time: {most_efficient['Inference Time (ms)']:.4f}ms per sample")
        
        return results_df
    

    def plot_tsne_projections(self, X_test, y_test, all_predictions, results_df):
        """Plot t-SNE projections for each model."""
        print("    Generating t-SNE projections...")
        
        # Compute t-SNE (only once for efficiency)
        print("      Computing t-SNE embedding...")
        # Sample if dataset is too large (t-SNE is slow)
        max_samples = 5000
        if len(X_test) > max_samples:
            indices = np.random.choice(len(X_test), max_samples, replace=False)
            X_test_sample = X_test[indices]
            y_test_sample = y_test[indices]
            predictions_sample = {k: v[indices] for k, v in all_predictions.items()}
        else:
            X_test_sample = X_test
            y_test_sample = y_test
            predictions_sample = all_predictions
        
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        X_test_2d = tsne.fit_transform(X_test_sample)
        
        # Color map for classes
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        
        for model_name, y_pred in predictions_sample.items():
            # Get model metrics
            model_row = results_df[results_df['Model'] == model_name].iloc[0]
            accuracy = model_row['Accuracy']
            f1_macro = model_row['F1-Score (Macro)']
            
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            fig.suptitle(f'{model_name} - t-SNE Projection\n'
                        f'Accuracy: {accuracy:.3f} | F1-Macro: {f1_macro:.3f}',
                        fontsize=14, fontweight='bold')
            
            # Ground truth
            ax = axes[0]
            for i, class_name in enumerate(self.class_names):
                mask = (y_test_sample == i)
                ax.scatter(X_test_2d[mask, 0], X_test_2d[mask, 1],
                          c=colors[i], label=class_name, alpha=0.5, s=20, edgecolors='none')
            ax.set_xlabel('t-SNE Dimension 1')
            ax.set_ylabel('t-SNE Dimension 2')
            ax.set_title('Ground Truth Labels', fontweight='bold')
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)
            
            # Predictions
            ax = axes[1]
            for i, class_name in enumerate(self.class_names):
                mask = (y_pred == i)
                ax.scatter(X_test_2d[mask, 0], X_test_2d[mask, 1],
                          c=colors[i], label=f'Predicted {class_name}', 
                          alpha=0.5, s=20, edgecolors='none')
            ax.set_xlabel('t-SNE Dimension 1')
            ax.set_ylabel('t-SNE Dimension 2')
            ax.set_title('Model Predictions', fontweight='bold')
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            safe_name = model_name.lower().replace(' ', '_')
            fig_path = self.fig_dir / f'tsne_{safe_name}.png'
            plt.savefig(fig_path, bbox_inches='tight')
            
            if self.logger:
                self.logger.log_figure(fig, f"task4_{self.mode}/tsne_{safe_name}")
            plt.close(fig)
        
        print(f"      ✓ Generated {len(all_predictions)} t-SNE plots")
    
    def plot_cluster_mappings(self, all_mappings):
        """Visualize cluster-to-class mappings for interpretability."""
        print("    Generating cluster mapping visualization...")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Cluster-to-Class Mappings ({self.mode.upper()} Mode)',
                    fontsize=14, fontweight='bold')
        
        axes = axes.flatten()
        
        for idx, (model_name, mapping) in enumerate(all_mappings.items()):
            if idx >= 4:
                break
            
            ax = axes[idx]
            
            # Create mapping matrix (5x5)
            # Rows = Cluster IDs, Columns = Class Labels
            mapping_matrix = np.zeros((5, 5))
            for cluster_id, class_id in mapping.items():
                if cluster_id < 5 and class_id < 5:
                    mapping_matrix[cluster_id, class_id] = 1
            
            # Plot heatmap
            sns.heatmap(mapping_matrix, annot=True, fmt='.0f', cmap='RdYlGn',
                    xticklabels=self.class_names,
                    yticklabels=[f'Cluster {i}' for i in range(5)],
                    cbar=False, ax=ax, vmin=0, vmax=1, linewidths=1, linecolor='gray')
            
            ax.set_title(f'{model_name}', fontweight='bold')
            ax.set_xlabel('Mapped Class', fontweight='bold')
            ax.set_ylabel('Cluster ID', fontweight='bold')
        
        plt.tight_layout()
        fig_path = self.fig_dir / 'cluster_mappings.png'
        plt.savefig(fig_path, bbox_inches='tight')
        print(f"      Saved: {fig_path}")
        
        if self.logger:
            self.logger.log_figure(fig, f"task4_{self.mode}/cluster_mappings")
        plt.close(fig)


def run_task4(config: Dict[str, Any], logger=None) -> Dict:
    """
    Execute Task 4: Multi-Class Intrusion Detection.
    
    FIXED: Final summary now uses lowercase class names.
    """
    results = {}
    
    # Run both modes: CORE (10 features) and FULL (all features)
    # Run all four modes
    for mode in config['mode']:
        print(f"\n{'#'*70}")
        print(f"# STARTING {mode.upper()} MODE ANALYSIS")
        print(f"{'#'*70}\n")
        
        # Initialize Task 4 detector
        task4 = Task4MultiClassDetection(config, logger, mode=mode)
        
        # Step 1: Load combined dataset (all attack types)
        print("\n" + "="*70)
        print("STEP 1: LOADING DATA")
        print("="*70)
        train_df, test_df = task4.load_data()
        
        # Step 2: Preprocess data
        print("\n" + "="*70)
        print("STEP 2: PREPROCESSING")
        print("="*70)
        X_train, y_train, X_test, y_test, features, scaler = task4.preprocess_data(
            train_df, test_df
        )
        
        # Step 3: Train and evaluate all models
        print("\n" + "="*70)
        print("STEP 3: TRAINING AND EVALUATION")
        print("="*70)
        results_df, all_predictions, all_mappings = task4.train_and_evaluate(
            X_train, y_train, X_test, y_test
        )
        
        # Step 4: Generate visualizations
        print("\n" + "="*70)
        print("STEP 4: GENERATING VISUALIZATIONS")
        print("="*70)
        task4.visualize_results(
            results_df, X_test, y_test, all_predictions, all_mappings
        )
        
        # Step 5: Generate comprehensive summary
        print("\n" + "="*70)
        print("STEP 5: GENERATING SUMMARY")
        print("="*70)
        summary_df = task4.generate_summary(results_df)
        
        # Log to Weights & Biases if available
        if logger:
            logger.log_dataframe(summary_df, f"task4_{mode}/model_comparison")
            
        
        # Store results for this mode
        results[mode] = {
            "summary": summary_df,
            "predictions": all_predictions,
            "mappings": all_mappings,
            "features": features,
            "scaler": scaler
        }
        
        print(f"\n✓ Completed {mode.upper()} mode analysis")
        print(f"  - Models trained: {len(all_predictions)}")
        print(f"  - Best model: {summary_df.iloc[0]['Model']}")
        print(f"  - Best F1-Macro: {summary_df.iloc[0]['F1-Score (Macro)']:.4f}")
    
    # Final summary
    print("\n" + "="*70)
    print("TASK 4 COMPLETE - BOTH MODES FINISHED")
    print("="*70)
    print("\nComparison between modes:")
    for mode in config['mode']:
        best_f1 = results[mode]['summary'].iloc[0]['F1-Score (Macro)']
        best_model = results[mode]['summary'].iloc[0]['Model']
        print(f"  {mode.upper():10s} best F1-Macro: {best_f1:.4f} ({best_model})")

    print("\nPer-class performance (best models):")
    class_names = ['normal', 'injection', 'masquerade', 'poisoning', 'replay']

    for mode in config['mode']:
        best = results[mode]['summary'].iloc[0]
        print(f"\n  {mode.upper()} mode ({best['Model']}):")
        for class_name in class_names:
            f1_score = best[f'F1_{class_name}']
            print(f"    {class_name:12s}: {f1_score:.4f}")

    print("\nResults saved to:")
    for mode in config['mode']:
        print(f"  - outputs/figures/task4_{mode}/ and outputs/tables/task4_{mode}/")
    
    return results


if __name__ == '__main__':
    """
    Main entry point for running Task 4 standalone.
    """
    import yaml
    
    print("="*70)
    print("TASK 4: UNSUPERVISED MULTI-CLASS ATTACK DETECTION")
    print("="*70)
    print("\nObjective:")
    print("  Perform unsupervised multi-class classification to distinguish")
    print("  between Normal traffic and four attack types using clustering.")
    print("\nApproach:")
    print("  1. Train clustering models (K-Means, GMM, Hierarchical)")
    print("  2. Learn cluster-to-class mappings using Hungarian algorithm")
    print("  3. Evaluate using macro-averaged Precision, Recall, F1-score")
    print("  4. Analyze per-class performance and confusion matrices")
    print("="*70)
    
    # Load configuration
    config_path = 'config/config.yaml'
    print(f"\nLoading configuration from: {config_path}")
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print("✓ Configuration loaded successfully")
    except FileNotFoundError:
        print(f"ERROR: Config file not found at {config_path}")
        print("Please ensure config.yaml exists with correct data paths.")
        exit(1)
    
    # Run Task 4
    print("\nStarting Task 4 execution...")
    results = run_task4(config, logger=None)
    
    # Print final comparison
    print("\n" + "="*70)
    print("FINAL RESULTS SUMMARY")
    print("="*70)
    
    class_names = ['normal', 'injection', 'masquerade', 'poisoning', 'replay']
    
    for mode in config['mode']:
        print(f"\n{mode.upper()} Mode:")
        summary = results[mode]['summary']
        best = summary.iloc[0]
        
        print(f"  Best Model: {best['Model']}")
        print(f"  Macro F1-Score: {best['F1-Score (Macro)']:.4f}")
        print(f"  Accuracy: {best['Accuracy']:.4f}")
        print(f"  Train Time: {best['Train Time (s)']:.2f}s")
        print(f"\n  Per-Class F1-Scores:")
        for class_name in class_names:
            print(f"    {class_name:12s}: {best[f'F1_{class_name}']:.4f}")
    
    print("\n" + "="*70)
    print("Task 4 execution completed successfully!")
    print("="*70)