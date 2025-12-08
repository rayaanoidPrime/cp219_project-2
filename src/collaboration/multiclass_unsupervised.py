import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

# Sklearn imports
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    accuracy_score, classification_report, precision_score, recall_score, f1_score,
    adjusted_rand_score, normalized_mutual_info_score
)

# PyTorch imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Import ResourceProfiler for resource tracking
sys.path.insert(0, str(Path(__file__).parent.parent / 'SV_try' / 'utility'))
try:
    from resource_usage import ResourceProfiler
    RESOURCE_PROFILER_AVAILABLE = True
except ImportError:
    RESOURCE_PROFILER_AVAILABLE = False
    print("Warning: ResourceProfiler not available. Resource metrics will be disabled.")

warnings.filterwarnings('ignore')


# =============================================================================
# BASE CLASSES FOR CLUSTERING-BASED DETECTORS
# =============================================================================

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
        
        # No downsampling here - it's handled in fit_and_map
        self.model.fit(X)
        
        # For Hierarchical, store full data
        if isinstance(self.model, AgglomerativeClustering):
            self.X_train_ = X.copy()
            self.train_cluster_labels_ = self.model.labels_.copy()

        self.train_time = time.time() - start
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
        Uses proportional downsampling if dataset is large.
        """
        # --- Proportional downsampling for large datasets ---
        if len(X_train) > 40000:
            target_size = 20000
            print(f"      ⚠️  Large dataset ({len(X_train)}). Proportional downsampling to {target_size} samples.")
            
            # Calculate samples per class proportionally
            unique_classes, class_counts = np.unique(y_train_true, return_counts=True)
            total_samples = len(X_train)
            
            sampled_indices = []
            np.random.seed(42)
            
            for cls, count in zip(unique_classes, class_counts):
                # Calculate proportional sample size for this class
                cls_target = int((count / total_samples) * target_size)
                # Ensure at least some samples from each class (min 10 or available count)
                cls_target = max(min(10, count), cls_target)
                
                # Get indices for this class
                cls_indices = np.where(y_train_true == cls)[0]
                
                # Sample (with or without replacement depending on availability)
                if len(cls_indices) <= cls_target:
                    sampled_indices.extend(cls_indices)
                else:
                    sampled = np.random.choice(cls_indices, cls_target, replace=False)
                    sampled_indices.extend(sampled)
            
            # Shuffle the combined indices
            np.random.shuffle(sampled_indices)
            sampled_indices = np.array(sampled_indices)
            
            # Downsample both X and y
            X_train_sample = X_train[sampled_indices]
            y_train_sample = y_train_true[sampled_indices]
            
            # Print sampling stats
            unique_sampled, sampled_counts = np.unique(y_train_sample, return_counts=True)
        else:
            X_train_sample = X_train
            y_train_sample = y_train_true
        
        # Fit model on sampled data
        self.fit(X_train_sample)
        
        # Get cluster predictions on sampled training set
        if hasattr(self.model, 'labels_') and len(self.model.labels_) == len(X_train_sample):
            y_train_clusters = self.model.labels_
        else:
            y_train_clusters = self.predict_clusters(X_train_sample)
        
        # Learn optimal mapping using SAMPLED data
        self.cluster_to_class_map_ = self._find_optimal_mapping(
            y_train_sample,  # Use sampled labels, not original
            y_train_clusters
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
        """
        total_bytes = 0
        
        # 1. Base model object
        total_bytes += sys.getsizeof(self.model)
        
        # 2. Model-specific memory (numpy arrays in sklearn models)
        try:
            if hasattr(self.model, 'cluster_centers_'):
                total_bytes += self.model.cluster_centers_.nbytes
            
            if hasattr(self.model, 'labels_'):
                total_bytes += self.model.labels_.nbytes
            
            if hasattr(self.model, 'means_'):
                total_bytes += self.model.means_.nbytes
            
            if hasattr(self.model, 'covariances_'):
                total_bytes += self.model.covariances_.nbytes
            
            if hasattr(self.model, 'weights_'):
                total_bytes += self.model.weights_.nbytes
            
            if hasattr(self.model, 'precisions_'):
                total_bytes += self.model.precisions_.nbytes
            
            if hasattr(self.model, 'precisions_cholesky_'):
                total_bytes += self.model.precisions_cholesky_.nbytes
            
        except Exception as e:
            pass
        
        # 3. Cached training data (for Hierarchical Clustering)
        if self.X_train_ is not None:
            total_bytes += self.X_train_.nbytes
        
        if self.train_cluster_labels_ is not None:
            total_bytes += self.train_cluster_labels_.nbytes
        
        # 4. Cluster-to-class mapping
        if self.cluster_to_class_map_ is not None:
            total_bytes += sys.getsizeof(self.cluster_to_class_map_)
            for k, v in self.cluster_to_class_map_.items():
                total_bytes += sys.getsizeof(k) + sys.getsizeof(v)
        
        return total_bytes / (1024 * 1024)
    
    def get_class_resources(self):
        """Return per-class resource metrics (not applicable for clustering)."""
        return None


# =============================================================================
# AUTOENCODER MODEL (PYTORCH)
# =============================================================================

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


# =============================================================================
# ONE-VS-ALL AUTOENCODER DETECTOR
# =============================================================================

class OneVsAllAutoencoderDetector:
    """
    One-vs-All Autoencoder anomaly detection for multi-class classification.
    
    Trains a separate autoencoder for each class. Each autoencoder learns to 
    reconstruct samples from its class. During prediction, the class whose 
    autoencoder produces the lowest reconstruction error is selected.
    """
    
    def __init__(self, name: str, config: Dict):
        self.name = name
        self.config = config
        self.ae_config = config.get('autoencoder', {})
        self.autoencoders = {}  # class_id -> trained autoencoder
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Timing and resource metrics
        self.train_time = 0
        self.inference_time = 0
        self.class_resources = {}  # class_id -> resource metrics dict
        
    def _train_single_autoencoder(self, X_class, class_id, input_dim):
        """Train a single autoencoder on data from one class."""
        
        # Build autoencoder
        ae = AutoencoderModel(
            input_dim=input_dim,
            encoder_layers=self.ae_config.get('architecture', {}).get('encoder_layers', [64, 32]),
            latent_dim=self.ae_config.get('architecture', {}).get('latent_dim', 16),
            decoder_layers=self.ae_config.get('architecture', {}).get('decoder_layers', [32, 64]),
            activation=self.ae_config.get('activation', 'relu'),
            dropout=self.ae_config.get('dropout', 0.2)
        ).to(self.device)
        
        # Training parameters
        train_config = self.ae_config.get('training', {})
        epochs = train_config.get('epochs', 50)
        batch_size = train_config.get('batch_size', 256)
        learning_rate = train_config.get('learning_rate', 0.001)
        patience = train_config.get('early_stopping_patience', 10)
        
        # Prepare data
        X_tensor = torch.FloatTensor(X_class).to(self.device)
        
        # Train/validation split (80/20)
        n_train = int(len(X_class) * 0.8)
        if n_train < 10:
            n_train = len(X_class)  # Too small, use all for training
        
        X_train_tensor = X_tensor[:n_train]
        X_val_tensor = X_tensor[n_train:] if n_train < len(X_class) else X_tensor
        
        train_dataset = TensorDataset(X_train_tensor, X_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Optimizer and loss
        optimizer = optim.Adam(ae.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        
        ae.train()
        for epoch in range(epochs):
            train_loss = 0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                output = ae(batch_X)
                loss = criterion(output, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # Validation
            ae.eval()
            with torch.no_grad():
                val_output = ae(X_val_tensor)
                val_loss = criterion(val_output, X_val_tensor).item()
            ae.train()
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break
        
        ae.eval()
        return ae
    
    def _compute_reconstruction_error(self, ae, X):
        """Compute per-sample reconstruction error using trained autoencoder."""
        ae.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            reconstructed = ae(X_tensor)
            # MSE per sample
            errors = torch.mean((X_tensor - reconstructed) ** 2, dim=1).cpu().numpy()
        return errors
    
    def fit_and_map(self, X_train, y_train_true):
        """Train one autoencoder per class with resource tracking."""
        start_total = time.time()
        unique_classes = np.unique(y_train_true)
        input_dim = X_train.shape[1]
        
        print(f"      Training {len(unique_classes)} One-vs-All Autoencoders...")
        
        for class_id in unique_classes:
            # Extract samples belonging to this class
            class_mask = y_train_true == class_id
            X_class = X_train[class_mask]
            
            print(f"        Class {class_id}: {len(X_class)} samples...", end=' ')
            
            # Train with resource profiling
            if RESOURCE_PROFILER_AVAILABLE:
                with ResourceProfiler() as profiler:
                    ae = self._train_single_autoencoder(X_class, class_id, input_dim)
                
                self.class_resources[int(class_id)] = {
                    'train_time_ns': profiler.wall_nanoseconds,
                    'train_time_s': profiler.wall_seconds,
                    'peak_ram_mb': profiler.peak_ram_mb,
                    'cpu_avg_pct': profiler.cpu_avg_machine_pct,
                    'cpu_peak_pct': profiler.cpu_peak_machine_pct
                }
            else:
                start_class = time.time()
                ae = self._train_single_autoencoder(X_class, class_id, input_dim)
                elapsed = time.time() - start_class
                
                self.class_resources[int(class_id)] = {
                    'train_time_ns': int(elapsed * 1e9),
                    'train_time_s': elapsed,
                    'peak_ram_mb': 0.0,
                    'cpu_avg_pct': 0.0,
                    'cpu_peak_pct': 0.0
                }
            
            self.autoencoders[class_id] = ae
            print(f"✓ ({self.class_resources[int(class_id)]['train_time_s']:.2f}s)")
        
        self.train_time = time.time() - start_total
        print(f"      Total training time: {self.train_time:.2f}s")
        
        return self
    
    def predict(self, X):
        """Predict class by finding which autoencoder gives lowest reconstruction error."""
        start = time.time()
        
        # Compute reconstruction errors for all classes
        all_errors = {}
        for class_id, ae in self.autoencoders.items():
            all_errors[class_id] = self._compute_reconstruction_error(ae, X)
        
        # For each sample, pick class with lowest reconstruction error
        predictions = np.zeros(len(X), dtype=int)
        for i in range(len(X)):
            sample_errors = {cid: all_errors[cid][i] for cid in all_errors}
            predictions[i] = min(sample_errors, key=sample_errors.get)
        
        self.inference_time = (time.time() - start) / len(X)
        return predictions
    
    def get_memory_mb(self):
        """Estimate memory usage of all autoencoders."""
        total_bytes = 0
        for class_id, ae in self.autoencoders.items():
            for param in ae.parameters():
                total_bytes += param.data.nelement() * param.data.element_size()
        return total_bytes / (1024 * 1024)
    
    def get_class_resources(self):
        """Return per-class resource metrics."""
        return self.class_resources


# =============================================================================
# ONE-VS-ALL ISOLATION FOREST DETECTOR
# =============================================================================

class OneVsAllIsolationForestDetector:
    """
    One-vs-All Isolation Forest for multi-class classification.
    
    Trains a separate Isolation Forest for each class. Each forest learns the 
    "normal" distribution of its class. During prediction, the class whose 
    forest gives the highest score (least anomalous) is selected.
    """
    
    def __init__(self, name: str, config: Dict):
        self.name = name
        self.config = config
        self.forests = {}  # class_id -> trained IsolationForest
        
        # Timing and resource metrics
        self.train_time = 0
        self.inference_time = 0
        self.class_resources = {}  # class_id -> resource metrics dict
        
    def fit_and_map(self, X_train, y_train_true):
        """Train one Isolation Forest per class with resource tracking."""
        start_total = time.time()
        unique_classes = np.unique(y_train_true)
        
        print(f"      Training {len(unique_classes)} One-vs-All Isolation Forests...")
        
        for class_id in unique_classes:
            # Extract samples belonging to this class
            class_mask = y_train_true == class_id
            X_class = X_train[class_mask]
            
            print(f"        Class {class_id}: {len(X_class)} samples...", end=' ')
            
            # Train with resource profiling
            if RESOURCE_PROFILER_AVAILABLE:
                with ResourceProfiler() as profiler:
                    forest = IsolationForest(
                        n_estimators=self.config.get('n_estimators', 100),
                        contamination='auto',
                        random_state=42,
                        n_jobs=-1
                    )
                    forest.fit(X_class)
                
                self.class_resources[int(class_id)] = {
                    'train_time_ns': profiler.wall_nanoseconds,
                    'train_time_s': profiler.wall_seconds,
                    'peak_ram_mb': profiler.peak_ram_mb,
                    'cpu_avg_pct': profiler.cpu_avg_machine_pct,
                    'cpu_peak_pct': profiler.cpu_peak_machine_pct
                }
            else:
                start_class = time.time()
                forest = IsolationForest(
                    n_estimators=self.config.get('n_estimators', 100),
                    contamination='auto',
                    random_state=42,
                    n_jobs=-1
                )
                forest.fit(X_class)
                elapsed = time.time() - start_class
                
                self.class_resources[int(class_id)] = {
                    'train_time_ns': int(elapsed * 1e9),
                    'train_time_s': elapsed,
                    'peak_ram_mb': 0.0,
                    'cpu_avg_pct': 0.0,
                    'cpu_peak_pct': 0.0
                }
            
            self.forests[class_id] = forest
            print(f"✓ ({self.class_resources[int(class_id)]['train_time_s']:.2f}s)")
        
        self.train_time = time.time() - start_total
        print(f"      Total training time: {self.train_time:.2f}s")
        
        return self
    
    def predict(self, X):
        """Predict class by finding which forest gives highest score (least anomalous)."""
        start = time.time()
        
        # Compute anomaly scores for all classes
        # score_samples returns: higher = more normal (less anomalous)
        all_scores = {}
        for class_id, forest in self.forests.items():
            all_scores[class_id] = forest.score_samples(X)
        
        # For each sample, pick class with highest score (most normal)
        predictions = np.zeros(len(X), dtype=int)
        for i in range(len(X)):
            sample_scores = {cid: all_scores[cid][i] for cid in all_scores}
            predictions[i] = max(sample_scores, key=sample_scores.get)
        
        self.inference_time = (time.time() - start) / len(X)
        return predictions
    
    def get_memory_mb(self):
        """Estimate memory usage of all forests."""
        total_bytes = 0
        for class_id, forest in self.forests.items():
            total_bytes += sys.getsizeof(forest)
            # Estimate tree memory (rough approximation)
            if hasattr(forest, 'estimators_'):
                for tree in forest.estimators_:
                    total_bytes += sys.getsizeof(tree)
        return total_bytes / (1024 * 1024)
    
    def get_class_resources(self):
        """Return per-class resource metrics."""
        return self.class_resources


# =============================================================================
# DATA LOADING
# =============================================================================

class DatasetLoader:
    """Handles loading and aggregating datasets from directory structure."""
    
    def __init__(self, root_dir: str):
        self.root_dir = Path(root_dir)
        self.level1_structure = {}  # protocol -> devices -> scenarios
        
    def traverse_directory(self):
        """Traverse 3-level directory structure and build hierarchy."""
        print("\n" + "="*80)
        print("PHASE 1: TRAVERSING DIRECTORY STRUCTURE")
        print("="*80)
        
        # Level 1: Protocols
        for protocol_dir in sorted(self.root_dir.iterdir()):
            if not protocol_dir.is_dir():
                continue
                
            protocol_name = protocol_dir.name
            print(f"\n📁 Dataset: {protocol_name}")
            self.level1_structure[protocol_name] = {}
            
            # Level 2: Devices/Components
            for device_dir in sorted(protocol_dir.iterdir()):
                if not device_dir.is_dir():
                    continue
                    
                device_name = device_dir.name
                print(f"  📁 Device: {device_name}")
                self.level1_structure[protocol_name][device_name] = []
                
                # Level 3: Attack Scenarios
                for scenario_dir in sorted(device_dir.iterdir()):
                    if not scenario_dir.is_dir():
                        continue
                        
                    scenario_name = scenario_dir.name
                    train_dir = scenario_dir / 'train'
                    test_dir = scenario_dir / 'test'
                    
                    if train_dir.exists() and test_dir.exists():
                        print(f"    📁 Scenario: {scenario_name}")
                        self.level1_structure[protocol_name][device_name].append({
                            'name': scenario_name,
                            'train_dir': train_dir,
                            'test_dir': test_dir
                        })
        
        return self.level1_structure
    
    def load_level3_data(self, train_dir: Path, test_dir: Path, scenario_name: str) -> Optional[Tuple]:
        """Load train and test CSVs from a Level 3 scenario folder."""
        try:
            # Load train data
            train_csv = train_dir / 'attack_and_normal.csv'
            if not train_csv.exists():
                print(f"      ⚠️  Missing: {train_csv}")
                return None
            
            train_df = pd.read_csv(train_csv)
            
            # Load test data
            test_csv = test_dir / 'attack_and_normal.csv'
            if not test_csv.exists():
                print(f"      ⚠️  Missing: {test_csv}")
                return None
            
            test_df = pd.read_csv(test_csv)
            
            # Validate 'attack' column exists
            if 'attack' not in train_df.columns or 'attack' not in test_df.columns:
                print(f"      ⚠️  Missing 'attack' column in {scenario_name}")
                return None
            
            return train_df, test_df, scenario_name
            
        except Exception as e:
            print(f"      ❌ Error loading {scenario_name}: {str(e)}")
            return None
    
    def validate_and_aggregate_level2(self, protocol_name: str, device_name: str, 
                                     scenarios: List[Dict]) -> Optional[Tuple]:
        """
        Aggregate Level 3 datasets into Level 2, with column validation.
        Returns: (X_train, y_train, X_test, y_test, n_clusters, attack_mapping)
        """
        print(f"\n  🔄 Aggregating scenarios for {device_name}...")
        
        all_train_dfs = []
        all_test_dfs = []
        scenario_names = []
        reference_columns = None

        IMP_FEATURES = [
            'integer_7',
            'integer_8',
            'time_diff',
            'timestamp_diff',
            'integer_6',
            'stNum',
            'integer_5',
            'Length',
            'sqNum_diff',
            'sqNum',
            'floatvalue_3',
            'integer_3',
            'floatvalue_1',
            'stNum_diff',
            'time_from_start'
        ]

        
        # Load all scenarios
        for scenario in scenarios:
            result = self.load_level3_data(
                scenario['train_dir'], 
                scenario['test_dir'],
                scenario['name']
            )
            
            if result is None:
                continue
            
            train_df, test_df, scenario_name = result
            
            # Get feature columns (exclude 'attack')
            all_train_features = [col for col in train_df.columns if col != 'attack']
            all_test_features = [col for col in test_df.columns if col != 'attack']
            
            # Filter to only target columns that exist
            train_features = sorted([col for col in all_train_features if col in IMP_FEATURES])
            test_features = sorted([col for col in all_test_features if col in IMP_FEATURES])
            
            # Validate train and test have same features
            if train_features != test_features:
                print(f"      ⚠️  Feature mismatch between train/test in {scenario_name}")
                print(f"         Train: {len(train_features)} features")
                print(f"         Test: {len(test_features)} features")
                continue
            
            # Set reference or validate against reference
            if reference_columns is None:
                reference_columns = train_features
                print(f"      ✓ Reference schema: {len(reference_columns)} features")
            else:
                if train_features != reference_columns:
                    print(f"      ❌ COLUMN MISMATCH in {scenario_name}!")
                    print(f"         Expected: {len(reference_columns)} features")
                    print(f"         Got: {len(train_features)} features")
                    print(f"         ⚠️  SKIPPING ENTIRE DEVICE: {device_name}")
                    return None
            
            all_train_dfs.append(train_df)
            all_test_dfs.append(test_df)
            scenario_names.append(scenario_name)
            print(f"      ✓ Loaded {scenario_name}: train={len(train_df)}, test={len(test_df)}")
        
        # Check if we have any valid scenarios
        if len(all_train_dfs) == 0:
            print(f"      ⚠️  No valid scenarios found for {device_name}")
            return None
        
        for i, (df, scenario_name) in enumerate(zip(all_train_dfs, scenario_names)):
            df['attack_scenario'] = scenario_name
            
        for i, (df, scenario_name) in enumerate(zip(all_test_dfs, scenario_names)):
            df['attack_scenario'] = scenario_name

        # Concatenate all scenarios
        train_combined = pd.concat(all_train_dfs, ignore_index=True).drop_duplicates()
        test_combined = pd.concat(all_test_dfs, ignore_index=True)

        # NaN removal and infinity handling
        train_combined.replace([np.inf, -np.inf], np.nan, inplace=True)
        test_combined.replace([np.inf, -np.inf], np.nan, inplace=True)

        initial_features = reference_columns         
        # Find columns that have NaNs in EITHER Train or Test
        train_nans = train_combined[initial_features].columns[train_combined[initial_features].isna().any()].tolist()
        test_nans = test_combined[initial_features].columns[test_combined[initial_features].isna().any()].tolist()
        
        cols_to_drop = list(set(train_nans + test_nans))
        
        if cols_to_drop:
            print(f"      ⚠️  Dropping {len(cols_to_drop)} columns containing NaN/Inf: {cols_to_drop}")
            valid_features = [c for c in initial_features if c not in cols_to_drop]
        else:
            valid_features = initial_features

        if not valid_features:
            print("      ❌ Error: All feature columns contain NaN/Inf! Cannot proceed.")
            return None
        
        print(f"      ✓ Combined dataset: train={len(train_combined)}, test={len(test_combined)}")
        
        # Create attack label mapping: Normal=0, then attack types by folder order
        attack_mapping = {'Normal': 0}
        for idx, scenario_name in enumerate(sorted(scenario_names), start=1):
            attack_mapping[scenario_name] = idx
        
        n_clusters = len(attack_mapping)  # Normal + attack types
        print(f"      ✓ Attack types: {n_clusters} clusters")
        print(f"         Mapping: {attack_mapping}")
        
        # Use scenario_source to assign correct multi-class labels
        def map_attack_label(row):
            # If attack column is 0 or 'Normal', it's Normal
            if pd.isna(row['attack']) or str(row['attack']).strip() in ['0', 'Normal', 'normal']:
                return 0
            # If attack column is 1, use the scenario source
            else:
                scenario = row['attack_scenario']
                return attack_mapping.get(scenario, 0)
        
        train_combined['attack_numeric'] = train_combined.apply(map_attack_label, axis=1)
        test_combined['attack_numeric'] = test_combined.apply(map_attack_label, axis=1)

        # Drop the temporary scenario_source column
        train_combined = train_combined.drop('attack_scenario', axis=1)
        test_combined = test_combined.drop('attack_scenario', axis=1)
        
        X_train = train_combined[valid_features].values
        y_train = train_combined['attack_numeric'].values
        
        X_test = test_combined[valid_features].values
        y_test = test_combined['attack_numeric'].values
        
        print(f"      ✓ Final cleaned shapes: X_train={X_train.shape}, X_test={X_test.shape}")
        print(f"      ✓ Label distribution (train): {np.bincount(y_train.astype(int))}")
        print(f"      ✓ Label distribution (test): {np.bincount(y_test.astype(int))}")
        
        return X_train, y_train, X_test, y_test, n_clusters, attack_mapping


# =============================================================================
# EXPERIMENT RUNNER
# =============================================================================

class ExperimentRunner:
    """Runs experiments with multiple models and random seeds."""
    
    def __init__(self, random_seeds: List[int] = [42, 123, 456]):
        self.random_seeds = random_seeds
        self.device = torch.device('cpu')  # CPU only for PyTorch
        
    def initialize_models(self, n_clusters: int) -> Dict:
        """Initialize all models with proper configurations."""
        models = {}
        
        # 1. K-Means
        models['KMeans'] = lambda: MultiClassDetector(
            name='KMeans',
            model=KMeans(n_clusters=n_clusters, random_state=42, n_init=10, max_iter=300),
            config={'n_clusters': n_clusters}
        )
        
        # 2. Gaussian Mixture Model
        models['GMM'] = lambda: MultiClassDetector(
            name='GMM',
            model=GaussianMixture(n_components=n_clusters, random_state=42, max_iter=100),
            config={'n_clusters': n_clusters}
        )
        
        # 3. Hierarchical Clustering
        models['Hierarchical'] = lambda: MultiClassDetector(
            name='Hierarchical',
            model=AgglomerativeClustering(n_clusters=n_clusters, linkage='ward'),
            config={'n_clusters': n_clusters}
        )
        
        # 4. One-vs-All Autoencoder
        ae_config = {
            'autoencoder': {
                'architecture': {
                    'encoder_layers': [64, 32],
                    'latent_dim': 16,
                    'decoder_layers': [32, 64]
                },
                'activation': 'relu',
                'dropout': 0.2,
                'training': {
                    'epochs': 50,
                    'batch_size': 256,
                    'learning_rate': 0.001,
                    'early_stopping_patience': 10
                }
            }
        }
        models['OVA-Autoencoder'] = lambda: OneVsAllAutoencoderDetector(
            name='OVA-Autoencoder',
            config=ae_config
        )
        
        # 5. One-vs-All Isolation Forest
        if_config = {
            'n_estimators': 100
        }
        models['OVA-IsolationForest'] = lambda: OneVsAllIsolationForestDetector(
            name='OVA-IsolationForest',
            config=if_config
        )
        
        return models
    
    def run_single_experiment(self, model, X_train, y_train, X_test, y_test, 
                            seed: int) -> Dict:
        """Run a single experiment with one model and one seed."""
        try:
            # Set random seeds
            np.random.seed(seed)
            torch.manual_seed(seed)
            
            # Preprocess data
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train model with mapping
            start_time = time.time()
            model.fit_and_map(X_train_scaled, y_train)
            train_time = time.time() - start_time
            
            # Predict
            start_time = time.time()
            y_pred = model.predict(X_test_scaled)
            inference_time = (time.time() - start_time) / len(X_test)
            
            # Generate detailed Classification Report
            clf_report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
            
            # Base Metrics
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'train_time': train_time,
                'inference_time': inference_time,
                'memory_mb': model.get_memory_mb(),
                
                # Macro Averages (Standard)
                'macro_precision': clf_report['macro avg']['precision'],
                'macro_recall': clf_report['macro avg']['recall'],
                'macro_f1': clf_report['macro avg']['f1-score'],
                
                # Weighted Averages
                'weighted_precision': clf_report['weighted avg']['precision'],
                'weighted_recall': clf_report['weighted avg']['recall'],
                'weighted_f1': clf_report['weighted avg']['f1-score']
            }
            
            # Extract Per-Class Metrics
            per_class_metrics = {}
            for cls_label, scores in clf_report.items():
                if cls_label not in ['accuracy', 'macro avg', 'weighted avg']:
                    per_class_metrics[cls_label] = scores
            
            # Get per-class resource metrics if available
            class_resources = model.get_class_resources()
            
            return metrics, per_class_metrics, class_resources
            
        except Exception as e:
            print(f"        ❌ Error in experiment: {str(e)}")
            import traceback
            traceback.print_exc()
            return None, None, None
    
    def run_multiple_runs(self, model_name: str, model_factory, X_train, y_train, 
                         X_test, y_test, n_clusters: int, attack_mapping: Dict) -> Optional[Dict]:
        """Run 3 times, average summaries, and collect per-class data."""
        print(f"      🔬 Training {model_name}...")
        
        summary_metrics_list = []
        class_metrics_list = []  # List of dicts
        class_resources_list = []  # List of resource dicts

        for run_idx, seed in enumerate(self.random_seeds, start=1):
            print(f"        Run {run_idx}/3 (seed={seed})...", end=' ')
            
            # Create fresh model instance
            model = model_factory()
            if hasattr(model, 'n_clusters'):
                model.n_clusters = n_clusters
            
            # Run experiment
            summary, per_class, class_resources = self.run_single_experiment(
                model, X_train, y_train, X_test, y_test, seed
            )
            
            if summary is None:
                print("❌ Failed")
                continue
            
            summary_metrics_list.append(summary)
            class_metrics_list.append(per_class)
            if class_resources:
                class_resources_list.append(class_resources)
            
            print(f"✓ Acc={summary['accuracy']:.4f}, Macro F1={summary['macro_f1']:.4f}")
        
        if not summary_metrics_list:
            return None, None
        
        # 1. Average Summary Metrics (Scalar values)
        avg_summary = {}
        for key in summary_metrics_list[0].keys():
            avg_summary[key] = np.mean([m[key] for m in summary_metrics_list])
            avg_summary[f'{key}_std'] = np.std([m[key] for m in summary_metrics_list])
            
        # 2. Average Per-Class Metrics
        avg_class_metrics = []
        
        # Get all class IDs present in the report (as strings)
        class_ids = class_metrics_list[0].keys()
        
        # Reverse mapping for naming: {0: 'Normal', 1: 'Dos'}
        id_to_name = {v: k for k, v in attack_mapping.items()}
        
        for cid in class_ids:
            # Name of the class
            try:
                c_name = id_to_name.get(int(cid), f"Class_{cid}")
            except:
                c_name = f"Class_{cid}"
                
            # Aggregate stats for this specific class across seeds
            precisions = [run[cid]['precision'] for run in class_metrics_list]
            recalls = [run[cid]['recall'] for run in class_metrics_list]
            f1s = [run[cid]['f1-score'] for run in class_metrics_list]
            supports = [run[cid]['support'] for run in class_metrics_list]
            
            class_row = {
                'class_id': int(cid),
                'class_name': c_name,
                'precision': np.mean(precisions),
                'recall': np.mean(recalls),
                'f1_score': np.mean(f1s),
                'support': np.mean(supports)
            }
            
            # Add resource metrics if available (average across runs)
            if class_resources_list and int(cid) in class_resources_list[0]:
                resource_keys = ['train_time_ns', 'train_time_s', 'peak_ram_mb', 'cpu_avg_pct', 'cpu_peak_pct']
                for rkey in resource_keys:
                    vals = [r.get(int(cid), {}).get(rkey, 0) for r in class_resources_list]
                    class_row[rkey] = np.mean(vals) if vals else 0
            
            avg_class_metrics.append(class_row)
            
        return avg_summary, avg_class_metrics


# =============================================================================
# RESULTS AGGREGATION
# =============================================================================

class ResultsAggregator:
    """Aggregates and saves results at different levels."""
    
    def __init__(self, output_dir: str = 'results'):
        self.output_dir = Path(output_dir)
        
        # New Directory Structure
        self.level2_summary_dir = self.output_dir / 'level2_device_summary'
        self.level2_class_dir = self.output_dir / 'level2_per_class_details'
        self.level1_dir = self.output_dir / 'level1_dataset_summary'
        
        for d in [self.level2_summary_dir, self.level2_class_dir, self.level1_dir]:
            d.mkdir(parents=True, exist_ok=True)
        
        self.all_level2_summaries = []
        self.all_level2_class_details = []
        self.all_level1_summaries = []
    
    def save_level2_results(self, protocol: str, device: str, 
                          model_summaries: Dict, model_class_details: Dict):
        """Save both summary and granular reports for a device."""
        
        # 1. Process Summary Data
        device_summary_rows = []
        for model_name, metrics in model_summaries.items():
            if metrics:
                row = {'dataset': protocol, 'device': device, 'model': model_name, **metrics}
                device_summary_rows.append(row)
                self.all_level2_summaries.append(row)
        
        if device_summary_rows:
            df_summary = pd.DataFrame(device_summary_rows)
            df_summary.to_csv(self.level2_summary_dir / f"{protocol}_{device}_results.csv", index=False)
            
        # 2. Process Per-Class Data
        device_class_rows = []
        for model_name, class_list in model_class_details.items():
            if class_list:
                for item in class_list:
                    row = {
                        'dataset': protocol, 
                        'device': device, 
                        'model': model_name,
                        **item
                    }
                    device_class_rows.append(row)
                    self.all_level2_class_details.append(row)
        
        if device_class_rows:
            df_class = pd.DataFrame(device_class_rows)
            # Reorder columns for readability (include resource columns if present)
            base_cols = ['dataset', 'device', 'model', 'class_name', 'class_id', 
                        'precision', 'recall', 'f1_score', 'support']
            resource_cols = ['train_time_s', 'train_time_ns', 'peak_ram_mb', 'cpu_avg_pct', 'cpu_peak_pct']
            
            all_cols = base_cols + [c for c in resource_cols if c in df_class.columns]
            available_cols = [c for c in all_cols if c in df_class.columns]
            df_class = df_class[available_cols]
            df_class.to_csv(self.level2_class_dir / f"{protocol}_{device}_class_report.csv", index=False)
            print(f"      💾 Saved detailed class report to {self.level2_class_dir}")

    
    def aggregate_level1(self, protocol: str):
        """Aggregate summaries to dataset level."""
        # Filter for current protocol
        proto_data = [r for r in self.all_level2_summaries if r['dataset'] == protocol]
        
        if not proto_data:
            return

        model_groups = defaultdict(list)
        for r in proto_data:
            model_groups[r['model']].append(r)
            
        protocol_rows = []
        for model, records in model_groups.items():
            avg_row = {'dataset': protocol, 'model': model, 'n_devices': len(records)}
            # Average all numeric keys
            for key in records[0].keys():
                if isinstance(records[0][key], (int, float)) and key not in ['class_id']:
                    avg_row[key] = np.mean([r[key] for r in records])
            protocol_rows.append(avg_row)
            self.all_level1_summaries.append(avg_row)
            
        df = pd.DataFrame(protocol_rows)
        df.to_csv(self.level1_dir / f"{protocol}_summary.csv", index=False)
        print(f"    💾 Saved Level 1 summary to {self.level1_dir}")

    def save_final_summaries(self):
        """Save master CSVs."""
        if self.all_level2_summaries:
            pd.DataFrame(self.all_level2_summaries).to_csv(
                self.level2_summary_dir / 'level2_all_summary.csv', index=False)
            
        if self.all_level2_class_details:
            pd.DataFrame(self.all_level2_class_details).to_csv(
                self.level2_class_dir / 'detailed_class_all.csv', index=False)
            
        if self.all_level1_summaries:
            pd.DataFrame(self.all_level1_summaries).to_csv(
                self.level1_dir / 'level1_all_results.csv', index=False)


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    """Main orchestrator."""
    
    # Configuration
    ROOT_DIR = r"C:\Users\sengu\Documents\cp219_project-2\data\Final_Datasets\Final_Datasets\preprocessed_new\preprocessed_new"
    
    print("\n" + "="*80)
    print("UNSUPERVISED MULTICLASS ATTACK CLASSIFICATION")
    print("One-vs-All Anomaly Detection Approach")
    print("="*80)
    print(f"Root Directory: {ROOT_DIR}")
    print(f"Device: CPU (PyTorch)")
    print(f"Random Seeds: [42, 123, 456]")
    print(f"ResourceProfiler Available: {RESOURCE_PROFILER_AVAILABLE}")
    
    # Initialize components
    loader = DatasetLoader(ROOT_DIR)
    runner = ExperimentRunner()
    aggregator = ResultsAggregator()
    
    # Phase 1: Traverse directory
    structure = loader.traverse_directory()
    
    # Phase 2 & 3: Train and evaluate
    print("\n" + "="*80)
    print("PHASE 2 & 3: TRAINING AND EVALUATION")
    print("="*80)
    
    for protocol_name, devices in structure.items():
        print(f"\n{'='*80}")
        print(f"DATASET: {protocol_name}")
        print(f"{'='*80}")
        
        
        for device_name, scenarios in devices.items():
            print(f"\n  {'─'*76}")
            print(f"  DEVICE: {device_name}")
            print(f"  {'─'*76}")
            
            if len(scenarios) == 0:
                print("    ⚠️  No scenarios found, skipping...")
                continue
            
            # Aggregate Level 3 to Level 2
            result = loader.validate_and_aggregate_level2(
                protocol_name, device_name, scenarios
            )
            
            if result is None:
                print(f"    ⚠️  Skipping device {device_name} due to validation errors")
                continue
            
            X_train, y_train, X_test, y_test, n_clusters, attack_mapping = result
            
            # Initialize models
            models = runner.initialize_models(n_clusters)
            
            # Run experiments for each model
            device_summaries = {}
            device_class_details = {}

            for model_name, model_factory in models.items():
                summary, class_details = runner.run_multiple_runs(
                    model_name, model_factory, 
                    X_train, y_train, X_test, y_test,
                    n_clusters, attack_mapping
                )
                device_summaries[model_name] = summary
                device_class_details[model_name] = class_details
            
            # Save Level 2 results
            aggregator.save_level2_results(protocol_name, device_name, device_summaries, device_class_details)
        
        # Aggregate to Level 1
        aggregator.aggregate_level1(protocol_name)
    
    # Phase 4: Overall summary
    aggregator.save_final_summaries()
    
    print("\n" + "="*80)
    print("✅ TRAINING COMPLETE!")
    print("="*80)
    print(f"Results saved in: {aggregator.output_dir}")


if __name__ == "__main__":
    main()