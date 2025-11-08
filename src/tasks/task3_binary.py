"""
Task 3: Unsupervised Binary Intrusion Detection (Per-Attack Training)
Compare multiple unsupervised models for binary classification (Normal vs Attack).
Trains separate models for each attack type for better attack-specific insights.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any, List, Tuple
import time
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve
)
from src.preprocessing import (
    preprocess_dataframe,
    load_combined_datasets,
    standardize_schema,
    engineer_features,
    get_numeric_features,
    CORE_FIELDS,
    ALLOWED_FIELDS
)


class BinaryDetector:
    """Base class for binary anomaly detectors."""
    
    def __init__(self, name: str, model, config: Dict = None):
        self.name = name
        self.model = model
        self.config = config or {}
        self.train_time = 0
        self.inference_time = 0
        self.train_labels_ = None  # Store training labels for DBSCAN
        
    def fit(self, X):
        """Train the model."""
        start = time.time()
        
        if isinstance(self.model, DBSCAN):
            # DBSCAN fits and labels in one step
            self.train_labels_ = self.model.fit_predict(X)
        else:
            self.model.fit(X)
            
        self.train_time = time.time() - start
        return self
    
    def predict(self, X):
        """Predict anomalies."""
        start = time.time()
        
        # Handle DBSCAN separately
        if isinstance(self.model, DBSCAN):
            # DBSCAN doesn't have predict() - use fit_predict on test data
            predictions = self.model.fit_predict(X)
        else:
            predictions = self.model.predict(X)
        
        self.inference_time = (time.time() - start) / len(X)
        
        # Convert to binary (0=normal, 1=attack)
        if isinstance(self.model, (IsolationForest, OneClassSVM)):
            # These use -1 for anomalies, 1 for normal
            predictions = np.where(predictions == -1, 1, 0)
        elif isinstance(self.model, DBSCAN):
            # DBSCAN: -1 = noise/outliers (treat as attacks), others = clusters (normal)
            predictions = np.where(predictions == -1, 1, 0)
        elif isinstance(self.model, (KMeans, GaussianMixture)):
            # For clustering: assume smaller cluster is attacks
            unique, counts = np.unique(predictions, return_counts=True)
            minority_cluster = unique[np.argmin(counts)]
            predictions = np.where(predictions == minority_cluster, 1, 0)
        
        return predictions
    
    def get_memory_mb(self):
        """Estimate memory footprint."""
        import sys
        return sys.getsizeof(self.model) / (1024 * 1024)


class Task3BinaryDetection:
    """Task 3: Binary Intrusion Detection (Per-Attack Training)."""
    
    def __init__(self, config: Dict[str, Any], logger=None, mode: str = 'core'):
        """
        Args:
            mode: 'core' for CORE_FIELDS only, 'full' for CORE + ALLOWED + ENGINEERED
        """
        self.config = config
        self.logger = logger
        self.mode = mode
        
        # Create output directories with mode suffix
        self.fig_dir = Path(f'outputs/figures/task3_{mode}')
        self.table_dir = Path(f'outputs/tables/task3_{mode}')
        self.model_dir = Path(f'outputs/models/task3_{mode}')
        self.fig_dir.mkdir(parents=True, exist_ok=True)
        self.table_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.dpi'] = 300
        
        print(f"\n{'='*70}")
        print(f"RUNNING TASK 3 IN '{mode.upper()}' MODE (PER-ATTACK TRAINING)")
        if mode == 'core':
            print("Using ONLY the 10 CORE_FIELDS")
        else:
            print("Using CORE + ALLOWED + ENGINEERED features")
        print(f"{'='*70}\n")
        
    def load_attack_data(self, attack_type: str):
        """Load and preprocess data for a single attack type."""
        print(f"\nLoading {attack_type} dataset...")
        data_dir = Path(self.config['data']['raw_dir'])
        
        train_file = self.config['data']['train_files'][attack_type]
        test_file = self.config['data']['test_files'][attack_type]
        
        # Load raw data
        train_df = pd.read_csv(data_dir / train_file)
        test_df = pd.read_csv(data_dir / test_file)
        
        # Select fields based on mode
        if self.mode == 'core':
            keep_train = [c for c in CORE_FIELDS if c in train_df.columns]
            keep_test = [c for c in CORE_FIELDS if c in test_df.columns]
        else:  # full mode
            keep_train = [c for c in ALLOWED_FIELDS if c in train_df.columns]
            keep_test = [c for c in ALLOWED_FIELDS if c in test_df.columns]
        
        train_df = train_df[keep_train].copy()
        test_df = test_df[keep_test].copy()
        
        # Apply preprocessing
        print(f"  Preprocessing {attack_type}...")
        train_df = preprocess_dataframe(train_df)
        train_df = standardize_schema(train_df)
        test_df = preprocess_dataframe(test_df)
        test_df = standardize_schema(test_df)
        
        # Engineer features (only in full mode)
        if self.mode == 'full':
            train_df = engineer_features(train_df)
            test_df = engineer_features(test_df)
        
        print(f"  {attack_type}: Train={len(train_df)}, Test={len(test_df)}, "
              f"Features={len(train_df.columns)}")
        
        return train_df, test_df
        
    def preprocess_attack_data(self, train_df, test_df, attack_type: str):
        """Preprocess data for a specific attack type."""
        print(f"\nPreprocessing {attack_type} data...")
        
        # Get numeric features based on mode (excluding 'attack')
        numeric_cols = get_numeric_features(train_df, mode=self.mode)
        
        print(f"  Using {len(numeric_cols)} numeric features from {self.mode.upper()} mode")
        
        # Extract features and labels
        X_train = train_df[numeric_cols].copy()
        y_train = train_df['attack'].values
        X_test = test_df[numeric_cols].copy()
        y_test = test_df['attack'].values
        
        # Clean data
        print("  Cleaning data...")
        X_train = X_train.replace([np.inf, -np.inf], np.nan)
        X_test = X_test.replace([np.inf, -np.inf], np.nan)
        
        # Drop columns with too many NaN values (>50%)
        threshold = len(X_train) * 0.5
        cols_passing_threshold = X_train.columns[X_train.notna().sum() > threshold]
        cols_to_drop = set(X_train.columns) - set(cols_passing_threshold)
        
        # Ensure 'boolean_1' is never dropped
        if 'boolean_1' in cols_to_drop:
            print("    Force-keeping 'boolean_1' column despite high NaN count.")
            cols_to_drop.remove('boolean_1')
            
        final_valid_cols = [col for col in X_train.columns if col not in cols_to_drop]
        X_train = X_train[final_valid_cols]
        X_test = X_test[final_valid_cols]
        
        print(f"  Features after dropping high-NaN columns: {len(X_train.columns)}")
        
        # Fill remaining NaN values with median
        print("  Filling NaN values with column median...")
        for col in X_train.columns:
            median_val = X_train[col].median()
            if pd.isna(median_val):
                median_val = 0
            
            if col == 'boolean_1' and X_train[col].isna().any():
                print(f"    - Imputing 'boolean_1' NaN values with median: {median_val}")
                
            X_train[col].fillna(median_val, inplace=True)
            X_test[col].fillna(median_val, inplace=True)
        
        # Drop columns with zero variance
        variances = X_train.var()
        valid_cols_variance = variances[variances > 0].index
        cols_to_drop_variance = set(X_train.columns) - set(valid_cols_variance)

        if 'boolean_1' in cols_to_drop_variance:
            print(f"    Warning: 'boolean_1' has zero variance after imputation.")
            
        final_valid_cols = [col for col in X_train.columns if col not in cols_to_drop_variance]
        X_train = X_train[final_valid_cols]
        X_test = X_test[final_valid_cols]
        
        print(f"  Features after cleaning: {len(X_train.columns)}")
        print(f"  Train samples: {len(X_train)}, Attack ratio: {y_train.mean():.2%}")
        print(f"  Test samples: {len(X_test)}, Attack ratio: {y_test.mean():.2%}")
        
        # Store feature names before scaling
        feature_names = list(X_train.columns)
        
        # Standardize features
        print("  Standardizing features...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Handle any remaining NaN after scaling
        X_train_scaled = np.nan_to_num(X_train_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        X_test_scaled = np.nan_to_num(X_test_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        
        return X_train_scaled, y_train, X_test_scaled, y_test, feature_names, scaler
    
    def create_models(self, contamination: float = 0.1):
        """Create all models to compare."""
        models = {
            'Isolation Forest': BinaryDetector(
                'Isolation Forest',
                IsolationForest(
                    n_estimators=100,
                    contamination=contamination,
                    random_state=42,
                    n_jobs=-1
                )
            ),
            'One-Class SVM': BinaryDetector(
                'One-Class SVM',
                OneClassSVM(
                    kernel='rbf',
                    gamma='auto',
                    nu=contamination
                )
            ),
            'K-Means': BinaryDetector(
                'K-Means',
                KMeans(
                    n_clusters=2,
                    random_state=42,
                    n_init=10
                )
            ),
            'GMM': BinaryDetector(
                'GMM',
                GaussianMixture(
                    n_components=2,
                    random_state=42,
                    covariance_type='full'
                )
            ),
            # 'DBSCAN': BinaryDetector(
            #     'DBSCAN',
            #     DBSCAN(
            #         eps=3.0,
            #         min_samples=50,
            #         n_jobs=-1
            #     )
            # )
        }
        
        return models
    
    def balance_combined_test_set(self, test_df: pd.DataFrame,
                              strategy: str = 'equal_per_attack',
                              random_state: int = 42,
                              allow_replacement: bool = False) -> pd.DataFrame:
        """
        Balanced test set where each attack_type contributes:
        - n_pos attack rows and n_pos normal rows (parity within each attack type).

        Args:
            test_df: must contain 'attack_type' and 'attack' (0/1) columns.
            strategy: currently supports 'equal_per_attack' (recommended) and a simple 'downsample_majority' fallback.
            random_state: reproducible sampling seed.
            allow_replacement: if True, sampling will use replacement when a group is too small.

        Returns:
            balanced_df: shuffled DataFrame with equal (attack, normal) counts per attack_type.
        """
        if 'attack_type' not in test_df.columns:
            raise KeyError("test_df must contain 'attack_type' column")
        if 'attack' not in test_df.columns:
            raise KeyError("test_df must contain 'attack' column (0/1)")

        rng = np.random.RandomState(random_state)
        print(f"\n  Balancing test set (strategy: {strategy})...")

        # count positives per attack_type
        pos_counts = test_df[test_df['attack'] == 1].groupby('attack_type').size()
        if pos_counts.empty:
            raise ValueError("No positive (attack==1) samples found in test_df")

        if strategy == 'equal_per_attack':
            n_pos = int(pos_counts.min())  # number of positive samples to take per attack_type
            if n_pos < 1:
                raise ValueError("Insufficient positive samples to build balanced set")

            parts = []
            for atype in pos_counts.index:
                attacks_of_type = test_df[(test_df['attack_type'] == atype) & (test_df['attack'] == 1)]
                replace_att = allow_replacement and (len(attacks_of_type) < n_pos)
                sampled_attacks = attacks_of_type.sample(n=n_pos, replace=replace_att,
                                                        random_state=rng.randint(0, 2**31-1))

                normals_same = test_df[(test_df['attack_type'] == atype) & (test_df['attack'] == 0)]
                if len(normals_same) >= n_pos:
                    sampled_normals = normals_same.sample(n=n_pos, replace=False,
                                                        random_state=rng.randint(0, 2**31-1))
                else:
                    # fallback: sample normals from global pool (avoid running out)
                    normals_global = test_df[test_df['attack'] == 0]
                    replace_norm = allow_replacement and (len(normals_global) < n_pos)
                    sampled_normals = normals_global.sample(n=n_pos, replace=replace_norm,
                                                            random_state=rng.randint(0, 2**31-1))

                parts.append(pd.concat([sampled_attacks, sampled_normals], ignore_index=True))

            balanced_df = pd.concat(parts, ignore_index=True)

        elif strategy == 'downsample_majority':
            # Simple fallback: limit largest group(s) so distribution is less dominated
            # Here we downsample each attack_type to median count and then ensure parity by sampling normals per-type similarly.
            median_total = int(test_df['attack_type'].value_counts().median())
            parts = []
            for atype in test_df['attack_type'].unique():
                grp = test_df[test_df['attack_type'] == atype]
                # pick positives up to median_total (or less)
                pos_grp = grp[grp['attack'] == 1]
                neg_grp = grp[grp['attack'] == 0]
                n_pos = min(len(pos_grp), median_total)
                if n_pos == 0:
                    continue
                replace_pos = allow_replacement and (len(pos_grp) < n_pos)
                sampled_pos = pos_grp.sample(n=n_pos, replace=replace_pos, random_state=rng.randint(0, 2**31-1))

                if len(neg_grp) >= n_pos:
                    sampled_neg = neg_grp.sample(n=n_pos, replace=False, random_state=rng.randint(0, 2**31-1))
                else:
                    normals_global = test_df[test_df['attack'] == 0]
                    replace_norm = allow_replacement and (len(normals_global) < n_pos)
                    sampled_neg = normals_global.sample(n=n_pos, replace=replace_norm, random_state=rng.randint(0, 2**31-1))

                parts.append(pd.concat([sampled_pos, sampled_neg], ignore_index=True))

            if not parts:
                raise ValueError("No groups available to downsample")
            balanced_df = pd.concat(parts, ignore_index=True)

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        # shuffle final DF
        balanced_df = balanced_df.sample(frac=1, random_state=rng.randint(0, 2**31-1)).reset_index(drop=True)

        # debug print
        print("  Balanced distribution (per attack_type):")
        for at, cnt in balanced_df['attack_type'].value_counts().items():
            print(f"    {at}: {cnt} rows (should be 2 * n_pos for equal_per_attack)")
        print(f"  Total: {len(test_df)} -> {len(balanced_df)}")

        return balanced_df

    
    def evaluate_model(self, model: BinaryDetector, X_test, y_test):
        """Evaluate a single model."""
        y_pred = model.predict(X_test)
        
        # Handle DBSCAN outliers (-1) as attacks
        if isinstance(model.model, DBSCAN):
            y_pred = np.where(y_pred == -1, 1, 0)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        cm = confusion_matrix(y_test, y_pred)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm,
            'train_time': model.train_time,
            'inference_time_ms': model.inference_time * 1000,
            'memory_mb': model.get_memory_mb()
        }
    
    def train_and_evaluate_attack(self, attack_type: str, X_train, y_train, X_test, y_test):
        """Train and evaluate all models for a specific attack type."""
        print(f"\n{'='*70}")
        print(f"TRAINING MODELS FOR {attack_type.upper()} ATTACK")
        print(f"{'='*70}")
        
        # Calculate contamination based on actual attack ratio
        contamination = max(0.01, min(0.5, y_train.mean()))
        print(f"Using contamination={contamination:.3f} (attack ratio: {y_train.mean():.3f})")
        
        models = self.create_models(contamination=contamination)
        results = []
        all_predictions = {}
        
        for model_name, model in models.items():
            print(f"\n{'-'*60}")
            print(f"Model: {model_name}")
            print(f"{'-'*60}")
            
            try:
                # Train
                print(f"  Training...")
                model.fit(X_train)
                print(f"  Training time: {model.train_time:.2f}s")
                
                # Evaluate
                print(f"  Evaluating...")
                metrics = self.evaluate_model(model, X_test, y_test)
                all_predictions[model_name] = model.predict(X_test)
                
                # Store results
                result = {
                    'Attack': attack_type,
                    'Model': model_name,
                    'Accuracy': metrics['accuracy'],
                    'Precision': metrics['precision'],
                    'Recall': metrics['recall'],
                    'F1-Score': metrics['f1_score'],
                    'Train Time (s)': metrics['train_time'],
                    'Inference Time (ms)': metrics['inference_time_ms'],
                    'Memory (MB)': metrics['memory_mb']
                }
                results.append(result)
                
                # Print results
                print(f"  Results:")
                print(f"    Accuracy:  {metrics['accuracy']:.4f}")
                print(f"    Precision: {metrics['precision']:.4f}")
                print(f"    Recall:    {metrics['recall']:.4f}")
                print(f"    F1-Score:  {metrics['f1_score']:.4f}")
                
                
                
                # Save model
                import joblib
                attack_model_dir = self.model_dir / attack_type
                attack_model_dir.mkdir(exist_ok=True)
                model_path = attack_model_dir / f"{model_name.lower().replace(' ', '_')}.pkl"
                joblib.dump(model.model, model_path)
                
            except Exception as e:
                print(f"  ERROR: {str(e)}")
                import traceback
                traceback.print_exc()
                continue
        
        return pd.DataFrame(results), all_predictions
    
    def visualize_attack_results(self, attack_type: str, results_df: pd.DataFrame, 
                                X_test, y_test, all_predictions):
        """Create visualizations for a specific attack type."""
        print(f"\n  Generating visualizations for {attack_type}...")
        
        # Create attack-specific directory
        attack_fig_dir = self.fig_dir / attack_type
        attack_fig_dir.mkdir(exist_ok=True)
        
        # 1. Performance metrics comparison
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'{attack_type.upper()} Attack - Model Performance ({self.mode.upper()} Mode)', 
                     fontsize=14, fontweight='bold')
        
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        for idx, metric in enumerate(metrics):
            ax = axes[idx // 2, idx % 2]
            data = results_df.sort_values(metric, ascending=False)
            
            bars = ax.barh(data['Model'], data[metric], color='steelblue', alpha=0.7)
            ax.set_xlabel(metric)
            ax.set_title(f'{metric} by Model', fontweight='bold')
            ax.set_xlim([0, 1])
            ax.grid(True, alpha=0.3, axis='x')
            
            # Add value labels
            for i, (bar, val) in enumerate(zip(bars, data[metric])):
                ax.text(val, i, f' {val:.3f}', va='center')
        
        plt.tight_layout()
        fig_path = attack_fig_dir / 'model_performance_comparison.png'
        plt.savefig(fig_path, bbox_inches='tight')
        print(f"    Saved: {fig_path}")
        
        if self.logger:
            self.logger.log_figure(fig, f"task3_{self.mode}/{attack_type}/model_performance")
        plt.close(fig)
        
        # 2. PCA Projections for each model
        self.visualize_attack_projections(attack_type, X_test, y_test, all_predictions, results_df)
    
    def visualize_attack_projections(self, attack_type: str, X_test, y_test,
                                    all_predictions, results_df: pd.DataFrame):
        """
        Visualize PCA projections for the passed X_test/y_test and predictions.
        This version validates that X_test, y_test, and predictions all have matching lengths.
        """
        print(f"    Generating PCA projections for {attack_type}...")

        # Ensure X_test is numpy array of shape (n_samples, n_features)
        import numpy as _np
        X_test = _np.asarray(X_test)
        n_rows = X_test.shape[0]

        # Basic shape checks
        if len(y_test) != n_rows:
            raise ValueError(f"y_test length ({len(y_test)}) does not match X_test rows ({n_rows}). "
                            "Ensure you pass balanced X and y produced by train_and_evaluate_combined().")

        # Ensure predictions exist and match length
        for model_name, preds in list(all_predictions.items()):
            # allow dict with 'balanced' key (returned by train_and_evaluate_combined)
            if isinstance(preds, dict):
                model_preds = preds.get('balanced', None)
            else:
                model_preds = preds

            if model_preds is None:
                print(f"    WARNING: missing balanced predictions for {model_name}; skipping projection for this model.")
                # remove from items to avoid plotting
                all_predictions.pop(model_name, None)
                continue

            if len(model_preds) != n_rows:
                print(f"    WARNING: predictions length for {model_name} ({len(model_preds)}) "
                    f"does not match X_test rows ({n_rows}); skipping this model.")
                all_predictions.pop(model_name, None)
                continue

        # PCA transform
        pca = PCA(n_components=2, random_state=42)
        X_test_2d = pca.fit_transform(X_test)
        explained_var = pca.explained_variance_ratio_

        # For each model present in results_df & all_predictions, plot projections
        for model_name, y_pred in all_predictions.items():
            # ensure model present in results_df
            rows = results_df[results_df['Model'] == model_name]
            if rows.empty:
                print(f"    WARNING: no metrics row for {model_name}; skipping labeling in projection.")
                accuracy = np.nan
                f1 = np.nan
            else:
                model_row = rows.iloc[0]
                accuracy = model_row.get('Accuracy', np.nan)
                f1 = model_row.get('F1-Score', np.nan)

            # If preds stored as dict entry, get the array
            if isinstance(y_pred, dict):
                y_pred = y_pred.get('balanced', None)
                if y_pred is None:
                    continue

            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            fig.suptitle(f'{attack_type.upper()} - {model_name} - PCA Projection\n'
                        f'Accuracy: {accuracy:.3f} | F1-Score: {f1:.3f}',
                        fontsize=14, fontweight='bold')

            # Ground truth plot
            ax = axes[0]
            for label, color, name in [(0, 'blue', 'Normal'), (1, 'red', 'Attack')]:
                mask = (y_test == label)
                ax.scatter(X_test_2d[mask, 0], X_test_2d[mask, 1],
                        c=color, label=name, alpha=0.5, s=20, edgecolors='none')
            ax.set_xlabel(f'PC1 ({explained_var[0]:.1%} variance)')
            ax.set_ylabel(f'PC2 ({explained_var[1]:.1%} variance)')
            ax.set_title('Ground Truth Labels', fontweight='bold')
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)

            # Predictions plot
            ax = axes[1]
            for label, color, name in [(0, 'blue', 'Normal'), (1, 'red', 'Attack')]:
                mask = (y_pred == label)
                ax.scatter(X_test_2d[mask, 0], X_test_2d[mask, 1],
                        c=color, label=f'Predicted {name}', alpha=0.5, s=20, edgecolors='none')
            ax.set_xlabel(f'PC1 ({explained_var[0]:.1%} variance)')
            ax.set_ylabel(f'PC2 ({explained_var[1]:.1%} variance)')
            ax.set_title('Model Predictions', fontweight='bold')
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            safe_name = model_name.lower().replace(' ', '_').replace('-', '_')
            attack_fig_dir = self.fig_dir / attack_type
            attack_fig_dir.mkdir(exist_ok=True)
            fig_path = attack_fig_dir / f'projection_{safe_name}.png'
            plt.savefig(fig_path, bbox_inches='tight')
            if self.logger:
                self.logger.log_figure(fig, f"task3_{self.mode}/{attack_type}/projection_{safe_name}")
            plt.close(fig)

        print(f"      ✓ Generated {len(all_predictions)} projection plots")

    
    def generate_attack_summary(self, attack_type: str, results_df: pd.DataFrame):
        """Generate summary for a specific attack type."""
        print(f"\n{'='*70}")
        print(f"{attack_type.upper()} ATTACK - SUMMARY")
        print(f"{'='*70}")
        
        # Sort by F1-Score
        results_df = results_df.sort_values('F1-Score', ascending=False)
        
        print(f"\n{attack_type.upper()} Model Comparison:")
        print(results_df[['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score']].to_string(index=False))
        
        # Save to CSV
        attack_table_dir = self.table_dir
        results_df.to_csv(attack_table_dir / f'{attack_type}_model_comparison.csv', index=False)
        
        # Best model
        best_model = results_df.iloc[0]
        print(f"\nBest Model for {attack_type}: {best_model['Model']}")
        print(f"  F1-Score: {best_model['F1-Score']:.4f}")
        print(f"  Accuracy: {best_model['Accuracy']:.4f}")
        
        return results_df
    
    def generate_cross_attack_summary(self, all_results: Dict[str, pd.DataFrame]):
        """Generate aggregated summary across all attack types."""
        print(f"\n{'='*70}")
        print(f"CROSS-ATTACK SUMMARY ({self.mode.upper()} MODE)")
        print(f"{'='*70}")
        
        # Combine all results
        combined_df = pd.concat(all_results.values(), ignore_index=True)
        
        # Calculate average metrics per model
        summary_df = combined_df.groupby('Model').agg({
            'Accuracy': ['mean', 'std'],
            'Precision': ['mean', 'std'],
            'Recall': ['mean', 'std'],
            'F1-Score': ['mean', 'std'],
            'Train Time (s)': 'mean',
            'Inference Time (ms)': 'mean',
            'Memory (MB)': 'mean'
        }).round(4)
        
        summary_df.columns = ['_'.join(col).strip() for col in summary_df.columns.values]
        summary_df = summary_df.reset_index()
        
        print("\nAverage Performance Across All Attacks:")
        print(summary_df.to_string(index=False))
        
        # Save summary
        summary_df.to_csv(self.table_dir / 'summary_all_attacks.csv', index=False)
        print(f"\nSaved: {self.table_dir / 'summary_all_attacks.csv'}")
        
        # Best model per attack
        print("\n" + "="*70)
        print("BEST MODEL PER ATTACK TYPE")
        print("="*70)
        for attack_type, results_df in all_results.items():
            best = results_df.loc[results_df['F1-Score'].idxmax()]
            print(f"{attack_type.upper():15s}: {best['Model']:20s} "
                  f"(F1={best['F1-Score']:.4f}, Acc={best['Accuracy']:.4f})")
        
        # Overall best model (highest average F1-Score)
        best_overall = summary_df.loc[summary_df['F1-Score_mean'].idxmax()]
        print(f"\n{'='*70}")
        print(f"OVERALL BEST MODEL: {best_overall['Model']}")
        print(f"{'='*70}")
        print(f"Average F1-Score: {best_overall['F1-Score_mean']:.4f} ± {best_overall['F1-Score_std']:.4f}")
        print(f"Average Accuracy: {best_overall['Accuracy_mean']:.4f} ± {best_overall['Accuracy_std']:.4f}")
        
        # Create cross-attack heatmap
        self.create_cross_attack_heatmap(combined_df)
        
        return summary_df
    
    def create_cross_attack_heatmap(self, combined_df: pd.DataFrame):
        """Create heatmap showing F1-score per model per attack."""
        print("\n  Generating cross-attack heatmap...")
        
        # Pivot data for heatmap
        heatmap_data = combined_df.pivot(index='Model', columns='Attack', values='F1-Score')
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='YlOrRd', 
                   cbar_kws={'label': 'F1-Score'}, ax=ax, vmin=0, vmax=1)
        ax.set_title(f'Model Performance Across Attack Types ({self.mode.upper()} Mode)', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Attack Type', fontweight='bold')
        ax.set_ylabel('Model', fontweight='bold')
        
        plt.tight_layout()
        fig_path = self.fig_dir / 'cross_attack_heatmap.png'
        plt.savefig(fig_path, bbox_inches='tight')
        print(f"    Saved: {fig_path}")
        
        if self.logger:
            self.logger.log_figure(fig, f"task3_{self.mode}/cross_attack_heatmap")
        plt.close(fig)

    def train_and_evaluate_combined(self, X_train, y_train, X_test, y_test,
                                test_df: pd.DataFrame, scaler, features,
                                balance_strategy: str = 'equal_per_attack'):
        """
        Train and evaluate all models on combined dataset with balanced evaluation.

        Returns:
            results_df, all_predictions, balanced_test_df, X_balanced_scaled, y_balanced
        all_predictions will be a dict: { model_name: {'imbalanced': arr, 'balanced': arr} }
        """
        print(f"\n{'='*70}")
        print(f"TRAINING MODELS ON COMBINED DATASET")
        print(f"{'='*70}")

        # Calculate contamination based on actual attack ratio
        contamination = max(0.01, min(0.5, y_train.mean()))
        print(f"Using contamination={contamination:.3f} (attack ratio: {y_train.mean():.3f})")

        models = self.create_models(contamination=contamination)
        results = []
        all_predictions = {}

        # We'll create the balanced test DF once (so all models evaluate on the SAME balanced sample)
        balanced_test_df = self.balance_combined_test_set(test_df, strategy=balance_strategy)

        # Build balanced arrays and scaled version once
        features_present = [c for c in features if c in balanced_test_df.columns]
        if not features_present:
            raise RuntimeError("No feature columns found in balanced test_df. Check `features` list.")

        X_balanced = balanced_test_df[features_present].values
        y_balanced = balanced_test_df['attack'].values

        X_balanced_scaled = scaler.transform(X_balanced)
        X_balanced_scaled = np.nan_to_num(X_balanced_scaled, nan=0.0, posinf=0.0, neginf=0.0)

        # Iterate models
        for model_name, model in models.items():
            print(f"\n{'-'*60}")
            print(f"Model: {model_name}")
            print(f"{'-'*60}")

            # prepare storage shape placeholders
            all_predictions[model_name] = {'imbalanced': None, 'balanced': None}

            try:
                # Train
                print(f"  Training...")
                model.fit(X_train)
                print(f"  Training time: {model.train_time:.2f}s")

                # Evaluate on IMBALANCED test set (for reference)
                print(f"  Evaluating on imbalanced test set...")
                try:
                    y_pred_imbalanced = model.predict(X_test)
                except Exception as e:
                    print(f"  WARNING: model.predict failed on imbalanced X_test: {e}")
                    y_pred_imbalanced = np.zeros(len(X_test), dtype=int)

                metrics_imbalanced = self.evaluate_model(model, X_test, y_test)
                all_predictions[model_name]['imbalanced'] = y_pred_imbalanced

                print(f"  Imbalanced Results:")
                print(f"    Accuracy:  {metrics_imbalanced['accuracy']:.4f}")
                print(f"    Precision: {metrics_imbalanced['precision']:.4f}")
                print(f"    Recall:    {metrics_imbalanced['recall']:.4f}")
                print(f"    F1-Score:  {metrics_imbalanced['f1_score']:.4f}")

                # Evaluate on BALANCED test set (the same balanced_test_df for all models)
                print(f"\n  Evaluating on balanced test set (strategy={balance_strategy})...")
                try:
                    y_pred_balanced = model.predict(X_balanced_scaled)
                except Exception as e:
                    print(f"  WARNING: model.predict failed on balanced X: {e}")
                    y_pred_balanced = np.zeros(len(X_balanced_scaled), dtype=int)

                all_predictions[model_name]['balanced'] = y_pred_balanced

                balanced_metrics = {
                    'accuracy': accuracy_score(y_balanced, y_pred_balanced),
                    'precision': precision_score(y_balanced, y_pred_balanced, zero_division=0),
                    'recall': recall_score(y_balanced, y_pred_balanced, zero_division=0),
                    'f1_score': f1_score(y_balanced, y_pred_balanced, zero_division=0)
                }

                print(f"  Balanced Results:")
                print(f"    Accuracy:  {balanced_metrics['accuracy']:.4f}")
                print(f"    Precision: {balanced_metrics['precision']:.4f}")
                print(f"    Recall:    {balanced_metrics['recall']:.4f}")
                print(f"    F1-Score:  {balanced_metrics['f1_score']:.4f}")

                # Store results (balanced metrics used as primary)
                result = {
                    'Attack': 'Combined',
                    'Model': model_name,
                    'Accuracy': balanced_metrics['accuracy'],
                    'Precision': balanced_metrics['precision'],
                    'Recall': balanced_metrics['recall'],
                    'F1-Score': balanced_metrics['f1_score'],
                    'Train Time (s)': metrics_imbalanced['train_time'],
                    'Inference Time (ms)': metrics_imbalanced['inference_time_ms'],
                    'Memory (MB)': metrics_imbalanced['memory_mb'],
                    # Also store imbalanced metrics for reference
                    'Imbalanced_Accuracy': metrics_imbalanced['accuracy'],
                    'Imbalanced_Precision': metrics_imbalanced['precision'],
                    'Imbalanced_Recall': metrics_imbalanced['recall'],
                    'Imbalanced_F1': metrics_imbalanced['f1_score']
                }
                results.append(result)

                # Save model artifact
                import joblib
                combined_model_dir = self.model_dir / 'combined'
                combined_model_dir.mkdir(exist_ok=True)
                model_path = combined_model_dir / f"{model_name.lower().replace(' ', '_')}.pkl"
                joblib.dump(model.model, model_path)

            except Exception as e:
                print(f"  ERROR: {str(e)}")
                import traceback
                traceback.print_exc()
                # ensure missing preds remain None to detect later
                all_predictions[model_name] = all_predictions.get(model_name, {'imbalanced': None, 'balanced': None})
                continue

        results_df = pd.DataFrame(results)
        # Return balanced test artifacts so caller and visualizer can use them
        return results_df, all_predictions, balanced_test_df, X_balanced_scaled, y_balanced

    
    def visualize_combined_results(self, results_df: pd.DataFrame, X_test, y_test, all_predictions):
        """Create visualizations for combined dataset."""
        print(f"\n  Generating visualizations for combined dataset...")
        
        # Create combined-specific directory
        combined_fig_dir = self.fig_dir / 'combined'
        combined_fig_dir.mkdir(exist_ok=True)
        
        # 1. Performance metrics comparison
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Combined Dataset - Model Performance ({self.mode.upper()} Mode)', 
                    fontsize=14, fontweight='bold')
        
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        for idx, metric in enumerate(metrics):
            ax = axes[idx // 2, idx % 2]
            data = results_df.sort_values(metric, ascending=False)
            
            bars = ax.barh(data['Model'], data[metric], color='steelblue', alpha=0.7)
            ax.set_xlabel(metric)
            ax.set_title(f'{metric} by Model', fontweight='bold')
            ax.set_xlim([0, 1])
            ax.grid(True, alpha=0.3, axis='x')
            
            # Add value labels
            for i, (bar, val) in enumerate(zip(bars, data[metric])):
                ax.text(val, i, f' {val:.3f}', va='center')
        
        plt.tight_layout()
        fig_path = combined_fig_dir / 'model_performance_comparison.png'
        plt.savefig(fig_path, bbox_inches='tight')
        print(f"    Saved: {fig_path}")
        
        if self.logger:
            self.logger.log_figure(fig, f"task3_{self.mode}/combined/model_performance")
        plt.close(fig)
        
        # 2. PCA Projections for each model
        self.visualize_attack_projections('combined', X_test, y_test, all_predictions, results_df)

    def generate_combined_summary(self, results_df: pd.DataFrame):
        """Generate summary for combined dataset."""
        print(f"\n{'='*70}")
        print(f"COMBINED DATASET - SUMMARY")
        print(f"{'='*70}")
        
        # Sort by F1-Score
        results_df = results_df.sort_values('F1-Score', ascending=False)
        
        print(f"\nCombined Dataset Model Comparison (Balanced Test Set):")
        display_cols = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score']
        print(results_df[display_cols].to_string(index=False))
        
        print(f"\nImbalanced Test Set Reference:")
        imbalanced_cols = ['Model', 'Imbalanced_Accuracy', 'Imbalanced_Precision', 
                        'Imbalanced_Recall', 'Imbalanced_F1']
        print(results_df[imbalanced_cols].to_string(index=False))
        
        # Save to CSV
        combined_table_dir = self.table_dir
        results_df.to_csv(combined_table_dir / 'combined_model_comparison.csv', index=False)
        
        # Best model
        best_model = results_df.iloc[0]
        print(f"\nBest Model for Combined Dataset: {best_model['Model']}")
        print(f"  F1-Score (Balanced): {best_model['F1-Score']:.4f}")
        print(f"  Accuracy (Balanced): {best_model['Accuracy']:.4f}")
        print(f"  F1-Score (Imbalanced): {best_model['Imbalanced_F1']:.4f}")
        
        return results_df


def run_task3(config: Dict[str, Any], logger=None) -> Dict:
    """
    Execute Task 3: Binary Intrusion Detection with per-attack training.
    
    Returns results for both CORE-only and FULL modes, with separate models per attack.
    """
    results = {}
    attack_types = ['replay', 'masquerade', 'injection', 'poisoning']
    
    # Run both modes
    for mode in ['core', 'full']:
        print(f"\n{'#'*70}")
        print(f"# STARTING {mode.upper()} MODE ANALYSIS (PER-ATTACK TRAINING)")
        print(f"{'#'*70}\n")
        
        task3 = Task3BinaryDetection(config, logger, mode=mode)
        
        all_attack_results = {}
        
        # Process each attack type separately
        for attack_type in attack_types:
            print(f"\n{'*'*70}")
            print(f"* PROCESSING {attack_type.upper()} ATTACK")
            print(f"{'*'*70}")
            
            # Load attack-specific data
            train_df, test_df = task3.load_attack_data(attack_type)
            
            # Preprocess
            X_train, y_train, X_test, y_test, features, scaler = task3.preprocess_attack_data(
                train_df, test_df, attack_type
            )
            
            # Train and evaluate all models for this attack
            results_df, all_predictions = task3.train_and_evaluate_attack(
                attack_type, X_train, y_train, X_test, y_test
            )
            
            # Visualize results
            task3.visualize_attack_results(
                attack_type, results_df, X_test, y_test, all_predictions
            )
            
            # Generate attack-specific summary
            results_df = task3.generate_attack_summary(attack_type, results_df)
            
            all_attack_results[attack_type] = results_df
            
            print(f"\n✓ Completed {attack_type} attack analysis")
        
        # ===== COMBINED TRAINING =====
        print(f"\n{'*'*70}")
        print("* PART 2: COMBINED TRAINING")
        print(f"{'*'*70}")

        # Load combined datasets (returns combined_train, combined_test DataFrames)
        train_combined, test_combined = load_combined_datasets(
            Path(config['data']['raw_dir']),
            config['data']['train_files'],
            config['data']['test_files'],
            mode=mode
        )

        # Preprocess combined data to get X_train_comb, y_train_comb, scaled X_test_comb etc.
        X_train_comb, y_train_comb, X_test_comb, y_test_comb, features_comb, scaler_comb = \
            task3.preprocess_attack_data(train_combined, test_combined, 'combined')

        # Train & evaluate on combined dataset — returns balanced test artifacts as well
        results_df_comb, all_predictions_comb, balanced_df_comb, X_bal_scaled, y_bal = \
            task3.train_and_evaluate_combined(
                X_train_comb, y_train_comb, X_test_comb, y_test_comb,
                test_combined, scaler_comb, features_comb, balance_strategy='equal_per_attack'
            )

        # Build a dict mapping model -> balanced predictions (visualizer expects predictions aligned with X_bal_scaled)
        all_predictions_balanced = {}
        for model_name, preds in all_predictions_comb.items():
            if isinstance(preds, dict) and preds.get('balanced') is not None:
                all_predictions_balanced[model_name] = preds['balanced']
            elif isinstance(preds, np.ndarray) and len(preds) == len(X_bal_scaled):
                all_predictions_balanced[model_name] = preds
            else:
                # fallback: attempt to predict now on the balanced X
                try:
                    # If you have the model object saved in disk one could reload; here we try to skip
                    print(f"Warning: No balanced preds for {model_name}; skipping projection for this model.")
                except Exception:
                    pass

        # Visualize using the BALANCED arrays (so plots match balanced metrics)
        task3.visualize_combined_results(
            results_df_comb, X_bal_scaled, y_bal, all_predictions_balanced
        )
        # Generate combined dataset summary
        results_df_comb = task3.generate_combined_summary(results_df_comb)
        
        # Add to results
        all_attack_results['combined'] = results_df_comb
        
        print(f"\n✓ Completed combined dataset analysis")
        
        # Generate cross-attack summary (excluding combined)
        attack_only_results = {k: v for k, v in all_attack_results.items() if k != 'combined'}
        summary_df = task3.generate_cross_attack_summary(attack_only_results)
        
        # Log to W&B
        if logger:
            logger.log_dataframe(summary_df, f"task3_{mode}/cross_attack_summary")
            logger.log_dataframe(results_df_comb, f"task3_{mode}/combined_summary")
        
        # Collect results for this mode
        results[mode] = {
            "summary": summary_df,
            "per_attack": all_attack_results,
            "combined": results_df_comb
        }

    print("\n" + "="*70)
    print("TASK 3 COMPLETE - BOTH MODES FINISHED")
    print("="*70)
    print("\nResults available in:")
    print("  - outputs/figures/task3_core/ and outputs/tables/task3_core/")
    print("  - outputs/figures/task3_full/ and outputs/tables/task3_full/")
    
    return results


if __name__ == '__main__':
    import yaml
    
    # Load config
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Run task
    results = run_task3(config, logger=None)