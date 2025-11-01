"""
Task 3: Unsupervised Binary Intrusion Detection
Compare multiple unsupervised models for binary classification (Normal vs Attack).
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


class BinaryDetector:
    """Base class for binary anomaly detectors."""
    
    def __init__(self, name: str, model, config: Dict = None):
        self.name = name
        self.model = model
        self.config = config or {}
        self.train_time = 0
        self.inference_time = 0
        
    def fit(self, X):
        """Train the model."""
        start = time.time()
        self.model.fit(X)
        self.train_time = time.time() - start
        return self
    
    def predict(self, X):
        """Predict anomalies."""
        start = time.time()
        predictions = self.model.predict(X)
        self.inference_time = (time.time() - start) / len(X)
        
        # Convert to binary (0=normal, 1=attack)
        # Most sklearn models use -1 for anomalies, 1 for normal
        if hasattr(self.model, 'predict'):
            if isinstance(self.model, (IsolationForest, OneClassSVM)):
                predictions = np.where(predictions == -1, 1, 0)
        
        return predictions
    
    def get_memory_mb(self):
        """Estimate memory footprint."""
        import sys
        return sys.getsizeof(self.model) / (1024 * 1024)


class Task3BinaryDetection:
    """Task 3: Binary Intrusion Detection."""
    
    def __init__(self, config: Dict[str, Any], logger=None):
        self.config = config
        self.logger = logger
        self.train_data = {}
        self.test_data = {}
        
        # Create output directories
        self.fig_dir = Path('outputs/figures/task3')
        self.table_dir = Path('outputs/tables/task3')
        self.model_dir = Path('outputs/models/task3')
        self.fig_dir.mkdir(parents=True, exist_ok=True)
        self.table_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.dpi'] = 300
        
    def load_data(self):
        """Load all datasets."""
        print("Loading datasets...")
        data_dir = Path(self.config['data']['raw_dir'])
        
        attack_types = ['replay', 'masquerade', 'injection', 'poisoning']
        
        for attack in attack_types:
            train_file = self.config['data']['train_files'][attack]
            test_file = self.config['data']['test_files'][attack]
            
            self.train_data[attack] = pd.read_csv(data_dir / train_file)
            self.test_data[attack] = pd.read_csv(data_dir / test_file)
            
            print(f"  {attack}: Train={len(self.train_data[attack])}, "
                  f"Test={len(self.test_data[attack])}")
        
        print(f"\nTotal train samples: {sum(len(df) for df in self.train_data.values())}")
        print(f"Total test samples: {sum(len(df) for df in self.test_data.values())}")
    
    def preprocess_data(self):
        """Preprocess and combine all data."""
        print("\nPreprocessing data...")
        
        # Combine all training data
        train_combined = pd.concat(list(self.train_data.values()), ignore_index=True)
        test_combined = pd.concat(list(self.test_data.values()), ignore_index=True)
        
        # Select numeric features only
        numeric_cols = train_combined.select_dtypes(include=[np.number]).columns
        numeric_cols = [c for c in numeric_cols if c not in ['attack']]
        
        # Extract features and labels
        X_train = train_combined[numeric_cols].copy()
        y_train = train_combined['attack'].values
        X_test = test_combined[numeric_cols].copy()
        y_test = test_combined['attack'].values
        
        # Clean data
        print("Cleaning data...")
        X_train = X_train.replace([np.inf, -np.inf], np.nan)
        X_test = X_test.replace([np.inf, -np.inf], np.nan)
        
        # Drop columns with too many NaN values (>50%)
        threshold = len(X_train) * 0.5
        valid_cols = X_train.columns[X_train.notna().sum() > threshold]
        X_train = X_train[valid_cols]
        X_test = X_test[valid_cols]
        
        # Fill remaining NaN values
        for col in X_train.columns:
            median_val = X_train[col].median()
            if pd.isna(median_val):
                median_val = 0
            X_train[col].fillna(median_val, inplace=True)
            X_test[col].fillna(median_val, inplace=True)
        
        # Drop columns with zero variance
        variances = X_train.var()
        valid_cols = variances[variances > 0].index
        X_train = X_train[valid_cols]
        X_test = X_test[valid_cols]
        
        print(f"Features after cleaning: {len(X_train.columns)}")
        print(f"Train samples: {len(X_train)}, Attack ratio: {y_train.mean():.2%}")
        print(f"Test samples: {len(X_test)}, Attack ratio: {y_test.mean():.2%}")
        
        # Standardize features
        print("Standardizing features...")
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Handle any remaining NaN after scaling
        X_train_scaled = np.nan_to_num(X_train_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        X_test_scaled = np.nan_to_num(X_test_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        
        return X_train_scaled, y_train, X_test_scaled, y_test
    
    def create_models(self):
        """Create all models to compare."""
        contamination = 0.1  # Approximate attack ratio
        
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
            'DBSCAN': BinaryDetector(
                'DBSCAN',
                DBSCAN(
                    eps=3.0,
                    min_samples=50,
                    n_jobs=-1
                )
            )
        }
        
        return models
    
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
    
    def train_and_evaluate_all(self, X_train, y_train, X_test, y_test):
        """Train and evaluate all models."""
        print("\n" + "="*70)
        print("TRAINING AND EVALUATING MODELS")
        print("="*70)
        
        models = self.create_models()
        results = []
        all_predictions = {}
        
        for model_name, model in models.items():
            print(f"\n{'='*60}")
            print(f"Model: {model_name}")
            print(f"{'='*60}")
            
            try:
                # Train
                print(f"Training {model_name}...")
                model.fit(X_train)
                print(f"Training time: {model.train_time:.2f}s")
                
                # Evaluate
                print(f"Evaluating {model_name}...")
                metrics = self.evaluate_model(model, X_test, y_test)
                all_predictions[model_name] = model.predict(X_test)
                
                # Store results
                result = {
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
                print(f"\nResults for {model_name}:")
                print(f"  Accuracy:  {metrics['accuracy']:.4f}")
                print(f"  Precision: {metrics['precision']:.4f}")
                print(f"  Recall:    {metrics['recall']:.4f}")
                print(f"  F1-Score:  {metrics['f1_score']:.4f}")
                print(f"  Train Time: {metrics['train_time']:.2f}s")
                print(f"  Inference: {metrics['inference_time_ms']:.2f}ms per sample")
                print(f"  Memory:    {metrics['memory_mb']:.2f}MB")
                
                # Log to W&B
                if self.logger:
                    self.logger.log_metrics({
                        f"task3/{model_name}/accuracy": metrics['accuracy'],
                        f"task3/{model_name}/precision": metrics['precision'],
                        f"task3/{model_name}/recall": metrics['recall'],
                        f"task3/{model_name}/f1_score": metrics['f1_score'],
                        f"task3/{model_name}/train_time": metrics['train_time'],
                        f"task3/{model_name}/inference_time_ms": metrics['inference_time_ms']
                    })
                    
                    # Log confusion matrix
                    self.logger.log_confusion_matrix(
                        y_test, all_predictions[model_name],
                        ['Normal', 'Attack'],
                        f"task3/{model_name}_confusion_matrix"
                    )
                
                # Save model
                import joblib
                model_path = self.model_dir / f"{model_name.lower().replace(' ', '_')}.pkl"
                joblib.dump(model.model, model_path)
                
            except Exception as e:
                print(f"\nError training {model_name}: {str(e)}")
                import traceback
                traceback.print_exc()
                continue
        
        return pd.DataFrame(results), all_predictions
    
    def visualize_results(self, results_df: pd.DataFrame):
        """Create visualization of model comparison."""
        print("\nGenerating visualizations...")
        
        # 1. Performance metrics comparison
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Model Performance Comparison', fontsize=14, fontweight='bold')
        
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
        fig_path = self.fig_dir / 'model_performance_comparison.png'
        plt.savefig(fig_path, bbox_inches='tight')
        print(f"Saved: {fig_path}")
        
        if self.logger:
            self.logger.log_figure(fig, "task3/model_performance_comparison")
        plt.close(fig)
        
        # 2. Performance vs Efficiency tradeoff
        fig, ax = plt.subplots(figsize=(10, 6))
        
        scatter = ax.scatter(
            results_df['Inference Time (ms)'],
            results_df['F1-Score'],
            s=results_df['Memory (MB)'] * 50,
            alpha=0.6,
            c=range(len(results_df)),
            cmap='viridis'
        )
        
        for idx, row in results_df.iterrows():
            ax.annotate(
                row['Model'],
                (row['Inference Time (ms)'], row['F1-Score']),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3)
            )
        
        ax.set_xlabel('Inference Time per Sample (ms)')
        ax.set_ylabel('F1-Score')
        ax.set_title('Performance vs Efficiency Tradeoff\n(bubble size = memory footprint)',
                    fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        fig_path = self.fig_dir / 'performance_efficiency_tradeoff.png'
        plt.savefig(fig_path, bbox_inches='tight')
        print(f"Saved: {fig_path}")
        
        if self.logger:
            self.logger.log_figure(fig, "task3/performance_efficiency_tradeoff")
        plt.close(fig)
        
        # 3. Computational metrics
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle('Computational Performance', fontsize=14, fontweight='bold')
        
        # Training time
        data = results_df.sort_values('Train Time (s)', ascending=False)
        axes[0].barh(data['Model'], data['Train Time (s)'], color='coral', alpha=0.7)
        axes[0].set_xlabel('Training Time (seconds)')
        axes[0].set_title('Training Time', fontweight='bold')
        axes[0].grid(True, alpha=0.3, axis='x')
        
        # Inference time
        data = results_df.sort_values('Inference Time (ms)', ascending=False)
        axes[1].barh(data['Model'], data['Inference Time (ms)'], color='lightgreen', alpha=0.7)
        axes[1].set_xlabel('Inference Time (ms/sample)')
        axes[1].set_title('Inference Time', fontweight='bold')
        axes[1].grid(True, alpha=0.3, axis='x')
        
        # Memory footprint
        data = results_df.sort_values('Memory (MB)', ascending=False)
        axes[2].barh(data['Model'], data['Memory (MB)'], color='skyblue', alpha=0.7)
        axes[2].set_xlabel('Memory Footprint (MB)')
        axes[2].set_title('Memory Usage', fontweight='bold')
        axes[2].grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        fig_path = self.fig_dir / 'computational_performance.png'
        plt.savefig(fig_path, bbox_inches='tight')
        print(f"Saved: {fig_path}")
        
        if self.logger:
            self.logger.log_figure(fig, "task3/computational_performance")
        plt.close(fig)
    
    def generate_summary(self, results_df: pd.DataFrame):
        """Generate summary report."""
        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)
        
        # Sort by F1-Score
        results_df = results_df.sort_values('F1-Score', ascending=False)
        
        print("\nModel Comparison Results:")
        print(results_df.to_string(index=False))
        
        # Save to CSV
        results_df.to_csv(self.table_dir / 'model_comparison.csv', index=False)
        print(f"\nResults saved to: {self.table_dir / 'model_comparison.csv'}")
        
        # Best model
        best_model = results_df.iloc[0]
        print(f"\n{'='*70}")
        print("BEST MODEL")
        print(f"{'='*70}")
        print(f"Model: {best_model['Model']}")
        print(f"F1-Score: {best_model['F1-Score']:.4f}")
        print(f"Accuracy: {best_model['Accuracy']:.4f}")
        print(f"Precision: {best_model['Precision']:.4f}")
        print(f"Recall: {best_model['Recall']:.4f}")
        print(f"Inference Time: {best_model['Inference Time (ms)']:.2f}ms per sample")
        
        # Recommendations
        print(f"\n{'='*70}")
        print("RECOMMENDATIONS FOR REAL-TIME DEPLOYMENT")
        print(f"{'='*70}")
        
        # Best for accuracy
        best_accuracy = results_df.loc[results_df['Accuracy'].idxmax()]
        print(f"\nBest Accuracy: {best_accuracy['Model']} ({best_accuracy['Accuracy']:.4f})")
        
        # Best for speed
        best_speed = results_df.loc[results_df['Inference Time (ms)'].idxmin()]
        print(f"Fastest: {best_speed['Model']} ({best_speed['Inference Time (ms)']:.2f}ms)")
        
        # Best balanced (F1-Score / Inference Time ratio)
        results_df['efficiency_score'] = results_df['F1-Score'] / (results_df['Inference Time (ms)'] + 1)
        best_balanced = results_df.loc[results_df['efficiency_score'].idxmax()]
        print(f"Best Balanced: {best_balanced['Model']} "
              f"(F1={best_balanced['F1-Score']:.4f}, Time={best_balanced['Inference Time (ms)']:.2f}ms)")
        
        return results_df


def run_task3(config: Dict[str, Any], logger=None) -> Dict:
    """
    Execute Task 3: Binary Intrusion Detection.
    
    Args:
        config: Configuration dictionary
        logger: WandbLogger instance (optional)
    
    Returns:
        Dictionary with results
    """
    task3 = Task3BinaryDetection(config, logger)
    
    # Load data
    task3.load_data()
    
    # Preprocess
    X_train, y_train, X_test, y_test = task3.preprocess_data()
    
    # Train and evaluate all models
    results_df, all_predictions = task3.train_and_evaluate_all(X_train, y_train, X_test, y_test)
    
    # Visualize results
    task3.visualize_results(results_df)
    
    # Generate summary
    results_df = task3.generate_summary(results_df)
    
    # Log to W&B
    if logger:
        logger.log_dataframe(results_df[['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score']], 
                            "task3/model_comparison")
    
    print("\n✓ Task 3 (Binary Intrusion Detection) completed successfully!")
    print(f"All outputs saved to outputs/figures/task3/ and outputs/tables/task3/")
    
    best_model = results_df.iloc[0]
    return {
        'status': 'completed',
        'best_model': best_model['Model'],
        'best_f1_score': float(best_model['F1-Score']),
        'best_accuracy': float(best_model['Accuracy']),
        'results_table': results_df,
        'all_predictions': all_predictions
    }


if __name__ == '__main__':
    import yaml
    
    # Load config
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Run task
    results = run_task3(config, logger=None)