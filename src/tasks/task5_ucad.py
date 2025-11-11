"""
Task 5: Advanced Analysis with UCAD
Replaces LSTM Autoencoder with UCAD for unsupervised cyberattack detection
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any, List, Tuple

# Advanced analysis libraries
import networkx as nx
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.metrics import (
    confusion_matrix, classification_report, 
    roc_curve, auc, precision_recall_curve
)

import torch

# Import UCAD
import sys
sys.path.append('src/models')
from src.models.ucad import UCADDetector

# Import preprocessing
from src.preprocessing import (
    load_combined_datasets
)

import warnings
warnings.filterwarnings('ignore')

# ============================================================
# UCAD FEATURE CONFIGURATION
# ============================================================

UCAD_FEATURES = [
    # Timing features
    'inter_arrival', 'jitter_rolling_std', 'msg_rate', 'byte_rate',
    
    # Protocol compliance
    'ttl_violation', 'ttl_margin', 'sqNum_jump', 'stNum_change',
    
    # Length dynamics
    'length_delta', 'Length', 'Frame length on the wire',
    
    # GOOSE rule violations
    'sq_reset_violation', 't_st_consistency_violation',
    'status_change_missing_on_event', 'heartbeat_interval_error',
    'sq_inc_violation_when_no_event', 'status_change_violation_on_heartbeat',
    'stNum_jump_gt1', 'sq_reset_without_st_change',
    
    # Event context
    'event_msg_rank'
]


# ============================================================
# ADVANCED ANALYZER WITH UCAD
# ============================================================

class AdvancedAnalyzerUCAD:
    """
    Advanced anomaly detection using UCAD (Unsupervised Cyberattack Detection).
    """
    def __init__(self, config: Dict[str, Any], logger=None):
        self.config = config
        self.logger = logger
        self.data_train: pd.DataFrame = None
        self.data_test: pd.DataFrame = None
        self.ucad_model: UCADDetector = None
        self.scaler: StandardScaler = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create output directories
        self.fig_dir = Path('outputs/figures/task5_ucad')
        self.table_dir = Path('outputs/tables/task5_ucad')
        self.model_dir = Path('outputs/models/task5_ucad')
        self.fig_dir.mkdir(parents=True, exist_ok=True)
        self.table_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # Style
        sns.set_style("whitegrid")
        plt.rcParams['figure.dpi'] = 300
        
        print(f"\n{'='*70}")
        print("RUNNING TASK 5: ADVANCED ANALYSES WITH UCAD")
        print(f"Using device: {self.device}")
        print(f"{'='*70}\n")

    def load_and_prepare_data(self):
        """Load and preprocess data using the centralized function."""
        print("Loading and preparing data...")
        
        train_df, test_df = load_combined_datasets(
            data_dir=self.config['data']['raw_dir'],
            train_files=self.config['data']['train_files'],
            test_files=self.config['data'].get('test_files', {}),
            mode='full',  # Full mode for all engineered features
        )
        
        self.data_train = train_df
        self.data_test = test_df
        
        if 'base_time' in self.data_train.columns:
            self.data_train.sort_values('base_time', inplace=True)
        if 'base_time' in self.data_test.columns:
            self.data_test.sort_values('base_time', inplace=True)
        
        print(f"Data loaded successfully:")
        print(f"  Train: {len(self.data_train)} samples, "
              f"Attack ratio: {self.data_train['attack'].mean():.2%}")
        print(f"  Test:  {len(self.data_test)} samples, "
              f"Attack ratio: {self.data_test['attack'].mean():.2%}")

    def prepare_ucad_features(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare features for UCAD training and detection.
        
        Returns:
            X_train, X_test, y_train, y_test
        """
        print("\nPreparing UCAD features...")
        
        # Check which features are available
        available_features = [f for f in UCAD_FEATURES if f in self.data_train.columns]
        missing_features = [f for f in UCAD_FEATURES if f not in self.data_train.columns]
        
        print(f"  Available features: {len(available_features)}/{len(UCAD_FEATURES)}")
        if missing_features:
            print(f"  Missing features: {missing_features}")
        
        # Handle missing values and inf
        for col in available_features:
            self.data_train[col] = self.data_train[col].fillna(0).replace([np.inf, -np.inf], 0)
            self.data_test[col] = self.data_test[col].fillna(0).replace([np.inf, -np.inf], 0)
        
        # Extract features and labels
        X_train = self.data_train[available_features].values
        X_test = self.data_test[available_features].values
        y_train = self.data_train['attack'].values
        y_test = self.data_test['attack'].values
        
        # Standardize features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"  Train shape: {X_train_scaled.shape}")
        print(f"  Test shape:  {X_test_scaled.shape}")
        print(f"  Feature names: {available_features}")
        
        # Store for later use
        self.feature_names = available_features
        
        return X_train_scaled, X_test_scaled, y_train, y_test

    # ==================== 1. UCAD DETECTION ====================

    def ucad_detection(
        self, 
        epochs: int = 50, 
        batch_size: int = 128,
        contamination_rate: float = 0.076
    ):
        """
        Train and evaluate UCAD for cyberattack detection.
        """
        print("\n" + "="*70)
        print("1. UCAD: UNSUPERVISED CYBERATTACK DETECTION")
        print("="*70)
        
        model_path = self.model_dir / 'ucad_model.pth'
        
        # Prepare data
        X_train, X_test, y_train, y_test = self.prepare_ucad_features()
        
        # Initialize UCAD
        self.ucad_model = UCADDetector(
            input_dim=X_train.shape[1],
            latent_dim=32,
            hidden_dims_encoder=[64, 128],
            hidden_dims_decoder=[128, 64],
            transformer_layers=2,
            transformer_heads=4,
            transformer_dim_feedforward=128,
            dropout=0.1,
            contamination_rate=contamination_rate,
            device=str(self.device)
        )
        
        if model_path.exists():
            print(f"\nLoading existing model from {model_path}")
            self.ucad_model.load_model(str(model_path))
        else:
            # Train UCAD
            print("\nTraining UCAD model...")
            self.ucad_model.train_ucad(
                X_train=X_train,
                epochs=epochs,
                batch_size=batch_size,
                lr_dae=1e-3,
                lr_classifier=1e-3,
                patience=7,
                verbose=True
            )
            
            # Save model
            self.ucad_model.save_model(str(model_path))
            print(f"Model saved to {model_path}")
            
            # Plot training history
            self._plot_training_history()
        
        # ========== DETECTION ON TEST SET ==========
        print("\nPerforming detection on test set...")
        y_pred, log_probs = self.ucad_model.detect(X_test)
        
        # Calculate metrics
        self._evaluate_detection(y_test, y_pred, log_probs)
        
        # Visualize results
        self._visualize_detection_results(y_test, y_pred, log_probs, X_test)
        
        # Analyze reconstruction errors
        self._analyze_reconstruction_errors(X_train, X_test, y_train, y_test)

    # ==================== 1B. PER-ATTACK UCAD DETECTION ====================

    def ucad_detection_per_attack(
        self, 
        epochs: int = 50, 
        batch_size: int = 128,
        contamination_rate: float = 0.076
    ):
        """
        Train and evaluate UCAD separately for each attack type.
        Compares per-attack performance vs combined detection.
        """
        print("\n" + "="*70)
        print("1B. UCAD: PER-ATTACK TYPE DETECTION")
        print("="*70)
        
        attack_types = list(self.config['data']['train_files'].keys())
        per_attack_results = {}
        
        for attack_type in attack_types:
            print(f"\n{'='*70}")
            print(f"Training UCAD for: {attack_type.upper()}")
            print(f"{'='*70}")
            
            # Load data for this specific attack
            from src.preprocessing import load_and_preprocess
            from pathlib import Path
            
            data_dir = Path(self.config['data']['raw_dir'])
            train_file = data_dir / self.config['data']['train_files'][attack_type]
            test_file = data_dir / self.config['data']['test_files'][attack_type]
            
            # Load and preprocess
            train_df = load_and_preprocess(str(train_file), mode='full')
            test_df = load_and_preprocess(str(test_file), mode='full')
            
            print(f"  Train: {len(train_df)} samples, "
                  f"Attack ratio: {train_df['attack'].mean():.2%}")
            print(f"  Test:  {len(test_df)} samples, "
                  f"Attack ratio: {test_df['attack'].mean():.2%}")
            
            # Prepare features
            available_features = [f for f in UCAD_FEATURES if f in train_df.columns]
            
            for col in available_features:
                train_df[col] = train_df[col].fillna(0).replace([np.inf, -np.inf], 0)
                test_df[col] = test_df[col].fillna(0).replace([np.inf, -np.inf], 0)
            
            X_train = train_df[available_features].values
            X_test = test_df[available_features].values
            y_train = train_df['attack'].values
            y_test = test_df['attack'].values
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Calculate attack-specific contamination rate
            attack_contamination = y_train.mean()
            print(f"  Calculated contamination rate: {attack_contamination:.2%}")
            
            # Initialize UCAD for this attack
            model_path = self.model_dir / f'ucad_model_{attack_type}.pth'
            
            ucad_model = UCADDetector(
                input_dim=X_train_scaled.shape[1],
                latent_dim=32,
                hidden_dims_encoder=[64, 128],
                hidden_dims_decoder=[128, 64],
                transformer_layers=2,
                transformer_heads=4,
                transformer_dim_feedforward=128,
                dropout=0.1,
                contamination_rate=attack_contamination,
                device=str(self.device)
            )
            
            if model_path.exists():
                print(f"  Loading existing model from {model_path}")
                ucad_model.load_model(str(model_path))
            else:
                # Train UCAD
                print(f"  Training UCAD for {attack_type}...")
                ucad_model.train_ucad(
                    X_train=X_train_scaled,
                    epochs=epochs,
                    batch_size=batch_size,
                    lr_dae=1e-3,
                    lr_classifier=1e-3,
                    patience=7,
                    verbose=True
                )
                
                # Save model
                ucad_model.save_model(str(model_path))
                print(f"  Model saved to {model_path}")
            
            # Detection on test set
            print(f"\n  Performing detection for {attack_type}...")
            y_pred, log_probs = ucad_model.detect(X_test_scaled)
            
            # Calculate metrics
            from sklearn.metrics import (
                accuracy_score, precision_score, recall_score, 
                f1_score, roc_auc_score, classification_report
            )
            
            probs_attack = np.exp(log_probs[:, 1])
            
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            
            try:
                roc_auc = roc_auc_score(y_test, probs_attack)
            except:
                roc_auc = 0.0
            
            # Store results
            per_attack_results[attack_type] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'roc_auc': roc_auc,
                'y_test': y_test,
                'y_pred': y_pred,
                'probs': probs_attack,
                'model': ucad_model,
                'scaler': scaler,
                'features': available_features
            }
            
            print(f"\n  {attack_type.upper()} Results:")
            print(f"    Accuracy:  {accuracy:.4f}")
            print(f"    Precision: {precision:.4f}")
            print(f"    Recall:    {recall:.4f}")
            print(f"    F1-Score:  {f1:.4f}")
            print(f"    ROC AUC:   {roc_auc:.4f}")
        
        # Create comparison visualization
        self._visualize_per_attack_comparison(per_attack_results)
        
        # Save comparison table
        self._save_per_attack_results(per_attack_results)
        
        return per_attack_results
    
    def _visualize_per_attack_comparison(self, results: Dict):
        """Create comparison visualizations for per-attack detection."""
        print("\nCreating per-attack comparison visualizations...")
        
        # Prepare data
        attack_types = list(results.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        
        data_for_plot = {metric: [results[at][metric] for at in attack_types] 
                         for metric in metrics}
        
        # Create comparison plot
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('UCAD Per-Attack Detection Performance Comparison', 
                     fontsize=16, fontweight='bold')
        
        # Plot each metric
        for idx, metric in enumerate(metrics):
            ax = axes[idx // 3, idx % 3]
            
            bars = ax.bar(range(len(attack_types)), data_for_plot[metric], 
                          color=['skyblue', 'lightcoral', 'lightgreen', 'khaki'])
            ax.set_xticks(range(len(attack_types)))
            ax.set_xticklabels(attack_types, rotation=45, ha='right')
            ax.set_ylabel(metric.replace('_', ' ').title())
            ax.set_title(f'{metric.replace("_", " ").title()} by Attack Type', 
                        fontweight='bold')
            ax.set_ylim(0, 1.0)
            ax.grid(axis='y', alpha=0.3)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}',
                       ha='center', va='bottom', fontsize=9)
        
        # Confusion matrices comparison
        ax = axes[1, 2]
        for idx, attack_type in enumerate(attack_types):
            y_test = results[attack_type]['y_test']
            y_pred = results[attack_type]['y_pred']
            
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(y_test, y_pred)
            
            # Normalize
            cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            
            # Plot as text
            ax.text(0.1, 0.9 - idx*0.2, f"{attack_type}:", 
                   fontweight='bold', transform=ax.transAxes)
            ax.text(0.3, 0.9 - idx*0.2, 
                   f"TN:{cm[0,0]} FP:{cm[0,1]} FN:{cm[1,0]} TP:{cm[1,1]}", 
                   fontsize=9, transform=ax.transAxes)
        
        ax.set_title('Confusion Matrices', fontweight='bold')
        ax.axis('off')
        
        plt.tight_layout()
        fig_path = self.fig_dir / 'per_attack_comparison.png'
        plt.savefig(fig_path, bbox_inches='tight')
        print(f"Saved per-attack comparison to {fig_path}")
        if self.logger:
            self.logger.log_figure(fig, "task5_ucad/per_attack_comparison")
        plt.close()
        
        # ROC curves comparison
        fig, ax = plt.subplots(figsize=(10, 8))
        
        from sklearn.metrics import roc_curve, auc
        colors = ['blue', 'red', 'green', 'orange']
        
        for idx, attack_type in enumerate(attack_types):
            y_test = results[attack_type]['y_test']
            probs = results[attack_type]['probs']
            
            if len(np.unique(y_test)) > 1:
                fpr, tpr, _ = roc_curve(y_test, probs)
                roc_auc = auc(fpr, tpr)
                
                ax.plot(fpr, tpr, color=colors[idx], lw=2,
                       label=f'{attack_type} (AUC = {roc_auc:.3f})')
        
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
               label='Random Classifier')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontweight='bold')
        ax.set_ylabel('True Positive Rate', fontweight='bold')
        ax.set_title('ROC Curves - Per-Attack Type Comparison', 
                    fontweight='bold', fontsize=14)
        ax.legend(loc="lower right")
        ax.grid(alpha=0.3)
        
        fig_path = self.fig_dir / 'per_attack_roc_comparison.png'
        plt.savefig(fig_path, bbox_inches='tight')
        print(f"Saved ROC comparison to {fig_path}")
        if self.logger:
            self.logger.log_figure(fig, "task5_ucad/per_attack_roc")
        plt.close()
    
    def _save_per_attack_results(self, results: Dict):
        """Save per-attack results to CSV."""
        print("\nSaving per-attack results table...")
        
        rows = []
        for attack_type, metrics in results.items():
            rows.append({
                'Attack Type': attack_type.upper(),
                'Accuracy': f"{metrics['accuracy']:.4f}",
                'Precision': f"{metrics['precision']:.4f}",
                'Recall': f"{metrics['recall']:.4f}",
                'F1-Score': f"{metrics['f1']:.4f}",
                'ROC AUC': f"{metrics['roc_auc']:.4f}"
            })
        
        df_results = pd.DataFrame(rows)
        
        # Calculate average
        avg_row = {
            'Attack Type': 'AVERAGE',
            'Accuracy': f"{df_results['Accuracy'].astype(float).mean():.4f}",
            'Precision': f"{df_results['Precision'].astype(float).mean():.4f}",
            'Recall': f"{df_results['Recall'].astype(float).mean():.4f}",
            'F1-Score': f"{df_results['F1-Score'].astype(float).mean():.4f}",
            'ROC AUC': f"{df_results['ROC AUC'].astype(float).mean():.4f}"
        }
        df_results = pd.concat([df_results, pd.DataFrame([avg_row])], ignore_index=True)
        
        # Save
        table_path = self.table_dir / 'per_attack_detection_results.csv'
        df_results.to_csv(table_path, index=False)
        print(f"Saved results table to {table_path}")
        
        # Print to console
        print("\n" + "="*70)
        print("PER-ATTACK DETECTION RESULTS SUMMARY")
        print("="*70)
        print(df_results.to_string(index=False))
        print("="*70)
        
        if self.logger:
            self.logger.log_dataframe(df_results, "task5_ucad/per_attack_results")

    def _plot_training_history(self):
        """Plot UCAD training curves."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Reconstruction loss
        axes[0].plot(self.ucad_model.history['recon_loss'], label='Reconstruction Loss', linewidth=2)
        axes[0].set_title('DAE Reconstruction Loss', fontweight='bold', fontsize=12)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('MSE Loss')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        # Classification loss
        axes[1].plot(self.ucad_model.history['cls_loss'], label='Classification Loss', 
                     color='orange', linewidth=2)
        axes[1].set_title('Pseudo-Classifier Loss', fontweight='bold', fontsize=12)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('NLL Loss')
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        fig_path = self.fig_dir / 'ucad_training_history.png'
        plt.savefig(fig_path)
        print(f"Saved training history to {fig_path}")
        if self.logger:
            self.logger.log_figure(fig, "task5_ucad/training_history")
        plt.close()

    def _evaluate_detection(self, y_true, y_pred, log_probs):
        """Evaluate detection performance."""
        print("\n" + "-"*70)
        print("DETECTION PERFORMANCE")
        print("-"*70)
        
        # Classification report
        report = classification_report(y_true, y_pred, target_names=['Normal', 'Attack'])
        print(report)
        
        # Save report
        report_path = self.table_dir / 'detection_report.txt'
        with open(report_path, 'w') as f:
            f.write("UCAD Detection Report\n")
            f.write("="*70 + "\n\n")
            f.write(report)
        print(f"Saved report to {report_path}")

    def _visualize_detection_results(self, y_true, y_pred, log_probs, X_test):
        """Create comprehensive detection visualizations."""
        
        # 1. Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        fig.suptitle('UCAD Detection Analysis', fontsize=16, fontweight='bold')
        
        # Confusion matrix
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0],
                    xticklabels=['Normal', 'Attack'], yticklabels=['Normal', 'Attack'])
        axes[0, 0].set_title('Confusion Matrix', fontweight='bold')
        axes[0, 0].set_ylabel('True Label')
        axes[0, 0].set_xlabel('Predicted Label')
        
        # ROC Curve
        probs_attack = np.exp(log_probs[:, 1])  # Convert log probs to probs
        fpr, tpr, _ = roc_curve(y_true, probs_attack)
        roc_auc = auc(fpr, tpr)
        
        axes[0, 1].plot(fpr, tpr, color='darkorange', lw=2, 
                       label=f'ROC curve (AUC = {roc_auc:.3f})')
        axes[0, 1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        axes[0, 1].set_xlim([0.0, 1.0])
        axes[0, 1].set_ylim([0.0, 1.05])
        axes[0, 1].set_xlabel('False Positive Rate')
        axes[0, 1].set_ylabel('True Positive Rate')
        axes[0, 1].set_title('ROC Curve', fontweight='bold')
        axes[0, 1].legend(loc="lower right")
        axes[0, 1].grid(alpha=0.3)
        
        # Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(y_true, probs_attack)
        pr_auc = auc(recall, precision)
        
        axes[1, 0].plot(recall, precision, color='green', lw=2,
                       label=f'PR curve (AUC = {pr_auc:.3f})')
        axes[1, 0].set_xlabel('Recall')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].set_title('Precision-Recall Curve', fontweight='bold')
        axes[1, 0].legend(loc="lower left")
        axes[1, 0].grid(alpha=0.3)
        
        # Prediction confidence distribution
        axes[1, 1].hist(probs_attack[y_true == 0], bins=50, alpha=0.6, 
                       label='Normal', color='skyblue', density=True)
        axes[1, 1].hist(probs_attack[y_true == 1], bins=50, alpha=0.6,
                       label='Attack', color='salmon', density=True)
        axes[1, 1].set_xlabel('Attack Probability')
        axes[1, 1].set_ylabel('Density')
        axes[1, 1].set_title('Prediction Confidence Distribution', fontweight='bold')
        axes[1, 1].legend()
        axes[1, 1].grid(alpha=0.3)
        
        plt.tight_layout()
        fig_path = self.fig_dir / 'ucad_detection_analysis.png'
        plt.savefig(fig_path)
        print(f"Saved detection analysis to {fig_path}")
        if self.logger:
            self.logger.log_figure(fig, "task5_ucad/detection_analysis")
        plt.close()

    def _analyze_reconstruction_errors(self, X_train, X_test, y_train, y_test):
        """Analyze reconstruction error distributions."""
        print("\nAnalyzing reconstruction errors...")
        
        # Get reconstruction errors
        train_errors = self.ucad_model.get_reconstruction_errors(X_train)
        test_errors = self.ucad_model.get_reconstruction_errors(X_test)
        
        # Calculate threshold based on contamination rate
        threshold_idx = int(len(train_errors) * (1 - self.ucad_model.contamination_rate))
        sorted_train_errors = np.sort(train_errors)
        threshold = sorted_train_errors[threshold_idx]
        
        print(f"  Contamination-based threshold: {threshold:.6f}")
        
        # Visualization
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('Reconstruction Error Analysis', fontsize=14, fontweight='bold')
        
        # Train error distribution
        axes[0].hist(train_errors[y_train == 0], bins=50, alpha=0.6, 
                    label='Normal', color='skyblue', density=True)
        axes[0].hist(train_errors[y_train == 1], bins=50, alpha=0.6,
                    label='Attack', color='salmon', density=True)
        axes[0].axvline(threshold, color='red', linestyle='--', linewidth=2,
                       label=f'Threshold (c={self.ucad_model.contamination_rate:.1%})')
        axes[0].set_xlabel('Reconstruction Error')
        axes[0].set_ylabel('Density')
        axes[0].set_title('Training Set', fontweight='bold')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        # Test error distribution
        axes[1].hist(test_errors[y_test == 0], bins=50, alpha=0.6,
                    label='Normal', color='skyblue', density=True)
        axes[1].hist(test_errors[y_test == 1], bins=50, alpha=0.6,
                    label='Attack', color='salmon', density=True)
        axes[1].axvline(threshold, color='red', linestyle='--', linewidth=2,
                       label=f'Threshold (c={self.ucad_model.contamination_rate:.1%})')
        axes[1].set_xlabel('Reconstruction Error')
        axes[1].set_ylabel('Density')
        axes[1].set_title('Test Set', fontweight='bold')
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        fig_path = self.fig_dir / 'reconstruction_error_analysis.png'
        plt.savefig(fig_path)
        print(f"Saved reconstruction error analysis to {fig_path}")
        if self.logger:
            self.logger.log_figure(fig, "task5_ucad/reconstruction_errors")
        plt.close()
        
        # Statistics
        print("\n  Reconstruction Error Statistics:")
        print(f"    Train Normal - Mean: {train_errors[y_train==0].mean():.6f}, "
              f"Std: {train_errors[y_train==0].std():.6f}")
        print(f"    Train Attack - Mean: {train_errors[y_train==1].mean():.6f}, "
              f"Std: {train_errors[y_train==1].std():.6f}")
        print(f"    Test Normal  - Mean: {test_errors[y_test==0].mean():.6f}, "
              f"Std: {test_errors[y_test==0].std():.6f}")
        print(f"    Test Attack  - Mean: {test_errors[y_test==1].mean():.6f}, "
              f"Std: {test_errors[y_test==1].std():.6f}")

    # ==================== 2. GRAPH-BASED ANALYSIS ====================

    def graph_based_analysis(self):
        """Model network traffic as a graph to identify anomalous communication patterns."""
        print("\n" + "="*70)
        print("2. GRAPH-BASED ANOMALY ANALYSIS")
        print("="*70)

        if 'Source' not in self.data_train.columns or 'Destination' not in self.data_train.columns:
            print("Skipping graph analysis: Source/destination MAC addresses not found.")
            return

        # Use combined train+test for comprehensive graph analysis
        data_combined = pd.concat([self.data_train, self.data_test], ignore_index=True)
        
        # Separate normal and attack dataframes
        df_normal = data_combined[data_combined['attack'] == 0]
        df_attack = data_combined[data_combined['attack'] == 1]

        # Create graphs
        G_normal = nx.from_pandas_edgelist(df_normal, 'Source', 'Destination', create_using=nx.DiGraph())
        G_attack = nx.from_pandas_edgelist(df_attack, 'Source', 'Destination', create_using=nx.DiGraph())

        # Calculate degree centrality
        centrality_normal = nx.degree_centrality(G_normal)
        centrality_attack = nx.degree_centrality(G_attack)

        # Convert to DataFrame for comparison
        df_centrality = pd.DataFrame([centrality_normal, centrality_attack]).T
        df_centrality.columns = ['Normal Centrality', 'Attack Centrality']
        df_centrality.fillna(0, inplace=True)
        df_centrality['Centrality Change'] = df_centrality['Attack Centrality'] - df_centrality['Normal Centrality']
        df_centrality.sort_values('Centrality Change', ascending=False, inplace=True)

        print("\nTop 5 nodes with largest increase in centrality during attacks:")
        print(df_centrality.head().to_string())
        
        out_path = self.table_dir / 'graph_centrality_analysis.csv'
        df_centrality.to_csv(out_path)
        print(f"Saved centrality analysis to: {out_path}")
        if self.logger:
            self.logger.log_dataframe(df_centrality, "task5_ucad/graph_centrality")

        # Visualize the graphs
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        fig.suptitle('Network Communication Graph Analysis', fontsize=16, fontweight='bold')
        
        pos_normal = nx.spring_layout(G_normal, seed=42)
        nx.draw(G_normal, pos_normal, ax=axes[0], with_labels=True, node_size=500, 
               node_color='skyblue', font_size=8, arrows=True)
        axes[0].set_title('Normal Communication Graph', fontweight='bold')
        
        pos_attack = nx.spring_layout(G_attack, seed=42)
        nx.draw(G_attack, pos_attack, ax=axes[1], with_labels=True, node_size=500,
               node_color='salmon', font_size=8, arrows=True)
        axes[1].set_title('Attack Communication Graph', fontweight='bold')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig_path = self.fig_dir / 'communication_graphs.png'
        plt.savefig(fig_path)
        print(f"Saved graph visualizations to {fig_path}")
        if self.logger:
            self.logger.log_figure(fig, "task5_ucad/communication_graphs")
        plt.close(fig)

    # ==================== 3. LATENT SPACE VISUALIZATION ====================

    def visualize_latent_space(self):
        """Visualize UCAD's latent space using t-SNE."""
        print("\n" + "="*70)
        print("3. VISUALIZATION OF UCAD LATENT SPACE (t-SNE)")
        print("="*70)

        if self.ucad_model is None:
            print("Skipping latent space visualization: UCAD model not trained.")
            return

        # Prepare data
        X_train, X_test, y_train, y_test = self.prepare_ucad_features()
        
        # Use a sample for faster t-SNE (optional)
        sample_size = min(5000, len(X_test))
        if len(X_test) > sample_size:
            print(f"Sampling {sample_size} points for faster t-SNE...")
            indices = np.random.choice(len(X_test), sample_size, replace=False)
            X_sample = X_test[indices]
            y_sample = y_test[indices]
        else:
            X_sample = X_test
            y_sample = y_test

        # Get latent representations
        print("Extracting latent representations...")
        latent_vectors = self.ucad_model.get_latent_representations(X_sample)
        
        print(f"Generated {latent_vectors.shape[0]} latent vectors of dimension {latent_vectors.shape[1]}")

        # t-SNE
        print("Running t-SNE... (this may take a moment)")
        perplexity = min(30.0, len(latent_vectors) - 1.0)
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, max_iter=1000)
        tsne_results = tsne.fit_transform(latent_vectors)
        
        # Create DataFrame
        df_tsne = pd.DataFrame({
            'tsne-1': tsne_results[:, 0],
            'tsne-2': tsne_results[:, 1],
            'label': ['Attack' if l == 1 else 'Normal' for l in y_sample]
        })

        # Plot
        plt.figure(figsize=(10, 8))
        sns.scatterplot(
            x="tsne-1", y="tsne-2",
            hue="label",
            palette={"Normal": "skyblue", "Attack": "salmon"},
            data=df_tsne,
            legend="full",
            alpha=0.6,
            s=50
        )
        plt.title('t-SNE Visualization of UCAD Latent Space', fontweight='bold', fontsize=14)
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        plt.grid(alpha=0.3)
        
        fig_path = self.fig_dir / 'tsne_latent_space.png'
        plt.savefig(fig_path, bbox_inches='tight')
        print(f"Saved t-SNE visualization to {fig_path}")
        if self.logger:
            self.logger.log_figure(plt.gcf(), "task5_ucad/tsne_latent_space")
        plt.close()
        
        return df_tsne


# ============================================================
# RUNNER FUNCTION
# ============================================================

def run_task5_ucad(config: Dict[str, Any], logger=None, mode: str = 'both') -> Dict:
    """
    Execute Task 5: Advanced Analyses with UCAD.
    
    Args:
        config: Configuration dictionary
        logger: Optional logger
        mode: 'combined', 'per_attack', or 'both'
            - 'combined': Train on combined dataset only
            - 'per_attack': Train separate model for each attack type
            - 'both': Run both analyses (default)
    
    Returns:
        Dictionary with results
    """
    results = {}
    
    analyzer = AdvancedAnalyzerUCAD(config, logger)
    
    # Load and prepare data
    analyzer.load_and_prepare_data()
    
    # Run based on mode
    if mode in ['combined', 'both']:
        print("\n" + "="*70)
        print("RUNNING COMBINED DETECTION (All Attacks Together)")
        print("="*70)
        
        # 1. Run UCAD detection on combined dataset
        analyzer.ucad_detection(
            epochs=50,
            batch_size=128,
            contamination_rate=0.076
        )
        
        # 2. Run graph-based analysis
        analyzer.graph_based_analysis()
        
        # 3. Visualize latent space
        tsne_df = analyzer.visualize_latent_space()
        
        # # 4. Feature importance analysis
        # importance_df = analyzer.analyze_feature_importance()
        
        results['combined'] = {
            'status': 'completed',
            'tsne_results': tsne_df,
            # 'feature_importance': importance_df
        }
    
    if mode in ['per_attack', 'both']:
        print("\n" + "="*70)
        print("RUNNING PER-ATTACK DETECTION (Separate Models)")
        print("="*70)
        
        # Run per-attack detection
        per_attack_results = analyzer.ucad_detection_per_attack(
            epochs=50,
            batch_size=128,
            contamination_rate=0.076  # Will be overridden per attack
        )
        
        results['per_attack'] = per_attack_results
    
    if mode == 'both':
        # Create combined vs per-attack comparison
        print("\n" + "="*70)
        print("Creating Combined vs Per-Attack Comparison...")
        print("="*70)
        analyzer._create_combined_vs_per_attack_comparison()
    
    print(f"\n{'='*70}")
    print("✓ Task 5 (Advanced Analyses with UCAD) completed successfully!")
    print(f"{'='*70}")
    print(f"All outputs saved to:")
    print(f"  Figures: {analyzer.fig_dir}/")
    print(f"  Tables:  {analyzer.table_dir}/")
    print(f"  Models:  {analyzer.model_dir}/")
    print(f"{'='*70}\n")
    
    return results


if __name__ == '__main__':
    import yaml
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    results = run_task5_ucad(config, logger=None)