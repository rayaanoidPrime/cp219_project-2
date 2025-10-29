"""
Task 3: Binary Intrusion Detection - Compare multiple unsupervised models
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from typing import Dict, Any
import matplotlib.pyplot as plt
import seaborn as sns

from src.data.loader import load_all_datasets
from src.data.preprocessor import preprocess_data
from src.models.isolation_forest import IsolationForestDetector
from src.models.clustering import GMMDetector, KMeansDetector, DBSCANDetector
from src.models.one_class_svm import OneClassSVMDetector
from src.models.autoencoder import AutoencoderDetector
from src.evaluation.metrics import compute_all_metrics
from src.evaluation.visualizer import plot_model_comparison, plot_roc_curves
from src.utils.wandb_utils import WandbLogger


def run_task3(config: Dict[str, Any], logger: WandbLogger = None) -> Dict:
    """
    Run Task 3: Binary Intrusion Detection.
    
    Args:
        config: Configuration dictionary
        logger: WandbLogger instance (optional)
    
    Returns:
        Dictionary with results
    """
    print("Loading datasets...")
    train_data, test_data = load_all_datasets(config)
    
    print("Preprocessing data...")
    X_train, y_train, X_test, y_test = preprocess_data(
        train_data, test_data, config
    )
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    print(f"Attack ratio in test: {y_test.mean():.2%}")
    
    # Define models to compare
    models = {
        'Isolation Forest': IsolationForestDetector(
            config=config.get('models', {}).get('isolation_forest', {}),
            use_wandb=(logger is not None)
        ),
        'GMM': GMMDetector(
            config=config.get('models', {}).get('gmm', {}),
            use_wandb=(logger is not None)
        ),
        'K-Means': KMeansDetector(
            config=config.get('models', {}).get('kmeans', {}),
            use_wandb=(logger is not None)
        ),
        'DBSCAN': DBSCANDetector(
            config=config.get('models', {}).get('dbscan', {}),
            use_wandb=(logger is not None)
        ),
        'One-Class SVM': OneClassSVMDetector(
            config=config.get('models', {}).get('one_class_svm', {}),
            use_wandb=(logger is not None)
        )
    }
    
    # Add Autoencoder if configured
    if config.get('models', {}).get('autoencoder', {}).get('enabled', False):
        models['Autoencoder'] = AutoencoderDetector(
            config=config.get('models', {}).get('autoencoder', {}),
            use_wandb=(logger is not None)
        )
    
    results = []
    all_predictions = {}
    
    # Train and evaluate each model
    for model_name, model in models.items():
        print(f"\n{'='*60}")
        print(f"Training and evaluating: {model_name}")
        print(f"{'='*60}")
        
        try:
            # Train model
            print(f"Training {model_name}...")
            model.fit(X_train)
            
            # Evaluate on test set
            print(f"Evaluating {model_name}...")
            metrics = model.evaluate(X_test, y_test, prefix="test")
            
            # Store predictions for ensemble later
            all_predictions[model_name] = model.predict(X_test)
            
            # Compute additional metrics
            memory_mb = model.get_memory_footprint()
            
            result = {
                'Model': model_name,
                'Accuracy': metrics['test/accuracy'],
                'Precision': metrics['test/precision'],
                'Recall': metrics['test/recall'],
                'F1-Score': metrics['test/f1_score'],
                'Train Time (s)': model.train_time,
                'Inference Time (ms)': model.inference_time * 1000,
                'Memory (MB)': memory_mb
            }
            results.append(result)
            
            # Print results
            print(f"\n{model_name} Results:")
            for key, value in result.items():
                if key != 'Model':
                    print(f"  {key}: {value:.4f}")
            
            # Log to wandb
            if logger is not None:
                logger.log_metrics({
                    f"binary/{model_name}/accuracy": metrics['test/accuracy'],
                    f"binary/{model_name}/precision": metrics['test/precision'],
                    f"binary/{model_name}/recall": metrics['test/recall'],
                    f"binary/{model_name}/f1_score": metrics['test/f1_score'],
                    f"binary/{model_name}/train_time": model.train_time,
                    f"binary/{model_name}/inference_time_ms": model.inference_time * 1000,
                    f"binary/{model_name}/memory_mb": memory_mb
                })
            
            # Save model
            model_path = Path('outputs/models') / f'{model_name.lower().replace(" ", "_")}_binary.pkl'
            model.save(str(model_path))
            
        except Exception as e:
            print(f"Error training {model_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    # Create results dataframe
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('F1-Score', ascending=False)
    
    print(f"\n{'='*60}")
    print("MODEL COMPARISON SUMMARY")
    print(f"{'='*60}")
    print(results_df.to_string(index=False))
    
    # Save results table
    table_path = Path('outputs/tables') / 'task3_model_comparison.csv'
    results_df.to_csv(table_path, index=False)
    print(f"\nResults saved to: {table_path}")
    
    # Log to wandb
    if logger is not None:
        logger.log_dataframe(results_df, "task3_model_comparison")
    
    # Create comparison visualizations
    print("\nGenerating comparison plots...")
    
    # 1. Performance metrics comparison
    fig = plot_model_comparison(results_df)
    fig_path = Path('outputs/figures') / 'task3_model_comparison.png'
    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {fig_path}")
    
    if logger is not None:
        logger.log_figure(fig, "task3_model_comparison")
    plt.close(fig)
    
    # 2. Trade-off analysis (F1 vs Inference Time)
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(
        results_df['Inference Time (ms)'],
        results_df['F1-Score'],
        s=results_df['Memory (MB)'] * 10,
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
            fontsize=9
        )
    
    ax.set_xlabel('Inference Time per Sample (ms)')
    ax.set_ylabel('F1-Score')
    ax.set_title('Model Performance vs Inference Time\n(bubble size = memory footprint)')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    tradeoff_path = Path('outputs/figures') / 'task3_performance_tradeoff.png'
    fig.savefig(tradeoff_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {tradeoff_path}")
    
    if logger is not None:
        logger.log_figure(fig, "task3_performance_tradeoff")
    plt.close(fig)
    
    # 3. Best model analysis
    best_model_name = results_df.iloc[0]['Model']
    print(f"\n{'='*60}")
    print(f"BEST MODEL: {best_model_name}")
    print(f"F1-Score: {results_df.iloc[0]['F1-Score']:.4f}")
    print(f"{'='*60}")
    
    # Return summary
    return {
        'best_model': best_model_name,
        'best_f1_score': float(results_df.iloc[0]['F1-Score']),
        'best_accuracy': float(results_df.iloc[0]['Accuracy']),
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
    print("\nTask 3 completed!")