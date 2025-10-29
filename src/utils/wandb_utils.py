import wandb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional, List
from pathlib import Path


class WandbLogger:
    """Utility class for W&B logging and experiment tracking."""
    
    def __init__(self, project_name: str, entity: Optional[str] = None,
                 config: Optional[Dict] = None, job_type: str = "train",
                 tags: Optional[List[str]] = None, group: Optional[str] = None):
        """
        Initialize wandb logger.
        
        Args:
            project_name: Name of the W&B project
            entity: W&B entity (team name)
            config: Configuration dictionary
            job_type: Type of job (train, eval, eda, etc.)
            tags: List of tags for the run
            group: Group name for organizing runs
        """
        self.run = wandb.init(
            project=project_name,
            entity=entity,
            config=config,
            job_type=job_type,
            tags=tags,
            group=group
        )
        
    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """Log metrics to wandb."""
        wandb.log(metrics, step=step)
    
    def log_figure(self, fig: plt.Figure, name: str, step: Optional[int] = None):
        """Log matplotlib figure to wandb."""
        wandb.log({name: wandb.Image(fig)}, step=step)
        plt.close(fig)
    
    def log_dataframe(self, df: pd.DataFrame, name: str):
        """Log pandas dataframe as wandb table."""
        wandb.log({name: wandb.Table(dataframe=df)})
    
    def log_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray,
                            class_names: List[str], name: str = "confusion_matrix"):
        """Log confusion matrix to wandb."""
        wandb.log({
            name: wandb.plot.confusion_matrix(
                probs=None,
                y_true=y_true,
                preds=y_pred,
                class_names=class_names
            )
        })
    
    def log_histogram(self, data: np.ndarray, name: str, step: Optional[int] = None):
        """Log histogram to wandb."""
        wandb.log({name: wandb.Histogram(data)}, step=step)
    
    def log_model_comparison(self, results: pd.DataFrame, name: str = "model_comparison"):
        """
        Log model comparison table.
        
        Args:
            results: DataFrame with model names and metrics
            name: Name for the table
        """
        self.log_dataframe(results, name)
        
        # Also log as summary metrics for easy comparison
        for idx, row in results.iterrows():
            model_name = row.get('model', idx)
            for col in results.columns:
                if col != 'model' and pd.api.types.is_numeric_dtype(results[col]):
                    wandb.run.summary[f"{model_name}/{col}"] = row[col]
    
    def log_artifact(self, artifact_path: str, artifact_type: str,
                     artifact_name: str, metadata: Optional[Dict] = None):
        """
        Log artifact to wandb.
        
        Args:
            artifact_path: Path to artifact file/directory
            artifact_type: Type of artifact (model, dataset, etc.)
            artifact_name: Name of the artifact
            metadata: Optional metadata dictionary
        """
        artifact = wandb.Artifact(artifact_name, type=artifact_type, metadata=metadata)
        artifact.add_file(artifact_path)
        wandb.log_artifact(artifact)
    
    def log_code(self, code_dir: str = "./src"):
        """Log source code directory."""
        wandb.run.log_code(code_dir)
    
    def finish(self):
        """Finish the wandb run."""
        wandb.finish()


def create_sweep_config(sweep_type: str = "bayes") -> Dict:
    """
    Create a wandb sweep configuration for hyperparameter tuning.
    
    Args:
        sweep_type: Type of sweep (bayes, grid, random)
    
    Returns:
        Sweep configuration dictionary
    """
    sweep_config = {
        'method': sweep_type,
        'metric': {
            'name': 'test/f1_score',
            'goal': 'maximize'
        },
        'parameters': {
            # Example parameters - customize based on your models
            'learning_rate': {
                'distribution': 'log_uniform_values',
                'min': 1e-5,
                'max': 1e-1
            },
            'n_estimators': {
                'values': [50, 100, 200, 300]
            },
            'contamination': {
                'distribution': 'uniform',
                'min': 0.01,
                'max': 0.3
            }
        }
    }
    return sweep_config


def compare_runs(project: str, entity: Optional[str] = None,
                tags: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Compare multiple wandb runs.
    
    Args:
        project: Project name
        entity: Entity name
        tags: Filter by tags
    
    Returns:
        DataFrame with run comparisons
    """
    api = wandb.Api()
    
    # Construct filters
    filters = {}
    if tags:
        filters["tags"] = {"$in": tags}
    
    # Get runs
    runs = api.runs(f"{entity}/{project}" if entity else project, filters=filters)
    
    # Extract data
    summary_list, config_list, name_list = [], [], []
    for run in runs:
        summary_list.append(run.summary._json_dict)
        config_list.append({k: v for k, v in run.config.items() if not k.startswith('_')})
        name_list.append(run.name)
    
    # Create dataframe
    summary_df = pd.DataFrame.from_records(summary_list)
    config_df = pd.DataFrame.from_records(config_list)
    summary_df.insert(0, 'run_name', name_list)
    
    return pd.concat([summary_df, config_df], axis=1)


class ExperimentTracker:
    """Track multiple experiments and aggregate results."""
    
    def __init__(self, project_name: str, entity: Optional[str] = None):
        self.project_name = project_name
        self.entity = entity
        self.experiments = []
    
    def add_experiment(self, experiment_name: str, config: Dict,
                      metrics: Dict, model_path: Optional[str] = None):
        """Add experiment results."""
        self.experiments.append({
            'name': experiment_name,
            'config': config,
            'metrics': metrics,
            'model_path': model_path
        })
    
    def get_best_experiment(self, metric: str = 'f1_score',
                           maximize: bool = True) -> Dict:
        """Get best experiment based on metric."""
        if not self.experiments:
            return None
        
        sorted_exps = sorted(
            self.experiments,
            key=lambda x: x['metrics'].get(metric, -float('inf') if maximize else float('inf')),
            reverse=maximize
        )
        return sorted_exps[0]
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert experiments to dataframe."""
        rows = []
        for exp in self.experiments:
            row = {'name': exp['name']}
            row.update(exp['config'])
            row.update(exp['metrics'])
            rows.append(row)
        return pd.DataFrame(rows)
    
    def save_summary(self, path: str):
        """Save summary to CSV."""
        df = self.to_dataframe()
        df.to_csv(path, index=False)
        
        # Log to wandb if in an active run
        if wandb.run is not None:
            wandb.log({"experiment_summary": wandb.Table(dataframe=df)})


def log_attack_distribution(train_data: Dict[str, pd.DataFrame],
                            test_data: Dict[str, pd.DataFrame],
                            logger: WandbLogger):
    """
    Log attack distribution statistics to wandb.
    
    Args:
        train_data: Dictionary of training dataframes by attack type
        test_data: Dictionary of test dataframes by attack type
        logger: WandbLogger instance
    """
    # Compute distribution
    train_dist = {attack: len(df) for attack, df in train_data.items()}
    test_dist = {attack: len(df) for attack, df in test_data.items()}
    
    # Create distribution dataframe
    dist_df = pd.DataFrame({
        'Attack Type': list(train_dist.keys()),
        'Train Count': list(train_dist.values()),
        'Test Count': list(test_dist.values())
    })
    
    logger.log_dataframe(dist_df, "data_distribution")
    
    # Log as bar chart
    data = [[attack, train_dist[attack], test_dist[attack]] 
            for attack in train_dist.keys()]
    table = wandb.Table(data=data, columns=["Attack Type", "Train", "Test"])
    logger.log_metrics({
        "data_distribution_chart": wandb.plot.bar(
            table, "Attack Type", "Train", title="Attack Distribution"
        )
    })