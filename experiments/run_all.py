"""
Master script to run all experiments for CP219 Project 2.

Usage:
    python experiments/run_all.py --config config/config.yaml --tasks all
    python experiments/run_all.py --config config/config.yaml --tasks task3,task4
    python experiments/run_all.py --config config/config.yaml --tasks task3 --use-wandb
"""

import argparse
import yaml
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.wandb_utils import WandbLogger
from src.tasks.task1_eda import run_task1
from src.tasks.task2_characterization import run_task2
from src.tasks.task2_core_vs_engg import run_task2_workflow
# from src.tasks.task3_binary import run_task3
# from src.tasks.task4_multiclass import run_task4
# from src.tasks.task5_advanced import run_task5


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def run_all_tasks(config: dict, tasks: list, use_wandb: bool = True):
    """
    Run all specified tasks.
    
    Args:
        config: Configuration dictionary
        tasks: List of task names to run
        use_wandb: Whether to use wandb logging
    """
    project_name = config['project']['name']
    entity = config['project'].get('entity')
    
    # Create output directories
    output_dir = Path('outputs')
    output_dir.mkdir(exist_ok=True)
    (output_dir / 'figures').mkdir(exist_ok=True)
    (output_dir / 'tables').mkdir(exist_ok=True)
    (output_dir / 'models').mkdir(exist_ok=True)
    
    task_functions = {
        'task1': ('Exploratory Data Analysis', run_task1),
        'task2': ('Feature & Attack Characterization', run_task2),
        # 'task3': ('Binary Intrusion Detection', run_task3),
        # 'task4': ('Multi-Class Attack Detection', run_task4),
        # 'task5': ('Advanced Analyses (Bonus)', run_task5)
    }
    
    results = {}
    
    for task_name in tasks:
        if task_name not in task_functions:
            print(f"Warning: Unknown task '{task_name}'. Skipping...")
            continue
        
        task_title, task_func = task_functions[task_name]
        print(f"\n{'='*70}")
        print(f"Running {task_name.upper()}: {task_title}")
        print(f"{'='*70}\n")
        
        try:
            # Initialize wandb for this task
            if use_wandb and config.get('wandb', {}).get('enabled', True):
                logger = WandbLogger(
                    project_name=project_name,
                    entity=entity,
                    config=config,
                    job_type=task_name,
                    tags=[task_name, task_title],
                    group=f"{task_name}_experiments"
                )
                
                # Run task
                task_results = task_func(config, logger)
                results[task_name] = task_results
                
                # Finish wandb run
                logger.finish()
            else:
                # Run without wandb
                task_results = task_func(config, None)
                results[task_name] = task_results
            
            print(f"\n✓ {task_name.upper()} completed successfully!")
            
        except Exception as e:
            print(f"\n✗ Error in {task_name.upper()}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    # Print summary
    print(f"\n{'='*70}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*70}\n")
    
    for task_name, task_results in results.items():
        print(f"{task_name.upper()}:")
        if isinstance(task_results, dict):
            for key, value in task_results.items():
                if isinstance(value, (int, float)):
                    print(f"  {key}: {value:.4f}")
                else:
                    print(f"  {key}: {value}")
        print()
    
    print(f"\nAll outputs saved to: {output_dir.absolute()}")
    print(f"Figures: {(output_dir / 'figures').absolute()}")
    print(f"Tables: {(output_dir / 'tables').absolute()}")
    print(f"Models: {(output_dir / 'models').absolute()}")


def main():
    parser = argparse.ArgumentParser(
        description='Run CP219 Project 2 Experiments'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--tasks',
        type=str,
        default='all',
        help='Comma-separated list of tasks to run (e.g., task1,task3) or "all"'
    )
    parser.add_argument(
        '--use-wandb',
        action='store_true',
        help='Enable wandb logging'
    )
    parser.add_argument(
        '--no-wandb',
        action='store_true',
        help='Disable wandb logging'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Determine wandb usage
    if args.no_wandb:
        use_wandb = False
    elif args.use_wandb:
        use_wandb = True
    else:
        use_wandb = config.get('wandb', {}).get('enabled', True)
    
    # Parse tasks
    if args.tasks.lower() == 'all':
        tasks = ['task1', 'task2', 'task3', 'task4']
        # Optionally include task5 if it exists
        # tasks.append('task5')
    else:
        tasks = [t.strip() for t in args.tasks.split(',')]
    
    print(f"\nConfiguration: {args.config}")
    print(f"Tasks to run: {', '.join(tasks)}")
    print(f"W&B logging: {'Enabled' if use_wandb else 'Disabled'}")
    print()
    
    # Run experiments
    run_all_tasks(config, tasks, use_wandb)


if __name__ == '__main__':
    main()