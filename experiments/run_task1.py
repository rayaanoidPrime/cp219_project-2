"""
Task 1 Runner: Exploratory Data Analysis
Execute comprehensive EDA on GOOSE IDS dataset.

Usage:
    python experiments/run_task1.py
    python experiments/run_task1.py --config config/config.yaml
    python experiments/run_task1.py --use-wandb
    python experiments/run_task1.py --no-wandb
"""

import sys
from pathlib import Path
import argparse
import yaml
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.tasks.task1_eda import run_task1
from src.utils.wandb_utils import WandbLogger


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run Task 1: Exploratory Data Analysis'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to configuration file (default: config/config.yaml)'
    )
    parser.add_argument(
        '--use-wandb',
        action='store_true',
        help='Enable Weights & Biases logging'
    )
    parser.add_argument(
        '--no-wandb',
        action='store_true',
        help='Disable Weights & Biases logging'
    )
    parser.add_argument(
        '--wandb-name',
        type=str,
        default=None,
        help='Custom name for W&B run'
    )
    parser.add_argument(
        '--wandb-tags',
        type=str,
        default='',
        help='Comma-separated tags for W&B run (e.g., "alice,v1")'
    )
    
    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_args()
    
    # Load configuration
    print("="*70)
    print("CP219 Project 2 - Task 1: Exploratory Data Analysis")
    print("="*70)
    print(f"\nLoading configuration from: {args.config}")
    
    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"ERROR: Configuration file not found: {args.config}")
        print("Please create config/config.yaml or specify correct path with --config")
        return 1
    except yaml.YAMLError as e:
        print(f"ERROR: Invalid YAML in configuration file: {e}")
        return 1
    
    # Determine if W&B should be used
    if args.no_wandb:
        use_wandb = False
    elif args.use_wandb:
        use_wandb = True
    else:
        use_wandb = config.get('wandb', {}).get('enabled', False)
    
    # Verify data directory exists
    data_dir = Path(config['data']['raw_dir'])
    if not data_dir.exists():
        print(f"\nWARNING: Data directory not found: {data_dir}")
        print("Creating directory...")
        data_dir.mkdir(parents=True, exist_ok=True)
        print("\nPlease place your dataset files in this directory:")
        for attack_type, filename in config['data']['train_files'].items():
            print(f"  - {filename}")
        for attack_type, filename in config['data']['test_files'].items():
            print(f"  - {filename}")
        return 1
    
    # Check if data files exist
    missing_files = []
    for attack_type, filename in {**config['data']['train_files'], 
                                  **config['data']['test_files']}.items():
        if not (data_dir / filename).exists():
            missing_files.append(filename)
    
    if missing_files:
        print(f"\nERROR: Missing data files in {data_dir}:")
        for f in missing_files:
            print(f"  - {f}")
        return 1
    
    # Initialize logger if W&B is enabled
    logger = None
    if use_wandb:
        try:
            print("\nInitializing Weights & Biases...")
            
            # Parse tags
            tags = ['task1', 'eda']
            if args.wandb_tags:
                tags.extend([t.strip() for t in args.wandb_tags.split(',')])
            
            logger = WandbLogger(
                project_name=config['project']['name'],
                entity=config['project'].get('entity'),
                config=config,
                job_type='task1_eda',
                tags=tags,
                group='task1_experiments'
            )
            
            # Set custom run name if provided
            if args.wandb_name:
                import wandb
                wandb.run.name = args.wandb_name
            
            print(f"W&B Run: {logger.run.name}")
            print(f"W&B URL: {logger.run.url}")
            
        except Exception as e:
            print(f"\nWARNING: Failed to initialize W&B: {e}")
            print("Continuing without W&B logging...")
            logger = None
    else:
        print("\nW&B logging: Disabled")
    
    # Run Task 1
    print("\n" + "="*70)
    print("Starting Task 1: Exploratory Data Analysis")
    print("="*70 + "\n")
    
    try:
        results = run_task1(config, logger)
        
        print("\n" + "="*70)
        print("TASK 1 COMPLETED SUCCESSFULLY! ✓")
        print("="*70)
        
        print("\nResults Summary:")
        print(f"  Total samples analyzed: {results.get('total_samples', 'N/A'):,}")
        print(f"  Attack ratio: {results.get('attack_ratio', 0):.2%}")
        
        print("\nOutputs saved to:")
        print(f"  Figures: outputs/figures/task1/")
        print(f"  Tables:  outputs/tables/task1/")
        
        # List generated files
        fig_dir = Path('outputs/figures/task1')
        table_dir = Path('outputs/tables/task1')
        
        if fig_dir.exists():
            figures = list(fig_dir.glob('*.png'))
            if figures:
                print(f"\nGenerated {len(figures)} figure(s):")
                for fig in sorted(figures):
                    print(f"  - {fig.name}")
        
        if table_dir.exists():
            tables = list(table_dir.glob('*.csv')) + list(table_dir.glob('*.txt'))
            if tables:
                print(f"\nGenerated {len(tables)} table(s):")
                for table in sorted(tables):
                    print(f"  - {table.name}")
        
        if logger:
            print(f"\nView results in W&B: {logger.run.url}")
            logger.finish()
        
        return 0
        
    except FileNotFoundError as e:
        print(f"\nERROR: File not found - {e}")
        print("Please check that all data files are in the correct location.")
        if logger:
            logger.finish()
        return 1
        
    except Exception as e:
        print(f"\nERROR: Task 1 failed with error:")
        print(f"  {type(e).__name__}: {e}")
        print("\nFor detailed traceback, run with Python's -v flag:")
        print(f"  python -v experiments/run_task1.py")
        
        import traceback
        traceback.print_exc()
        
        if logger:
            logger.finish()
        return 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)