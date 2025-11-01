"""
Task 3 Runner: Binary Intrusion Detection
Execute binary classification experiments comparing multiple unsupervised models.

Usage:
    python experiments/run_task3.py
    python experiments/run_task3.py --config config/config.yaml
    python experiments/run_task3.py --use-wandb
    python experiments/run_task3.py --no-wandb
"""

import sys
import argparse
import yaml
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tasks.task3_binary import run_task3
from src.utils.wandb_utils import WandbLogger


def main():
    """Main execution function."""
    
    # Parse arguments
    parser = argparse.ArgumentParser(
        description='Run Task 3: Binary Intrusion Detection'
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
        help='Enable W&B logging (overrides config)'
    )
    parser.add_argument(
        '--no-wandb',
        action='store_true',
        help='Disable W&B logging (overrides config)'
    )
    parser.add_argument(
        '--wandb-tags',
        type=str,
        default='',
        help='Comma-separated W&B tags (e.g., "alice,experiment-v1")'
    )
    parser.add_argument(
        '--wandb-name',
        type=str,
        default=None,
        help='Custom W&B run name'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    print(f"Loading configuration from: {args.config}")
    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file not found: {args.config}")
        print("Please create config/config.yaml or specify a valid config file.")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error parsing configuration file: {e}")
        sys.exit(1)
    
    # Determine W&B usage
    if args.no_wandb:
        use_wandb = False
    elif args.use_wandb:
        use_wandb = True
    else:
        use_wandb = config.get('wandb', {}).get('enabled', False)
    
    print("\n" + "="*70)
    print("CP219 PROJECT 2 - TASK 3")
    print("Binary Intrusion Detection (Normal vs Attack)")
    print("="*70)
    print(f"Configuration: {args.config}")
    print(f"W&B Logging: {'Enabled' if use_wandb else 'Disabled'}")
    
    # Verify data directory
    data_dir = Path(config['data']['raw_dir'])
    if not data_dir.exists():
        print(f"\nError: Data directory not found: {data_dir}")
        print("Please ensure your data files are in the correct location.")
        sys.exit(1)
    
    # Check for required data files
    required_files = []
    for attack_type in ['replay', 'masquerade', 'injection', 'poisoning']:
        train_file = config['data']['train_files'][attack_type]
        test_file = config['data']['test_files'][attack_type]
        required_files.append(data_dir / train_file)
        required_files.append(data_dir / test_file)
    
    missing_files = [f for f in required_files if not f.exists()]
    if missing_files:
        print("\nError: Missing data files:")
        for f in missing_files:
            print(f"  - {f}")
        print("\nPlease add all required training and test data files.")
        sys.exit(1)
    
    print(f"Data directory: {data_dir}")
    print(f"Output directory: outputs/")
    print()
    
    # Initialize logger if W&B is enabled
    logger = None
    if use_wandb:
        try:
            # Parse tags
            tags = ['task3', 'binary-detection']
            if args.wandb_tags:
                tags.extend([t.strip() for t in args.wandb_tags.split(',')])
            
            # Initialize W&B
            print("Initializing Weights & Biases...")
            logger = WandbLogger(
                project_name=config['project']['name'],
                entity=config['project'].get('entity'),
                config=config,
                job_type='task3_binary_detection',
                tags=tags,
                group='task3_experiments'
            )
            
            # Set custom run name if provided
            if args.wandb_name:
                import wandb
                wandb.run.name = args.wandb_name
            
            print(f"W&B Run: {logger.run.name}")
            print(f"W&B URL: {logger.run.url}")
            print()
            
        except Exception as e:
            print(f"Warning: Failed to initialize W&B: {e}")
            print("Continuing without W&B logging...")
            logger = None
    
    # Run Task 3
    try:
        print("Starting Task 3 analysis...")
        print("This may take several minutes depending on dataset size...")
        print("-" * 70)
        
        results = run_task3(config, logger)
        
        print("\n" + "="*70)
        print("TASK 3 COMPLETED SUCCESSFULLY!")
        print("="*70)
        
        # Print summary
        print("\nSummary:")
        print(f"  Status: {results.get('status', 'completed')}")
        print(f"  Best Model: {results.get('best_model', 'N/A')}")
        print(f"  Best F1-Score: {results.get('best_f1_score', 0):.4f}")
        print(f"  Best Accuracy: {results.get('best_accuracy', 0):.4f}")
        
        # Model comparison
        if 'results_table' in results:
            results_table = results['results_table']
            print("\n  Top 3 Models by F1-Score:")
            for idx, row in results_table.head(3).iterrows():
                print(f"    {idx+1}. {row['Model']}: F1={row['F1-Score']:.4f}, "
                      f"Inference={row['Inference Time (ms)']:.2f}ms")
        
        # Output locations
        print("\n  Output Locations:")
        print("    Figures: outputs/figures/task3/")
        print("    Tables:  outputs/tables/task3/")
        print("    Models:  outputs/models/task3/")
        
        # List generated files
        fig_dir = Path('outputs/figures/task3')
        table_dir = Path('outputs/tables/task3')
        model_dir = Path('outputs/models/task3')
        
        if fig_dir.exists():
            figures = list(fig_dir.glob('*.png'))
            if figures:
                print(f"\n  Generated Figures ({len(figures)}):")
                for fig in sorted(figures):
                    print(f"    - {fig.name}")
        
        if table_dir.exists():
            tables = list(table_dir.glob('*.csv'))
            if tables:
                print(f"\n  Generated Tables ({len(tables)}):")
                for table in sorted(tables):
                    print(f"    - {table.name}")
        
        if model_dir.exists():
            models = list(model_dir.glob('*.pkl'))
            if models:
                print(f"\n  Saved Models ({len(models)}):")
                for model in sorted(models):
                    print(f"    - {model.name}")
        
        # W&B info
        if logger:
            print(f"\n  W&B Dashboard: {logger.run.url}")
        
        print("\n" + "="*70)
        
        # Recommendations
        print("\nRECOMMENDATIONS:")
        print("-" * 70)
        print(f"For real-time substation deployment (<4ms latency requirement):")
        print(f"  • Best overall: {results.get('best_model', 'N/A')}")
        print(f"  • Consider trade-offs between accuracy and inference speed")
        print(f"  • Review the performance vs efficiency plot for deployment decision")
        
    except Exception as e:
        print("\n" + "="*70)
        print("ERROR DURING TASK 3 EXECUTION")
        print("="*70)
        print(f"\nError: {str(e)}")
        
        import traceback
        print("\nFull traceback:")
        traceback.print_exc()
        
        if logger:
            logger.finish()
        
        sys.exit(1)
    
    # Finish W&B run
    if logger:
        print("\nFinalizing W&B run...")
        logger.finish()
        print("W&B run finished.")
    
    print("\n✓ Task 3 completed successfully!")
    print("\nNext steps:")
    print("  1. Review model comparison in outputs/tables/task3/model_comparison.csv")
    print("  2. Examine visualizations in outputs/figures/task3/")
    print("  3. Check saved models in outputs/models/task3/")
    print("  4. Proceed to Task 4 (Multi-class Attack Detection)")
    if logger:
        print(f"  5. View detailed results at: {logger.run.url}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())