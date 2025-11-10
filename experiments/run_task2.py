"""
Task 2 Runner: Feature and Attack Characterization Analysis
Execute comprehensive feature analysis and attack signature characterization.

Usage:
    python experiments/run_task2.py
    python experiments/run_task2.py --config config/config.yaml
    python experiments/run_task2.py --use-wandb
    python experiments/run_task2.py --no-wandb
"""

import sys
import argparse
import yaml
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tasks.task2_characterization import run_task2
from src.utils.wandb_utils import WandbLogger


def main():
    """Main execution function."""
    
    # Parse arguments
    parser = argparse.ArgumentParser(
        description='Run Task 2: Feature and Attack Characterization Analysis'
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
    parser.add_argument(
        '--mode',
        type=str,
        default="full",
        help='comma separated list of modes to run: core, full, new, core_new'
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
    print("CP219 PROJECT 2 - TASK 2")
    print("Feature and Attack Characterization Analysis")
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
        required_files.append(data_dir / train_file)
    
    missing_files = [f for f in required_files if not f.exists()]
    if missing_files:
        print("\nError: Missing data files:")
        for f in missing_files:
            print(f"  - {f}")
        print("\nPlease add all required training data files.")
        sys.exit(1)
    
    print(f"Data directory: {data_dir}")
    print(f"Output directory: outputs/")
    print()
    
    # Initialize logger if W&B is enabled
    logger = None
    if use_wandb:
        try:
            # Parse tags
            tags = ['task2', 'feature-characterization']
            if args.wandb_tags:
                tags.extend([t.strip() for t in args.wandb_tags.split(',')])
            
            # Initialize W&B
            print("Initializing Weights & Biases...")
            logger = WandbLogger(
                project_name=config['project']['name'],
                entity=config['project'].get('entity'),
                config=config,
                job_type='task2_characterization',
                tags=tags,
                group='task2_experiments'
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

    # override mode if specified
    if args.mode:
        mode_list = [t.strip() for t in args.mode.split(',')]
        if any(m not in ['core', 'full', "new", "core_new"] for m in mode_list):
            print(f"\nError: Invalid mode specified: {args.mode}")
            print("Valid modes are 'core', 'full', 'new', 'core_new', or a comma-separated combination.")
            sys.exit(1)
        config['mode'] = mode_list    
    # Run Task 2
    try:
        print("Starting Task 2 analysis...")
        print("-" * 70)
        
        results = run_task2(config, logger)
        
        print("\n" + "="*70)
        print("TASK 2 COMPLETED SUCCESSFULLY!")
        print("="*70)
        
        # Print summary
        print("\nSummary:")
        print(f"  Status: {results.get('status', 'completed')}")
        
        if 'difficulty_ranking' in results and results['difficulty_ranking'] is not None:
            print("\n  Attack Detection Difficulty Ranking:")
            for idx, row in results['difficulty_ranking'].iterrows():
                print(f"    {idx+1}. {row['Attack Type']}: {row['Detection Difficulty']} "
                      f"(Separability: {row['Separability Score']:.2f})")
        
        # Output locations
        print("\n  Output Locations:")
        print("    Figures: outputs/figures/task2/")
        print("    Tables:  outputs/tables/task2/")
        
        # List generated files
        fig_dir = Path('outputs/figures/task2')
        table_dir = Path('outputs/tables/task2')
        
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
        
        # W&B info
        if logger:
            print(f"\n  W&B Dashboard: {logger.run.url}")
        
        print("\n" + "="*70)
        
    except Exception as e:
        print("\n" + "="*70)
        print("ERROR DURING TASK 2 EXECUTION")
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
    
    print("\n✓ Task 2 completed successfully!")
  
    if logger:
        print(f"  4. View detailed results at: {logger.run.url}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())