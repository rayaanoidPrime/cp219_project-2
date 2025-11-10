"""
Task 5 Runner: Optional Advanced Analyses
Executes advanced analyses including LSTM Autoencoders, Graph Analysis,
and Latent Space Visualization.

Usage:
    python experiments/run_task5.py
    python experiments/run_task5.py --config config/config.yaml
    python experiments/run_task5.py --use-wandb
"""

import sys
import argparse
import yaml
from pathlib import Path
# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tasks.task5_ucad import run_task5_ucad



# Assuming the PyTorch version of task5 is located at this path
from src.utils.wandb_utils import WandbLogger


def main():
    """Main execution function."""
    
    # Parse arguments
    parser = argparse.ArgumentParser(
        description='Run Task 5: Optional Advanced Analyses'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--use-wandb',
        action='store_true',
        help='Enable W&B logging',
        default=True
    )
    parser.add_argument(
        '--no-wandb',
        action='store_true',
        help='Disable W&B logging'
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
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error parsing configuration: {e}")
        sys.exit(1)
    
    # Determine W&B usage
    use_wandb = args.use_wandb and not args.no_wandb
    
    print("\n" + "="*70)
    print("TASK 5: OPTIONAL ADVANCED ANALYSES")
    print("LSTM Autoencoder, Graph Analysis, Latent Space Viz")
    print("="*70)
    print(f"Configuration: {args.config}")
    print(f"W&B Logging: {'Enabled' if use_wandb else 'Disabled'}")
    
    # Verify data directory
    data_dir = Path(config['data']['raw_dir'])
    if not data_dir.exists():
        print(f"\nError: Data directory not found: {data_dir}")
        sys.exit(1)
    
    # Check for required data files (Task 5 uses all training sets)
    required_files = [data_dir / f for f in config['data']['train_files'].values()]
    
    missing_files = [f for f in required_files if not f.exists()]
    if missing_files:
        print("\nError: Missing data files:")
        for f in missing_files:
            print(f"  - {f}")
        sys.exit(1)
    
    print(f"Data directory: {data_dir}")
    print(f"Output directory: outputs/\n")
    
    # Initialize W&B logger if enabled
    logger = None
    if use_wandb:
        try:
            print("Initializing Weights & Biases...")
            logger = WandbLogger(
                project_name=config['project']['name'],
                entity=config['project'].get('entity'),
                config=config,
                job_type='task5_advanced_analyses',
                tags=['task5', 'advanced', 'lstm', 'pytorch', 'graph'],
                group='task5_experiments'
            )
            
            if args.wandb_name:
                import wandb
                wandb.run.name = args.wandb_name
            
            print(f"W&B Run: {logger.run.name}")
            print(f"W&B URL: {logger.run.url}\n")
            
        except Exception as e:
            print(f"Warning: Failed to initialize W&B: {e}")
            print("Continuing without W&B logging...\n")
            logger = None

     # override mode if specified
    if args.mode:
        mode_list = [t.strip() for t in args.mode.split(',')]
        if any(m not in ['core', 'full', "new", "core_new"] for m in mode_list):
            print(f"\nError: Invalid mode specified: {args.mode}")
            print("Valid modes are 'core', 'full', 'new', 'core_new', or a comma-separated combination.")
            sys.exit(1)
        config['mode'] = mode_list 

    # Run Task 5
    try:
        print("Starting Task 5 analysis...")
       
        
        results = run_task5_ucad(config, logger)
        
        print("\n" + "="*70)
        print("TASK 5 COMPLETED SUCCESSFULLY!")
        print("="*70)
        
        

        # W&B info
        if logger:
            print(f"\nW&B Dashboard: {logger.run.url}")
        
    except Exception as e:
        print("\n" + "="*70)
        print("ERROR DURING TASK 5 EXECUTION")
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
    
    print("\n✓ Task 5 completed successfully!")
    return 0


if __name__ == '__main__':
    sys.exit(main())