"""
Task 4 Runner: Multi-Class Attack Detection
Execute multi-class classification experiments using unsupervised clustering.

Usage:
    python experiments/run_task4.py
    python experiments/run_task4.py --config config/config.yaml
    python experiments/run_task4.py --use-wandb
"""

import sys
import argparse
import yaml
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tasks.task4_multiclass_detection import run_task4
from src.utils.wandb_utils import WandbLogger


def main():
    """Main execution function."""
    
    # Parse arguments
    parser = argparse.ArgumentParser(
        description='Run Task 4: Multi-Class Attack Detection'
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
    use_wandb = args.use_wandb or (not args.no_wandb and config.get('wandb', {}).get('enabled', False))
    
    print("\n" + "="*70)
    print("TASK 4: MULTI-CLASS ATTACK DETECTION")
    print("Unsupervised Classification: Normal + 4 Attack Types")
    print("="*70)
    print(f"Configuration: {args.config}")
    print(f"W&B Logging: {'Enabled' if use_wandb else 'Disabled'}")
    
    # Verify data directory
    data_dir = Path(config['data']['raw_dir'])
    if not data_dir.exists():
        print(f"\nError: Data directory not found: {data_dir}")
        sys.exit(1)
    
    # Check for required data files
    required_files = []
    for attack_type in ['replay', 'masquerade', 'injection', 'poisoning']:
        required_files.extend([
            data_dir / config['data']['train_files'][attack_type],
            data_dir / config['data']['test_files'][attack_type]
        ])
    
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
                job_type='task4_multiclass_detection',
                tags=['task4', 'multi-class', 'clustering'],
                group='task4_experiments'
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
    
    # Run Task 4
    try:
        print("Starting Task 4 analysis...")
        print("This will run both CORE and FULL feature modes...")
        print("-" * 70)
        
        results = run_task4(config, logger)
        
        print("\n" + "="*70)
        print("TASK 4 COMPLETED SUCCESSFULLY!")
        print("="*70)
        
        # Print summary for both modes
        for mode in ['core', 'full']:
            print(f"\n{mode.upper()} Mode Results:")
            summary = results[mode]['summary']
            best = summary.iloc[0]
            
            print(f"  Best Model: {best['Model']}")
            print(f"  Macro F1-Score: {best['F1-Score (Macro)']:.4f}")
            print(f"  Accuracy: {best['Accuracy']:.4f}")
            print(f"  Train Time: {best['Train Time (s)']:.2f}s")
            
            print(f"\n  Top 3 Models by Macro F1:")
            for idx in range(min(3, len(summary))):
                row = summary.iloc[idx]
                print(f"    {idx+1}. {row['Model']}: F1={row['F1-Score (Macro)']:.4f}, "
                      f"Acc={row['Accuracy']:.4f}")
            
            print(f"\n  Per-Class F1-Scores (Best Model):")
            print(f"    Normal:      {best['F1_normal']:.4f}")
            print(f"    Injection:   {best['F1_injection']:.4f}")
            print(f"    Masquerade:  {best['F1_masquerade']:.4f}")
            print(f"    Poisoning:   {best['F1_poisoning']:.4f}")
            print(f"    Replay:      {best['F1_replay']:.4f}")
        
        # Output locations
        print("\n" + "="*70)
        print("OUTPUT LOCATIONS")
        print("="*70)
        print("\nCORE Mode:")
        print("  Figures: outputs/figures/task4_core/")
        print("  Tables:  outputs/tables/task4_core/")
        print("  Models:  outputs/models/task4_core/")
        print("\nFULL Mode:")
        print("  Figures: outputs/figures/task4_full/")
        print("  Tables:  outputs/tables/task4_full/")
        print("  Models:  outputs/models/task4_full/")
        
        # List key generated files
        for mode in ['core', 'full']:
            fig_dir = Path(f'outputs/figures/task4_{mode}')
            if fig_dir.exists():
                key_figures = [
                    'confusion_matrices.png',
                    'macro_metrics_comparison.png',
                    'per_class_f1_heatmap.png',
                    'cluster_mappings.png'
                ]
                existing = [f for f in key_figures if (fig_dir / f).exists()]
                if existing:
                    print(f"\n{mode.upper()} - Key Figures:")
                    for fig in existing:
                        print(f"  ✓ {fig}")
        
        # W&B info
        if logger:
            print(f"\nW&B Dashboard: {logger.run.url}")
        
        print("\n" + "="*70)
        print("KEY INSIGHTS")
        print("="*70)
        
        # Compare modes
        core_best_f1 = results['core']['summary'].iloc[0]['F1-Score (Macro)']
        full_best_f1 = results['full']['summary'].iloc[0]['F1-Score (Macro)']
        
        print(f"\nFeature Impact:")
        print(f"  CORE mode (10 features): F1 = {core_best_f1:.4f}")
        print(f"  FULL mode (all features): F1 = {full_best_f1:.4f}")
        improvement = ((full_best_f1 - core_best_f1) / core_best_f1) * 100
        print(f"  Improvement: {improvement:+.2f}%")
        
        # Identify hardest attack to detect
        print(f"\nAttack Detectability (FULL mode):")
        best_full = results['full']['summary'].iloc[0]
        attack_f1s = {
            'Normal': best_full['F1_normal'],
            'Injection': best_full['F1_injection'],
            'Masquerade': best_full['F1_masquerade'],
            'Poisoning': best_full['F1_poisoning'],
            'Replay': best_full['F1_replay']
        }
        sorted_attacks = sorted(attack_f1s.items(), key=lambda x: x[1], reverse=True)
        for attack, f1 in sorted_attacks:
            print(f"  {attack:12s}: {f1:.4f}")
        
        print(f"\nRecommendations:")
        print(f"  • Best overall model: {results['full']['summary'].iloc[0]['Model']}")
        print(f"  • Hardest to detect: {sorted_attacks[-1][0]} (F1={sorted_attacks[-1][1]:.4f})")
        print(f"  • Most reliable: {sorted_attacks[0][0]} (F1={sorted_attacks[0][1]:.4f})")
        print(f"  • Review confusion matrices for class-specific insights")
        
    except Exception as e:
        print("\n" + "="*70)
        print("ERROR DURING TASK 4 EXECUTION")
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
    
    print("\n✓ Task 4 completed successfully!")
    return 0


if __name__ == '__main__':
    sys.exit(main())