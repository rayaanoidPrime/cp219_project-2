"""
Distribution Visualization Script

Plots feature distributions (histograms + KDE) for top 15 features across all datasets,
grouped by attack type vs Normal. Used to motivate IQR-based anomaly detection approach.

Usage: python src/collaboration/distribution_plots.py
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    BASE_DIR = Path(r"C:\Users\sengu\Documents\cp219_project-2\data\Final_Datasets\Final_Datasets\preprocessed_new\preprocessed_new")
    OUTPUT_DIR = Path(r"C:\Users\sengu\Documents\cp219_project-2\results\distribution_plots")
    
    # Top 15 features from global_ranking.csv (ordered by importance)
    TOP_FEATURES = [
        'integer_7', 'integer_5', 'integer_8', 'integer_6',
        'timestamp_diff', 'stNum', 'time_diff',
        'floatvalue_3', 'floatvalue_1', 'freq',
        'Length', 'index', 'floatvalue_2',
        'stNum_diff', 'sqNum_diff'
    ]
    
    # Plot styling
    DPI = 330
    TITLE_FONTSIZE = 16
    LABEL_FONTSIZE = 14
    TICK_FONTSIZE = 12
    LEGEND_FONTSIZE = 12
    
    # Colors
    NORMAL_COLOR = '#4CAF50'  # Green
    ATTACK_COLOR = '#F44336'  # Red

# Create output directory
Config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# DATA DISCOVERY
# ============================================================================

def discover_datasets(base_dir: Path):
    """Find all attack_and_normal.csv files in train folders."""
    datasets = []
    
    print("🔍 Scanning for datasets...")
    
    for train_csv in base_dir.rglob("train/attack_and_normal.csv"):
        # Build dataset path info
        scenario_dir = train_csv.parent.parent
        device_dir = scenario_dir.parent
        protocol_dir = device_dir.parent
        
        dataset_info = {
            'csv_path': train_csv,
            'protocol': protocol_dir.name,
            'device': device_dir.name,
            'scenario': scenario_dir.name,
            'name': f"{protocol_dir.name}/{device_dir.name}/{scenario_dir.name}"
        }
        datasets.append(dataset_info)
        
    print(f"✅ Found {len(datasets)} datasets")
    return datasets

# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def plot_feature_distribution(df, feature, dataset_name, output_path):
    """
    Plot histogram with KDE for a single feature, comparing Normal vs Attack.
    
    Style matches the reference image: overlapping histograms with KDE curves.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Split data by attack label
    normal_data = df[df['attack'] == 0][feature].dropna()
    attack_data = df[df['attack'] == 1][feature].dropna()
    
    # Skip if insufficient data
    if len(normal_data) < 5 or len(attack_data) < 5:
        plt.close()
        return False
    
    # Remove extreme outliers for better visualization (keep 99.5% of data)
    def clip_outliers(data):
        q_low, q_high = np.percentile(data, [0.25, 99.75])
        return data[(data >= q_low) & (data <= q_high)]
    
    normal_clipped = clip_outliers(normal_data)
    attack_clipped = clip_outliers(attack_data)
    
    if len(normal_clipped) < 5 or len(attack_clipped) < 5:
        normal_clipped = normal_data
        attack_clipped = attack_data
    
    # Determine common bins
    all_data = pd.concat([normal_clipped, attack_clipped])
    bins = np.histogram_bin_edges(all_data, bins=50)
    
    # Plot histograms with transparency
    ax.hist(normal_clipped, bins=bins, density=True, alpha=0.5, 
            color=Config.NORMAL_COLOR, label='Normal', edgecolor='white', linewidth=0.5)
    ax.hist(attack_clipped, bins=bins, density=True, alpha=0.5, 
            color=Config.ATTACK_COLOR, label='Attack', edgecolor='white', linewidth=0.5)
    
    # Add KDE curves
    try:
        sns.kdeplot(normal_clipped, ax=ax, color=Config.NORMAL_COLOR, linewidth=2, label='_nolegend_')
        sns.kdeplot(attack_clipped, ax=ax, color=Config.ATTACK_COLOR, linewidth=2, label='_nolegend_')
    except Exception:
        pass  # KDE may fail for certain distributions
    
    # Styling
    ax.set_xlabel(feature, fontsize=Config.LABEL_FONTSIZE, fontweight='bold')
    ax.set_ylabel('Density', fontsize=Config.LABEL_FONTSIZE, fontweight='bold')
    ax.set_title(f'{feature}: Normal vs Attack\n{dataset_name}', 
                 fontsize=Config.TITLE_FONTSIZE, fontweight='bold')
    ax.tick_params(axis='both', labelsize=Config.TICK_FONTSIZE)
    ax.legend(fontsize=Config.LEGEND_FONTSIZE, loc='upper right')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=Config.DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return True

def plot_dataset_grid(df, features, dataset_name, output_path):
    """
    Plot a grid of all available features for a dataset.
    Creates a multi-subplot figure similar to the reference image.
    """
    n_features = len(features)
    if n_features == 0:
        return False
    
    # Calculate grid size
    n_cols = min(4, n_features)
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    
    # Flatten axes for easier iteration
    if n_features == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if n_rows > 1 or n_cols > 1 else [axes]
    
    # Split data
    normal_data = df[df['attack'] == 0]
    attack_data = df[df['attack'] == 1]
    
    for idx, feature in enumerate(features):
        ax = axes[idx]
        
        normal_feat = normal_data[feature].dropna()
        attack_feat = attack_data[feature].dropna()
        
        if len(normal_feat) < 5 or len(attack_feat) < 5:
            ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(feature, fontsize=10, fontweight='bold')
            continue
        
        # Clip outliers
        def clip_outliers(data):
            q_low, q_high = np.percentile(data, [0.5, 99.5])
            clipped = data[(data >= q_low) & (data <= q_high)]
            return clipped if len(clipped) >= 5 else data
        
        normal_clipped = clip_outliers(normal_feat)
        attack_clipped = clip_outliers(attack_feat)
        
        # Determine bins
        all_data = pd.concat([normal_clipped, attack_clipped])
        bins = np.histogram_bin_edges(all_data, bins=30)
        
        # Plot histograms
        ax.hist(normal_clipped, bins=bins, density=True, alpha=0.5, 
                color=Config.NORMAL_COLOR, label='Normal', edgecolor='white', linewidth=0.3)
        ax.hist(attack_clipped, bins=bins, density=True, alpha=0.5, 
                color=Config.ATTACK_COLOR, label='Attack', edgecolor='white', linewidth=0.3)
        
        # Add KDE
        try:
            sns.kdeplot(normal_clipped, ax=ax, color=Config.NORMAL_COLOR, linewidth=1.5)
            sns.kdeplot(attack_clipped, ax=ax, color=Config.ATTACK_COLOR, linewidth=1.5)
        except Exception:
            pass
        
        ax.set_title(feature, fontsize=10, fontweight='bold')
        ax.tick_params(axis='both', labelsize=8)
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(alpha=0.2)
    
    # Hide unused subplots
    for idx in range(len(features), len(axes)):
        axes[idx].set_visible(False)
    
    fig.suptitle(f'Feature Distributions: {dataset_name}', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=Config.DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return True

# ============================================================================
# MAIN PROCESSING
# ============================================================================

def process_dataset(dataset_info):
    """Process a single dataset and create distribution plots."""
    name = dataset_info['name']
    csv_path = dataset_info['csv_path']
    
    # Load data
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"  ❌ Error loading {name}: {e}")
        return None
    
    # Check for attack column
    if 'attack' not in df.columns:
        print(f"  ⚠️ No 'attack' column in {name}")
        return None
    
    # Find available top features in this dataset
    available_features = [f for f in Config.TOP_FEATURES if f in df.columns]
    missing_features = [f for f in Config.TOP_FEATURES if f not in df.columns]
    
    if not available_features:
        print(f"  ⚠️ No top features found in {name}")
        return None
    
    # Create output subdirectory for this dataset
    safe_name = name.replace('/', '_').replace('\\', '_').replace('$', '_')
    dataset_output_dir = Config.OUTPUT_DIR / safe_name
    dataset_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Stats
    n_normal = (df['attack'] == 0).sum()
    n_attack = (df['attack'] == 1).sum()
    
    result = {
        'name': name,
        'n_samples': len(df),
        'n_normal': n_normal,
        'n_attack': n_attack,
        'available_features': available_features,
        'missing_features': missing_features,
        'individual_plots': [],
        'grid_plot': None
    }
    
    # Create individual feature plots
    for feature in available_features:
        output_path = dataset_output_dir / f"{feature}.png"
        success = plot_feature_distribution(df, feature, name, output_path)
        if success:
            result['individual_plots'].append(str(output_path))
    
    # Create grid plot with all available features
    grid_path = dataset_output_dir / "all_features_grid.png"
    if plot_dataset_grid(df, available_features, name, grid_path):
        result['grid_plot'] = str(grid_path)
    
    return result

def main():
    """Main entry point."""
    print("=" * 80)
    print("📊 DISTRIBUTION VISUALIZATION SCRIPT")
    print("=" * 80)
    print(f"Looking for datasets in: {Config.BASE_DIR}")
    print(f"Output directory: {Config.OUTPUT_DIR}")
    print(f"Top features to plot: {len(Config.TOP_FEATURES)}")
    print()
    
    # Discover datasets
    datasets = discover_datasets(Config.BASE_DIR)
    
    if not datasets:
        print("❌ No datasets found!")
        return
    
    # Process each dataset
    results = []
    print("\n📈 Processing datasets...")
    
    for dataset_info in tqdm(datasets, desc="Generating plots"):
        result = process_dataset(dataset_info)
        if result:
            results.append(result)
    
    # Summary
    print("\n" + "=" * 80)
    print("✅ PROCESSING COMPLETE")
    print("=" * 80)
    print(f"Datasets processed: {len(results)}/{len(datasets)}")
    
    # Print feature availability summary
    print("\n📋 Feature Availability Summary:")
    feature_counts = {f: 0 for f in Config.TOP_FEATURES}
    for result in results:
        for feat in result['available_features']:
            feature_counts[feat] += 1
    
    for feat, count in sorted(feature_counts.items(), key=lambda x: -x[1]):
        print(f"  {feat}: {count}/{len(results)} datasets ({100*count/len(results):.0f}%)")
    
    print(f"\n📁 Plots saved to: {Config.OUTPUT_DIR}")
    
    # Save summary CSV
    summary_data = []
    for r in results:
        summary_data.append({
            'dataset': r['name'],
            'n_samples': r['n_samples'],
            'n_normal': r['n_normal'],
            'n_attack': r['n_attack'],
            'n_features_available': len(r['available_features']),
            'features_available': '; '.join(r['available_features']),
            'features_missing': '; '.join(r['missing_features'])
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(Config.OUTPUT_DIR / "summary.csv", index=False)
    print(f"📄 Summary saved to: {Config.OUTPUT_DIR / 'summary.csv'}")

if __name__ == "__main__":
    main()
