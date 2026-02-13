"""
Combined KDE Plots Script - 4x4 Grid Layout

Creates a single figure with 4 datasets (rows) × 4 features (columns).
- Left Y-axis: Shared "Density" label
- Right Y-axis: Dataset names
- Top: Feature column headers
- Single legend for entire figure

Usage: python src/collaboration/combined_kde_plots.py
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set up Computer Modern font (IEEE/LaTeX style)
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Computer Modern Roman', 'CMU Serif', 'Times New Roman'],
    'mathtext.fontset': 'cm',
    'axes.unicode_minus': False,
})

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    BASE_DIR = Path(r"C:\Users\sengu\Documents\cp219_project-2\data\Final_Datasets\Final_Datasets\preprocessed_new\preprocessed_new")
    OUTPUT_DIR = Path(r"C:\Users\sengu\Documents\cp219_project-2\results\distribution_plots")
    
    # Plot styling
    DPI = 330
    
    # Colors
    NORMAL_COLOR = '#4CAF50'  # Green
    ATTACK_COLOR = '#F44336'  # Red

# 4 Features (columns)
FEATURES = ['stNum_diff', 'Length', 'time_diff', 'sqNum_diff']

# 4 Datasets (rows) - path and display name
DATASETS = [
    ('Process_Bus/PIOC_TRSF1_CBStval/905_replay', 'Process_Bus - Replay'),
    ('NTT_2025/physical_disjunctor_1/injection', 'NTT_2025 - Injection'),
    ('GOOSE_secure/GOOSE_Ctrl15/spoofing_stable', 'GOOSE_secure - Spoofing'),
    ('Power_Duck/Abgang1_SIP1_CTRL_LLN0_Control_DataSet/flood1', 'Power_Duck - Flood'),
]

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_dataset(dataset_path):
    """Load dataset from path."""
    csv_path = Config.BASE_DIR / dataset_path / "train" / "attack_and_normal.csv"
    if not csv_path.exists():
        print(f"  [WARN] Not found: {csv_path}")
        return None
    return pd.read_csv(csv_path)

def clip_outliers(data):
    """Clip outliers keeping 99.5% of data."""
    if len(data) < 5:
        return data
    q_low, q_high = np.percentile(data, [0.25, 99.75])
    clipped = data[(data >= q_low) & (data <= q_high)]
    return clipped if len(clipped) >= 5 else data

def plot_single_kde(ax, df, feature, show_legend=False):
    """Plot a single KDE histogram on the given axis."""
    if feature not in df.columns:
        ax.text(0.5, 0.5, 'Feature N/A', ha='center', va='center', 
                transform=ax.transAxes, fontsize=12)
        return
    
    # Extract data
    normal_data = df[df['attack'] == 0][feature].dropna()
    attack_data = df[df['attack'] == 1][feature].dropna()
    
    if len(normal_data) < 5 or len(attack_data) < 1:
        ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center', 
                transform=ax.transAxes, fontsize=12)
        return
    
    # Clip outliers
    normal_clipped = clip_outliers(normal_data)
    attack_clipped = clip_outliers(attack_data)
    
    # Determine bins
    all_data = pd.concat([normal_clipped, attack_clipped])
    bins = np.histogram_bin_edges(all_data, bins=40)
    
    # Plot histograms
    ax.hist(normal_clipped, bins=bins, density=True, alpha=0.5, 
            color=Config.NORMAL_COLOR, label='Normal', edgecolor='white', linewidth=0.3)
    ax.hist(attack_clipped, bins=bins, density=True, alpha=0.5, 
            color=Config.ATTACK_COLOR, label='Attack', edgecolor='white', linewidth=0.3)
    
    # Add KDE curves
    try:
        sns.kdeplot(normal_clipped, ax=ax, color=Config.NORMAL_COLOR, linewidth=1.5)
        sns.kdeplot(attack_clipped, ax=ax, color=Config.ATTACK_COLOR, linewidth=1.5)
    except Exception:
        pass
    
    # Styling
    ax.tick_params(axis='both', labelsize=10)
    ax.grid(alpha=0.2)
    
    ax.legend(fontsize=12, loc='upper right')

# ============================================================================
# MAIN PLOT FUNCTION
# ============================================================================

def create_combined_plot():
    """Create a 4x4 grid: 4 datasets (rows) × 4 features (columns)."""
    Config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("COMBINED KDE PLOTS - 4x4 Grid")
    print("=" * 80)
    
    n_rows = len(DATASETS)
    n_cols = len(FEATURES)
    
    # Create figure and axes
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 12))
    
    # Load all datasets
    loaded_data = {}
    for dataset_path, display_name in DATASETS:
        print(f"Loading: {display_name}...")
        loaded_data[display_name] = load_dataset(dataset_path)
    
    # Plot each cell
    for row_idx, (dataset_path, display_name) in enumerate(DATASETS):
        df = loaded_data[display_name]
        
        for col_idx, feature in enumerate(FEATURES):
            ax = axes[row_idx, col_idx]
            
            if df is not None:
                # Only show legend for first subplot
                show_legend = (row_idx == 0 and col_idx == 0)
                plot_single_kde(ax, df, feature, show_legend=show_legend)
            else:
                ax.text(0.5, 0.5, 'Data not found', ha='center', va='center', 
                        transform=ax.transAxes, fontsize=12)
            
            # Column headers (feature names) - top row only
            if row_idx == 0:
                ax.set_title(feature, fontsize=16, pad=10)
            
            # X-axis labels - bottom row only
            if row_idx == n_rows - 1:
                ax.set_xlabel(feature, fontsize=12)
            else:
                ax.set_xlabel('')
                ax.tick_params(axis='x', labelbottom=False)
            
            # Hide individual Y-labels
            ax.set_ylabel('')
            
            # Dataset names on right side - rightmost column only
            if col_idx == n_cols - 1:
                ax2 = ax.twinx()
                ax2.set_ylabel(display_name, fontsize=14, rotation=270, labelpad=20)
                ax2.set_yticks([])
    
    # Shared Y-axis label on the left
    fig.text(0.02, 0.5, 'Density', va='center', rotation='vertical', fontsize=18)
    
    # Adjust layout
    plt.tight_layout(rect=[0.04, 0, 1, 1])  # Leave space for left Y-label
    
    # Save figure
    output_path = Config.OUTPUT_DIR / "combined_kde_4x4.png"
    plt.savefig(output_path, dpi=Config.DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"\n[OK] Saved: {output_path}")
    print("=" * 80)
    print("COMBINED PLOT COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    create_combined_plot()
