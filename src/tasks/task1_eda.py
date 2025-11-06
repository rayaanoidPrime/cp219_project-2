"""
Task 1: Exploratory Data Analysis (EDA)
Comprehensive analysis of GOOSE protocol dataset to understand patterns in normal and attack traffic.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Optional, Tuple
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import confusion_matrix, classification_report

from src.utils.wandb_utils import WandbLogger


class GOOSEDataExplorer:
    """Comprehensive EDA for GOOSE protocol data."""
    
    def __init__(self, output_dir: str = "outputs/figures"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 6)
        plt.rcParams['font.size'] = 10
    
    def load_data(self, data_dir: str) -> pd.DataFrame:
        """Load all training datasets only."""
        data_dir = Path(data_dir)
        
        attack_types = ['replay', 'masquerade', 'injection', 'poisoning']
        
        # Load training data only
        train_dfs = []
        for attack in attack_types:
            df = pd.read_csv(data_dir / f'{attack}_labelled_train.csv')
            df['attack_type'] = attack
            train_dfs.append(df)
        train_data = pd.concat(train_dfs, ignore_index=True)
        
        print(f"Training data shape: {train_data.shape}")
        print(f"Test data shape: {test_data.shape}")
        # Drop irrelevant columns
        keep_columns = ['gocbRef',
                        'timeAllowedtoLive',
                        'Time',
                        't',
                        'stNum',
                        'sqNum',
                        'Length',
                        'Boolean',
                        'bit-string',
                        'attack',
                        'attack_type'
                        ]
        train_data = train_data.filter(items=keep_columns)
        test_data = test_data.filter(items=keep_columns)
        print(f"Training data shape after dropping irrelevant columns: {train_data.shape}")
        print(f"Test data shape after dropping irrelevant columns: {test_data.shape}")
        return train_data, test_data
    
    def basic_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate basic statistical summary."""
        print("\n" + "="*70)
        print("BASIC STATISTICS")
        print("="*70)
        
        # Overall statistics
        print(f"\nTotal samples: {len(df)}")
        print(f"Normal samples: {(df['attack'] == 0).sum()}")
        print(f"Attack samples: {(df['attack'] == 1).sum()}")
        print(f"Attack ratio: {df['attack'].mean():.2%}")
        
        # Per attack type
        print("\nSamples per attack type:")
        attack_dist = df.groupby(['attack_type', 'attack']).size().unstack(fill_value=0)
        attack_dist.columns = ['Normal', 'Attack']
        print(attack_dist)
        
        # Unique gocbRef values
        print(f"\nUnique gocbRef values: {df['gocbRef'].nunique()}")
        gocbref_counts = df['gocbRef'].value_counts()
        print(f"Most common gocbRef: {gocbref_counts.index[0]} ({gocbref_counts.iloc[0]} samples)")
        
        print("\n" + "="*70)
        print("FEATURE STATISTICS SUMMARY (ALL COLUMNS)")
        print("="*70)
        
        # Create a list of columns to describe, excluding 'attack' if it exists
        if 'attack' in df.columns:
            cols_to_describe = [col for col in df.columns if col != 'attack']
        else:
            cols_to_describe = df.columns.tolist()

        if not cols_to_describe:
            print("No features to describe.")
            return pd.DataFrame() # Return empty dataframe

        # Use describe(include='all') to get stats for all data types
        stats_df = df[cols_to_describe].describe(include='all')
        
        return stats_df
    
    def plot_attack_distribution(self, df: pd.DataFrame, logger: WandbLogger = None):
        """Visualize attack distribution."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Overall distribution
        attack_counts = df['attack'].value_counts()
        colors = ['#2ecc71', '#e74c3c']
        axes[0].bar(['Normal', 'Attack'], attack_counts.values, color=colors, alpha=0.7)
        axes[0].set_ylabel('Count')
        axes[0].set_title('Overall Attack vs Normal Distribution')
        axes[0].grid(axis='y', alpha=0.3)
        
        # Per attack type
        attack_type_dist = df.groupby(['attack_type', 'attack']).size().unstack()
        attack_type_dist.plot(kind='bar', ax=axes[1], color=colors, alpha=0.7)
        axes[1].set_xlabel('Attack Type')
        axes[1].set_ylabel('Count')
        axes[1].set_title('Distribution by Attack Type')
        axes[1].legend(['Normal', 'Attack'])
        axes[1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # Save
        save_path = self.output_dir / 'task1_attack_distribution.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
        
        if logger:
            logger.log_figure(fig, "task1_attack_distribution")
        
        plt.close()
    
    def plot_temporal_patterns(self, df: pd.DataFrame, logger: WandbLogger = None):
        """Analyze temporal patterns in GOOSE messages."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Time intervals between messages (by attack type)
        if 'Time' in df.columns:
            for attack_type in df['attack_type'].unique():
                subset = df[df['attack_type'] == attack_type].sort_values('Time')
                if len(subset) > 1:
                    time_diff = subset['Time'].diff().dropna()
                    axes[0, 0].hist(time_diff[time_diff < time_diff.quantile(0.99)], 
                                   bins=50, alpha=0.5, label=attack_type)
            axes[0, 0].set_xlabel('Time Difference (s)')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].set_title('Inter-Message Time Distribution')
            axes[0, 0].legend()
            axes[0, 0].set_yscale('log')
        
        # 2. stNum changes over time
        if 'stNum' in df.columns:
            for i, attack_type in enumerate(df['attack_type'].unique()[:4]):
                subset = df[df['attack_type'] == attack_type].head(1000)
                color = 'red' if (subset['attack'] == 1).any() else 'blue'
                axes[0, 1].plot(range(len(subset)), subset['stNum'], 
                              alpha=0.3, label=attack_type, linewidth=0.5)
            axes[0, 1].set_xlabel('Message Index')
            axes[0, 1].set_ylabel('stNum')
            axes[0, 1].set_title('Status Number Evolution')
            axes[0, 1].legend()
        
        # 3. sqNum patterns
        if 'sqNum' in df.columns:
            normal_data = df[df['attack'] == 0]['sqNum']
            attack_data = df[df['attack'] == 1]['sqNum']
            
            axes[1, 0].hist(normal_data, bins=50, alpha=0.5, label='Normal', color='blue')
            axes[1, 0].hist(attack_data, bins=50, alpha=0.5, label='Attack', color='red')
            axes[1, 0].set_xlabel('sqNum')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].set_title('Sequence Number Distribution')
            axes[1, 0].legend()
            axes[1, 0].set_yscale('log')
        
        # 4. timeAllowedtoLive distribution
        if 'timeAllowedtoLive' in df.columns:
            df_sample = df.sample(min(10000, len(df)))
            for attack in [0, 1]:
                subset = df_sample[df_sample['attack'] == attack]['timeAllowedtoLive']
                axes[1, 1].hist(subset, bins=30, alpha=0.5, 
                              label='Normal' if attack == 0 else 'Attack')
            axes[1, 1].set_xlabel('timeAllowedtoLive (ms)')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].set_title('Time-To-Live Distribution')
            axes[1, 1].legend()
        
        plt.tight_layout()
        
        save_path = self.output_dir / 'task1_temporal_patterns.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
        
        if logger:
            logger.log_figure(fig, "task1_temporal_patterns")
        
        plt.close()

    def plot_heartbeat_by_gocbref_split_by_attacktype(
        self,
        df: pd.DataFrame,
        logger: WandbLogger = None,
        max_gocbref_plots: int | None = None,
        downsample_max_points: int = 15000,
        attack_types_order: List[str] = ("replay", "masquerade", "injection", "poisoning"),
    ):
        """
        For each gocbRef and attack_type:
          - x: relative time (s) from first sample in that subset
          - y-left: stNum (blue step line)
          - y-right: sqNum (orange step line)
          - Red shaded regions = attack periods
          - Blue verticals = stNum changes
          - Orange dots = sqNum resets to 0
        Saves plots to outputs/figures/heartbeat_split/<gocbRef>__<attack_type>.png
        """
        req = {'gocbRef', 'Time', 'stNum', 'sqNum', 'attack', 'attack_type'}
        miss = req - set(df.columns)
        if miss:
            print(f"[heartbeat_split] Missing columns: {sorted(miss)}")
            return

        out_dir = self.output_dir / "heartbeat_split"
        out_dir.mkdir(parents=True, exist_ok=True)

        def _sanitize(s: str) -> str:
            for ch in '<>:"/\\|?*':
                s = s.replace(ch, '_')
            return s[:90]

        # ---- handle Time ----
        df = df.copy()
        if pd.api.types.is_datetime64_any_dtype(df['Time']):
            pass
        elif pd.api.types.is_string_dtype(df['Time']):
            parsed = pd.to_datetime(df['Time'], errors='coerce', utc=True)
            if parsed.notna().mean() > 0.8:
                df['Time'] = parsed
            else:
                df['Time'] = pd.to_numeric(df['Time'], errors='coerce')
        else:
            df['Time'] = pd.to_numeric(df['Time'], errors='coerce')

        g_list = list(df.groupby('gocbRef'))
        if max_gocbref_plots is not None:
            g_list = g_list[:max_gocbref_plots]

        for gname, gdf in g_list:
            gdf = gdf.sort_values('Time').dropna(subset=['Time'])
            if len(gdf) < 2:
                continue

            atypes = [a for a in attack_types_order if a in gdf['attack_type'].unique()]
            for atype in atypes:
                sub = gdf[gdf['attack_type'] == atype].copy()
                if len(sub) < 2:
                    continue

                # --- relative time ---
                if pd.api.types.is_datetime64_any_dtype(sub['Time']):
                    t0 = sub['Time'].iloc[0]
                    sub['t_rel'] = (sub['Time'] - t0).dt.total_seconds()
                else:
                    t0 = float(sub['Time'].iloc[0])
                    sub['t_rel'] = sub['Time'].astype(float) - t0

                sub = sub.sort_values('t_rel')
                if len(sub) > downsample_max_points:
                    step = int(np.ceil(len(sub) / downsample_max_points))
                    sub = sub.iloc[::step, :]

                sub['st_change'] = sub['stNum'].diff().fillna(0).ne(0)
                sub['sq_reset'] = sub['sqNum'].eq(0)

                fig, axL = plt.subplots(figsize=(14, 6))
                axR = axL.twinx()

                # Shade attack spans
                attk = sub['attack'].astype(int).values
                t = sub['t_rel'].values
                if attk.any():
                    idx = np.where(np.diff(np.r_[0, attk, 0]) != 0)[0]
                    spans = list(zip(idx[0::2], idx[1::2]))
                    for s, e in spans:
                        axL.axvspan(
                            t[s], t[e - 1 if e - 1 < len(t) else -1],
                            color='red', alpha=0.08, lw=0
                        )

                # Step lines
                l1, = axL.plot(
                    sub['t_rel'], sub['stNum'],
                    drawstyle='steps-post', linewidth=1.4,
                    color='tab:blue', label='stNum'
                )
                l2, = axR.plot(
                    sub['t_rel'], sub['sqNum'],
                    drawstyle='steps-post', linewidth=1.2,
                    color='tab:orange', alpha=0.9, label='sqNum'
                )

                # Event markers
                for x in sub.loc[sub['st_change'], 't_rel']:
                    axL.axvline(x, color='tab:blue', alpha=0.15, lw=0.8)
                axR.scatter(
                    sub.loc[sub['sq_reset'], 't_rel'],
                    sub.loc[sub['sq_reset'], 'sqNum'],
                    s=14, color='tab:orange', alpha=0.8,
                    zorder=5, label='sqNum reset=0'
                )

                # Axes
                axL.set_xlabel('Time (s, relative)')
                axL.set_ylabel('stNum', color='tab:blue')
                axR.set_ylabel('sqNum', color='tab:orange')
                axL.grid(alpha=0.3)

                # Legend
                h1, lab1 = axL.get_legend_handles_labels()
                h2, lab2 = axR.get_legend_handles_labels()
                seen, H, L = set(), [], []
                for h, l in list(zip(h1 + h2, lab1 + lab2)):
                    if l not in seen:
                        H.append(h); L.append(l); seen.add(l)
                axL.legend(H, L, loc='upper left', fontsize=9)

                axL.set_title(
                    f"Heartbeat — gocbRef: {str(gname)[:80]} | attack_type: {atype} | n={len(sub)}",
                    pad=10
                )

                # Margins
                xmin, xmax = sub['t_rel'].min(), sub['t_rel'].max()
                rng = xmax - xmin if xmax > xmin else 1.0
                axL.set_xlim(xmin - 0.01 * rng, xmax + 0.01 * rng)

                fname = f"{_sanitize(str(gname))}__{atype}.png"
                path = out_dir / fname
                fig.savefig(path, dpi=300, bbox_inches='tight')
                print(f"Saved: {path}")

                if logger:
                    logger.log_figure(fig, f"heartbeat_split/{_sanitize(str(gname))}__{atype}")

                plt.close(fig)

    def plot_feature_distributions(self, df: pd.DataFrame, logger: WandbLogger = None):
        """Plot distributions of key features."""
        # Select numerical features
        numerical_cols = ['stNum', 'sqNum', 'Length', 'timeAllowedtoLive']
        numerical_cols = [col for col in numerical_cols if col in df.columns]
        
        if len(numerical_cols) == 0:
            print("No numerical features found for distribution plots")
            return
        
        n_cols = 2
        n_rows = (len(numerical_cols) + 1) // 2
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4*n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
        
        for idx, col in enumerate(numerical_cols):
            ax = axes[idx]
            
            # Sample data for efficiency
            df_sample = df.sample(min(10000, len(df)))
            
            normal_data = df_sample[df_sample['attack'] == 0][col]
            attack_data = df_sample[df_sample['attack'] == 1][col]
            
            # Use log scale if range is large
            if normal_data.max() > 1000:
                bins = np.logspace(np.log10(max(normal_data.min(), 1)), 
                                 np.log10(normal_data.max()), 50)
                ax.set_xscale('log')
            else:
                bins = 50
            
            ax.hist(normal_data, bins=bins, alpha=0.5, label='Normal', color='blue')
            ax.hist(attack_data, bins=bins, alpha=0.5, label='Attack', color='red')
            ax.set_xlabel(col)
            ax.set_ylabel('Frequency')
            ax.set_title(f'Distribution of {col}')
            ax.legend()
            ax.grid(alpha=0.3)
        
        # Hide empty subplots
        for idx in range(len(numerical_cols), len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        
        save_path = self.output_dir / 'task1_feature_distributions.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
        
        if logger:
            logger.log_figure(fig, "task1_feature_distributions")
        
        plt.close()
    
    def plot_correlation_analysis(self, df: pd.DataFrame, logger: WandbLogger = None):
        """Correlation analysis between features."""
        # Select numerical features
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        numerical_cols = [col for col in numerical_cols if col not in ['attack']]
        
        if len(numerical_cols) < 2:
            print("Not enough numerical features for correlation analysis")
            return
        
        
        df_num = df[numerical_cols]
        
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        
        # Overall correlation
        corr = df_num.corr()
        
        # Increase figure size and font for clarity
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                   ax=axes[0], square=True, linewidths=1, 
                   annot_kws={'size': 11}, cbar_kws={'shrink': 0.8})
        axes[0].set_title('Feature Correlation Heatmap', fontsize=14, pad=15)
        
        # Rotate labels for better readability
        axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45, ha='right')
        axes[0].set_yticklabels(axes[0].get_yticklabels(), rotation=0)
        
        # Correlation with attack label
        if len(numerical_cols) > 0:
            df_with_label = df[numerical_cols + ['attack']].sample(min(10000, len(df)))
            attack_corr = df_with_label.corr()['attack'].drop('attack').sort_values()
            
            colors = ['red' if x < 0 else 'green' for x in attack_corr.values]
            bars = axes[1].barh(range(len(attack_corr)), attack_corr.values, color=colors, alpha=0.7)
            axes[1].set_yticks(range(len(attack_corr)))
            axes[1].set_yticklabels(attack_corr.index, fontsize=11)
            axes[1].set_xlabel('Correlation with Attack Label', fontsize=12)
            axes[1].set_title('Feature Correlation with Attack', fontsize=14, pad=15)
            axes[1].axvline(0, color='black', linewidth=0.8)
            axes[1].grid(axis='x', alpha=0.3)
            
            # Add value labels on bars
            for i, (bar, val) in enumerate(zip(bars, attack_corr.values)):
                axes[1].text(val + (0.01 if val > 0 else -0.01), i, f'{val:.3f}',
                           va='center', ha='left' if val > 0 else 'right', fontsize=9)
        
        plt.tight_layout()
        
        save_path = self.output_dir / 'task1_correlation_analysis.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
        
        if logger:
            logger.log_figure(fig, "task1_correlation_analysis")
        
        plt.close()
    
    def plot_dimensionality_reduction(self, df: pd.DataFrame, logger: WandbLogger = None):
        """PCA and t-SNE visualization per attack type using ALL data."""
        print("\nPerforming dimensionality reduction...")
        
        # Prepare data
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        numerical_cols = [col for col in numerical_cols if col not in ['attack']]
        
        if len(numerical_cols) < 2:
            print("Not enough features for dimensionality reduction")
            return
        
        attack_types = ['replay', 'masquerade', 'injection', 'poisoning']
        
        # PCA projections - one figure with 2x2 grid
        fig_pca, axes_pca = plt.subplots(2, 2, figsize=(16, 14))
        axes_pca = axes_pca.flatten()
        
        # t-SNE projections - one figure with 2x2 grid
        fig_tsne, axes_tsne = plt.subplots(2, 2, figsize=(16, 14))
        axes_tsne = axes_tsne.flatten()
        
        for idx, attack_type in enumerate(attack_types):
            print(f"Processing {attack_type}...")
            
            # Get subset for this attack type - USE ALL DATA
            df_subset = df[df['attack_type'] == attack_type]
            
            X = df_subset[numerical_cols].fillna(0)
            y = df_subset['attack'].values
            
            # Standardize
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # PCA
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_scaled)
            
            ax_pca = axes_pca[idx]
            for attack in [0, 1]:
                mask = y == attack
                ax_pca.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                              alpha=0.6, s=30,
                              label='Normal' if attack == 0 else 'Attack',
                              c='#3498db' if attack == 0 else '#e74c3c')
            ax_pca.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} var)')
            ax_pca.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} var)')
            ax_pca.set_title(f'{attack_type.capitalize()} - PCA Projection (n={len(df_subset)})')
            ax_pca.legend()
            ax_pca.grid(alpha=0.3)
            
            # t-SNE (sample for computational efficiency)
            sample_size = min(5000, len(df_subset))
            sample_indices = np.random.choice(len(X_scaled), sample_size, replace=False)
            X_scaled_sample = X_scaled[sample_indices]
            y_sample = y[sample_indices]
            
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, sample_size//4))
            X_tsne = tsne.fit_transform(X_scaled_sample)
            
            ax_tsne = axes_tsne[idx]
            for attack in [0, 1]:
                mask = y_sample == attack
                ax_tsne.scatter(X_tsne[mask, 0], X_tsne[mask, 1], 
                               alpha=0.6, s=30,
                               label='Normal' if attack == 0 else 'Attack',
                               c='#3498db' if attack == 0 else '#e74c3c')
            ax_tsne.set_xlabel('t-SNE 1')
            ax_tsne.set_ylabel('t-SNE 2')
            ax_tsne.set_title(f'{attack_type.capitalize()} - t-SNE Projection (sampled n={sample_size})')
            ax_tsne.legend()
            ax_tsne.grid(alpha=0.3)
        
        # Save PCA figure
        fig_pca.tight_layout()
        save_path_pca = self.output_dir / 'task1_pca_by_attack_type.png'
        fig_pca.savefig(save_path_pca, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path_pca}")
        
        if logger:
            logger.log_figure(fig_pca, "task1_pca_by_attack_type")
        
        plt.close(fig_pca)
        
        # Save t-SNE figure
        fig_tsne.tight_layout()
        save_path_tsne = self.output_dir / 'task1_tsne_by_attack_type.png'
        fig_tsne.savefig(save_path_tsne, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path_tsne}")
        
        if logger:
            logger.log_figure(fig_tsne, "task1_tsne_by_attack_type")
        
        plt.close(fig_tsne)
    
    def analyze_temporal_and_categorical_patterns(self, df: pd.DataFrame, logger: WandbLogger = None):
        """
        Phase 2: Deep dive into temporal patterns and categorical features.
        Analyzes time-based patterns, boolean field behavior, and gocbRef distributions.
        """
        print("\n" + "="*70)
        print("PHASE 2: TEMPORAL AND CATEGORICAL DEEP DIVE")
        print("="*70)
        
        fig = plt.figure(figsize=(22, 16))
        gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.35)
        
        # Parse datetime if 't' column exists
        if 't' in df.columns:
            df_temp = df.copy()
            df_temp['t'] = pd.to_datetime(df_temp['t'], utc=True, errors='coerce')
            df_temp['hour'] = df_temp['t'].dt.hour
            df_temp['dayofweek'] = df_temp['t'].dt.dayofweek
            df_temp['minute'] = df_temp['t'].dt.minute
            
            # 1. Hour of day distribution
            ax1 = fig.add_subplot(gs[0, 0])
            for attack in [0, 1]:
                subset = df_temp[df_temp['attack'] == attack]['hour'].value_counts().sort_index()
                ax1.plot(subset.index, subset.values, marker='o', linewidth=2, markersize=6,
                        label='Normal' if attack == 0 else 'Attack',
                        color='blue' if attack == 0 else 'red', alpha=0.7)
            ax1.set_xlabel('Hour of Day', fontsize=11)
            ax1.set_ylabel('Message Count', fontsize=11)
            ax1.set_title('Messages by Hour of Day', fontsize=13, pad=10)
            ax1.legend(fontsize=10)
            ax1.grid(alpha=0.3)
            
            # 2. Day of week distribution
            ax2 = fig.add_subplot(gs[0, 1])
            day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
            for attack in [0, 1]:
                subset = df_temp[df_temp['attack'] == attack]['dayofweek'].value_counts().sort_index()
                ax2.bar(subset.index + (0.2 if attack == 1 else -0.2), subset.values, 
                       width=0.4, label='Normal' if attack == 0 else 'Attack',
                       color='blue' if attack == 0 else 'red', alpha=0.7)
            ax2.set_xticks(range(7))
            ax2.set_xticklabels(day_names, fontsize=10)
            ax2.set_xlabel('Day of Week', fontsize=11)
            ax2.set_ylabel('Message Count', fontsize=11)
            ax2.set_title('Messages by Day of Week', fontsize=13, pad=10)
            ax2.legend(fontsize=10)
            ax2.grid(axis='y', alpha=0.3)
            
            # 3. Attack distribution across hours (heatmap style)
            ax3 = fig.add_subplot(gs[0, 2])
            attack_by_hour = df_temp[df_temp['attack'] == 1].groupby(['attack_type', 'hour']).size().unstack(fill_value=0)
            if not attack_by_hour.empty:
                sns.heatmap(attack_by_hour, cmap='Reds', ax=ax3, 
                           cbar_kws={'label': 'Attack Count', 'shrink': 0.8},
                           annot=False, fmt='d')
                ax3.set_xlabel('Hour of Day', fontsize=11)
                ax3.set_ylabel('Attack Type', fontsize=11)
                ax3.set_title('Attack Distribution by Hour', fontsize=13, pad=10)
                ax3.set_yticklabels(ax3.get_yticklabels(), rotation=0, fontsize=10)
        
        # 4. Boolean field analysis
        if 'boolean' in df.columns:
            ax4 = fig.add_subplot(gs[1, 0])
            # Count number of True values in boolean string
            df_temp = df.copy()
            df_temp['bool_true_count'] = df_temp['boolean'].astype(str).apply(
                lambda x: x.count('True') if pd.notna(x) else 0
            )
            
            for attack in [0, 1]:
                subset = df_temp[df_temp['attack'] == attack]['bool_true_count']
                ax4.hist(subset, bins=range(0, int(subset.max()) + 2), alpha=0.5,
                        label='Normal' if attack == 0 else 'Attack',
                        color='blue' if attack == 0 else 'red')
            ax4.set_xlabel('Number of True Values in Boolean Field', fontsize=11)
            ax4.set_ylabel('Frequency', fontsize=11)
            ax4.set_title('Boolean Field True Count Distribution', fontsize=13, pad=10)
            ax4.legend(fontsize=10)
            ax4.grid(alpha=0.3)
            ax4.set_yscale('log')
        
        # 5. gocbRef attack distribution
        ax5 = fig.add_subplot(gs[1, 1])
        top_gocbrefs = df['gocbRef'].value_counts().head(10).index
        gocbref_attack = df[df['gocbRef'].isin(top_gocbrefs)].groupby(['gocbRef', 'attack']).size().unstack(fill_value=0)
        gocbref_attack_pct = gocbref_attack.div(gocbref_attack.sum(axis=1), axis=0) * 100
        
        gocbref_attack_pct.plot(kind='barh', stacked=True, ax=ax5, color=['blue', 'red'], alpha=0.7)
        ax5.set_xlabel('Percentage', fontsize=11)
        ax5.set_ylabel('gocbRef (Top 10)', fontsize=11)
        ax5.set_title('Attack Distribution by gocbRef', fontsize=13, pad=10)
        ax5.legend(['Normal', 'Attack'], fontsize=10)
        ax5.grid(axis='x', alpha=0.3)
        ax5.tick_params(axis='y', labelsize=9)
        
        # 6. Attack count by gocbRef and type
        ax6 = fig.add_subplot(gs[1, 2])
        attack_by_gocb = df[df['attack'] == 1].groupby(['gocbRef', 'attack_type']).size().unstack(fill_value=0)
        if not attack_by_gocb.empty:
            attack_by_gocb_top = attack_by_gocb.sum(axis=1).nlargest(10)
            attack_by_gocb.loc[attack_by_gocb_top.index].plot(kind='barh', stacked=True, ax=ax6, alpha=0.7)
            ax6.set_xlabel('Attack Count', fontsize=11)
            ax6.set_ylabel('gocbRef (Top 10)', fontsize=11)
            ax6.set_title('Attack Types by gocbRef', fontsize=13, pad=10)
            ax6.legend(title='Attack Type', fontsize=9)
            ax6.grid(axis='x', alpha=0.3)
            ax6.tick_params(axis='y', labelsize=9)
        
        # 7. Time interval between messages (Normal vs Attack)
        if 'Time' in df.columns:
            ax7 = fig.add_subplot(gs[2, 0])
            for attack_type in df['attack_type'].unique():
                for attack in [0, 1]:
                    subset = df[(df['attack_type'] == attack_type) & (df['attack'] == attack)].sort_values('Time')
                    if len(subset) > 1:
                        time_diff = subset['Time'].diff().dropna()
                        time_diff = time_diff[time_diff < time_diff.quantile(0.99)]
                        ax7.hist(time_diff, bins=50, alpha=0.3, 
                                label=f"{attack_type}-{'Attack' if attack == 1 else 'Normal'}")
            ax7.set_xlabel('Time Interval (s)', fontsize=11)
            ax7.set_ylabel('Frequency', fontsize=11)
            ax7.set_title('Inter-Message Time Intervals', fontsize=13, pad=10)
            ax7.set_yscale('log')
            ax7.legend(fontsize=8, ncol=2)
            ax7.grid(alpha=0.3)
        
        # 8. Boolean field changes analysis
        if 'boolean' in df.columns:
            ax8 = fig.add_subplot(gs[2, 1])
            df_sorted = df.sort_values(['gocbRef', 'Time'])
            df_sorted['boolean_changed'] = df_sorted.groupby('gocbRef')['boolean'].shift() != df_sorted['boolean']
            
            change_stats = df_sorted.groupby(['attack_type', 'attack'])['boolean_changed'].mean() * 100
            change_stats = change_stats.unstack()
            
            x = np.arange(len(change_stats))
            width = 0.35
            ax8.bar(x - width/2, change_stats[0], width, label='Normal', color='blue', alpha=0.7)
            ax8.bar(x + width/2, change_stats[1], width, label='Attack', color='red', alpha=0.7)
            ax8.set_xticks(x)
            ax8.set_xticklabels(change_stats.index, rotation=45, ha='right', fontsize=10)
            ax8.set_ylabel('Boolean Change Rate (%)', fontsize=11)
            ax8.set_title('Boolean Field Change Frequency', fontsize=13, pad=10)
            ax8.legend(fontsize=10)
            ax8.grid(axis='y', alpha=0.3)
        
        # 9. timeAllowedtoLive vs actual intervals
        if 'timeAllowedtoLive' in df.columns and 'Time' in df.columns:
            ax9 = fig.add_subplot(gs[2, 2])
            df_sorted = df.sort_values(['gocbRef', 'Time'])
            df_sorted['actual_interval'] = df_sorted.groupby('gocbRef')['Time'].diff()
            df_sorted['expected_interval'] = df_sorted['timeAllowedtoLive'] / 2000  # Convert ms to s
            
            sample_data = df_sorted[['actual_interval', 'expected_interval', 'attack']].dropna().sample(min(5000, len(df_sorted)))
            
            for attack in [0, 1]:
                subset = sample_data[sample_data['attack'] == attack]
                ax9.scatter(subset['expected_interval'], subset['actual_interval'], 
                           alpha=0.3, s=15, label='Normal' if attack == 0 else 'Attack',
                           c='blue' if attack == 0 else 'red')
            
            # Add diagonal line
            max_val = max(sample_data['expected_interval'].max(), sample_data['actual_interval'].max())
            ax9.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, linewidth=1.5)
            
            ax9.set_xlabel('Expected Interval (timeAllowedtoLive/2)', fontsize=11)
            ax9.set_ylabel('Actual Interval (s)', fontsize=11)
            ax9.set_title('Expected vs Actual Message Intervals', fontsize=13, pad=10)
            ax9.legend(fontsize=10)
            ax9.grid(alpha=0.3)
            ax9.set_xscale('log')
            ax9.set_yscale('log')
        
        plt.tight_layout()
        
        save_path = self.output_dir / 'task1_phase2_temporal_categorical.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
        
        if logger:
            logger.log_figure(fig, "task1_phase2_temporal_categorical")
        
        plt.close()
    
    def plot_event_occurrence_timeline(self, df: pd.DataFrame, logger: WandbLogger = None):
        """
        Visualize when events occurred vs normal heartbeats across time.
        Shows stNum changes (events) and their relationship to attacks.
        """
        print("\n" + "="*70)
        print("GOOSE EVENT OCCURRENCE ANALYSIS")
        print("="*70)
        
        # Analyze each attack type separately
        attack_types = ['replay', 'masquerade', 'injection', 'poisoning']
        
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        axes = axes.flatten()
        
        for idx, attack_type in enumerate(attack_types):
            ax = axes[idx]
            
            # Get data for this attack type
            subset = df[df['attack_type'] == attack_type].sort_values('Time')
            
            if len(subset) < 2 or 'stNum' not in subset.columns:
                ax.text(0.5, 0.5, f'Insufficient data for {attack_type}', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{attack_type.capitalize()} - Event Timeline')
                continue
            
            # Detect stNum changes (events)
            subset = subset.copy()
            subset['stNum_changed'] = subset['stNum'].diff() != 0
            subset['stNum_changed'] = subset['stNum_changed'].fillna(False)
            
            # Sample for visualization (use first N messages for clarity)
            sample_size = min(2000, len(subset))
            subset_vis = subset.head(sample_size)
            
            # Create time index
            time_idx = np.arange(len(subset_vis))
            
            # Plot heartbeat messages (normal, no event)
            heartbeat_mask = (~subset_vis['stNum_changed']) & (subset_vis['attack'] == 0)
            ax.scatter(time_idx[heartbeat_mask], 
                      np.zeros(heartbeat_mask.sum()),
                      c='gray', alpha=0.3, s=10, label='Heartbeat', marker='.')
            
            # Plot events (stNum changed, not attack)
            event_mask = subset_vis['stNum_changed'] & (subset_vis['attack'] == 0)
            ax.scatter(time_idx[event_mask], 
                      np.ones(event_mask.sum()) * 0.5,
                      c='green', alpha=0.7, s=60, label='Event', marker='^')
            
            # Plot attacks
            attack_mask = subset_vis['attack'] == 1
            ax.scatter(time_idx[attack_mask], 
                      np.ones(attack_mask.sum()),
                      c='red', alpha=0.8, s=80, label='Attack', marker='x', linewidths=2)
            
            # Calculate statistics
            total_msgs = len(subset)
            total_events = subset['stNum_changed'].sum()
            total_attacks = (subset['attack'] == 1).sum()
            events_with_attacks = ((subset['stNum_changed']) & (subset['attack'] == 1)).sum()
            
            # Add statistics box
            stats_text = (f'Total: {total_msgs}\n'
                         f'Events: {total_events}\n'
                         f'Attacks: {total_attacks}\n'
                         f'Events+Attack: {events_with_attacks}')
            ax.text(0.98, 0.97, stats_text,
                   transform=ax.transAxes,
                   verticalalignment='top',
                   horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                   fontsize=9)
            
            ax.set_xlabel('Message Index')
            ax.set_ylabel('Message Type')
            ax.set_yticks([0, 0.5, 1.0])
            ax.set_yticklabels(['Heartbeat', 'Event', 'Attack'])
            ax.set_title(f'{attack_type.capitalize()} - Event Timeline (n={sample_size}/{total_msgs})')
            ax.legend(loc='upper left', fontsize=8)
            ax.grid(alpha=0.3)
            ax.set_ylim(-0.2, 1.2)
        
        plt.tight_layout()
        
        save_path = self.output_dir / 'task1_event_timeline.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
        
        if logger:
            logger.log_figure(fig, "task1_event_timeline")
        
        plt.close()

    def detect_goose_events(self, df: pd.DataFrame, logger: Optional[WandbLogger]) -> pd.DataFrame:
        df = df.sort_values(['gocbRef', 'Time']).copy()
        events = []
        for gocb, group in df.groupby('gocbRef'):
            group = group.reset_index(drop=True)
            group['ΔstNum'] = group['stNum'].diff()
            group['ΔsqNum'] = group['sqNum'].diff()
            group['ΔTime']  = group['Time'].diff()
            # Rule 1: stNum change but sqNum not reset
            rule1_violation = (group['ΔstNum'] > 0) & (group['sqNum'] != 0)
            # Rule 2: sqNum jump or reset without stNum change
            rule2_violation = (group['ΔstNum'] == 0) & (group['ΔsqNum'] > 1)

            # Rule 3: timeAllowedtoLive unreasonable
            rule3_violation = (group['timeAllowedtoLive'] < 1) | (group['timeAllowedtoLive'] > 10000)
            # Rule 4: inter-frame interval > timeAllowedtoLive
            rule4_violation = group['ΔTime'] > group['timeAllowedtoLive']

            group['rule1_v'] = rule1_violation
            group['rule2_v'] = rule2_violation
            group['rule3_v'] = rule3_violation
            group['rule4_v'] = rule4_violation

            events.append(group)

        df_events = pd.concat(events, ignore_index=True)

    # Save and summarize
        out_path = Path('outputs/tables/task1_goose_event_flags.csv')
        df_events.to_csv(out_path, index=False)
        print(f"Saved event-flagged data: {out_path}")

        summary = (df_events[['rule1_v','rule2_v','rule3_v','rule4_v']]
               .sum().rename('violations').to_frame())
        print("\nGOOSE rule violation summary:\n", summary)

        return df_events

    def analyze_goose_rules(self, df: pd.DataFrame, logger: WandbLogger = None):
        """Analyze adherence to GOOSE protocol rules."""
        print("\n" + "="*70)
        print("GOOSE PROTOCOL RULE ANALYSIS")
        print("="*70)
        
        results = []
        
        # Group by gocbRef for proper analysis - analyze 10 gocbRefs
        for gocbref in df['gocbRef'].unique()[:10]:
            subset = df[df['gocbRef'] == gocbref].sort_values('Time')
            
            if len(subset) < 2:
                continue
            
            # Check Rule 1: stNum increment should reset sqNum
            if 'stNum' in subset.columns and 'sqNum' in subset.columns:
                stnum_changes = subset['stNum'].diff() != 0
                sqnum_resets = subset['sqNum'] == 0
                
                rule1_violations = stnum_changes & ~sqnum_resets
                rule1_violation_rate = rule1_violations.sum() / stnum_changes.sum() if stnum_changes.sum() > 0 else 0
                
                # Count violations that coincide with attacks
                violations_with_attacks = (rule1_violations & (subset['attack'] == 1)).sum()
                
                results.append({
                    'gocbRef': gocbref[:30],
                    'Total Messages': len(subset),
                    'stNum Changes': stnum_changes.sum(),
                    'Rule 1 Violations': rule1_violations.sum(),
                    'Violations+Attacks': violations_with_attacks,
                    'Rule 1 Violation Rate': f"{rule1_violation_rate:.2%}"
                })
        
        if results:
            results_df = pd.DataFrame(results)
            print("\nGOOSE Rule Violations by gocbRef:")
            print(results_df.to_string(index=False))
            
            # Print detailed examples
            print("\n" + "-"*70)
            print("DETAILED VIOLATION EXAMPLES:")
            print("-"*70)
            
            for i, row in results_df.head(3).iterrows():
                if row['Rule 1 Violations'] > 0:
                    print(f"\ngocbRef: {row['gocbRef']}")
                    print(f"  Violations: {row['Rule 1 Violations']} / {row['stNum Changes']} stNum changes")
                    print(f"  Violations during attacks: {row['Violations+Attacks']}")
            
            # Save table
            table_path = Path('outputs/tables') / 'task1_goose_rule_analysis.csv'
            table_path.parent.mkdir(exist_ok=True)
            results_df.to_csv(table_path, index=False)
            print(f"\nSaved: {table_path}")
            
            if logger:
                logger.log_dataframe(results_df, "task1_goose_rule_analysis")


def run_task1(config: Dict[str, Any], logger: WandbLogger = None) -> Dict:
    """
    Run Task 1: Exploratory Data Analysis.
    
    Args:
        config: Configuration dictionary
        logger: WandbLogger instance (optional)
    
    Returns:
        Dictionary with EDA results
    """
    print("\n" + "="*70)
    print("TASK 1: EXPLORATORY DATA ANALYSIS")
    print("="*70 + "\n")
    
    explorer = GOOSEDataExplorer()
    
    # Load data
    data_dir = config.get('data', {}).get('raw_dir', 'data/raw')
    train_data = explorer.load_data(data_dir)
    
    # Basic statistics
    stats_df = explorer.basic_statistics(train_data)
    
    # Save statistics
    stats_path = Path('outputs/tables') / 'task1_basic_statistics.csv'
    stats_path.parent.mkdir(exist_ok=True)
    stats_df.to_csv(stats_path)
    print(f"\nStatistics saved to: {stats_path}")
    
    if logger:
        logger.log_dataframe(stats_df, "task1_basic_statistics")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    
    explorer.plot_attack_distribution(all_data, logger)
    explorer.plot_temporal_patterns(all_data, logger)
    explorer.plot_feature_distributions(all_data, logger)
    explorer.plot_correlation_analysis(all_data, logger)
    explorer.plot_dimensionality_reduction(all_data, logger)
    explorer.analyze_goose_rules(all_data, logger)
    # Heartbeat plots per gocbRef (cap at, say, first 20 to keep it manageable)
    explorer.plot_heartbeat_by_gocbref_split_by_attacktype(
    train_data, logger, max_gocbref_plots=20
)


    df_events = explorer.detect_goose_events(all_data, logger)
    
    print("\n✓ Task 1 completed!")
    print(f"All figures saved to: {explorer.output_dir.absolute()}")
    # Compare with actual attack label
    for rule in ['rule1_v', 'rule2_v', 'rule3_v', 'rule4_v']:
        corr = df_events[rule].astype(int).corr(df_events['attack'])
        print(f"Correlation of {rule} violation with attack label: {corr:.3f}")
        # Or confusion-style table
        pd.crosstab(df_events['attack'], df_events[['rule1_v','rule2_v','rule3_v','rule4_v']].any(axis=1),
                    rownames=['Label'], colnames=['Any rule violated'])
    df_events["any_rule_v"] = df_events[["rule1_v","rule2_v","rule3_v","rule4_v"]].any(axis=1)
    df_events["predicted_attack"] = df_events["any_rule_v"].astype(int)

    print(confusion_matrix(df_events["attack"], df_events["predicted_attack"]))
    print(classification_report(df_events["attack"], df_events["predicted_attack"]))

    return {
        'total_samples': len(train_data),
        'attack_ratio': train_data['attack'].mean(),
        'num_gocbref': train_data['gocbRef'].nunique(),
        'stats': stats_df
    }


if __name__ == '__main__':
    import yaml
    
    # Load config
    config_path = 'config/config.yaml'
    if Path(config_path).exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = {'data': {'raw_dir': 'data/raw'}}
    
    # Run task
    results = run_task1(config, logger=None)
    print("\nTask 1 completed successfully!")