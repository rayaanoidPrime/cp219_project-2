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
    
    def load_data(self, data_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load all training and test datasets."""
        data_dir = Path(data_dir)
        
        attack_types = ['replay', 'masquerade', 'injection', 'poisoning']
        
        # Load training data
        train_dfs = []
        for attack in attack_types:
            df = pd.read_csv(data_dir / f'{attack}_labelled_train.csv')
            df['attack_type'] = attack
            train_dfs.append(df)
        train_data = pd.concat(train_dfs, ignore_index=True)
        
        # Load test data
        test_dfs = []
        for attack in attack_types:
            df = pd.read_csv(data_dir / f'{attack}_labelled_test.csv')
            df['attack_type'] = attack
            test_dfs.append(df)
        test_data = pd.concat(test_dfs, ignore_index=True)
        
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
        
        # Numerical features summary
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        numerical_cols = [col for col in numerical_cols if col not in ['attack']]
        
        stats_df = df[numerical_cols].describe()
        
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
        
        # Sample for efficiency
        df_sample = df[numerical_cols].sample(min(10000, len(df)))
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Overall correlation
        corr = df_sample.corr()
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                   ax=axes[0], square=True, linewidths=0.5)
        axes[0].set_title('Feature Correlation Heatmap')
        
        # Correlation with attack label
        if len(numerical_cols) > 0:
            df_with_label = df[numerical_cols + ['attack']].sample(min(10000, len(df)))
            attack_corr = df_with_label.corr()['attack'].drop('attack').sort_values()
            
            colors = ['red' if x < 0 else 'green' for x in attack_corr.values]
            axes[1].barh(range(len(attack_corr)), attack_corr.values, color=colors, alpha=0.7)
            axes[1].set_yticks(range(len(attack_corr)))
            axes[1].set_yticklabels(attack_corr.index)
            axes[1].set_xlabel('Correlation with Attack Label')
            axes[1].set_title('Feature Correlation with Attack')
            axes[1].axvline(0, color='black', linewidth=0.5)
            axes[1].grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        
        save_path = self.output_dir / 'task1_correlation_analysis.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
        
        if logger:
            logger.log_figure(fig, "task1_correlation_analysis")
        
        plt.close()
    
    def plot_dimensionality_reduction(self, df: pd.DataFrame, logger: WandbLogger = None):
        """PCA and t-SNE visualization."""
        print("\nPerforming dimensionality reduction...")
        
        # Prepare data
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        numerical_cols = [col for col in numerical_cols if col not in ['attack']]
        
        if len(numerical_cols) < 2:
            print("Not enough features for dimensionality reduction")
            return
        
        # Sample for efficiency
        sample_size = min(5000, len(df))
        df_sample = df.sample(sample_size, random_state=42)
        
        X = df_sample[numerical_cols].fillna(0)
        y = df_sample['attack'].values
        attack_types = df_sample['attack_type'].values
        
        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # PCA
        print("Running PCA...")
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        
        for attack in [0, 1]:
            mask = y == attack
            axes[0].scatter(X_pca[mask, 0], X_pca[mask, 1], 
                          alpha=0.5, s=20,
                          label='Normal' if attack == 0 else 'Attack',
                          c='blue' if attack == 0 else 'red')
        axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} var)')
        axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} var)')
        axes[0].set_title('PCA Projection')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        # t-SNE
        print("Running t-SNE...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        X_tsne = tsne.fit_transform(X_scaled)
        
        for attack in [0, 1]:
            mask = y == attack
            axes[1].scatter(X_tsne[mask, 0], X_tsne[mask, 1], 
                          alpha=0.5, s=20,
                          label='Normal' if attack == 0 else 'Attack',
                          c='blue' if attack == 0 else 'red')
        axes[1].set_xlabel('t-SNE 1')
        axes[1].set_ylabel('t-SNE 2')
        axes[1].set_title('t-SNE Projection')
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        
        save_path = self.output_dir / 'task1_dimensionality_reduction.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
        
        if logger:
            logger.log_figure(fig, "task1_dimensionality_reduction")
        
        plt.close()
        
        # Also plot by attack type
        fig, ax = plt.subplots(figsize=(10, 8))
        
        colors_map = {
            'Replay': '#e74c3c',
            'Masquerade': '#f39c12', 
            'Injection': '#9b59b6',
            'Poisoning': '#3498db'
        }
        
        for attack_type in df_sample['attack_type'].unique():
            mask = (attack_types == attack_type) & (y == 1)
            ax.scatter(X_tsne[mask, 0], X_tsne[mask, 1], 
                      alpha=0.6, s=30, label=attack_type,
                      c=colors_map.get(attack_type, '#95a5a6'))
        
        # Normal data
        mask = y == 0
        ax.scatter(X_tsne[mask, 0], X_tsne[mask, 1], 
                  alpha=0.3, s=20, label='Normal', c='lightgray')
        
        ax.set_xlabel('t-SNE 1')
        ax.set_ylabel('t-SNE 2')
        ax.set_title('t-SNE Projection by Attack Type')
        ax.legend()
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        
        save_path = self.output_dir / 'task1_tsne_by_attack_type.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
        
        if logger:
            logger.log_figure(fig, "task1_tsne_by_attack_type")
        
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
        
        # Group by gocbRef for proper analysis
        for gocbref in df['gocbRef'].unique()[:5]:  # Analyze first 5
            subset = df[df['gocbRef'] == gocbref].sort_values('Time')
            
            if len(subset) < 2:
                continue
            
            # Check Rule 1: stNum increment should reset sqNum
            if 'stNum' in subset.columns and 'sqNum' in subset.columns:
                stnum_changes = subset['stNum'].diff() != 0
                sqnum_resets = subset['sqNum'] == 0
                
                rule1_violations = stnum_changes & ~sqnum_resets
                rule1_violation_rate = rule1_violations.sum() / stnum_changes.sum() if stnum_changes.sum() > 0 else 0
                
                results.append({
                    'gocbRef': gocbref[:30],
                    'Total Messages': len(subset),
                    'stNum Changes': stnum_changes.sum(),
                    'Rule 1 Violations': rule1_violations.sum(),
                    'Rule 1 Violation Rate': f"{rule1_violation_rate:.2%}"
                })
        
        if results:
            results_df = pd.DataFrame(results)
            print("\nGOOSE Rule Violations by gocbRef:")
            print(results_df.to_string(index=False))
            
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
    train_data, test_data = explorer.load_data(data_dir)
    
    # Combine for overall EDA
    all_data = pd.concat([train_data, test_data], ignore_index=True)
    
    # Basic statistics
    stats_df = explorer.basic_statistics(all_data)
    
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
        'total_samples': len(all_data),
        'attack_ratio': all_data['attack'].mean(),
        'num_gocbref': all_data['gocbRef'].nunique(),
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