import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any, List, Tuple
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Import shared preprocessing utilities
from src.preprocessing import (
    CORE_FIELDS,
    ALLOWED_FIELDS,
    engineer_features_new,
    preprocess_dataframe,
    standardize_schema,
    engineer_features,
    get_numeric_features,
    load_and_preprocess
)


# ==============================================================
# Task 2: Feature & Attack Characterization (GOOSE-only fields)
# FOUR ANALYSIS MODES: CORE / FULL / NEW / CORE_NEW
# ==============================================================

_EPS = 1e-6


class AttackCharacterizer:
    """Characterize features and attack signatures using only the agreed fields."""

    def __init__(self, config: Dict[str, Any], logger=None, mode: str = 'core'):
        """
        Args:
            mode: 'core', 'full', 'new', or 'core_new'
        """
        self.config = config
        self.logger = logger
        self.mode = mode
        self.train_data: Dict[str, pd.DataFrame] = {}
        self.feature_importance = {}

        # Create output directories with mode suffix
        self.fig_dir = Path(f'outputs/figures/task2_{mode}')
        self.table_dir = Path(f'outputs/tables/task2_{mode}')
        self.fig_dir.mkdir(parents=True, exist_ok=True)
        self.table_dir.mkdir(parents=True, exist_ok=True)

        # Style
        sns.set_style("whitegrid")
        plt.rcParams['figure.dpi'] = 300

        print(f"\n{'='*70}")
        print(f"RUNNING TASK 2 IN '{mode.upper()}' MODE")
        if mode == 'core':
            print("Using ONLY the 10 CORE_FIELDS")
        elif mode == 'full':
            print("Using CORE + ALLOWED + ENGINEERED features")
        elif mode == "new":
            print("Using ONLY NEW engineered features")
        elif mode == "core_new":
            print("Using CORE fields + NEW engineered features")
        else:
            raise ValueError(f"Unknown mode: {mode}. Must be 'core', 'full', 'new', or 'core_new'")
        print(f"{'='*70}\n")

    # ----------------------- Data Loading -----------------------

    def load_data(self):
        """Load datasets and subselect fields based on mode."""
        print("Loading datasets...")
        data_dir = Path(self.config['data']['raw_dir'])

        attack_types = ['replay', 'masquerade', 'injection', 'poisoning']

        for attack in attack_types:
            train_file = self.config['data']['train_files'][attack]
            file_path = data_dir / train_file
            
            # Load raw data first
            df = pd.read_csv(file_path)
            
            # Select fields based on mode
            if self.mode == 'core':
                keep = [c for c in CORE_FIELDS if c in df.columns]
            elif self.mode == 'full':
                keep = [c for c in ALLOWED_FIELDS if c in df.columns]
            elif self.mode in ['new', 'core_new']:
                # For new modes, we need minimal fields to generate new features
                
                keep = [c for c in CORE_FIELDS if c in df.columns]
            else:
                raise ValueError(f"Unknown mode: {self.mode}")
            
            df = df[keep].copy()
            
            # Apply preprocessing (splits boolean, handles bit-string)
            df = preprocess_dataframe(df)
            df = standardize_schema(df)
            
            # Apply feature engineering based on mode
            if self.mode == 'full':
                df = engineer_features(df)
            elif self.mode in ['new', 'core_new']:
                df = engineer_features_new(df)
                
                # CRITICAL: For 'new' mode, drop the original CORE numeric features
                # We only want the NEW engineered features
                if self.mode == 'new':
                    core_numeric_to_drop = CORE_FIELDS
                    df.drop(columns=[c for c in core_numeric_to_drop if c in df.columns and c != 'attack'], 
                           inplace=True, errors='ignore')
                    print(f"    (NEW mode: dropped original core features, keeping only engineered)")

            if 'attack' not in df.columns:
                raise ValueError(f"'attack' column missing in {train_file}.")
            df['attack_type'] = attack

            self.train_data[attack] = df
            print(f"  {attack}: {len(df)} samples, {len(df.columns)} columns")

        used_fields = sorted(set().union(*[set(d.columns) for d in self.train_data.values()]))
        used_fields_df = pd.DataFrame({'used_fields': used_fields})
        used_fields_df.to_csv(self.table_dir / 'used_fields.csv', index=False)
        print(f"Saved used field list to {self.table_dir / 'used_fields.csv'}")
        if self.logger is not None:
            self.logger.log_dataframe(used_fields_df, f"task2_{self.mode}/used_fields")

    # ------------------- Analysis Feature Set -------------------

    def _numeric_analysis_features(self, df: pd.DataFrame) -> List[str]:
        """Build the list of numeric features for stats/importance (no leakage)."""
        return get_numeric_features(df, mode=self.mode)

    # -------------------- Feature Statistics --------------------

    def compute_feature_statistics(self):
        """Compute feature statistics per attack type (Mann–Whitney U)."""
        print("\n" + "="*70)
        print(f"FEATURE STATISTICS BY ATTACK TYPE ({self.mode.upper()} MODE)")
        print("="*70)

        # Data is already preprocessed and engineered in load_data()
        all_data = [df for df in self.train_data.values()]
        combined = pd.concat(all_data, ignore_index=True)
        numeric_cols = self._numeric_analysis_features(combined)

        print(f"\nAnalyzing {len(numeric_cols)} numeric features...")

        results = []
        for attack_type in combined['attack_type'].unique():
            subset = combined[combined['attack_type'] == attack_type]
            normal = subset[subset['attack'] == 0]
            attack = subset[subset['attack'] == 1]

            for col in numeric_cols:
                normal_vals = normal[col].replace([np.inf, -np.inf], np.nan).dropna()
                attack_vals = attack[col].replace([np.inf, -np.inf], np.nan).dropna()
                if len(normal_vals) == 0 or len(attack_vals) == 0:
                    continue
                try:
                    statistic, pvalue = stats.mannwhitneyu(
                        normal_vals, attack_vals, alternative='two-sided'
                    )
                    results.append({
                        'Attack Type': attack_type.capitalize(),
                        'Feature': col,
                        'Normal Mean': normal_vals.mean(),
                        'Normal Std': normal_vals.std(),
                        'Attack Mean': attack_vals.mean(),
                        'Attack Std': attack_vals.std(),
                        'Mean Difference': abs(attack_vals.mean() - normal_vals.mean()),
                        'P-value': pvalue,
                        'Significant (p<0.01)': pvalue < 0.01
                    })
                except Exception:
                    continue

        if not results:
            print("No statistical results computed (check data availability).")
            return None

        stats_df = pd.DataFrame(results).sort_values(
            ['Attack Type', 'Mean Difference'], ascending=[True, False]
        )

        print("\nTop discriminative features by attack type:")
        for attack_type in stats_df['Attack Type'].unique():
            print(f"\n{attack_type}:")
            subset = stats_df[stats_df['Attack Type'] == attack_type].head(5)
            print(subset[['Feature', 'Normal Mean', 'Attack Mean',
                          'Mean Difference', 'P-value']].to_string(index=False))

        out_path = self.table_dir / 'feature_statistics.csv'
        stats_df.to_csv(out_path, index=False)
        print(f"Saved: {out_path}")
        if self.logger is not None:
            self.logger.log_dataframe(stats_df, f"task2_{self.mode}/feature_statistics")
        return stats_df

    # ---------------- Unsupervised Importance ----------------

    def unsupervised_feature_importance(self):
        """Estimate feature importance via variance & PCA (unsupervised)."""
        print("\n" + "="*70)
        print(f"UNSUPERVISED FEATURE IMPORTANCE ({self.mode.upper()} MODE)")
        print("="*70)

        # Data is already preprocessed and engineered in load_data()
        all_data = [df for df in self.train_data.values()]
        df_all = pd.concat(all_data, ignore_index=True)

        numeric_cols = self._numeric_analysis_features(df_all)
        X = df_all[numeric_cols].replace([np.inf, -np.inf], np.nan)

        threshold = len(X) * 0.5
        X = X.dropna(axis=1, thresh=threshold)

        for col in X.columns:
            if X[col].isna().any():
                med = X[col].median()
                X[col].fillna(0 if pd.isna(med) else med, inplace=True)

        X = X.loc[:, X.var() > 0]
        X = X.replace([np.inf, -np.inf], np.nan).dropna(axis=1)

        print(f"\nFeatures after cleaning: {len(X.columns)}")

        variances = X.var().sort_values(ascending=False)
        print("\nTop 15 features by variance:")
        print(variances.head(15).to_string())

        pc1_importance = pd.Series(dtype=float)
        try:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            if np.isnan(X_scaled).any():
                X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)

            pca = PCA(n_components=min(10, X_scaled.shape[1]))
            pca.fit(X_scaled)

            pc1_importance = pd.Series(
                np.abs(pca.components_[0]),
                index=X.columns
            ).sort_values(ascending=False)

            print("\nTop 15 features by PCA (PC1) importance:")
            print(pc1_importance.head(15).to_string())

        except Exception as e:
            print(f"\nWarning: PCA analysis failed: {e}")
            print("Skipping PCA-based importance analysis.")

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(f'Unsupervised Feature Importance ({self.mode.upper()} Mode)', 
                     fontsize=14, fontweight='bold')

        top_var = variances.head(15)
        axes[0].barh(range(len(top_var)), top_var.values)
        axes[0].set_yticks(range(len(top_var)))
        axes[0].set_yticklabels(top_var.index, fontsize=9)
        axes[0].set_xlabel('Variance')
        axes[0].set_title('Top 15 Features by Variance', fontweight='bold')
        axes[0].invert_yaxis()

        if not pc1_importance.empty:
            top_pca = pc1_importance.head(15)
            axes[1].barh(range(len(top_pca)), top_pca.values)
            axes[1].set_yticks(range(len(top_pca)))
            axes[1].set_yticklabels(top_pca.index, fontsize=9)
            axes[1].set_xlabel('|PC1 Loading|')
            axes[1].set_title('Top 15 Features by PCA Importance', fontweight='bold')
            axes[1].invert_yaxis()
        else:
            axes[1].text(0.5, 0.5, 'PCA analysis not available',
                         ha='center', va='center', transform=axes[1].transAxes)
            axes[1].set_title('PCA Importance (Failed)', fontweight='bold')

        plt.tight_layout()
        fig_path = self.fig_dir / 'feature_importance_unsupervised.png'
        plt.savefig(fig_path, bbox_inches='tight')
        print(f"\nSaved: {fig_path}")
        if self.logger:
            self.logger.log_figure(fig, f"task2_{self.mode}/feature_importance_unsupervised")
        plt.close(fig)

        importance_df = pd.DataFrame({
            'Feature': variances.index,
            'Variance': variances.values,
            'PCA_Importance': [pc1_importance.get(f, 0) for f in variances.index]
        })
        importance_df.to_csv(self.table_dir / 'feature_importance_scores.csv', index=False)
        if self.logger is not None:
            self.logger.log_dataframe(importance_df, f"task2_{self.mode}/feature_importance_scores")

        clean_features_df = pd.DataFrame({
            'clean_numeric_features': list(X.columns)
        })
        clean_features_df.to_csv(self.table_dir / 'clean_numeric_features.csv', index=False)
        if self.logger is not None:
            self.logger.log_dataframe(clean_features_df, f"task2_{self.mode}/clean_numeric_features")

        return importance_df

    # --------------- Supervised Validation (RF) ---------------

    def supervised_feature_importance(self):
        """Supervised RF importance (validation only; not for IDS training)."""
        print("\n" + "="*70)
        print(f"SUPERVISED FEATURE IMPORTANCE ({self.mode.upper()} MODE - For Validation Only)")
        print("="*70)

        # Data is already preprocessed and engineered in load_data()
        all_data = [df for df in self.train_data.values()]
        df_all = pd.concat(all_data, ignore_index=True)

        numeric_cols = self._numeric_analysis_features(df_all)
        X = df_all[numeric_cols].replace([np.inf, -np.inf], np.nan)

        threshold = len(X) * 0.5
        X = X.dropna(axis=1, thresh=threshold)

        for col in X.columns:
            if X[col].isna().any():
                med = X[col].median()
                X[col].fillna(0 if pd.isna(med) else med, inplace=True)

        X = X.loc[:, X.var() > 0]
        X = X.replace([np.inf, -np.inf], np.nan).dropna(axis=1)

        y = df_all['attack']

        print(f"\nFeatures for RF training: {len(X.columns)}")
        print(f"Samples: {len(X)}")

        rf = RandomForestClassifier(
            n_estimators=200, random_state=42, n_jobs=-1, max_depth=None
        )
        rf.fit(X, y)

        importance = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
        print("\nTop 15 features by Random Forest importance:")
        print(importance.head(15).to_string())

        fig, ax = plt.subplots(figsize=(10, 8))
        top_20 = importance.head(20)
        ax.barh(range(len(top_20)), top_20.values)
        ax.set_yticks(range(len(top_20)))
        ax.set_yticklabels(top_20.index, fontsize=9)
        ax.set_xlabel('Feature Importance', fontsize=11)
        ax.set_title(f'Top 20 Features by Random Forest Importance\n({self.mode.upper()} Mode - For Validation Only)',
                    fontweight='bold', fontsize=12)
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        fig_path = self.fig_dir / 'feature_importance_supervised.png'
        plt.savefig(fig_path, bbox_inches='tight')
        print(f"Saved: {fig_path}")
        if self.logger:
            self.logger.log_figure(fig, f"task2_{self.mode}/feature_importance_supervised")
        plt.close(fig)

        importance.to_csv(self.table_dir / 'feature_importance_rf.csv', header=['Importance'])
        if self.logger is not None:
            self.logger.log_dataframe(importance.reset_index().rename(columns={'index': 'Feature', 0: 'Importance'})
                                      if isinstance(importance, pd.Series) else importance,
                                      f"task2_{self.mode}/feature_importance_rf")
        return importance

    # --------------- Attack Signature Characterization ---------------

    def characterize_attack_signatures(self):
        """Summarize statistical & temporal signatures per attack type."""
        print("\n" + "="*70)
        print(f"ATTACK SIGNATURE CHARACTERIZATION ({self.mode.upper()} MODE)")
        print("="*70)

        signatures = {}

        for attack_type, df in self.train_data.items():
            # Data is already preprocessed and engineered in load_data()
            normal = df[df['attack'] == 0]
            attack = df[df['attack'] == 1]

            sig = {
                'attack_type': attack_type,
                'num_samples': len(df),
                'num_attack': len(attack),
                'attack_ratio': (len(attack) / max(1, len(df))),
            }

            def _m(series): return series.mean() if series.size else np.nan

            if 'stNum' in df.columns:
                sig['stNum_mean_normal'] = _m(normal['stNum'])
                sig['stNum_mean_attack'] = _m(attack['stNum'])

            if 'sqNum' in df.columns:
                sig['sqNum_mean_normal'] = _m(normal['sqNum'])
                sig['sqNum_mean_attack'] = _m(attack['sqNum'])

            if 'timeAllowedtoLive' in df.columns:
                sig['ttl_mean_normal'] = _m(normal['timeAllowedtoLive'])
                sig['ttl_mean_attack'] = _m(attack['timeAllowedtoLive'])

            length_col = 'Frame length on the wire' if 'Frame length on the wire' in df.columns else 'Length'
            if length_col in df.columns:
                sig['length_mean_normal'] = _m(normal[length_col])
                sig['length_mean_attack'] = _m(attack[length_col])

            # Engineered timing & sequence signals (only in full mode)
            if self.mode == 'full':
                for k in [
                    'inter_arrival', 'msg_rate', 'byte_rate',
                    'ttl_violation', 'sqNum_jump', 'stNum_change',
                    'heartbeat_interval_error', 'heartbeat_within_tol',
                    'sq_reset_violation', 't_st_consistency_violation',
                    'status_change_missing_on_event', 'status_change_violation_on_heartbeat'
                ]:
                    if k in df.columns:
                        sig[f'{k}_mean_normal'] = _m(normal[k])
                        sig[f'{k}_mean_attack'] = _m(attack[k])

            signatures[attack_type] = sig

            # Console summary
            print(f"\n{attack_type.upper()} Attack:")
            for k in ['stNum', 'sqNum', 'timeAllowedtoLive', length_col]:
                k_norm = f'{k}_mean_normal' if 'mean' not in k else k
                k_att = f'{k}_mean_attack' if 'mean' not in k else k
                if k_norm in sig and k_att in sig:
                    print(f"  {k}: Normal={sig[k_norm]:.4f}, Attack={sig[k_att]:.4f}")
            
            if self.mode == 'full':
                for k in ['inter_arrival', 'msg_rate', 'ttl_violation', 'sqNum_jump', 'heartbeat_interval_error']:
                    k_norm = f'{k}_mean_normal'
                    k_att = f'{k}_mean_attack'
                    if k_norm in sig and k_att in sig:
                        print(f"  {k}: Normal={sig[k_norm]:.4f}, Attack={sig[k_att]:.4f}")

        self._visualize_attack_signatures(signatures)

        sig_df = pd.DataFrame(signatures).T
        out_path = self.table_dir / 'attack_signatures.csv'
        sig_df.to_csv(out_path)
        print(f"Saved: {out_path}")
        if self.logger is not None:
            self.logger.log_dataframe(sig_df.reset_index().rename(columns={'index': 'attack_type'}),
                                      f"task2_{self.mode}/attack_signatures")

        return signatures

    def _visualize_attack_signatures(self, signatures: Dict):
        """Visualize key signature comparisons across attack types."""
        if not signatures:
            return

        attack_types = list(signatures.keys())

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Attack Signature Characteristics ({self.mode.upper()} Mode)', 
                     fontsize=14, fontweight='bold')

        # 1) Attack ratio
        ratios = [signatures[a]['attack_ratio'] for a in attack_types]
        axes[0, 0].bar(attack_types, ratios)
        axes[0, 0].set_ylabel('Attack Ratio')
        axes[0, 0].set_title('Attack Prevalence by Type')
        axes[0, 0].set_ylim([0, max(ratios) * 1.2])
        for i, v in enumerate(ratios):
            axes[0, 0].text(i, v, f'{v:.2%}', ha='center', va='bottom')

        # 2) stNum mean (Normal vs Attack)
        if all('stNum_mean_attack' in signatures[a] for a in attack_types):
            stnum_normal = [signatures[a].get('stNum_mean_normal', np.nan) for a in attack_types]
            stnum_attack = [signatures[a].get('stNum_mean_attack', np.nan) for a in attack_types]
            x = np.arange(len(attack_types)); width = 0.35
            axes[0, 1].bar(x - width/2, stnum_normal, width, label='Normal', alpha=0.8)
            axes[0, 1].bar(x + width/2, stnum_attack, width, label='Attack', alpha=0.8)
            axes[0, 1].set_ylabel('Mean stNum')
            axes[0, 1].set_title('Status Number Distribution')
            axes[0, 1].set_xticks(x)
            axes[0, 1].set_xticklabels([a.capitalize() for a in attack_types])
            axes[0, 1].legend()

        # 3) sqNum mean (Normal vs Attack)
        if all('sqNum_mean_attack' in signatures[a] for a in attack_types):
            sqnum_normal = [signatures[a].get('sqNum_mean_normal', np.nan) for a in attack_types]
            sqnum_attack = [signatures[a].get('sqNum_mean_attack', np.nan) for a in attack_types]
            x = np.arange(len(attack_types)); width = 0.35
            axes[1, 0].bar(x - width/2, sqnum_normal, width, label='Normal', alpha=0.8)
            axes[1, 0].bar(x + width/2, sqnum_attack, width, label='Attack', alpha=0.8)
            axes[1, 0].set_ylabel('Mean sqNum')
            axes[1, 0].set_title('Sequence Number Distribution')
            axes[1, 0].set_xticks(x)
            axes[1, 0].set_xticklabels([a.capitalize() for a in attack_types])
            axes[1, 0].legend()

        # 4) Length mean (Normal vs Attack)
        length_keys = ('length_mean_normal', 'length_mean_attack')
        if all(all(k in signatures[a] for k in length_keys) for a in attack_types):
            length_normal = [signatures[a]['length_mean_normal'] for a in attack_types]
            length_attack = [signatures[a]['length_mean_attack'] for a in attack_types]
            x = np.arange(len(attack_types)); width = 0.35
            axes[1, 1].bar(x - width/2, length_normal, width, label='Normal', alpha=0.8)
            axes[1, 1].bar(x + width/2, length_attack, width, label='Attack', alpha=0.8)
            axes[1, 1].set_ylabel('Mean Length (bytes)')
            axes[1, 1].set_title('Message Length Distribution')
            axes[1, 1].set_xticks(x)
            axes[1, 1].set_xticklabels([a.capitalize() for a in attack_types])
            axes[1, 1].legend()

        plt.tight_layout()
        fig_path = self.fig_dir / 'attack_signatures_comparison.png'
        plt.savefig(fig_path, bbox_inches='tight')
        print(f"\nSaved: {fig_path}")
        if self.logger:
            self.logger.log_figure(fig, f"task2_{self.mode}/attack_signatures")
        plt.close(fig)

    # ---------------- Detection Difficulty Ranking ----------------

    def analyze_attack_difficulty(self):
        """Rank attack types by separability (inter vs. intra-class)."""
        print("\n" + "="*70)
        print(f"ATTACK DETECTION DIFFICULTY ANALYSIS ({self.mode.upper()} MODE)")
        print("="*70)

        # Data is already preprocessed and engineered in load_data()
        all_data = [df for df in self.train_data.values()]
        df_all = pd.concat(all_data, ignore_index=True)

        numeric_cols = self._numeric_analysis_features(df_all)
        X = df_all[numeric_cols].replace([np.inf, -np.inf], np.nan)

        threshold = len(X) * 0.5
        X = X.dropna(axis=1, thresh=threshold)

        for col in X.columns:
            if X[col].isna().any():
                med = X[col].median()
                X[col].fillna(0 if pd.isna(med) else med, inplace=True)

        X = X.loc[:, X.var() > 0]
        X = X.replace([np.inf, -np.inf], np.nan).dropna(axis=1)

        print(f"\nFeatures for difficulty analysis: {len(X.columns)}")

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)

        difficulty = []
        for attack_type in df_all['attack_type'].unique():
            attack_mask = (df_all['attack_type'] == attack_type) & (df_all['attack'] == 1)
            normal_mask = (df_all['attack_type'] == attack_type) & (df_all['attack'] == 0)

            attack_samples = X_scaled[attack_mask]
            normal_samples = X_scaled[normal_mask]

            if len(attack_samples) > 0 and len(normal_samples) > 0:
                attack_center = attack_samples.mean(axis=0)
                normal_center = normal_samples.mean(axis=0)
                inter_class_distance = np.linalg.norm(attack_center - normal_center)

                attack_var = attack_samples.var(axis=0).mean()
                normal_var = normal_samples.var(axis=0).mean()
                intra_class_variance = (attack_var + normal_var) / 2

                separability = inter_class_distance / (np.sqrt(intra_class_variance) + _EPS)
                difficulty.append({
                    'Attack Type': attack_type.capitalize(),
                    'Inter-class Distance': inter_class_distance,
                    'Intra-class Variance': intra_class_variance,
                    'Separability Score': separability,
                    'Detection Difficulty': 'Easy' if separability > 5 else
                                             ('Medium' if separability > 2 else 'Hard')
                })

        difficulty_df = pd.DataFrame(difficulty).sort_values('Separability Score', ascending=False)

        print("\nAttack Detection Difficulty Ranking:")
        if not difficulty_df.empty:
            print(difficulty_df.to_string(index=False))
        else:
            print("No ranking produced (check data availability).")

        out_path = self.table_dir / 'attack_difficulty.csv'
        difficulty_df.to_csv(out_path, index=False)
        if self.logger is not None and not difficulty_df.empty:
            self.logger.log_dataframe(difficulty_df, f"task2_{self.mode}/attack_difficulty")

        if not difficulty_df.empty:
            fig, ax = plt.subplots(figsize=(10, 6))
            colors = ['green' if d == 'Easy' else 'orange' if d == 'Medium' else 'red'
                      for d in difficulty_df['Detection Difficulty']]
            ax.barh(difficulty_df['Attack Type'], difficulty_df['Separability Score'], color=colors, alpha=0.8)
            ax.set_xlabel('Separability Score (Higher = Easier to Detect)', fontsize=11)
            ax.set_title(f'Attack Detection Difficulty ({self.mode.upper()} Mode)\nBased on Class Separability',
                         fontweight='bold', fontsize=12)
            ax.grid(True, alpha=0.3, axis='x')
            for i, row in difficulty_df.reset_index(drop=True).iterrows():
                ax.text(row['Separability Score'], i, f"  {row['Detection Difficulty']}",
                        va='center', fontweight='bold')
            plt.tight_layout()
            fig_path = self.fig_dir / 'attack_difficulty.png'
            plt.savefig(fig_path, bbox_inches='tight')
            print(f"Saved: {fig_path}")
            if self.logger:
                self.logger.log_figure(fig, f"task2_{self.mode}/attack_difficulty")
            plt.close(fig)

        return difficulty_df

    def corr_delta_heatmap(self):
        """Generate correlation delta heatmap (Attack - Normal)."""
        # Data is already preprocessed and engineered in load_data()
        all_data = [df for df in self.train_data.values()]
        df = pd.concat(all_data, ignore_index=True)
        
        features = self._numeric_analysis_features(df)
        mats = {}
        for cls in [0, 1]:
            sub = df[df.attack == cls][features].copy()
            sub = sub.dropna(axis=1, thresh=int(0.5 * len(sub))).fillna(sub.median(numeric_only=True))
            mats[cls] = sub.corr().clip(-1, 1)
        delta = mats[1].reindex_like(mats[0]) - mats[0]
        fig = plt.figure(figsize=(9, 7))
        sns.heatmap(delta, center=0, cmap="coolwarm", vmin=-1, vmax=1)
        plt.title(f"Correlation delta (Attack – Normal) - {self.mode.upper()} Mode")
        plt.tight_layout()
        fig_path = self.fig_dir / 'corr_delta_heatmap.png'
        plt.savefig(fig_path, bbox_inches='tight')
        print(f"Saved: {fig_path}")
        if self.logger:
            self.logger.log_figure(fig, f"task2_{self.mode}/corr_delta_heatmap")
        plt.close(fig)


# ---------------------------- Runner ----------------------------

def run_task2(config: Dict[str, Any], logger=None) -> Dict:
    """
    Execute Task 2: Feature & Attack Characterization in ALL modes.
    
    Returns results for core, full, new, and core_new modes.
    """
    results = {}
    
    if not config.get('mode'):
        config['mode'] = ['core', 'full', 'new', 'core_new']

    # Run all modes
    for mode in config['mode']:
        print(f"\n{'#'*70}")
        print(f"# STARTING {mode.upper()} MODE ANALYSIS")
        print(f"{'#'*70}\n")
        
        characterizer = AttackCharacterizer(config, logger, mode=mode)
        
        # Load data (preprocessing and feature engineering done here once)
        characterizer.load_data()
        
        # Analyses (all use the already-processed data from self.train_data)
        stats_df = characterizer.compute_feature_statistics()
        unsup_importance = characterizer.unsupervised_feature_importance()
        sup_importance = characterizer.supervised_feature_importance()
        signatures = characterizer.characterize_attack_signatures()
        difficulty_df = characterizer.analyze_attack_difficulty()
        characterizer.corr_delta_heatmap()
        
        results[mode] = {
            'status': 'completed',
            'feature_statistics': stats_df,
            'unsupervised_importance': unsup_importance,
            'supervised_importance': sup_importance,
            'attack_signatures': signatures,
            'difficulty_ranking': difficulty_df
        }
        
        print(f"\n✓ Task 2 ({mode.upper()} mode) completed successfully!")
        print(f"All outputs saved to outputs/figures/task2_{mode}/ and outputs/tables/task2_{mode}/")
    
    print("\n" + "="*70)
    print("TASK 2 COMPLETE - ALL MODES FINISHED")
    print("="*70)
    print("\nResults available in:")
    for mode in config['mode']:
        print(f"  - outputs/figures/task2_{mode}/ and outputs/tables/task2_{mode}/")
    
    return results


if __name__ == '__main__':
    import yaml
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    results = run_task2(config, logger=None)