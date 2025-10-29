"""
Task 2: Feature and Attack Characterization Analysis
Analyze feature behavior and attack signatures in detail.
"""

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


class AttackCharacterizer:
    """Characterize features and attack signatures."""
    
    def __init__(self, config: Dict[str, Any], logger=None):
        self.config = config
        self.logger = logger
        self.train_data = {}
        self.test_data = {}
        self.feature_importance = {}
        
        # Create output directories
        self.fig_dir = Path('outputs/figures/task2')
        self.table_dir = Path('outputs/tables/task2')
        self.fig_dir.mkdir(parents=True, exist_ok=True)
        self.table_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.dpi'] = 300
    
    def load_data(self):
        """Load datasets."""
        print("Loading datasets...")
        data_dir = Path(self.config['data']['raw_dir'])
        
        attack_types = ['replay', 'masquerade', 'injection', 'poisoning']
        
        for attack in attack_types:
            train_file = self.config['data']['train_files'][attack]
            self.train_data[attack] = pd.read_csv(data_dir / train_file)
            self.train_data[attack]['attack_type'] = attack
            
            print(f"  {attack}: {len(self.train_data[attack])} samples")
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create derived features for better characterization."""
        df = df.copy()
        
        # Temporal features
        if 'Time' in df.columns:
            # Sort by gocbRef and Time
            df = df.sort_values(['gocbRef', 'Time'])
            df['time_delta'] = df.groupby('gocbRef')['Time'].diff()
            df['time_delta_std'] = df.groupby('gocbRef')['time_delta'].transform('std')
        
        # Sequence features
        if 'sqNum' in df.columns:
            df['sqNum_delta'] = df.groupby('gocbRef')['sqNum'].diff()
            df['sqNum_anomaly'] = (df['sqNum_delta'] != 1) & (df['sqNum_delta'].notna())
        
        if 'stNum' in df.columns:
            df['stNum_delta'] = df.groupby('gocbRef')['stNum'].diff()
            df['stNum_change'] = (df['stNum_delta'] != 0) & (df['stNum_delta'].notna())
        
        # Rate features
        if 'Time' in df.columns and 'Length' in df.columns:
            # Messages per second (approximate)
            df['msg_rate'] = 1.0 / (df['time_delta'] + 1e-6)
            df['byte_rate'] = df['Length'] / (df['time_delta'] + 1e-6)
        
        # Consistency features
        if 'timeAllowedtoLive' in df.columns and 'Time' in df.columns:
            df['ttl_violation'] = df['time_delta'] > (df['timeAllowedtoLive'] / 1000.0)
        
        return df
    
    def compute_feature_statistics(self):
        """Compute feature statistics per attack type."""
        print("\n" + "="*70)
        print("FEATURE STATISTICS BY ATTACK TYPE")
        print("="*70)
        
        # Combine all data with engineered features
        all_data = []
        for attack, df in self.train_data.items():
            df_eng = self.engineer_features(df)
            all_data.append(df_eng)
        
        combined = pd.concat(all_data, ignore_index=True)
        
        # Select numeric features
        numeric_cols = combined.select_dtypes(include=[np.number]).columns
        numeric_cols = [c for c in numeric_cols if c not in ['attack']]
        
        # Compute statistics by attack type and attack label
        results = []
        
        for attack_type in combined['attack_type'].unique():
            attack_data = combined[combined['attack_type'] == attack_type]
            
            # Normal traffic
            normal = attack_data[attack_data['attack'] == 0]
            # Attack traffic
            attack = attack_data[attack_data['attack'] == 1]
            
            for col in numeric_cols[:10]:  # Top 10 features
                if col not in normal.columns or col not in attack.columns:
                    continue
                
                # Clean data
                normal_vals = normal[col].replace([np.inf, -np.inf], np.nan).dropna()
                attack_vals = attack[col].replace([np.inf, -np.inf], np.nan).dropna()
                
                if len(normal_vals) == 0 or len(attack_vals) == 0:
                    continue
                
                # Statistical test
                try:
                    statistic, pvalue = stats.mannwhitneyu(normal_vals, attack_vals,
                                                           alternative='two-sided')
                    
                    results.append({
                        'Attack Type': attack_type.capitalize(),
                        'Feature': col,
                        'Normal Mean': normal_vals.mean(),
                        'Normal Std': normal_vals.std(),
                        'Attack Mean': attack_vals.mean(),
                        'Attack Std': attack_vals.std(),
                        'Mean Difference': abs(attack_vals.mean() - normal_vals.mean()),
                        'P-value': pvalue,
                        'Significant': pvalue < 0.01
                    })
                except Exception as e:
                    continue
        
        if results:
            stats_df = pd.DataFrame(results)
            stats_df = stats_df.sort_values(['Attack Type', 'Mean Difference'], 
                                           ascending=[True, False])
            
            print("\nTop discriminative features by attack type:")
            for attack_type in stats_df['Attack Type'].unique():
                print(f"\n{attack_type}:")
                subset = stats_df[stats_df['Attack Type'] == attack_type].head(5)
                print(subset[['Feature', 'Normal Mean', 'Attack Mean', 
                             'Mean Difference', 'P-value']].to_string(index=False))
            
            stats_df.to_csv(self.table_dir / 'feature_statistics.csv', index=False)
            
            return stats_df
        
        return None
    
    def unsupervised_feature_importance(self):
        """Estimate feature importance using unsupervised methods."""
        print("\n" + "="*70)
        print("UNSUPERVISED FEATURE IMPORTANCE")
        print("="*70)
        
        # Method 1: Variance-based importance
        all_data = pd.concat(list(self.train_data.values()), ignore_index=True)
        
        numeric_cols = all_data.select_dtypes(include=[np.number]).columns
        numeric_cols = [c for c in numeric_cols if c not in ['attack']]
        
        # Clean data
        data_clean = all_data[numeric_cols].replace([np.inf, -np.inf], np.nan)
        data_clean = data_clean.fillna(data_clean.median())
        
        # Compute variance
        variances = data_clean.var().sort_values(ascending=False)
        
        print("\nTop 15 features by variance:")
        print(variances.head(15).to_string())
        
        # Method 2: PCA-based importance
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data_clean)
        
        pca = PCA(n_components=min(10, data_scaled.shape[1]))
        pca.fit(data_scaled)
        
        # Feature importance from first PC
        pc1_importance = pd.Series(
            np.abs(pca.components_[0]),
            index=numeric_cols
        ).sort_values(ascending=False)
        
        print("\nTop 15 features by PCA (PC1) importance:")
        print(pc1_importance.head(15).to_string())
        
        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Variance-based
        top_var = variances.head(15)
        axes[0].barh(range(len(top_var)), top_var.values)
        axes[0].set_yticks(range(len(top_var)))
        axes[0].set_yticklabels(top_var.index, fontsize=9)
        axes[0].set_xlabel('Variance')
        axes[0].set_title('Top 15 Features by Variance', fontweight='bold')
        axes[0].invert_yaxis()
        
        # PCA-based
        top_pca = pc1_importance.head(15)
        axes[1].barh(range(len(top_pca)), top_pca.values)
        axes[1].set_yticks(range(len(top_pca)))
        axes[1].set_yticklabels(top_pca.index, fontsize=9)
        axes[1].set_xlabel('|PC1 Loading|')
        axes[1].set_title('Top 15 Features by PCA Importance', fontweight='bold')
        axes[1].invert_yaxis()
        
        plt.tight_layout()
        fig_path = self.fig_dir / 'feature_importance_unsupervised.png'
        plt.savefig(fig_path, bbox_inches='tight')
        print(f"\nSaved: {fig_path}")
        
        if self.logger:
            self.logger.log_figure(fig, "task2/feature_importance_unsupervised")
        plt.close(fig)
        
        # Save importance scores
        importance_df = pd.DataFrame({
            'Feature': variances.index,
            'Variance': variances.values,
            'PCA_Importance': [pc1_importance.get(f, 0) for f in variances.index]
        })
        importance_df.to_csv(self.table_dir / 'feature_importance_scores.csv', index=False)
        
        return importance_df
    
    def supervised_feature_importance(self):
        """Use supervised methods for validation (labels not for training IDS)."""
        print("\n" + "="*70)
        print("SUPERVISED FEATURE IMPORTANCE (For Validation Only)")
        print("="*70)
        
        all_data = pd.concat(list(self.train_data.values()), ignore_index=True)
        
        numeric_cols = all_data.select_dtypes(include=[np.number]).columns
        numeric_cols = [c for c in numeric_cols if c not in ['attack']]
        
        # Clean data
        X = all_data[numeric_cols].replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median())
        y = all_data['attack']
        
        # Random Forest for feature importance
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X, y)
        
        importance = pd.Series(rf.feature_importances_, index=numeric_cols)
        importance = importance.sort_values(ascending=False)
        
        print("\nTop 15 features by Random Forest importance:")
        print(importance.head(15).to_string())
        
        # Visualize
        fig, ax = plt.subplots(figsize=(10, 8))
        top_20 = importance.head(20)
        ax.barh(range(len(top_20)), top_20.values, color='steelblue')
        ax.set_yticks(range(len(top_20)))
        ax.set_yticklabels(top_20.index, fontsize=9)
        ax.set_xlabel('Feature Importance', fontsize=11)
        ax.set_title('Top 20 Features by Random Forest Importance\n(For Validation Only)',
                    fontweight='bold', fontsize=12)
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        fig_path = self.fig_dir / 'feature_importance_supervised.png'
        plt.savefig(fig_path, bbox_inches='tight')
        print(f"Saved: {fig_path}")
        
        if self.logger:
            self.logger.log_figure(fig, "task2/feature_importance_supervised")
        plt.close(fig)
        
        importance.to_csv(self.table_dir / 'feature_importance_rf.csv', 
                         header=['Importance'])
        
        return importance
    
    def characterize_attack_signatures(self):
        """Characterize each attack type's signature."""
        print("\n" + "="*70)
        print("ATTACK SIGNATURE CHARACTERIZATION")
        print("="*70)
        
        signatures = {}
        
        for attack_type, df in self.train_data.items():
            print(f"\n{attack_type.upper()} Attack:")
            
            normal = df[df['attack'] == 0]
            attack = df[df['attack'] == 1]
            
            sig = {
                'attack_type': attack_type,
                'num_samples': len(attack),
                'attack_ratio': len(attack) / len(df)
            }
            
            # Analyze key protocol fields
            if 'stNum' in df.columns:
                sig['stNum_mean_normal'] = normal['stNum'].mean()
                sig['stNum_mean_attack'] = attack['stNum'].mean()
                sig['stNum_std_attack'] = attack['stNum'].std()
                print(f"  stNum: Normal={sig['stNum_mean_normal']:.2f}, "
                      f"Attack={sig['stNum_mean_attack']:.2f}")
            
            if 'sqNum' in df.columns:
                sig['sqNum_mean_normal'] = normal['sqNum'].mean()
                sig['sqNum_mean_attack'] = attack['sqNum'].mean()
                sig['sqNum_std_attack'] = attack['sqNum'].std()
                print(f"  sqNum: Normal={sig['sqNum_mean_normal']:.2f}, "
                      f"Attack={sig['sqNum_mean_attack']:.2f}")
            
            if 'timeAllowedtoLive' in df.columns:
                sig['ttl_mean_normal'] = normal['timeAllowedtoLive'].mean()
                sig['ttl_mean_attack'] = attack['timeAllowedtoLive'].mean()
                print(f"  TTL: Normal={sig['ttl_mean_normal']:.2f}, "
                      f"Attack={sig['ttl_mean_attack']:.2f}")
            
            if 'Length' in df.columns:
                sig['length_mean_normal'] = normal['Length'].mean()
                sig['length_mean_attack'] = attack['Length'].mean()
                print(f"  Length: Normal={sig['length_mean_normal']:.2f}, "
                      f"Attack={sig['length_mean_attack']:.2f}")
            
            signatures[attack_type] = sig
        
        # Create comparison visualization
        self._visualize_attack_signatures(signatures)
        
        # Save signatures
        sig_df = pd.DataFrame(signatures).T
        sig_df.to_csv(self.table_dir / 'attack_signatures.csv')
        
        return signatures
    
    def _visualize_attack_signatures(self, signatures: Dict):
        """Visualize attack signature characteristics."""
        
        # Extract relevant metrics for visualization
        attack_types = list(signatures.keys())
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Attack Signature Characteristics', fontsize=14, fontweight='bold')
        
        # 1. Attack ratio comparison
        ratios = [signatures[a]['attack_ratio'] for a in attack_types]
        axes[0, 0].bar(attack_types, ratios, color='coral')
        axes[0, 0].set_ylabel('Attack Ratio')
        axes[0, 0].set_title('Attack Prevalence by Type')
        axes[0, 0].set_ylim([0, max(ratios) * 1.2])
        for i, v in enumerate(ratios):
            axes[0, 0].text(i, v, f'{v:.2%}', ha='center', va='bottom')
        
        # 2. stNum characteristics
        if all('stNum_mean_attack' in signatures[a] for a in attack_types):
            stnum_normal = [signatures[a]['stNum_mean_normal'] for a in attack_types]
            stnum_attack = [signatures[a]['stNum_mean_attack'] for a in attack_types]
            
            x = np.arange(len(attack_types))
            width = 0.35
            axes[0, 1].bar(x - width/2, stnum_normal, width, label='Normal', alpha=0.8)
            axes[0, 1].bar(x + width/2, stnum_attack, width, label='Attack', alpha=0.8)
            axes[0, 1].set_ylabel('Mean stNum')
            axes[0, 1].set_title('Status Number Distribution')
            axes[0, 1].set_xticks(x)
            axes[0, 1].set_xticklabels([a.capitalize() for a in attack_types])
            axes[0, 1].legend()
        
        # 3. sqNum characteristics
        if all('sqNum_mean_attack' in signatures[a] for a in attack_types):
            sqnum_normal = [signatures[a]['sqNum_mean_normal'] for a in attack_types]
            sqnum_attack = [signatures[a]['sqNum_mean_attack'] for a in attack_types]
            
            x = np.arange(len(attack_types))
            axes[1, 0].bar(x - width/2, sqnum_normal, width, label='Normal', alpha=0.8)
            axes[1, 0].bar(x + width/2, sqnum_attack, width, label='Attack', alpha=0.8)
            axes[1, 0].set_ylabel('Mean sqNum')
            axes[1, 0].set_title('Sequence Number Distribution')
            axes[1, 0].set_xticks(x)
            axes[1, 0].set_xticklabels([a.capitalize() for a in attack_types])
            axes[1, 0].legend()
        
        # 4. Length characteristics
        if all('length_mean_attack' in signatures[a] for a in attack_types):
            length_normal = [signatures[a]['length_mean_normal'] for a in attack_types]
            length_attack = [signatures[a]['length_mean_attack'] for a in attack_types]
            
            x = np.arange(len(attack_types))
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
            self.logger.log_figure(fig, "task2/attack_signatures")
        plt.close(fig)
    
    def analyze_attack_difficulty(self):
        """Analyze relative difficulty of detecting each attack type."""
        print("\n" + "="*70)
        print("ATTACK DETECTION DIFFICULTY ANALYSIS")
        print("="*70)
        
        # Combine all data
        all_data = pd.concat(list(self.train_data.values()), ignore_index=True)
        
        numeric_cols = all_data.select_dtypes(include=[np.number]).columns
        numeric_cols = [c for c in numeric_cols if c not in ['attack']]
        
        # Clean data
        X = all_data[numeric_cols].replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median())
        
        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Compute separability metrics per attack type
        difficulty = []
        
        for attack_type in all_data['attack_type'].unique():
            attack_mask = (all_data['attack_type'] == attack_type) & (all_data['attack'] == 1)
            normal_mask = (all_data['attack_type'] == attack_type) & (all_data['attack'] == 0)
            
            attack_samples = X_scaled[attack_mask]
            normal_samples = X_scaled[normal_mask]
            
            if len(attack_samples) > 0 and len(normal_samples) > 0:
                # Compute mean distance between classes
                attack_center = attack_samples.mean(axis=0)
                normal_center = normal_samples.mean(axis=0)
                inter_class_distance = np.linalg.norm(attack_center - normal_center)
                
                # Compute intra-class variance
                attack_var = attack_samples.var(axis=0).mean()
                normal_var = normal_samples.var(axis=0).mean()
                intra_class_variance = (attack_var + normal_var) / 2
                
                # Separability score (higher = easier to detect)
                separability = inter_class_distance / (np.sqrt(intra_class_variance) + 1e-6)
                
                difficulty.append({
                    'Attack Type': attack_type.capitalize(),
                    'Inter-class Distance': inter_class_distance,
                    'Intra-class Variance': intra_class_variance,
                    'Separability Score': separability,
                    'Detection Difficulty': 'Easy' if separability > 5 else 
                                           ('Medium' if separability > 2 else 'Hard')
                })
        
        difficulty_df = pd.DataFrame(difficulty)
        difficulty_df = difficulty_df.sort_values('Separability Score', ascending=False)
        
        print("\nAttack Detection Difficulty Ranking:")
        print(difficulty_df.to_string(index=False))
        
        difficulty_df.to_csv(self.table_dir / 'attack_difficulty.csv', index=False)
        
        # Visualize
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ['green' if d == 'Easy' else 'orange' if d == 'Medium' else 'red' 
                 for d in difficulty_df['Detection Difficulty']]
        
        bars = ax.barh(difficulty_df['Attack Type'], difficulty_df['Separability Score'],
                      color=colors, alpha=0.7)
        ax.set_xlabel('Separability Score (Higher = Easier to Detect)', fontsize=11)
        ax.set_title('Attack Detection Difficulty\nBased on Class Separability',
                    fontweight='bold', fontsize=12)
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add difficulty labels
        for i, (idx, row) in enumerate(difficulty_df.iterrows()):
            ax.text(row['Separability Score'], i, f"  {row['Detection Difficulty']}",
                   va='center', fontweight='bold')
        
        plt.tight_layout()
        fig_path = self.fig_dir / 'attack_difficulty.png'
        plt.savefig(fig_path, bbox_inches='tight')
        print(f"\nSaved: {fig_path}")
        
        if self.logger:
            self.logger.log_figure(fig, "task2/attack_difficulty")
        plt.close(fig)
        
        return difficulty_df


def run_task2(config: Dict[str, Any], logger=None) -> Dict:
    """
    Execute Task 2: Feature and Attack Characterization.
    
    Args:
        config: Configuration dictionary
        logger: WandbLogger instance (optional)
    
    Returns:
        Dictionary with analysis results
    """
    characterizer = AttackCharacterizer(config, logger)
    
    # Load data
    characterizer.load_data()
    
    # Run analyses
    stats_df = characterizer.compute_feature_statistics()
    unsup_importance = characterizer.unsupervised_feature_importance()
    sup_importance = characterizer.supervised_feature_importance()
    signatures = characterizer.characterize_attack_signatures()
    difficulty_df = characterizer.analyze_attack_difficulty()
    
    print("\n✓ Task 2 (Feature & Attack Characterization) completed successfully!")
    print(f"All outputs saved to outputs/figures/task2/ and outputs/tables/task2/")
    
    return {
        'status': 'completed',
        'feature_statistics': stats_df,
        'unsupervised_importance': unsup_importance,
        'supervised_importance': sup_importance,
        'attack_signatures': signatures,
        'difficulty_ranking': difficulty_df
    }


if __name__ == '__main__':
    import yaml
    
    # Load config
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Run task
    results = run_task2(config, logger=None)