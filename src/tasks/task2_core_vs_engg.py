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

# ==============================================================
# Task 2: Full Workflow — Baseline vs Augmented + Comparison
# ==============================================================

# -------------------- Core vs Augmented Fields --------------------

# Your requested 10 core/raw features
CORE_FIELDS: List[str] = [
    "gocbRef",
    "timeAllowedtoLive",
    "Time",
    "t",
    "stNum",
    "sqNum",
    "Length",
    "boolean",
    "bit-string",
    "attack",
]

# Superset of allowed fields for the augmented run (kept from earlier)
ALLOWED_FIELDS: List[str] = list({
    # Core / semantic
    "gocbRef", "timeAllowedtoLive", "Time", "t", "stNum", "sqNum", "Length",
    "boolean", "bit-string", "attack",
    # Timing intervals
    "Epoch Time", "Arrival Time", "Time delta from previous captured frame",
    "Time delta from previous displayed frame", "Time since reference or first frame",
    "Time shift for this packet",
    # Rate stats & metadata
    "Frame length on the wire", "Frame length stored into the capture file",
    "Protocol", "Interface name", "Encapsulation type", "numDatSetEntries",
    # Sequence transitions / identifiers
    "confRev", "datSet", "goID",
})

_EPS = 1e-6


# ------------------------- Utilities -------------------------

def _coerce_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors='coerce')

def _parse_to_epoch(col: pd.Series) -> pd.Series:
    if col.dtype == object:
        dt = pd.to_datetime(col, errors='coerce', utc=True)
        return dt.astype('int64') / 1e9  # ns -> s
    return pd.to_numeric(col, errors='coerce')

def _bitstring_to_int(s):
    if pd.isna(s): return np.nan
    s = str(s).strip().lower().replace('0x', '')
    try: return int(s, 16)
    except: return np.nan

def _bitstring_popcount(s):
    if pd.isna(s): return np.nan
    s = str(s).strip().lower().replace('0x', '')
    try: return bin(int(s, 16))[2:].count('1')
    except: return np.nan

def _map_boolean(col: pd.Series) -> pd.Series:
    if col.dtype == object:
        return col.astype(str).str.strip().str.lower().map({
            'true': 1, '1': 1, 'yes': 1, 'y': 1, 'on': 1,
            'false': 0, '0': 0, 'no': 0, 'n': 0, 'off': 0
        }).astype('float64')
    return pd.to_numeric(col, errors='coerce')


# ------------------------- Characterizer -------------------------

class AttackCharacterizer:
    """
    Characterize features and attack signatures.

    Modes:
      - use_only_core=True  -> Baseline (core fields only, minimal numeric conversions)
      - use_only_core=False -> Augmented (raw + engineered features)
    """
    def __init__(self, config: Dict[str, Any], run_name: str, use_only_core: bool, logger=None):
        self.config = config
        self.run_name = run_name
        self.use_only_core = use_only_core
        self.logger = logger

        self.train_data: Dict[str, pd.DataFrame] = {}
        self.feature_importance = {}

        # Output dirs segmented by run (baseline/augmented)
        self.fig_dir = Path(f'outputs/figures/task2/{self.run_name}')
        self.table_dir = Path(f'outputs/tables/task2/{self.run_name}')
        self.fig_dir.mkdir(parents=True, exist_ok=True)
        self.table_dir.mkdir(parents=True, exist_ok=True)

        sns.set_style("whitegrid")
        plt.rcParams['figure.dpi'] = 300

    # ----------------------- Data Loading -----------------------

    def load_data(self):
        print(f"[{self.run_name}] Loading datasets...")
        data_dir = Path(self.config['data']['raw_dir'])
        attack_types = ['replay', 'masquerade', 'injection', 'poisoning']

        for attack in attack_types:
            train_file = self.config['data']['train_files'][attack]
            df = pd.read_csv(data_dir / train_file)

            # Select columns
            keep = CORE_FIELDS if self.use_only_core else [c for c in ALLOWED_FIELDS if c in df.columns]
            keep = [c for c in keep if c in df.columns]
            df = df[keep].copy()

            if 'attack' not in df.columns:
                raise ValueError(f"'attack' column missing in {train_file}.")
            df['attack_type'] = attack

            # Standardize schema (types + timestamps)
            df = self._standardize_schema(df)

            self.train_data[attack] = df
            print(f"  {attack}: {len(df)} samples, {len(df.columns)} cols")

        used_fields = sorted(set().union(*[set(d.columns) for d in self.train_data.values()]))
        pd.Series(used_fields, name='used_fields').to_csv(self.table_dir / 'used_fields.csv', index=False)
        print(f"[{self.run_name}] Saved used fields list.")

    def _standardize_schema(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Map boolean to numeric if present
        if 'boolean' in df.columns:
            df['boolean'] = _map_boolean(df['boolean'])

        # Convert bit-string into two numeric surrogates (even in baseline so the field is usable)
        if 'bit-string' in df.columns:
            df['bitstring_int'] = df['bit-string'].apply(_bitstring_to_int).astype('float64')
            df['bitstring_bitcount'] = df['bit-string'].apply(_bitstring_popcount).astype('float64')

        # Numeric coercions (safe if columns exist)
        numeric_like = [
            "Epoch Time", "Arrival Time", "Length", "Frame length on the wire",
            "Frame length stored into the capture file", "timeAllowedtoLive",
            "stNum", "sqNum", "confRev", "numDatSetEntries",
            "Time delta from previous captured frame",
            "Time delta from previous displayed frame",
            "Time since reference or first frame",
            "Time shift for this packet", "attack",
        ]
        for col in numeric_like:
            if col in df.columns:
                df[col] = _coerce_numeric(df[col])

        # base_time (seconds) choosing best available: Epoch, Arrival, parsed Time, parsed t
        epoch = df['Epoch Time'] if 'Epoch Time' in df.columns else pd.Series([np.nan] * len(df))
        arriv = df['Arrival Time'] if 'Arrival Time' in df.columns else pd.Series([np.nan] * len(df))
        time_parsed = _parse_to_epoch(df['Time']) if 'Time' in df.columns else pd.Series([np.nan] * len(df))
        t_parsed = _parse_to_epoch(df['t']) if 't' in df.columns else pd.Series([np.nan] * len(df))
        base_time = epoch.copy().fillna(arriv).fillna(time_parsed).fillna(t_parsed)
        df['base_time'] = pd.to_numeric(base_time, errors='coerce')

        if 'gocbRef' in df.columns:
            df['gocbRef'] = df['gocbRef'].astype(str)

        # Clean infs
        for c in df.columns:
            if pd.api.types.is_numeric_dtype(df[c]):
                df[c].replace([np.inf, -np.inf], np.nan, inplace=True)

        return df

    # --------------------- Feature Engineering ---------------------

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        For augmented run: add timing/sequence/TTL/rule features.
        For baseline: return df (only core fields + minimal numeric conversions already done).
        """
        df = df.copy()
        if self.use_only_core:
            return df  # baseline = no extra engineered features

        # ---- Augmented feature engineering (same as earlier design) ----

        # Sort within device stream
        sort_key = 'base_time' if 'base_time' in df.columns else None
        if 'gocbRef' in df.columns and sort_key is not None:
            df = df.sort_values(['gocbRef', sort_key])

        # Inter-arrival
        if sort_key is not None and 'gocbRef' in df.columns:
            df['inter_arrival'] = df.groupby('gocbRef')[sort_key].diff()
        else:
            if 'Time delta from previous captured frame' in df.columns:
                df['inter_arrival'] = df['Time delta from previous captured frame']
            elif 'Time delta from previous displayed frame' in df.columns:
                df['inter_arrival'] = df['Time delta from previous displayed frame']
            else:
                df['inter_arrival'] = np.nan

        # Jitter
        if 'gocbRef' in df.columns and 'inter_arrival' in df.columns:
            df['jitter_rolling_std'] = (
                df.groupby('gocbRef')['inter_arrival'].rolling(window=10, min_periods=3)
                  .std().reset_index(level=0, drop=True)
            )

        # Rates
        length_col = 'Frame length on the wire' if 'Frame length on the wire' in df.columns else 'Length'
        if length_col in df.columns and 'inter_arrival' in df.columns:
            df['msg_rate'] = 1.0 / (df['inter_arrival'].astype(float) + _EPS)
            df['byte_rate'] = df[length_col].astype(float) / (df['inter_arrival'].astype(float) + _EPS)

        # TTL
        if 'timeAllowedtoLive' in df.columns and 'inter_arrival' in df.columns:
            df['ttl_violation'] = (df['inter_arrival'] > (df['timeAllowedtoLive'] / 1000.0)).astype(float)
            df['ttl_margin'] = (df['timeAllowedtoLive'] / 1000.0) - df['inter_arrival']

        # Sequence transitions
        if 'gocbRef' in df.columns and 'sqNum' in df.columns:
            df['sqNum_delta'] = df.groupby('gocbRef')['sqNum'].diff()
            df['sqNum_jump'] = ((df['sqNum_delta'] != 1) & df['sqNum_delta'].notna()).astype(float)

        if 'gocbRef' in df.columns and 'stNum' in df.columns:
            df['stNum_delta'] = df.groupby('gocbRef')['stNum'].diff()
            df['stNum_change'] = ((df['stNum_delta'] != 0) & df['stNum_delta'].notna()).astype(float)

        if 'gocbRef' in df.columns and 'confRev' in df.columns:
            df['confRev_delta'] = df.groupby('gocbRef')['confRev'].diff()
            df['confRev_change'] = ((df['confRev_delta'] != 0) & df['confRev_delta'].notna()).astype(float)

        if all(c in df.columns for c in ['stNum_change', 'sqNum']):
            df['sq_reset_on_st_change'] = ((df['stNum_change'] == 1.0) & (df['sqNum'].isin({0, 1}))).astype(float)

        # Length dynamics
        if 'gocbRef' in df.columns and length_col in df.columns:
            df['length_delta'] = df.groupby('gocbRef')[length_col].diff()

        # Composite ratios
        if 'msg_rate' in df.columns and 'stNum_change' in df.columns:
            df['event_rate_spike'] = df['msg_rate'] * df['stNum_change']
        if 'bitstring_bitcount' in df.columns and length_col in df.columns:
            df['bit_density'] = df['bitstring_bitcount'] / (df[length_col].astype(float) * 8.0 + _EPS)

        # ---- Rule-based features from GOOSE spec ----
        _RESET_SET = {0, 1}

        # st change (alias)
        if 'stNum_change' in df.columns:
            df['st_change'] = df['stNum_change'].astype(float)
        else:
            df['st_change'] = np.nan

        # t change
        if 't' in df.columns and 'gocbRef' in df.columns:
            df['t_raw'] = df['t'].astype(str)
            df['t_change'] = (df.groupby('gocbRef')['t_raw'].shift() != df['t_raw']).astype(float)
            df.loc[df.groupby('gocbRef').head(1).index, 't_change'] = np.nan
            df.drop(columns=['t_raw'], inplace=True, errors='ignore')
        else:
            df['t_change'] = np.nan

        # status_bit_change: proxy for allData changes (boolean or bit-string)
        changed_flags = []
        if 'boolean' in df.columns and 'gocbRef' in df.columns:
            ch = (df['boolean'] != df.groupby('gocbRef')['boolean'].shift()).astype(float)
            ch.loc[df.groupby('gocbRef').head(1).index] = np.nan
            changed_flags.append(ch)
        if 'bit-string' in df.columns and 'gocbRef' in df.columns:
            ch = (df['bit-string'] != df.groupby('gocbRef')['bit-string'].shift()).astype(float)
            ch.loc[df.groupby('gocbRef').head(1).index] = np.nan
            changed_flags.append(ch)
        if changed_flags:
            df['status_bit_change'] = np.nanmax(np.column_stack([c.fillna(0) for c in changed_flags]), axis=1)
            if 'gocbRef' in df.columns:
                df.loc[df.groupby('gocbRef').head(1).index, 'status_bit_change'] = np.nan
        else:
            df['status_bit_change'] = np.nan

        # Event consistency
        if 'sqNum' in df.columns and 'st_change' in df.columns and 'gocbRef' in df.columns:
            df['sq_reset_expected'] = ((df['st_change'] == 1.0) & (df['sqNum'].isin(list(_RESET_SET)))).astype(float)
            df['sq_reset_violation'] = ((df['st_change'] == 1.0) & (~df['sqNum'].isin(list(_RESET_SET)))).astype(float)
        else:
            df['sq_reset_expected'] = np.nan
            df['sq_reset_violation'] = np.nan

        if 't_change' in df.columns and 'st_change' in df.columns:
            df['t_st_consistency_violation'] = (df['t_change'].fillna(0) != df['st_change'].fillna(0)).astype(float)
            df.loc[df['t_change'].isna(), 't_st_consistency_violation'] = np.nan
        else:
            df['t_st_consistency_violation'] = np.nan

        if 'status_bit_change' in df.columns and 'st_change' in df.columns:
            df['status_change_missing_on_event'] = ((df['st_change'] == 1.0) & (df['status_bit_change'] != 1.0)).astype(float)
            df.loc[df['status_bit_change'].isna(), 'status_change_missing_on_event'] = np.nan

        for cfg_col in ['goID', 'datSet', 'confRev']:
            colname = f'cfg_change_{cfg_col}'
            if cfg_col in df.columns and 'gocbRef' in df.columns:
                df[colname] = (df[cfg_col] != df.groupby('gocbRef')[cfg_col].shift()).astype(float)
                df.loc[df.groupby('gocbRef').head(1).index, colname] = np.nan
            else:
                df[colname] = np.nan

        cfg_cols = [f'cfg_change_{c}' for c in ['goID', 'datSet', 'confRev']]
        if set(cfg_cols).issubset(df.columns):
            df['config_change_violation_on_event'] = (
                (df['st_change'] == 1.0) & (df[cfg_cols].fillna(0).sum(axis=1) > 0)
            ).astype(float)
        else:
            df['config_change_violation_on_event'] = np.nan

        # Heartbeat consistency (expected interval ~ TTL/2)
        if 'timeAllowedtoLive' in df.columns and 'inter_arrival' in df.columns:
            df['ttl_half'] = (df['timeAllowedtoLive'] / 2000.0).astype(float)
            tol = 0.05 * df['ttl_half'] + 0.005
            df['heartbeat_interval_error'] = (df['inter_arrival'] - df['ttl_half']).abs()
            df['heartbeat_within_tol'] = (df['heartbeat_interval_error'] <= tol).astype(float)
        else:
            df['ttl_half'] = np.nan
            df['heartbeat_interval_error'] = np.nan
            df['heartbeat_within_tol'] = np.nan

        if 'sqNum_delta' in df.columns and 'st_change' in df.columns:
            df['sq_inc_expected'] = ((df['st_change'] == 0.0) & (df['sqNum_delta'] == 1)).astype(float)
            df['sq_inc_violation_when_no_event'] = ((df['st_change'] == 0.0) & (df['sqNum_delta'] != 1)).astype(float)
        else:
            df['sq_inc_expected'] = np.nan
            df['sq_inc_violation_when_no_event'] = np.nan

        if 'status_bit_change' in df.columns and 'st_change' in df.columns:
            df['status_change_violation_on_heartbeat'] = ((df['st_change'] == 0.0) & (df['status_bit_change'] == 1.0)).astype(float)
            df.loc[df['status_bit_change'].isna(), 'status_change_violation_on_heartbeat'] = np.nan
        else:
            df['status_change_violation_on_heartbeat'] = np.nan

        # General
        if 'stNum_delta' in df.columns:
            df['stNum_jump_gt1'] = (df['stNum_delta'] > 1).astype(float)
        else:
            df['stNum_jump_gt1'] = np.nan

        if 'sqNum' in df.columns and 'st_change' in df.columns:
            df['sq_reset_without_st_change'] = ((df['st_change'] == 0.0) & (df['sqNum'].isin({0, 1}))).astype(float)
        else:
            df['sq_reset_without_st_change'] = np.nan

        if 'gocbRef' in df.columns and 'st_change' in df.columns:
            df['event_id'] = df.groupby('gocbRef')['st_change'].fillna(0).cumsum()
            df['msgs_since_last_event'] = df.groupby(['gocbRef', 'event_id']).cumcount()
            if 'base_time' in df.columns:
                first_time_in_event = df.groupby(['gocbRef', 'event_id'])['base_time'].transform('first')
                df['time_since_last_event'] = df['base_time'] - first_time_in_event
            else:
                df['time_since_last_event'] = np.nan
            df['event_msg_rank'] = df['msgs_since_last_event'].astype(float)
        else:
            df['event_id'] = np.nan
            df['msgs_since_last_event'] = np.nan
            df['time_since_last_event'] = np.nan
            df['event_msg_rank'] = np.nan

        return df

    # ------------------- Analysis Feature Set -------------------

    def _numeric_analysis_features(self, df: pd.DataFrame) -> List[str]:
        """
        Baseline: only numeric versions of the 10 core fields (no extra engineering).
        Augmented: raw numeric + engineered features.
        """
        if self.use_only_core:
            raw_numeric = [
                'timeAllowedtoLive', 'stNum', 'sqNum', 'Length',
                'boolean', 'bitstring_int', 'bitstring_bitcount',
                # we avoid 'attack' label, and time strings; 'base_time' ok for plotting but exclude from baseline stats
            ]
            cols = [c for c in raw_numeric if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
            return cols

        # Augmented: raw + engineered
        raw_numeric = [
            'Epoch Time', 'Arrival Time', 'Length', 'Frame length on the wire',
            'Frame length stored into the capture file', 'timeAllowedtoLive',
            'stNum', 'sqNum', 'confRev', 'numDatSetEntries',
            'Time delta from previous captured frame',
            'Time delta from previous displayed frame',
            'Time since reference or first frame',
            'Time shift for this packet',
            'boolean','bitstring_int', 'bitstring_bitcount',
        ]
        raw_numeric = [c for c in raw_numeric if c in df.columns]

        engineered = [
            'base_time', 'inter_arrival', 'jitter_rolling_std',
            'msg_rate', 'byte_rate',
            'ttl_violation', 'ttl_margin',
            'sqNum_delta', 'sqNum_jump',
            'stNum_delta', 'stNum_change',
            'confRev_delta', 'confRev_change',
            'sq_reset_on_st_change',
            'length_delta',
            'event_rate_spike', 'bit_density',
            # rule-based
            'st_change', 't_change', 'status_bit_change',
            'sq_reset_expected', 'sq_reset_violation',
            't_st_consistency_violation',
            'status_change_missing_on_event',
            'cfg_change_goID', 'cfg_change_datSet', 'cfg_change_confRev',
            'config_change_violation_on_event',
            'ttl_half', 'heartbeat_interval_error', 'heartbeat_within_tol',
            'sq_inc_expected', 'sq_inc_violation_when_no_event',
            'status_change_violation_on_heartbeat',
            'stNum_jump_gt1', 'sq_reset_without_st_change',
            'event_id', 'msgs_since_last_event', 'time_since_last_event', 'event_msg_rank',
        ]
        engineered = [c for c in engineered if c in df.columns]

        numeric_cols = raw_numeric + engineered
        numeric_cols = [c for c in numeric_cols if c != 'attack']
        numeric_cols = [c for c in numeric_cols if pd.api.types.is_numeric_dtype(df[c])]
        return numeric_cols

    # -------------------- Core Analyses --------------------

    def compute_feature_statistics(self):
        """Mann–Whitney U stats per attack type."""
        print(f"\n[{self.run_name}] FEATURE STATISTICS")
        all_data = []
        for atk, df in self.train_data.items():
            df_eng = self.engineer_features(df)
            all_data.append(df_eng)
        combined = pd.concat(all_data, ignore_index=True)

        numeric_cols = self._numeric_analysis_features(combined)
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
                    _, pvalue = stats.mannwhitneyu(normal_vals, attack_vals, alternative='two-sided')
                    results.append({
                        'Attack Type': attack_type.capitalize(),
                        'Feature': col,
                        'Normal Mean': normal_vals.mean(),
                        'Attack Mean': attack_vals.mean(),
                        'Mean Difference': abs(attack_vals.mean() - normal_vals.mean()),
                        'P-value': pvalue,
                        'Significant (p<0.01)': pvalue < 0.01
                    })
                except Exception:
                    continue

        if not results:
            print(f"[{self.run_name}] No stats computed.")
            return None

        stats_df = pd.DataFrame(results).sort_values(['Attack Type','Mean Difference'], ascending=[True, False])
        out_path = self.table_dir / 'feature_statistics.csv'
        stats_df.to_csv(out_path, index=False)
        print(f"[{self.run_name}] Saved: {out_path}")
        return stats_df

    def unsupervised_feature_importance(self):
        """Variance + PCA loadings (unsupervised)."""
        print(f"\n[{self.run_name}] UNSUPERVISED FEATURE IMPORTANCE")
        all_data = []
        for _, df in self.train_data.items():
            all_data.append(self.engineer_features(df))
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

        variances = X.var().sort_values(ascending=False)
        pc1_importance = pd.Series(dtype=float)
        try:
            scaler = StandardScaler()
            Xs = scaler.fit_transform(X)
            Xs = np.nan_to_num(Xs, nan=0.0, posinf=0.0, neginf=0.0)
            pca = PCA(n_components=min(10, Xs.shape[1]))
            pca.fit(Xs)
            pc1_importance = pd.Series(np.abs(pca.components_[0]), index=X.columns).sort_values(ascending=False)
        except Exception as e:
            print(f"[{self.run_name}] PCA failed: {e}")

        # save barplots
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        top_var = variances.head(15)
        axes[0].barh(range(len(top_var)), top_var.values)
        axes[0].set_yticks(range(len(top_var))); axes[0].set_yticklabels(top_var.index, fontsize=9)
        axes[0].invert_yaxis(); axes[0].set_title('Top 15 by Variance')

        if not pc1_importance.empty:
            top_pca = pc1_importance.head(15)
            axes[1].barh(range(len(top_pca)), top_pca.values)
            axes[1].set_yticks(range(len(top_pca))); axes[1].set_yticklabels(top_pca.index, fontsize=9)
            axes[1].invert_yaxis(); axes[1].set_title('Top 15 by |PC1|')
        else:
            axes[1].text(0.5,0.5,'PCA unavailable',ha='center',va='center',transform=axes[1].transAxes)

        plt.tight_layout()
        fig_path = self.fig_dir / 'feature_importance_unsupervised.png'
        plt.savefig(fig_path, bbox_inches='tight'); plt.close(fig)
        print(f"[{self.run_name}] Saved: {fig_path}")

        importance_df = pd.DataFrame({
            'Feature': variances.index,
            'Variance': variances.values,
            'PCA_Importance': [pc1_importance.get(f, 0) for f in variances.index]
        })
        importance_df.to_csv(self.table_dir / 'feature_importance_unsupervised.csv', index=False)
        return importance_df, pc1_importance

    def supervised_feature_importance(self):
        """RandomForest importance (validation only)."""
        print(f"\n[{self.run_name}] SUPERVISED FEATURE IMPORTANCE (Validation)")
        all_data = []
        for _, df in self.train_data.items():
            all_data.append(self.engineer_features(df))
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

        rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1, max_depth=None)
        rf.fit(X, y)
        imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)

        # plot
        fig, ax = plt.subplots(figsize=(10, 8))
        top_20 = imp.head(20)
        ax.barh(range(len(top_20)), top_20.values)
        ax.set_yticks(range(len(top_20))); ax.set_yticklabels(top_20.index, fontsize=9)
        ax.invert_yaxis()
        ax.set_title('Top 20 RF Feature Importances (Validation)')
        plt.tight_layout()
        fig_path = self.fig_dir / 'feature_importance_supervised.png'
        plt.savefig(fig_path, bbox_inches='tight'); plt.close(fig)
        print(f"[{self.run_name}] Saved: {fig_path}")

        imp.to_csv(self.table_dir / 'feature_importance_rf.csv', header=['Importance'])
        return imp

    def analyze_attack_difficulty(self):
        """Separability score per attack type (inter vs intra variance)."""
        print(f"\n[{self.run_name}] ATTACK DETECTION DIFFICULTY")
        all_data = []
        for _, df in self.train_data.items():
            all_data.append(self.engineer_features(df))
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

        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
        Xs = np.nan_to_num(Xs, nan=0.0, posinf=0.0, neginf=0.0)

        difficulty = []
        for attack_type in df_all['attack_type'].unique():
            attack_mask = (df_all['attack_type'] == attack_type) & (df_all['attack'] == 1)
            normal_mask = (df_all['attack_type'] == attack_type) & (df_all['attack'] == 0)
            A = Xs[attack_mask]; N = Xs[normal_mask]
            if len(A) == 0 or len(N) == 0: continue
            ac, nc = A.mean(axis=0), N.mean(axis=0)
            inter_d = np.linalg.norm(ac - nc)
            intra_v = (A.var(axis=0).mean() + N.var(axis=0).mean()) / 2
            sep = inter_d / (np.sqrt(intra_v) + _EPS)
            difficulty.append({
                'Attack Type': attack_type.capitalize(),
                'Inter-class Distance': inter_d,
                'Intra-class Variance': intra_v,
                'Separability Score': sep
            })

        diff_df = pd.DataFrame(difficulty).sort_values('Separability Score', ascending=False)
        out_path = self.table_dir / 'attack_difficulty.csv'
        diff_df.to_csv(out_path, index=False)
        print(f"[{self.run_name}] Saved: {out_path}")
        return diff_df


# ---------------------- Comparison Helpers ----------------------

def rank_join(unsup_df: pd.DataFrame, rf_series: pd.Series) -> pd.DataFrame:
    """
    Join unsupervised variance ranks with RF ranks on feature name.
    Robust to empty inputs and column-name drift.
    Returns columns: Feature, Variance, var_rank, rf_importance, rf_rank
    """
    # ---- Guard inputs (no truthiness on pandas) ----
    if unsup_df is None or (isinstance(unsup_df, pd.DataFrame) and unsup_df.empty):
        return pd.DataFrame(columns=['Feature', 'Variance', 'var_rank', 'rf_importance', 'rf_rank'])

    u = unsup_df.copy()

    # Normalize column names
    u.columns = [str(c).strip() for c in u.columns]

    # Identify feature/variance columns
    feature_col = None
    var_col = None

    if 'Feature' in u.columns:
        feature_col = 'Feature'
    elif 'feature' in u.columns:
        feature_col = 'feature'
    else:
        feature_col = u.columns[0]  # best effort fallback

    if 'Variance' in u.columns:
        var_col = 'Variance'
    elif 'variance' in u.columns:
        var_col = 'variance'
    elif 'PCA_Importance' in u.columns:
        var_col = 'PCA_Importance'
    else:
        u['Variance'] = np.nan
        var_col = 'Variance'

    v = u[[feature_col, var_col]].copy()
    v.columns = ['Feature', 'Variance']  # standardize names

    # Variance rank (lower rank = more important)
    v['var_rank'] = v['Variance'].rank(ascending=False, method='dense')

    # ---- RF importance handling ----
    if rf_series is None or (isinstance(rf_series, pd.Series) and rf_series.empty):
        v['rf_importance'] = np.nan
        v['rf_rank'] = np.nan
        return v[['Feature', 'Variance', 'var_rank', 'rf_importance', 'rf_rank']]

    rf_df = rf_series.rename('rf_importance').to_frame()
    rf_df.index = rf_df.index.astype(str)
    rf_df.index.name = 'Feature'
    rf_df = rf_df.reset_index()
    rf_df['rf_rank'] = rf_df['rf_importance'].rank(ascending=False, method='dense')

    m = v.merge(rf_df[['Feature', 'rf_importance', 'rf_rank']], on='Feature', how='inner')

    # Ensure all expected columns exist
    for col in ['Feature', 'Variance', 'var_rank', 'rf_importance', 'rf_rank']:
        if col not in m.columns:
            m[col] = np.nan

    return m[['Feature', 'Variance', 'var_rank', 'rf_importance', 'rf_rank']]



def importance_rank_scatter(m: pd.DataFrame, save_path: Path, title: str):
    plt.figure(figsize=(6,5))
    plt.scatter(m['var_rank'], m['rf_rank'], alpha=0.6)
    plt.xlabel("Unsupervised variance rank (lower=better)")
    plt.ylabel("RF importance rank (lower=better)")
    plt.gca().invert_xaxis(); plt.gca().invert_yaxis()
    plt.title(title)
    plt.tight_layout(); plt.savefig(save_path, bbox_inches='tight'); plt.close()

def compare_runs(baseline: Dict[str, Any], augmented: Dict[str, Any], out_dir_fig: Path, out_dir_tbl: Path):
    out_dir_fig.mkdir(parents=True, exist_ok=True)
    out_dir_tbl.mkdir(parents=True, exist_ok=True)

    # ---- Pull inputs safely ----
    b_unsup_df = None
    a_unsup_df = None
    b_rf = None
    a_rf = None

    if isinstance(baseline.get('unsupervised_importance'), (list, tuple)) and len(baseline['unsupervised_importance']) >= 1:
        b_unsup_df = baseline['unsupervised_importance'][0]
    if isinstance(augmented.get('unsupervised_importance'), (list, tuple)) and len(augmented['unsupervised_importance']) >= 1:
        a_unsup_df = augmented['unsupervised_importance'][0]

    b_rf = baseline.get('supervised_importance', pd.Series(dtype=float))
    a_rf = augmented.get('supervised_importance', pd.Series(dtype=float))

    # ---- Build rank tables (no ambiguous truthiness) ----
    b_merge = rank_join(b_unsup_df, b_rf)
    a_merge = rank_join(a_unsup_df, a_rf)

    # If either side empty, save notes & any partial plots, then try separability compare
    if b_merge.empty or a_merge.empty:
        note = out_dir_tbl / 'comparison_note.txt'
        with open(note, 'w') as f:
            f.write(
                "Comparison skipped or partial because one of the inputs is empty.\n"
                f"baseline ranks empty? {b_merge.empty}\n"
                f"augmented ranks empty? {a_merge.empty}\n"
            )
        print(f"[compare] {note} — one side had no features to compare.")

        if not b_merge.empty:
            importance_rank_scatter(
                b_merge, out_dir_fig / 'baseline_var_vs_rf_rankscatter.png',
                title='Baseline: Variance vs RF Rank'
            )
        if not a_merge.empty:
            importance_rank_scatter(
                a_merge, out_dir_fig / 'augmented_var_vs_rf_rankscatter.png',
                title='Augmented: Variance vs RF Rank'
            )

        # Separability comparison (if both present)
        b_diff = baseline.get('difficulty_ranking', pd.DataFrame())
        a_diff = augmented.get('difficulty_ranking', pd.DataFrame())
        if (isinstance(b_diff, pd.DataFrame) and not b_diff.empty and 'Attack Type' in b_diff.columns
                and isinstance(a_diff, pd.DataFrame) and not a_diff.empty and 'Attack Type' in a_diff.columns):
            diff_cmp = b_diff.merge(a_diff, on='Attack Type', suffixes=('_base', '_aug'))
            diff_cmp['delta_sep'] = diff_cmp['Separability Score_aug'] - diff_cmp['Separability Score_base']
            diff_cmp.sort_values('delta_sep', ascending=False, inplace=True)
            diff_cmp.to_csv(out_dir_tbl / 'separability_comparison.csv', index=False)
            plt.figure(figsize=(7,4))
            plt.bar(diff_cmp['Attack Type'], diff_cmp['delta_sep'])
            plt.axhline(0, linestyle='--', linewidth=1)
            plt.ylabel('Δ Separability (Aug - Base)')
            plt.title('Change in Separability by Attack Type (Augmented vs Baseline)')
            plt.tight_layout()
            plt.savefig(out_dir_fig / 'separability_delta.png', bbox_inches='tight')
            plt.close()
        return

    # ---- Full comparison when both sides present ----
    cmp = b_merge.merge(a_merge, on='Feature', suffixes=('_base','_aug'))
    cmp['delta_var_rank'] = cmp['var_rank_aug'] - cmp['var_rank_base']
    cmp['delta_rf_rank'] = cmp['rf_rank_aug'] - cmp['rf_rank_base']
    cmp.sort_values(['delta_rf_rank','delta_var_rank'], inplace=True)

    cmp_path = out_dir_tbl / 'importance_rank_comparison.csv'
    cmp.to_csv(cmp_path, index=False)

    importance_rank_scatter(
        b_merge, out_dir_fig / 'baseline_var_vs_rf_rankscatter.png',
        title='Baseline: Variance vs RF Rank'
    )
    importance_rank_scatter(
        a_merge, out_dir_fig / 'augmented_var_vs_rf_rankscatter.png',
        title='Augmented: Variance vs RF Rank'
    )

    # Separability compare
    b_diff = baseline.get('difficulty_ranking', pd.DataFrame())
    a_diff = augmented.get('difficulty_ranking', pd.DataFrame())
    if (isinstance(b_diff, pd.DataFrame) and not b_diff.empty and 'Attack Type' in b_diff.columns
            and isinstance(a_diff, pd.DataFrame) and not a_diff.empty and 'Attack Type' in a_diff.columns):
        diff_cmp = b_diff.merge(a_diff, on='Attack Type', suffixes=('_base', '_aug'))
        diff_cmp['delta_sep'] = diff_cmp['Separability Score_aug'] - diff_cmp['Separability Score_base']
        diff_cmp.sort_values('delta_sep', ascending=False, inplace=True)
        diff_cmp_path = out_dir_tbl / 'separability_comparison.csv'
        diff_cmp.to_csv(diff_cmp_path, index=False)

        plt.figure(figsize=(7,4))
        plt.bar(diff_cmp['Attack Type'], diff_cmp['delta_sep'])
        plt.axhline(0, linestyle='--', linewidth=1)
        plt.ylabel('Δ Separability (Aug - Base)')
        plt.title('Change in Separability by Attack Type (Augmented vs Baseline)')
        plt.tight_layout()
        plt.savefig(out_dir_fig / 'separability_delta.png', bbox_inches='tight')
        plt.close()

    print(f"[compare] Saved: {cmp_path}")




# ------------------------------ Runner ------------------------------

def run_single(config: Dict[str, Any], run_name: str, use_only_core: bool, logger=None) -> Dict[str, Any]:
    ac = AttackCharacterizer(config, run_name=run_name, use_only_core=use_only_core, logger=logger)
    ac.load_data()
    stats_df = ac.compute_feature_statistics()
    unsup_df, pc1_series = ac.unsupervised_feature_importance()
    sup_imp = ac.supervised_feature_importance()
    diff_df = ac.analyze_attack_difficulty()

    print(f"\n✓ {run_name} completed. Outputs in {ac.fig_dir} and {ac.table_dir}\n")
    return {
        'feature_statistics': stats_df,
        'unsupervised_importance': (unsup_df, pc1_series),
        'supervised_importance': sup_imp,
        'difficulty_ranking': diff_df,
        'fig_dir': ac.fig_dir,
        'table_dir': ac.table_dir,
    }

def run_task2_workflow(config: Dict[str, Any], logger=None):
    """
    Complete workflow:
      1) Baseline (core only)
      2) Augmented (raw + engineered)
      3) Comparison of ranks & separability
    """
    # 1) Baseline
    baseline = run_single(config, run_name='baseline', use_only_core=True, logger=logger)

    # 2) Augmented
    augmented = run_single(config, run_name='augmented', use_only_core=False, logger=logger)

    # 3) Compare
    out_fig = Path('outputs/figures/task2/compare')
    out_tbl = Path('outputs/tables/task2/compare')
    compare_runs(baseline, augmented, out_fig, out_tbl)

    print("\n✓ Comparison finished. See outputs/figures/task2/compare and outputs/tables/task2/compare\n")
    return {'baseline': baseline, 'augmented': augmented}


# ------------------------------ Main ------------------------------

if __name__ == '__main__':
    import yaml
    with open('config/config.yaml', 'r') as f:
        cfg = yaml.safe_load(f)

    _ = run_task2_workflow(cfg, logger=None)
