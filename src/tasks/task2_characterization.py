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
# Task 2: Feature & Attack Characterization (GOOSE-only fields)
# TWO ANALYSIS MODES: CORE-ONLY vs. FULL (CORE + ALLOWED + ENGINEERED)
# ==============================================================

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

# ---------- Allowed fields ----------
ALLOWED_FIELDS: List[str] = [
    # Core 
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

    # Message timing intervals
    "Epoch Time",
    "Arrival Time",
    "Time delta from previous captured frame",
    "Time delta from previous displayed frame",
    "Time since reference or first frame",
    "Time shift for this packet",

    # Rate-based statistics
    "Frame length on the wire",
    "Frame length stored into the capture file",
]

_EPS = 1e-6


class AttackCharacterizer:
    """Characterize features and attack signatures using only the agreed fields."""

    def __init__(self, config: Dict[str, Any], logger=None, mode: str = 'core'):
        """
        Args:
            mode: 'core' for CORE_FIELDS only, 'full' for CORE + ALLOWED + ENGINEERED
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
        else:
            print("Using CORE + ALLOWED + ENGINEERED features")
        print(f"{'='*70}\n")


    def _preprocess_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply mandatory preprocessing steps:
        1. Split boolean column into boolean_1, boolean_2, boolean_3
        2. Convert bitstring to numeric
        3. Drop attack column for feature calculations (keep for data understanding)
        """
        df = df.copy()
        
        # Step 1: Boolean column transformation
        if 'boolean' in df.columns:
            def parse_boolean_values(val):
                """Parse comma-separated boolean values or single boolean."""
                if pd.isna(val) or val == '':
                    return [np.nan, np.nan, np.nan]
                
                # Convert to string and handle case insensitivity
                val_str = str(val).strip().lower()
                
                # Split by comma if present
                parts = [p.strip() for p in val_str.split(',')]
                
                # Map to numeric (1 for true, 0 for false, nan for empty/invalid)
                def to_numeric(s):
                    if s in ['true', '1', 'yes', 'y', 'on']:
                        return 1.0
                    elif s in ['false', '0', 'no', 'n', 'off']:
                        return 0.0
                    else:
                        return np.nan
                
                # Convert each part
                result = [to_numeric(p) if p else np.nan for p in parts]
                
                # Pad to 3 values
                while len(result) < 3:
                    result.append(np.nan)
                
                return result[:3]  # Take only first 3 if more exist
            
            # Apply parsing
            boolean_split = df['boolean'].apply(parse_boolean_values)
            df['boolean_1'] = boolean_split.apply(lambda x: x[0])
            df['boolean_2'] = boolean_split.apply(lambda x: x[1])
            df['boolean_3'] = boolean_split.apply(lambda x: x[2])
            
            # Drop original boolean column
            df.drop(columns=['boolean'], inplace=True)
            print("  ✓ Split boolean column into boolean_1, boolean_2, boolean_3")
        
        # Step 2: Convert bitstring to numeric (if not already done)
        if 'bit-string' in df.columns:
            # Convert to numeric, coercing any non-numeric values
            df['bitstring_numeric'] = pd.to_numeric(df['bit-string'], errors='coerce')
            print(f"  ✓ Ensured bit-string is numeric (dtype: {df['bitstring_numeric'].dtype})")
        
        return df
    # ----------------------- Data Loading -----------------------

    def load_data(self):
        """Load datasets and subselect fields based on mode."""
        print("Loading datasets...")
        data_dir = Path(self.config['data']['raw_dir'])

        attack_types = ['replay', 'masquerade', 'injection', 'poisoning']

        for attack in attack_types:
            train_file = self.config['data']['train_files'][attack]
            df = pd.read_csv(data_dir / train_file)

            # Select fields based on mode
            if self.mode == 'core':
                keep = [c for c in CORE_FIELDS if c in df.columns]
            else:  # full mode
                keep = [c for c in ALLOWED_FIELDS if c in df.columns]
            
            df = df[keep].copy()
            # Apply preprocessing before standardizing schema
            df = self._preprocess_dataframe(df)

            if 'attack' not in df.columns:
                raise ValueError(f"'attack' column missing in {train_file}.")
            df['attack_type'] = attack

            # Standardize schema (types + timestamps)
            df = self._standardize_schema(df)

            self.train_data[attack] = df
            print(f"  {attack}: {len(df)} samples, {len(df.columns)} columns")

        used_fields = sorted(set().union(*[set(d.columns) for d in self.train_data.values()]))
        used_fields_df = pd.DataFrame({'used_fields': used_fields})
        used_fields_df.to_csv(self.table_dir / 'used_fields.csv', index=False)
        print(f"Saved used field list to {self.table_dir / 'used_fields.csv'}")
        if self.logger is not None:
            self.logger.log_dataframe(used_fields_df, f"task2_{self.mode}/used_fields")

    # ----------------------- Schema Utils -----------------------

    def _standardize_schema(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cast types, normalize times, and add base_time for engineering."""
        df = df.copy()

        numeric_like = [
            "Epoch Time", "Arrival Time", "Length", "Frame length on the wire",
            "Frame length stored into the capture file", "timeAllowedtoLive",
            "stNum", "sqNum", "confRev", "numDatSetEntries",
            "Time delta from previous captured frame",
            "Time delta from previous displayed frame",
            "Time since reference or first frame",
            "Time shift for this packet",
            "attack",
        ]
        for col in numeric_like:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # boolean may be strings: map to 0/1
        for bool_col in ['boolean_1', 'boolean_2', 'boolean_3']:
            if bool_col in df.columns:
                df[bool_col] = pd.to_numeric(df[bool_col], errors='coerce')

        if 'bitstring_numeric' in df.columns:
            df['bitstring_numeric'] = pd.to_numeric(df['bitstring_numeric'], errors='coerce')
            
        if 'bit-string' in df.columns:
            def _popcount_from_hex(x):
                if pd.isna(x): return np.nan
                s = str(x).strip().lower().replace('0x', '')
                try:
                    return bin(int(s, 16))[2:].count('1')
                except Exception:
                    return np.nan
        
            df['bitstring_bitcount'] = df['bit-string'].apply(_popcount_from_hex).astype('float64')

        # Create base_time seconds from best available: Epoch, Arrival, Time, t
        def _parse_to_epoch(col: pd.Series) -> pd.Series:
            if col.dtype == object:
                dt = pd.to_datetime(col, errors='coerce', utc=True)
                return dt.astype('int64') / 1e9  # ns -> s
            return pd.to_numeric(col, errors='coerce')

        epoch = df['Epoch Time'] if 'Epoch Time' in df.columns else pd.Series([np.nan] * len(df))
        arriv = df['Arrival Time'] if 'Arrival Time' in df.columns else pd.Series([np.nan] * len(df))
        time_parsed = _parse_to_epoch(df['Time']) if 'Time' in df.columns else pd.Series([np.nan] * len(df))
        t_parsed = _parse_to_epoch(df['t']) if 't' in df.columns else pd.Series([np.nan] * len(df))

        base_time = epoch.copy().fillna(arriv).fillna(time_parsed).fillna(t_parsed)
        df['base_time'] = pd.to_numeric(base_time, errors='coerce')

        if 'gocbRef' in df.columns:
            df['gocbRef'] = df['gocbRef'].astype(str)

        for c in df.columns:
            if pd.api.types.is_numeric_dtype(df[c]):
                df[c].replace([np.inf, -np.inf], np.nan, inplace=True)

        return df

    # --------------------- Feature Engineering ---------------------

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineered features using ONLY allowed/raw fields.
        Only runs in 'full' mode. In 'core' mode, returns df as-is.
        """
        if self.mode == 'core':
            return df.copy()
        
        df = df.copy()

        # Sort within device stream
        sort_key = 'base_time' if 'base_time' in df.columns else None
        if 'gocbRef' in df.columns and sort_key is not None:
            df = df.sort_values(['gocbRef', sort_key])

        # ----- Inter-arrival timing -----
        if sort_key is not None and 'gocbRef' in df.columns:
            df['inter_arrival'] = df.groupby('gocbRef')[sort_key].diff()
        else:
            if 'Time delta from previous captured frame' in df.columns:
                df['inter_arrival'] = df['Time delta from previous captured frame']
            elif 'Time delta from previous displayed frame' in df.columns:
                df['inter_arrival'] = df['Time delta from previous displayed frame']
            else:
                df['inter_arrival'] = np.nan

        # Jitter (rolling std of inter-arrival)
        if 'gocbRef' in df.columns and 'inter_arrival' in df.columns:
            df['jitter_rolling_std'] = (
                df.groupby('gocbRef')['inter_arrival']
                  .rolling(window=10, min_periods=3)
                  .std()
                  .reset_index(level=0, drop=True)
            )

        # ----- Rates -----
        length_col = 'Frame length on the wire' if 'Frame length on the wire' in df.columns else 'Length'
        if length_col in df.columns and 'inter_arrival' in df.columns:
            df['msg_rate'] = 1.0 / (df['inter_arrival'].astype(float) + _EPS)
            df['byte_rate'] = df[length_col].astype(float) / (df['inter_arrival'].astype(float) + _EPS)

        # ----- TTL consistency -----
        if 'timeAllowedtoLive' in df.columns and 'inter_arrival' in df.columns:
            df['ttl_violation'] = (df['inter_arrival'] > (df['timeAllowedtoLive'] / 1000.0)).astype(float)
            df['ttl_margin'] = (df['timeAllowedtoLive'] / 1000.0) - df['inter_arrival']

        # ----- Sequence transitions -----
        if 'gocbRef' in df.columns and 'sqNum' in df.columns:
            df['sqNum_delta'] = df.groupby('gocbRef')['sqNum'].diff()
            df['sqNum_jump'] = ((df['sqNum_delta'] != 1) & df['sqNum_delta'].notna()).astype(float)

        if 'gocbRef' in df.columns and 'stNum' in df.columns:
            df['stNum_delta'] = df.groupby('gocbRef')['stNum'].diff()
            df['stNum_change'] = ((df['stNum_delta'] != 0) & df['stNum_delta'].notna()).astype(float)

        if all(c in df.columns for c in ['stNum_change', 'sqNum']):
            df['sq_reset_on_st_change'] = ((df['stNum_change'] == 1.0) & (df['sqNum'].isin({0, 1}))).astype(float)

        # ----- Length dynamics -----
        if 'gocbRef' in df.columns and length_col in df.columns:
            df['length_delta'] = df.groupby('gocbRef')[length_col].diff()

        # ----- Optional composite ratios -----
        if 'msg_rate' in df.columns and 'stNum_change' in df.columns:
            df['event_rate_spike'] = df['msg_rate'] * df['stNum_change']

        if 'bitstring_bitcount' in df.columns and length_col in df.columns:
            df['bit_density'] = df['bitstring_bitcount'] / (df[length_col].astype(float) * 8.0 + _EPS)

        # ======== ENGINEERED FEATURES FROM GOOSE RULES ========

        _RESET_SET = {0, 1}

        # 1) Basic change flags / timestamp change
        if 'stNum_change' in df.columns:
            df['st_change'] = df['stNum_change'].astype(float)
        else:
            df['st_change'] = np.nan

        if 't' in df.columns and 'gocbRef' in df.columns:
            df['t_raw'] = df['t'].astype(str)
            df['t_change'] = (df.groupby('gocbRef')['t_raw'].shift() != df['t_raw']).astype(float)
            df.loc[df.groupby('gocbRef').head(1).index, 't_change'] = np.nan
            df.drop(columns=['t_raw'], inplace=True, errors='ignore')
        else:
            df['t_change'] = np.nan

        # allData proxy: boolean / bit-string changes
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

        # 2) Event consistency checks (Case 1)
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


        # 3) Heartbeat consistency (Case 2)
        if 'timeAllowedtoLive' in df.columns and 'inter_arrival' in df.columns:
            df['ttl_half'] = (df['timeAllowedtoLive'] / 2000.0).astype(float)
            tol = 0.05 * df['ttl_half'] + 0.005  # 5% + 5ms
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

        # 4) General hygiene
        if 'stNum_delta' in df.columns:
            df['stNum_jump_gt1'] = (df['stNum_delta'] > 1).astype(float)
        else:
            df['stNum_jump_gt1'] = np.nan

        if 'sqNum' in df.columns and 'st_change' in df.columns:
            df['sq_reset_without_st_change'] = ((df['st_change'] == 0.0) & (df['sqNum'].isin(list(_RESET_SET)))).astype(float)
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
        """Build the list of numeric features for stats/importance (no leakage)."""
        if self.mode == 'core':
            # CORE mode: only numeric fields from CORE_FIELDS
            core_numeric = [
                'timeAllowedtoLive', 'stNum', 'sqNum', 'Length',
                'boolean_1', 'boolean_2', 'boolean_3',
                'bitstring_numeric', 'bitstring_bitcount'
            ]
            numeric_cols = [c for c in core_numeric if c in df.columns]
        else:
            # FULL mode: raw + engineered
            raw_numeric = [
                'Epoch Time', 'Arrival Time', 'Length', 'Frame length on the wire',
                'Frame length stored into the capture file', 'timeAllowedtoLive',
                'stNum', 'sqNum',
                'Time delta from previous captured frame',
                'Time delta from previous displayed frame',
                'Time since reference or first frame',
                'Time shift for this packet',
                'boolean_1', 'boolean_2', 'boolean_3',  # Updated
                'bitstring_numeric', 'bitstring_bitcount',  # Updated
            ]
            raw_numeric = [c for c in raw_numeric if c in df.columns]

            engineered = [
                'base_time', 'inter_arrival', 'jitter_rolling_std',
                'msg_rate', 'byte_rate',
                'ttl_violation', 'ttl_margin',
                'sqNum_delta', 'sqNum_jump',
                'stNum_delta', 'stNum_change',
                'sq_reset_on_st_change',
                'length_delta',
                'event_rate_spike',
                'bit_density',
                'st_change', 't_change', 'status_bit_change',
                'sq_reset_expected', 'sq_reset_violation',
                't_st_consistency_violation',
                'status_change_missing_on_event',
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

    # -------------------- Feature Statistics --------------------

    def compute_feature_statistics(self):
        """Compute feature statistics per attack type (Mann–Whitney U)."""
        print("\n" + "="*70)
        print(f"FEATURE STATISTICS BY ATTACK TYPE ({self.mode.upper()} MODE)")
        print("="*70)

        all_data = []
        for attack, df in self.train_data.items():
            df_eng = self.engineer_features(df)
            all_data.append(df_eng)

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
            df_eng = self.engineer_features(df)
            normal = df_eng[df_eng['attack'] == 0]
            attack = df_eng[df_eng['attack'] == 1]

            sig = {
                'attack_type': attack_type,
                'num_samples': len(df_eng),
                'num_attack': len(attack),
                'attack_ratio': (len(attack) / max(1, len(df_eng))),
            }

            def _m(series): return series.mean() if series.size else np.nan

            if 'stNum' in df_eng.columns:
                sig['stNum_mean_normal'] = _m(normal['stNum'])
                sig['stNum_mean_attack'] = _m(attack['stNum'])

            if 'sqNum' in df_eng.columns:
                sig['sqNum_mean_normal'] = _m(normal['sqNum'])
                sig['sqNum_mean_attack'] = _m(attack['sqNum'])

            if 'timeAllowedtoLive' in df_eng.columns:
                sig['ttl_mean_normal'] = _m(normal['timeAllowedtoLive'])
                sig['ttl_mean_attack'] = _m(attack['timeAllowedtoLive'])

            length_col = 'Frame length on the wire' if 'Frame length on the wire' in df_eng.columns else 'Length'
            if length_col in df_eng.columns:
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
                    if k in df_eng.columns:
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
        all_data = []
        for _, df in self.train_data.items():
            all_data.append(self.engineer_features(df))
        df = pd.concat(all_data, ignore_index=True)
        features = self._numeric_analysis_features(df)
        mats = {}
        for cls in [0,1]:
            sub = df[df.attack==cls][features].copy()
            sub = sub.dropna(axis=1, thresh=int(0.5*len(sub))).fillna(sub.median(numeric_only=True))
            mats[cls] = sub.corr().clip(-1,1)
        delta = mats[1].reindex_like(mats[0]) - mats[0]
        fig = plt.figure(figsize=(9,7))
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
    Execute Task 2: Feature & Attack Characterization in BOTH modes.
    
    Returns results for both CORE-only and FULL (CORE+ALLOWED+ENGINEERED) modes.
    """
    results = {}
    
    # Run both modes
    for mode in ['core', 'full']:
        print(f"\n{'#'*70}")
        print(f"# STARTING {mode.upper()} MODE ANALYSIS")
        print(f"{'#'*70}\n")
        
        characterizer = AttackCharacterizer(config, logger, mode=mode)
        
        # Load data
        characterizer.load_data()
        
        # Analyses
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
    print("TASK 2 COMPLETE - BOTH MODES FINISHED")
    print("="*70)
    print("\nResults available in:")
    print("  - outputs/figures/task2_core/ and outputs/tables/task2_core/")
    print("  - outputs/figures/task2_full/ and outputs/tables/task2_full/")
    
    return results


if __name__ == '__main__':
    import yaml
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    results = run_task2(config, logger=None)