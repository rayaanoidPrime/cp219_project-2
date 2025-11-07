"""
Preprocessing.py code for dataset loading and feature engineering.
"""

import numpy as np
import pandas as pd
from typing import List

# Core GOOSE protocol fields
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

# Allowed fields for full mode
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


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply mandatory preprocessing steps:
    1. Split boolean column into boolean_1, boolean_2, boolean_3
    2. Convert bitstring to numeric
    3. Keep attack column (will be dropped later for feature analysis)
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
    
    # Step 2: Ensure bitstring is numeric (likely already is)
    if 'bit-string' in df.columns:
        # Convert to numeric, coercing any non-numeric values
        df['bitstring_numeric'] = pd.to_numeric(df['bit-string'], errors='coerce')
        print(f"  ✓ Ensured bit-string is numeric (dtype: {df['bitstring_numeric'].dtype})")
    
    return df


def standardize_schema(df: pd.DataFrame) -> pd.DataFrame:
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

    # Handle the new boolean_1, boolean_2, boolean_3 columns (already processed)
    for bool_col in ['boolean_1', 'boolean_2', 'boolean_3']:
        if bool_col in df.columns:
            df[bool_col] = pd.to_numeric(df[bool_col], errors='coerce')

    # bit-string should now be bitstring_numeric (already converted)
    if 'bitstring_numeric' in df.columns:
        df['bitstring_numeric'] = pd.to_numeric(df['bitstring_numeric'], errors='coerce')
    
    # Also create bitstring_bitcount for analysis
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


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineered features using ONLY allowed/raw fields.
    Should be used only in 'full' mode.
    """
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
    for bool_col in ['boolean_1', 'boolean_2', 'boolean_3']:
        if bool_col in df.columns and 'gocbRef' in df.columns:
            ch = (df[bool_col] != df.groupby('gocbRef')[bool_col].shift()).astype(float)
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


def get_numeric_features(df: pd.DataFrame, mode: str = 'core') -> List[str]:
    """
    Get list of numeric features for analysis (excluding 'attack').
    
    Args:
        df: DataFrame with features
        mode: 'core' for CORE_FIELDS only, 'full' for CORE + ALLOWED + ENGINEERED
    
    Returns:
        List of numeric feature column names
    """
    if mode == 'core':
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
            'boolean_1', 'boolean_2', 'boolean_3',
            'bitstring_numeric', 'bitstring_bitcount',
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

    # Exclude attack column and verify numeric dtype
    numeric_cols = [c for c in numeric_cols if c != 'attack']
    numeric_cols = [c for c in numeric_cols if pd.api.types.is_numeric_dtype(df[c])]
    
    return numeric_cols


def load_and_preprocess(filepath: str, mode: str = 'core') -> pd.DataFrame:
    """
    Load and preprocess a single dataset file.
    
    Args:
        filepath: Path to CSV file
        mode: 'core' or 'full'
    
    Returns:
        Preprocessed DataFrame
    """
    df = pd.read_csv(filepath)
    
    # Select fields based on mode
    if mode == 'core':
        keep = [c for c in CORE_FIELDS if c in df.columns]
    else:  # full mode
        keep = [c for c in ALLOWED_FIELDS if c in df.columns]
    
    df = df[keep].copy()
    
    # Apply preprocessing
    df = preprocess_dataframe(df)
    df = standardize_schema(df)
    
    # Engineer features (only in full mode)
    if mode == 'full':
        df = engineer_features(df)
    
    return df


def load_combined_datasets(data_dir, train_files: dict, test_files: dict, mode: str = 'core'):
    """
    Load and combine all attack datasets into single training and test sets.
    
    Args:
        data_dir: Path to data directory
        train_files: Dict mapping attack_type -> train filename
        test_files: Dict mapping attack_type -> test filename
        mode: 'core' or 'full'
    
    Returns:
        Tuple of (combined_train_df, combined_test_df)
    """
    from pathlib import Path
    
    data_dir = Path(data_dir)
    train_dfs = []
    test_dfs = []
    
    print("\nLoading and combining all attack datasets...")
    
    for attack_type in (train_files.keys()):
        # if attack_type.lower() == 'poisoning':
        #     print(f"  ➤ Skipping {attack_type} dataset.")
        #     continue
        print(f"  Loading {attack_type}...")
        
        # Load train
        train_file = data_dir / train_files[attack_type]
        train_df = pd.read_csv(train_file)
        
        # Load test
        test_file = data_dir / test_files[attack_type]
        test_df = pd.read_csv(test_file)
        
        # Select fields based on mode
        if mode == 'core':
            keep_train = [c for c in CORE_FIELDS if c in train_df.columns]
            keep_test = [c for c in CORE_FIELDS if c in test_df.columns]
        else:  # full mode
            keep_train = [c for c in ALLOWED_FIELDS if c in train_df.columns]
            keep_test = [c for c in ALLOWED_FIELDS if c in test_df.columns]
        
        train_df = train_df[keep_train].copy()
        test_df = test_df[keep_test].copy()
        
        # Apply preprocessing
        train_df = preprocess_dataframe(train_df)
        train_df = standardize_schema(train_df)
        test_df = preprocess_dataframe(test_df)
        test_df = standardize_schema(test_df)
        
        # Engineer features (only in full mode)
        if mode == 'full':
            train_df = engineer_features(train_df)
            test_df = engineer_features(test_df)

         # Tag dataset rows with their attack source (required for balanced evaluation)
        train_df['attack_type'] = attack_type
        test_df['attack_type'] = attack_type
        
        train_dfs.append(train_df)
        test_dfs.append(test_df)
        
        print(f"    {attack_type}: Train={len(train_df)}, Test={len(test_df)}")
    
    # Combine all datasets
    combined_train = pd.concat(train_dfs, ignore_index=True)
    combined_test = pd.concat(test_dfs, ignore_index=True)
    
    print(f"\n  Combined datasets:")
    print(f"    Total Train: {len(combined_train)}, Attack ratio: {combined_train['attack'].mean():.2%}")
    print(f"    Total Test: {len(combined_test)}, Attack ratio: {combined_test['attack'].mean():.2%}")
    
    return combined_train, combined_test