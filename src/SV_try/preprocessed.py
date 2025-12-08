import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

RAW_PATH = "/home/mahalakshmi/Journal_May2025/SV_dec/SV Dataset/sv_iai_lab_normal_fault_replay_injection_full_dataset.csv"
BASE = "/home/mahalakshmi/Journal_May2025/SV_dec/SV Dataset/preprocessed"


def split_df(d: pd.DataFrame):
    train, temp = train_test_split(d, test_size=0.30, random_state=42, shuffle=True)
    val, test = train_test_split(temp, test_size=0.50, random_state=42, shuffle=True)
    return train, val, test


def _sanitize_name(name: str) -> str:
    # for filenames: lowercase, replace spaces and slashes
    return name.lower().replace(" ", "_").replace("/", "_")


def prepare_and_save_splits(raw_path: str = RAW_PATH, base_dir: str = BASE, save_fault: bool = True):
    df = pd.read_csv(raw_path)
    print("Class counts:")
    print(df["class"].value_counts())

    # base subsets
    df_normal = df[df["class"] == "Normal"].copy()
    df_fault = df[df["class"] == "Fault"].copy()

    # Any class that is not Normal or Fault is treated as an "attack class"
    attack_classes = sorted(
        c for c in df["class"].unique() if c not in ["Normal", "Fault"]
    )
    attack_subsets = {
        attack_name: df[df["class"] == attack_name].copy()
        for attack_name in attack_classes
    }

    # Add binary attack label
    df_normal["attack"] = 0
    df_fault["attack"] = 0
    for attack_name, sub_df in attack_subsets.items():
        sub_df["attack"] = 1
        attack_subsets[attack_name] = sub_df  # reassign to keep the column

    print("\nAttack classes found:", attack_classes)
    for attack_name, sub_df in attack_subsets.items():
        print(f"{attack_name}: {sub_df.shape[0]} rows")

    # Split each subset
    normal_train, normal_val, normal_test = split_df(df_normal)
    fault_train, fault_val, fault_test = split_df(df_fault) if not df_fault.empty else (None, None, None)

    # For attacks, keep splits in a dict
    attack_splits = {}
    for attack_name, sub_df in attack_subsets.items():
        a_train, a_val, a_test = split_df(sub_df)
        attack_splits[attack_name] = {
            "train": a_train,
            "val": a_val,
            "test": a_test,
        }

    base = Path(base_dir)

    # Save to disk
    for split_name in ["train", "val", "test"]:
        split_dir = base / split_name
        split_dir.mkdir(parents=True, exist_ok=True)

        # normal
        if split_name == "train":
            split_df_normal = normal_train
        elif split_name == "val":
            split_df_normal = normal_val
        else:
            split_df_normal = normal_test
        split_df_normal.to_csv(split_dir / "normal.csv", index=False)

        # fault (if requested and available)
        if save_fault and fault_train is not None:
            if split_name == "train":
                split_df_fault = fault_train
            elif split_name == "val":
                split_df_fault = fault_val
            else:
                split_df_fault = fault_test
            split_df_fault.to_csv(split_dir / "fault.csv", index=False)

        # attacks: one CSV per attack class in subfolder attack/
        attack_dir = split_dir / "attack"
        attack_dir.mkdir(parents=True, exist_ok=True)

        for attack_name, splits in attack_splits.items():
            df_split = splits[split_name]
            fname = _sanitize_name(attack_name) + ".csv"
            df_split.to_csv(attack_dir / fname, index=False)

    print("Preprocessed splits saved under:", base_dir)

    # Optional: return nested dict of splits if you want to inspect them in memory
    return {
        "normal": {"train": normal_train, "val": normal_val, "test": normal_test},
        "fault": {"train": fault_train, "val": fault_val, "test": fault_test},
        "attack": attack_splits,
    }

def load_preprocessed_for_attack(attack_name: str, base_dir: str = BASE):
    base = Path(base_dir)

    # Normal
    normal_train = pd.read_csv(base / "train" / "normal.csv")
    normal_val   = pd.read_csv(base / "val" / "normal.csv")
    normal_test  = pd.read_csv(base / "test" / "normal.csv")

    # Specific attack (already has attack=1, class=<AttackName>)
    attack_train = pd.read_csv(base / "train" / "attack" / f"{attack_name}.csv")
    attack_val   = pd.read_csv(base / "val" / "attack" / f"{attack_name}.csv")
    attack_test  = pd.read_csv(base / "test" / "attack" / f"{attack_name}.csv")
    
    fault_train = pd.read_csv(base / "train" / "fault.csv")
    fault_val   = pd.read_csv(base / "val" / "fault.csv")
    fault_test  = pd.read_csv(base / "test" / "fault.csv")

    train_df = pd.concat([normal_train, fault_train, attack_train], ignore_index=True)
    val_df   = pd.concat([normal_val,   fault_val,   attack_val],   ignore_index=True)
    
    
    normal_test['class'] = 0
    attack_test['class'] = 1
    fault_test['class']  = 2
    test_df  = pd.concat([normal_test,  fault_test,  attack_test],  ignore_index=True)


    print(f"\n=== Dataset for attack: {attack_name} ===")
    print("Shapes (train, val, test):")
    print(train_df.shape, val_df.shape, test_df.shape)
    print("Attack counts (train/val/test):")
    print(train_df["attack"].value_counts())
    print(val_df["attack"].value_counts())
    print(test_df["attack"].value_counts())


    
    cols_to_drop = ['eth.dst.oui', 'eth.addr.oui', 'eth.src.oui', 'timestamp']
    train_df = train_df.drop(columns=cols_to_drop)
    val_df   = val_df.drop(columns=cols_to_drop)
    test_df  = test_df.drop(columns=cols_to_drop)
    
    # Same feature selection logic as before
    numeric_cols = (
        train_df.drop(["class", "attack"], axis=1)
        .select_dtypes(include=["number"])
        .columns
    )
    
    print("Numeric cols are:")
    print(numeric_cols)

    


    X_train = train_df[numeric_cols]
    y_train = train_df["attack"]

    X_val = val_df[numeric_cols]
    y_val = val_df["attack"]

    X_test = test_df[numeric_cols]
    y_test = test_df["attack"]
    y_test_original_labels = test_df["class"]

    print("Ok")
    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    print(f"Features used: {len(numeric_cols)}")

    return X_train, y_train, X_val, y_val, X_test, y_test, list(numeric_cols), y_test_original_labels
