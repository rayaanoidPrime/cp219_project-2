from sklearn.ensemble import RandomForestClassifier
import unsupervised_helper as uh
import os
import pandas as pd
import itertools

root_directory = 'D:/IISC/powerGrid/Final_Results_20_Nov/Final_Datasets/preprocessed_new'

def run_rf(df_val, df_test, scale):
    X_train, X_val, X_test, \
    y_train, y_val, y_test, \
    df_train, df_val, df_test, \
    cols = uh.get_trainable_data(
        df_train=df_val,
        df_val=df_val,
        df_test=df_test,
        scaled_input=scale,
        use_freq=True,
        use_features='all'
    )

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    importance = clf.feature_importances_



def find_dataset_paths(root_dir, validation=False):
    all_paths = []
    for dirpath, dirnames, _ in os.walk(root_dir):
        if 'train' in dirnames and 'test' in dirnames:
            train_dir = os.path.join(dirpath, 'train')
            test_dir = os.path.join(dirpath, 'test')
            train_input = os.path.join(train_dir, 'normal_only.csv')
            val_input = os.path.join(train_dir, 'attack_and_normal.csv') if validation else None
            test_input = os.path.join(test_dir, 'attack_and_normal.csv')
            parts = os.path.relpath(dirpath, root_dir).split(os.sep)
            if len(parts) >= 3:
                d = {'dataset': parts[0], 'goid': parts[1], 'attack_type': parts[2],
                     'train_input_path': train_input, 'test_input_path': test_input}
                if validation:
                    d['validation_input_path'] = val_input
                all_paths.append(d)
    return all_paths

def main():

    scaled_input=[True, False]

    datasets = find_dataset_paths(root_directory, validation=True)
    for scale in scaled_input:
        for i, ds in enumerate(datasets):
            df_val=pd.read_csv(ds.get('validation_input_path'))
            df_test=pd.read_csv(ds['test_input_path'])
            run_rf(df_train=df_val, df_val=df_val, df_test=df_test, )
