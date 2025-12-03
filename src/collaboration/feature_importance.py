import os
import pandas as pd
import numpy as np
from pathlib import Path
import json
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import shap
from mrmr import mrmr_classif

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Configuration
class Config:
    BASE_DIR = r"C:\Users\sengu\Documents\cp219_project-2\data\Final_Datasets\Final_Datasets\preprocessed_new\preprocessed_new"
    OUTPUT_DIR = "results"
    TOP_K_FEATURES = 15
    TARGET_COL = "attack"
    RANDOM_STATE = 42
    
    # Model parameters
    XGB_PARAMS = {
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_estimators': 100,
        'objective': 'binary:logistic',
        'random_state': RANDOM_STATE,
        'n_jobs': -1
    }
    
    RF_PARAMS = {
        'n_estimators': 100,
        'max_depth': 10,
        'random_state': RANDOM_STATE,
        'n_jobs': -1
    }

# Create output directories
os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
os.makedirs(f"{Config.OUTPUT_DIR}/dataset_rankings", exist_ok=True)
os.makedirs(f"{Config.OUTPUT_DIR}/visualizations", exist_ok=True)
os.makedirs(f"{Config.OUTPUT_DIR}/visualizations/per_dataset", exist_ok=True)


def discover_datasets(base_dir):
    """Recursively find all train/test dataset pairs."""
    datasets = []
    base_path = Path(base_dir)
    
    for train_dir in base_path.rglob("train"):
        test_dir = train_dir.parent / "test"
        
        if test_dir.exists():
            # Find CSV files
            train_files = list(train_dir.glob("*.csv"))
            test_files = list(test_dir.glob("*.csv"))
            
            if train_files and test_files:
                # Create readable dataset name from path
                relative_path = train_dir.parent.relative_to(base_path)
                dataset_name = str(relative_path).replace(os.sep, "_")
                
                datasets.append({
                    'name': dataset_name,
                    'train_path': train_files[0],  # Take first CSV
                    'test_path': test_files[0],
                    'category': str(relative_path).split(os.sep)[0]
                })
    
    return datasets


def load_and_prepare_data(train_path, test_path, target_col):
    """Load train/test data and prepare for modeling."""
    try:
        # Load data
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        
        # Verify target column exists
        if target_col not in train_df.columns:
            print(f"⚠️ Target column '{target_col}' not found. Available: {train_df.columns.tolist()}")
            return None
        
        # Separate features and target
        X_train = train_df.drop(columns=[target_col])
        y_train = train_df[target_col]
        X_test = test_df.drop(columns=[target_col]) if target_col in test_df.columns else test_df
        y_test = test_df[target_col] if target_col in test_df.columns else None
        
        # Handle categorical features
        categorical_cols = X_train.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            le = LabelEncoder()
            for col in categorical_cols:
                X_train[col] = le.fit_transform(X_train[col].astype(str))
                if col in X_test.columns:
                    X_test[col] = le.transform(X_test[col].astype(str))
        
        # Handle missing values
        X_train = X_train.fillna(X_train.mean())
        X_test = X_test.fillna(X_train.mean())
        
        # Remove constant features
        constant_cols = X_train.columns[X_train.nunique() <= 1]
        X_train = X_train.drop(columns=constant_cols)
        X_test = X_test.drop(columns=constant_cols)
        
        return X_train, X_test, y_train, y_test
    
    except Exception as e:
        print(f"❌ Error loading data: {str(e)}")
        return None

def calculate_xgboost_shap(X_train, X_test, y_train):
    """Calculate feature importance using XGBoost + SHAP."""
    try:
        model = xgb.XGBClassifier(**Config.XGB_PARAMS)
        model.fit(X_train, y_train)
        
        # Calculate SHAP values
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_train)
        
        # Get mean absolute SHAP values
        mean_shap = np.abs(shap_values).mean(axis=0)
        
        return dict(zip(X_train.columns, mean_shap)), model, explainer, shap_values
    
    except Exception as e:
        print(f"❌ XGBoost error: {str(e)}")
        return None, None, None, None

def calculate_random_forest(X_train, X_test, y_train):
    """Calculate feature importance using Random Forest."""
    try:
        model = RandomForestClassifier(**Config.RF_PARAMS)
        model.fit(X_train, y_train)
        
        importances = model.feature_importances_
        return dict(zip(X_train.columns, importances)), model
    
    except Exception as e:
        print(f"❌ Random Forest error: {str(e)}")
        return None, None

def calculate_mrmr(X_train, y_train, K=None):
    """Calculate feature importance using MRMR."""
    try:
        if K is None:
            K = min(len(X_train.columns), Config.TOP_K_FEATURES * 2)
        
        # MRMR requires discrete target
        selected_features = mrmr_classif(X=X_train, y=y_train, K=K)
        
        # Create scores (higher rank = higher importance)
        scores = {feat: len(selected_features) - i for i, feat in enumerate(selected_features)}
        
        return scores
    
    except Exception as e:
        print(f"❌ MRMR error: {str(e)}")
        return None

def normalize_scores(scores_dict):
    """Normalize scores to 0-1 range."""
    if not scores_dict or len(scores_dict) == 0:
        return {}
    
    values = np.array(list(scores_dict.values()))
    min_val, max_val = values.min(), values.max()
    
    if max_val == min_val:
        return {k: 1.0 for k in scores_dict.keys()}
    
    return {k: (v - min_val) / (max_val - min_val) for k, v in scores_dict.items()}

def process_single_dataset(dataset_info):
    """Process a single dataset and calculate all importance scores."""
    name = dataset_info['name']
    print(f"\n{'='*80}")
    print(f"Processing: {name}")
    print(f"{'='*80}")
    
    # Load data
    data = load_and_prepare_data(
        dataset_info['train_path'],
        dataset_info['test_path'],
        Config.TARGET_COL
    )
    
    if data is None:
        return None
    
    X_train, X_test, y_train, y_test = data
    print(f"📦 Data shape: {X_train.shape}, Features: {len(X_train.columns)}")
    
    results = {
        'dataset_name': name,
        'category': dataset_info['category'],
        'n_features': len(X_train.columns),
        'n_samples': len(X_train),
        'features': list(X_train.columns)
    }
    
    # Method 1: XGBoost + SHAP
    print("🔧 Calculating XGBoost + SHAP...")
    xgb_scores, xgb_model, explainer, shap_values = calculate_xgboost_shap(X_train, X_test, y_train)
    if xgb_scores:
        results['xgb_shap_scores'] = normalize_scores(xgb_scores)
        results['xgb_model'] = xgb_model
        results['shap_explainer'] = explainer
        results['shap_values'] = shap_values
    
    # Method 2: Random Forest
    print("🌲 Calculating Random Forest...")
    rf_scores, rf_model = calculate_random_forest(X_train, X_test, y_train)
    if rf_scores:
        results['rf_scores'] = normalize_scores(rf_scores)
        results['rf_model'] = rf_model
    
    # Method 3: MRMR
    print("🎯 Calculating MRMR...")
    mrmr_scores = calculate_mrmr(X_train, y_train)
    if mrmr_scores:
        results['mrmr_scores'] = normalize_scores(mrmr_scores)
    
    # Calculate combined score (equal weights)
    if all(k in results for k in ['xgb_shap_scores', 'rf_scores', 'mrmr_scores']):
        combined = defaultdict(list)
        for method in ['xgb_shap_scores', 'rf_scores', 'mrmr_scores']:
            for feat, score in results[method].items():
                combined[feat].append(score)
        
        results['combined_scores'] = {
            feat: np.mean(scores) for feat, scores in combined.items()
        }
    
    # Store data for visualization
    results['X_train'] = X_train
    
    return results



if __name__ == "__main__":
    print("🔍 Phase 1: Discovering datasets...")
    datasets = discover_datasets(Config.BASE_DIR)
    print(f"✅ Found {len(datasets)} dataset pairs")
    print(f"📊 Categories: {set(d['category'] for d in datasets)}")
   
    print("\n" + "="*80)
    print("🚀 Phase 2: Processing all datasets...")
    print("="*80)

    all_results = []
    failed_datasets = []

    for dataset_info in tqdm(datasets, desc="Processing datasets"):
        try:
            result = process_single_dataset(dataset_info)
            if result:
                all_results.append(result)
            else:
                failed_datasets.append(dataset_info['name'])
        except Exception as e:
            print(f"❌ Failed to process {dataset_info['name']}: {str(e)}")
            failed_datasets.append(dataset_info['name'])

    print(f"\n✅ Successfully processed: {len(all_results)}/{len(datasets)} datasets")
    if failed_datasets:
        print(f"⚠️ Failed datasets: {', '.join(failed_datasets)}")


    print("\n" + "="*80)
    print("📊 Phase 3: Aggregating results...")
    print("="*80)

    # Aggregate feature scores across datasets
    feature_global_scores = defaultdict(lambda: {'scores': [], 'datasets': [], 'count': 0})

    for result in all_results:
        if 'combined_scores' in result:
            for feat, score in result['combined_scores'].items():
                feature_global_scores[feat]['scores'].append(score)
                feature_global_scores[feat]['datasets'].append(result['dataset_name'])
                feature_global_scores[feat]['count'] += 1

    # Calculate global rankings
    global_ranking = []
    for feat, data in feature_global_scores.items():
        global_ranking.append({
            'feature': feat,
            'avg_importance': np.mean(data['scores']),
            'std_importance': np.std(data['scores']),
            'appearances': data['count'],
            'datasets': '; '.join(data['datasets'][:3]) + ('...' if len(data['datasets']) > 3 else '')
        })

    global_ranking_df = pd.DataFrame(global_ranking)
    global_ranking_df = global_ranking_df.sort_values('avg_importance', ascending=False)

    # Save global ranking
    global_ranking_df.to_csv(f"{Config.OUTPUT_DIR}/global_ranking.csv", index=False)
    print(f"💾 Saved global ranking to {Config.OUTPUT_DIR}/global_ranking.csv")

    print("\n" + "="*80)
    print("📈 Phase 4: Creating visualizations...")
    print("="*80)

    # 1. Global Top K Features
    plt.figure(figsize=(12, 8))
    top_features = global_ranking_df.head(Config.TOP_K_FEATURES)
    plt.barh(range(len(top_features)), top_features['avg_importance'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Average Normalized Importance Score')
    plt.title(f'Top {Config.TOP_K_FEATURES} Features Globally (Combined Methods)')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(f"{Config.OUTPUT_DIR}/visualizations/global_top{Config.TOP_K_FEATURES}.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Per-dataset rankings
    for result in tqdm(all_results[:10], desc="Creating per-dataset visualizations"):  # Limit to first 10 for demo
        if 'combined_scores' not in result:
            continue
        
        dataset_name = result['dataset_name']
        scores_df = pd.DataFrame([
            {'feature': k, 'score': v} 
            for k, v in result['combined_scores'].items()
        ]).sort_values('score', ascending=False).head(Config.TOP_K_FEATURES)
        
        plt.figure(figsize=(10, 6))
        plt.barh(range(len(scores_df)), scores_df['score'])
        plt.yticks(range(len(scores_df)), scores_df['feature'])
        plt.xlabel('Combined Importance Score')
        plt.title(f'Top {Config.TOP_K_FEATURES} Features: {dataset_name}')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        safe_name = dataset_name.replace('/', '_').replace('\\', '_')
        plt.savefig(f"{Config.OUTPUT_DIR}/visualizations/per_dataset/{safe_name}.png", dpi=300, bbox_inches='tight')
        plt.close()

    # 3. Method Comparison (Correlation)
    print("📊 Creating method comparison...")
    method_scores = defaultdict(lambda: {'xgb': [], 'rf': [], 'mrmr': []})

    for result in all_results:
        if all(k in result for k in ['xgb_shap_scores', 'rf_scores', 'mrmr_scores']):
            for feat in result['features']:
                if feat in result['xgb_shap_scores']:
                    method_scores[feat]['xgb'].append(result['xgb_shap_scores'].get(feat, 0))
                    method_scores[feat]['rf'].append(result['rf_scores'].get(feat, 0))
                    method_scores[feat]['mrmr'].append(result['mrmr_scores'].get(feat, 0))

    # Calculate average scores per method
    method_avg = {
        'XGBoost+SHAP': [],
        'Random Forest': [],
        'MRMR': []
    }

    for feat_data in method_scores.values():
        if feat_data['xgb']:
            method_avg['XGBoost+SHAP'].append(np.mean(feat_data['xgb']))
            method_avg['Random Forest'].append(np.mean(feat_data['rf']))
            method_avg['MRMR'].append(np.mean(feat_data['mrmr']))

    comparison_df = pd.DataFrame(method_avg)
    correlation_matrix = comparison_df.corr()

    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title('Correlation Between Feature Importance Methods')
    plt.tight_layout()
    plt.savefig(f"{Config.OUTPUT_DIR}/visualizations/method_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()

    # ============================================================================
    # FINAL SUMMARY
    # ============================================================================

    print("\n" + "="*80)
    print("✅ EXECUTION COMPLETE!")
    print("="*80)
    print(f"\n📁 Results saved to: {Config.OUTPUT_DIR}/")
    print(f"\n🏆 Top 10 Most Important Features Globally:")
    print(global_ranking_df.head(10)[['feature', 'avg_importance', 'appearances']].to_string(index=False))

    # Save metadata
    metadata = {
        'total_datasets': len(datasets),
        'processed_datasets': len(all_results),
        'failed_datasets': failed_datasets,
        'total_unique_features': len(global_ranking_df),
        'top_features': global_ranking_df.head(20).to_dict('records'),
        'categories': list(set(d['category'] for d in datasets))
    }

    with open(f"{Config.OUTPUT_DIR}/metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n📄 Full results available in:")
    print(f"   - Global ranking: {Config.OUTPUT_DIR}/global_ranking.csv")
    print(f"   - Visualizations: {Config.OUTPUT_DIR}/visualizations/")
    print(f"   - Metadata: {Config.OUTPUT_DIR}/metadata.json")