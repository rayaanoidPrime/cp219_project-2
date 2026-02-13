import os
import pandas as pd
import numpy as np
from pathlib import Path
import json
from collections import defaultdict
import warnings
import traceback
import pickle
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import xgboost as xgb
import shap

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Set up Computer Modern font (IEEE/LaTeX style)
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Computer Modern Roman', 'CMU Serif', 'Times New Roman'],
    'mathtext.fontset': 'cm',
    'axes.unicode_minus': False,
})

# Configuration
class Config:
    BASE_DIR = r"C:\Users\sengu\Documents\cp219_project-2\data\Final_Datasets\Final_Datasets\preprocessed_new\preprocessed_new"
    OUTPUT_DIR = "results"
    TOP_K_FEATURES = 15
    TARGET_COL = "attack"
    RANDOM_STATE = 42
    
    # File names
    TRAIN_FILE = "attack_and_normal.csv"
    TEST_FILE = "attack_and_normal.csv"
    
    # Model parameters
    XGB_PARAMS = {
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_estimators': 100,
        'objective': 'binary:logistic',
        'base_score': 0.5,
        'random_state': RANDOM_STATE,
        'n_jobs': -1,
        'verbosity': 0
    }
    
    RF_PARAMS = {
        'n_estimators': 100,
        'max_depth': 10,
        'random_state': RANDOM_STATE,
        'n_jobs': -1
    }
    
    # PCA parameters
    PCA_VARIANCE_THRESHOLD = 0.95

# Create output directories
os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
os.makedirs(f"{Config.OUTPUT_DIR}/visualizations", exist_ok=True)

# ============================================================================
# PHASE 1: DATA DISCOVERY
# ============================================================================

def discover_datasets(base_dir):
    """Recursively find all train/test dataset pairs with specific file names."""
    datasets = []
    base_path = Path(base_dir)
    
    print("🔍 Scanning directory structure...")
    
    for train_dir in base_path.rglob("train"):
        test_dir = train_dir.parent / "test"
        
        # Look for specific file names
        train_file = train_dir / Config.TRAIN_FILE
        test_file = test_dir / Config.TEST_FILE
        
        if train_file.exists() and test_file.exists():
            # Create readable dataset name from path
            relative_path = train_dir.parent.relative_to(base_path)
            dataset_name = str(relative_path).replace(os.sep, "_")
            
            datasets.append({
                'name': dataset_name,
                'train_path': train_file,
                'test_path': test_file,
                'category': str(relative_path).split(os.sep)[0]
            })
            print(f"  ✓ Found: {dataset_name}")
        else:
            # Log missing files
            if not train_file.exists():
                print(f"  ✗ Missing train file: {train_file}")
            if not test_file.exists():
                print(f"  ✗ Missing test file: {test_file}")
    
    return datasets

print("🔍 Phase 1: Discovering datasets...")
datasets = discover_datasets(Config.BASE_DIR)
print(f"\n✅ Found {len(datasets)} dataset pairs")

if len(datasets) == 0:
    print("❌ No datasets found! Please check the BASE_DIR path.")
    exit(1)

print(f"📊 Categories found: {set(d['category'] for d in datasets)}")

# ============================================================================
# PHASE 2: FEATURE IMPORTANCE CALCULATION
# ============================================================================

def load_and_prepare_data(train_path, test_path, target_col):
    """Load train/test data and prepare for modeling."""
    try:
        # Load data
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        
        print(f"  📦 Train shape: {train_df.shape}, Test shape: {test_df.shape}")
        
        # Verify target column exists
        if target_col not in train_df.columns:
            available_cols = train_df.columns.tolist()
            print(f"  ⚠️ Target column '{target_col}' not found.")
            print(f"  Available columns: {available_cols[:10]}...")
            return None
        
        # Separate features and target
        X_train = train_df.drop(columns=[target_col])
        y_train = train_df[target_col]
        X_test = test_df.drop(columns=[target_col]) if target_col in test_df.columns else test_df
        y_test = test_df[target_col] if target_col in test_df.columns else None
        
        # Exclude 'index' column from features (not meaningful for classification)
        if 'index' in X_train.columns:
            print(f"  🔧 Excluding 'index' column from features")
            X_train = X_train.drop(columns=['index'])
            if 'index' in X_test.columns:
                X_test = X_test.drop(columns=['index'])
        
        # Exclude 'freq' column from features (not using it in our models)
        if 'freq' in X_train.columns:
            print(f"  🔧 Excluding 'freq' column from features")
            X_train = X_train.drop(columns=['freq'])
            if 'freq' in X_test.columns:
                X_test = X_test.drop(columns=['freq'])
        
        # Store original feature names
        original_features = X_train.columns.tolist()
        
        # Handle non-numeric columns
        non_numeric_cols = X_train.select_dtypes(include=['object']).columns
        if len(non_numeric_cols) > 0:
            print(f"  🔧 Dropping {len(non_numeric_cols)} non-numeric columns")
            X_train = X_train.drop(columns=non_numeric_cols)
            X_test = X_test.drop(columns=non_numeric_cols)
        
        # Handle missing values
        if X_train.isnull().any().any():
            print(f"  🔧 Filling missing values")
            X_train = X_train.fillna(X_train.mean())
            X_test = X_test.fillna(X_train.mean())
        
        # Handle infinite values
        X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(X_train.mean())
        X_test = X_test.replace([np.inf, -np.inf], np.nan).fillna(X_train.mean())
        
        print(f"  ✓ Final feature count: {len(X_train.columns)}")
        
        return X_train, X_test, y_train, y_test
    
    except Exception as e:
        print(f"  ❌ Error loading data: {str(e)}")
        traceback.print_exc()
        return None

def calculate_xgboost_shap(X_train, X_test, y_train):
    """Calculate feature importance using XGBoost + SHAP (Supervised)."""
    try:
        print("  🔧 Training XGBoost model...")
        model = xgb.XGBClassifier(**Config.XGB_PARAMS)
        # Ensure labels are binary (0/1) for logistic objective
        y_binary = (y_train != 0).astype(int)
        model.fit(X_train, y_binary)
        
        print("  🔧 Calculating SHAP values...")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_train)
        
        # Get mean absolute SHAP values
        mean_shap = np.abs(shap_values).mean(axis=0)
        
        return dict(zip(X_train.columns, mean_shap)), model, explainer, shap_values
    
    except Exception as e:
        print(f"  ❌ XGBoost error: {str(e)}")
        traceback.print_exc()
        return None, None, None, None

def calculate_random_forest(X_train, X_test, y_train):
    """Calculate feature importance using Random Forest (Supervised)."""
    try:
        print("  🌲 Training Random Forest model...")
        model = RandomForestClassifier(**Config.RF_PARAMS)
        model.fit(X_train, y_train)
        
        importances = model.feature_importances_
        return dict(zip(X_train.columns, importances)), model
    
    except Exception as e:
        print(f"  ❌ Random Forest error: {str(e)}")
        traceback.print_exc()
        return None, None

def calculate_pca_importance(X_train):
    """
    Calculate feature importance using PCA (Unsupervised).
    Features that contribute more to principal components are more important.
    """
    try:
        print("  🎯 Calculating PCA-based importance...")
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train)
        
        # Fit PCA
        pca = PCA(n_components=Config.PCA_VARIANCE_THRESHOLD, random_state=Config.RANDOM_STATE)
        pca.fit(X_scaled)
        
        # Calculate feature importance as weighted sum of absolute loadings
        # Weight by explained variance ratio of each component
        components = np.abs(pca.components_)
        feature_importance = np.sum(
            components * pca.explained_variance_ratio_[:, np.newaxis],
            axis=0
        )
        
        scores = dict(zip(X_train.columns, feature_importance))
        
        print(f"  ✓ PCA retained {pca.n_components_} components explaining {pca.explained_variance_ratio_.sum():.2%} variance")
        
        return scores, pca
    
    except Exception as e:
        print(f"  ❌ PCA error: {str(e)}")
        traceback.print_exc()
        return None, None

def normalize_scores(scores_dict):
    """Normalize scores to 0-1 range."""
    if not scores_dict or len(scores_dict) == 0:
        return {}
    
    values = np.array(list(scores_dict.values()))
    min_val, max_val = values.min(), values.max()
    
    if max_val == min_val:
        return {k: 1.0 for k in scores_dict.keys()}
    
    return {k: (v - min_val) / (max_val - min_val) for k, v in scores_dict.items()}

def process_single_dataset(dataset_info, output_dir):
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
    
    if len(X_train.columns) == 0:
        print("  ❌ No features remaining after preprocessing")
        return None
    
    results = {
        'dataset_name': name,
        'category': dataset_info['category'],
        'n_features': len(X_train.columns),
        'n_samples': len(X_train),
        'n_attacks': int(y_train.sum()),
        'n_normal': int((y_train == 0).sum()),
        'features': list(X_train.columns)
    }
    
    # Method 1: XGBoost + SHAP (Supervised)
    print("🔧 Method 1: XGBoost + SHAP (Supervised)")
    xgb_scores, xgb_model, explainer, shap_values = calculate_xgboost_shap(X_train, X_test, y_train)
    if xgb_scores:
        results['xgb_shap_scores'] = normalize_scores(xgb_scores)
        results['xgb_model'] = xgb_model
        results['shap_explainer'] = explainer
        results['shap_values'] = shap_values
        print("  ✓ XGBoost + SHAP completed")
    
    # Method 2: Random Forest (Supervised)
    print("🌲 Method 2: Random Forest (Supervised)")
    rf_scores, rf_model = calculate_random_forest(X_train, X_test, y_train)
    if rf_scores:
        results['rf_scores'] = normalize_scores(rf_scores)
        results['rf_model'] = rf_model
        print("  ✓ Random Forest completed")
    
    # Method 3: PCA (Unsupervised)
    print("🎯 Method 3: PCA-based (Unsupervised)")
    pca_scores, pca_model = calculate_pca_importance(X_train)
    if pca_scores:
        results['pca_scores'] = normalize_scores(pca_scores)
        results['pca_model'] = pca_model
        print("  ✓ PCA completed")
    
    # Calculate combined score (equal weights)
    if all(k in results for k in ['xgb_shap_scores', 'rf_scores', 'pca_scores']):
        combined = defaultdict(list)
        for method in ['xgb_shap_scores', 'rf_scores', 'pca_scores']:
            for feat, score in results[method].items():
                combined[feat].append(score)
        
        results['combined_scores'] = {
            feat: np.mean(scores) for feat, scores in combined.items()
        }
        print("  ✓ Combined scores calculated")
    
    # Store data for visualization
    results['X_train'] = X_train
    
    return results

# ============================================================================
# PHASE 3: PROCESS ALL DATASETS 
# ============================================================================

print("\n" + "="*80)
print("🚀 Phase 2: Processing all datasets...")
print("="*80)

all_results = []
failed_datasets = []
processed_indices = set()

error_log = []

for idx, dataset_info in enumerate(tqdm(datasets, desc="Processing datasets")):
    # Skip if already processed
    if idx in processed_indices:
        continue
    
    try:
        result = process_single_dataset(dataset_info, Config.OUTPUT_DIR)
        if result:
            all_results.append(result)
            processed_indices.add(idx)
        else:
            failed_datasets.append(dataset_info['name'])
            error_log.append({
                'dataset': dataset_info['name'],
                'error': 'Processing returned None',
                'index': idx
            })
    except Exception as e:
        error_msg = str(e)
        error_trace = traceback.format_exc()
        print(f"❌ Failed to process {dataset_info['name']}: {error_msg}")
        failed_datasets.append(dataset_info['name'])
        error_log.append({
            'dataset': dataset_info['name'],
            'error': error_msg,
            'traceback': error_trace,
            'index': idx
        })

# Save error log
if error_log:
    error_df = pd.DataFrame(error_log)
    error_df.to_csv(f"{Config.OUTPUT_DIR}/error_log.csv", index=False)
    print(f"\n⚠️ Error log saved to {Config.OUTPUT_DIR}/error_log.csv")

print(f"\n✅ Successfully processed: {len(all_results)}/{len(datasets)} datasets")
if failed_datasets:
    print(f"⚠️ Failed datasets ({len(failed_datasets)}): {', '.join(failed_datasets[:5])}{'...' if len(failed_datasets) > 5 else ''}")

# ============================================================================
# PHASE 4: AGGREGATION & GLOBAL RANKING
# ============================================================================

print("\n" + "="*80)
print("📊 Phase 3: Aggregating results...")
print("="*80)

if len(all_results) == 0:
    print("❌ No results to aggregate! Check error log.")
    exit(1)

# Aggregate feature scores across datasets
feature_global_scores = defaultdict(lambda: {
    'xgb_scores': [], 'rf_scores': [], 'pca_scores': [], 'combined_scores': [],
    'datasets': [], 'count': 0
})

for result in all_results:
    for feat in result['features']:
        if feat in result.get('combined_scores', {}):
            feature_global_scores[feat]['combined_scores'].append(result['combined_scores'][feat])
            feature_global_scores[feat]['xgb_scores'].append(result.get('xgb_shap_scores', {}).get(feat, 0))
            feature_global_scores[feat]['rf_scores'].append(result.get('rf_scores', {}).get(feat, 0))
            feature_global_scores[feat]['pca_scores'].append(result.get('pca_scores', {}).get(feat, 0))
            feature_global_scores[feat]['datasets'].append(result['dataset_name'])
            feature_global_scores[feat]['count'] += 1

# Calculate global rankings
global_ranking = []
for feat, data in feature_global_scores.items():
    global_ranking.append({
        'feature': feat,
        'avg_combined_importance': np.mean(data['combined_scores']),
        'std_combined_importance': np.std(data['combined_scores']),
        'avg_xgb_importance': np.mean(data['xgb_scores']),
        'avg_rf_importance': np.mean(data['rf_scores']),
        'avg_pca_importance': np.mean(data['pca_scores']),
        'appearances': data['count'],
        'datasets': '; '.join(data['datasets'][:3]) + ('...' if len(data['datasets']) > 3 else '')
    })

global_ranking_df = pd.DataFrame(global_ranking)
global_ranking_df = global_ranking_df.sort_values('avg_combined_importance', ascending=False)

# Save global ranking
global_ranking_df.to_csv(f"{Config.OUTPUT_DIR}/global_ranking.csv", index=False)
print(f"💾 Saved global ranking to {Config.OUTPUT_DIR}/global_ranking.csv")

# ============================================================================
# PHASE 5: VISUALIZATION
# ============================================================================

print("\n" + "="*80)
print("📈 Phase 4: Creating visualizations...")
print("="*80)

# 1. Global Top K Features
plt.figure(figsize=(16, 10))
top_features = global_ranking_df.head(Config.TOP_K_FEATURES)
colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_features)))
plt.barh(range(len(top_features)), top_features['avg_combined_importance'], color=colors)
plt.yticks(range(len(top_features)), top_features['feature'], fontsize=26)
plt.xticks(fontsize=20)
plt.xlabel('Average Combined Importance Score', fontsize=24)
plt.ylabel('Feature Name', fontsize=22)
plt.title(f'Top {Config.TOP_K_FEATURES} Features Globally\n(XGBoost+SHAP + Random Forest + PCA)', fontsize=22)
plt.gca().invert_yaxis()
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(f"{Config.OUTPUT_DIR}/global_top{Config.TOP_K_FEATURES}.png", dpi=330, bbox_inches='tight')
plt.close()
print("✓ Global ranking visualization saved")

print(f"✓ Created {len(all_results)} per-dataset visualizations")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "="*80)
print("✅ EXECUTION COMPLETE!")
print("="*80)

print(f"\n📊 Processing Summary:")
print(f"  • Total datasets found: {len(datasets)}")
print(f"  • Successfully processed: {len(all_results)}")
print(f"  • Failed: {len(failed_datasets)}")
print(f"  • Unique features discovered: {len(global_ranking_df)}")

print(f"\n🏆 Top 10 Most Important Features Globally:")
print(global_ranking_df.head(10)[['feature', 'avg_combined_importance', 'appearances']].to_string(index=False))

# Save metadata
metadata = {
    'total_datasets': len(datasets),
    'processed_datasets': len(all_results),
    'failed_datasets': failed_datasets,
    'total_unique_features': len(global_ranking_df),
    'top_features': global_ranking_df.head(20).to_dict('records'),
    'categories': list(set(d['category'] for d in datasets)),
    'config': {
        'top_k_features': Config.TOP_K_FEATURES,
        'target_col': Config.TARGET_COL,
        'train_file': Config.TRAIN_FILE,
        'test_file': Config.TEST_FILE
    }
}

with open(f"{Config.OUTPUT_DIR}/metadata.json", 'w') as f:
    json.dump(metadata, f, indent=2, default=str)

print(f"\n📁 Results saved to: {Config.OUTPUT_DIR}/")
print(f"   ├── global_ranking.csv ({len(global_ranking_df)} features)")
print(f"   ├── global_top{Config.TOP_K_FEATURES}.png")
print(f"   ├── metadata.json")

if error_log:
    print(f"\n⚠️  Check error_log.csv for details on {len(failed_datasets)} failed datasets")

print("\n🎉 All done! Check the results directory for outputs.")