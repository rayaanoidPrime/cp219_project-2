from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import PowerTransformer
from sklearn.pipeline import Pipeline
import numpy as np

def model(X_train, y_train, X_val, y_val):
    print("Initializing Optimized Naive Bayes...")

    # --- 1. Robust Baseline ---
    # We use a basic GaussianNB for comparison
    baseline_nb = GaussianNB()
    baseline_nb.fit(X_train, y_train)
    base_pred = baseline_nb.predict(X_val)
    base_acc = accuracy_score(y_val, base_pred)
    print(f"Baseline NB Val Acc: {base_acc:.4f}")

    pipeline = Pipeline([
        ('scaler', PowerTransformer()), 
        ('nb', GaussianNB())
    ])

    print("Running Param Grid Search...")
    
    param_grid = {
        'nb__var_smoothing': np.logspace(0, -9, num=50)
    }

    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    gs = GridSearchCV(
        pipeline, 
        param_grid, 
        cv=cv_strategy, 
        n_jobs=-1, 
        scoring='accuracy'
    )
    
    gs.fit(X_train, y_train)

    print("\nBest Hyperparameters:", gs.best_params_)
    
    best_nb = gs.best_estimator_
    
    # Predict on validation set
    y_pred = best_nb.predict(X_val)
    tuned_acc = accuracy_score(y_val, y_pred)
    
    print(f"Tuned NB Val Acc:    {tuned_acc:.4f}")
    print(f"Improvement:         {(tuned_acc - base_acc) * 100:.2f}%")
    
    print(classification_report(y_val, y_pred))

    return best_nb