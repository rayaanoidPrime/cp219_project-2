from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

def model(X_train, y_train, X_val, y_val):
    print("Initializing Logistic Regression...")

    # Baseline (max_iter increased to avoid convergence warnings)
    baseline_lr = LogisticRegression(max_iter=1000, random_state=42)
    baseline_lr.fit(X_train, y_train)
    base_acc = accuracy_score(y_val, baseline_lr.predict(X_val))
    print(f"Baseline LR Val Acc is {base_acc:.4f}")

    print("param grid search...")
    param_grid = {
        'C': [0.01, 0.1, 1, 10, 100],
        'solver': ['liblinear', 'lbfgs'], 
        # Note: 'l1' penalty only works with 'liblinear' or 'saga'
        # We will test 'l2' here which works for both, or manage logic separately
    }
    
    
    gs = GridSearchCV(baseline_lr, param_grid, cv=3, n_jobs=-1, scoring='accuracy')
    gs.fit(X_train, y_train)

    print("\n Best Hyperparameters are", gs.best_params_)
    best_lr = gs.best_estimator_
    tuned_acc = accuracy_score(y_val, best_lr.predict(X_val))
    print(f"Tuned LR Validation Accuracy is {tuned_acc:.4f}")

    return best_lr