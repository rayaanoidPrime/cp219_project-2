from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

def model(X_train, y_train, X_val, y_val):
    print("Here...")

    baseline_rf = RandomForestClassifier(random_state=42)
    baseline_rf.fit(X_train, y_train)
    base_acc = accuracy_score(y_val, baseline_rf.predict(X_val))
    print(f"Baseline RF Val Acc is {base_acc:.4f}")

    print("param grid search...")
    param_grid = {
        "n_estimators": [100, 200, 400],
        "max_depth": [None, 10, 20, 40],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "bootstrap": [True, False],
    }

    gs = GridSearchCV(baseline_rf, param_grid, cv=3, n_jobs=-1, scoring='accuracy')
    gs.fit(X_train, y_train)

    print("\n Best Hyperparameters are", gs.best_params_)
    best_rf = gs.best_estimator_
    tuned_acc = accuracy_score(y_val, best_rf.predict(X_val))
    print(f"Tuned RF Validation Accuracy is {tuned_acc:.4f}")

    return best_rf
