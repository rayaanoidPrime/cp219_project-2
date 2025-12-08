from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

def model(X_train, y_train, X_val, y_val):
    print("Initializing KNN...")

    baseline_knn = KNeighborsClassifier()
    baseline_knn.fit(X_train, y_train)
    base_acc = accuracy_score(y_val, baseline_knn.predict(X_val))
    print(f"Baseline KNN Val Acc is {base_acc:.4f}")

    print("param grid search...")
    param_grid = {
        'n_neighbors': [3, 5, 7, 9, 11],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    }

    gs = GridSearchCV(baseline_knn, param_grid, cv=3, n_jobs=-1, scoring='accuracy')
    gs.fit(X_train, y_train)

    print("\n Best Hyperparameters are", gs.best_params_)
    best_knn = gs.best_estimator_
    tuned_acc = accuracy_score(y_val, best_knn.predict(X_val))
    print(f"Tuned KNN Validation Accuracy is {tuned_acc:.4f}")

    return best_knn