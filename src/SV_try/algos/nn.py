from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

def model(X_train, y_train, X_val, y_val):
    print("Initializing MLP (Neural Network)...")

    # Baseline (max_iter increased to ensure convergence)
    baseline_nn = MLPClassifier(max_iter=500, random_state=42)
    baseline_nn.fit(X_train, y_train)
    base_acc = accuracy_score(y_val, baseline_nn.predict(X_val))
    print(f"Baseline NN Val Acc is {base_acc:.4f}")

    print("param grid search...")
    param_grid = {
        'hidden_layer_sizes': [(50,), (100,), (50, 50)],
        'activation': ['relu', 'tanh'],
        'alpha': [0.0001, 0.05],
        'learning_rate': ['constant', 'adaptive']
    }

    gs = GridSearchCV(baseline_nn, param_grid, cv=3, n_jobs=-1, scoring='accuracy')
    gs.fit(X_train, y_train)

    print("\n Best Hyperparameters are", gs.best_params_)
    best_nn = gs.best_estimator_
    tuned_acc = accuracy_score(y_val, best_nn.predict(X_val))
    print(f"Tuned NN Validation Accuracy is {tuned_acc:.4f}")

    return best_nn