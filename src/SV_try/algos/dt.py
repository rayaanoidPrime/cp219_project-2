from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV


def model(X_train, y_train, X_val, y_val):
    print("DT Model Training")
    dt_base = DecisionTreeClassifier()
    dt_base.fit(X_train, y_train)
    y_val_pred_base = dt_base.predict(X_val)
    base_acc = accuracy_score(y_val, y_val_pred_base)
    print(f"Baseline DT Val Acc: {base_acc:.4f}")


    print("Hyperparameter tuning ...")

    param_grid = {
        "criterion": ["gini", "entropy"],
        "max_depth": [None, 5, 10, 20, 40],
        "min_samples_split": [2, 5, 10, 20],
        "min_samples_leaf": [1, 2, 4, 8],
        "max_features": [None, "sqrt", "log2"],
    }

    dt = DecisionTreeClassifier(random_state=42)

    gs = GridSearchCV(
        estimator=dt,
        param_grid=param_grid,
        cv=3,
        n_jobs=-1,
        scoring="f1",
    )
    gs.fit(X_train, y_train)

    best_dt = gs.best_estimator_
    tuned_acc = accuracy_score(y_val, best_dt.predict(X_val))

    print("\n Best Hyperparameters (DT):")
    for k, v in gs.best_params_.items():
        print(f"  {k}: {v}")
    print(f"\n Tuned DT Val Acc: {tuned_acc:.4f}")

    return best_dt

# from sklearn.tree import DecisionTreeClassifier
# from sklearn.metrics import accuracy_score
# from sklearn.model_selection import GridSearchCV
# import pandas as pd

# def model(X_train, y_train, X_val, y_val):
#     print("--- DT Model Training ---")
    
#     # Baseline
#     dt_base = DecisionTreeClassifier()
#     dt_base.fit(X_train, y_train)
    
#     # Hyperparameter tuning
#     print("Hyperparameter tuning ...")
#     param_grid = {
#         "criterion": ["gini", "entropy"],
#         "max_depth": [None, 10, 20],
#         "min_samples_split": [2, 10],
#         "min_samples_leaf": [1, 5],
#     }

#     dt = DecisionTreeClassifier(random_state=42)

#     gs = GridSearchCV(
#         estimator=dt,
#         param_grid=param_grid,
#         cv=3,
#         n_jobs=-1,
#         scoring="f1",
#     )
#     gs.fit(X_train, y_train)

#     best_dt = gs.best_estimator_
#     tuned_acc = accuracy_score(y_val, best_dt.predict(X_val))

#     print(f"Tuned DT Val Acc: {tuned_acc:.4f}")
#     return best_dt