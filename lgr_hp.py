import optuna
from sklearn.linear_model import LogisticRegression
from data import X_train, X_test, y_train, y_test
from sklearn.metrics import accuracy_score
import numpy as np


def objective_lgr(trial):
    
    params = {
        'C': trial.suggest_float('C', 1e-4, 10.0, log=True),  # Regularization strength
        'penalty': trial.suggest_categorical('penalty', ['l1', 'l2']),
        'solver': 'liblinear'
    }
    
    lgr_model = LogisticRegression(**params, random_state=42, n_jobs=-1)
    lgr_model.fit(X_train, y_train)

    preds = lgr_model.predict(X_test)
    accuracy = accuracy_score(y_test, preds)

    return accuracy



study = optuna.create_study(direction="maximize")
study.optimize(objective_lgr, n_trials=100)

trial = study.best_trial


print(f"Accuracy: {trial.value}")
print(f"Best hyperparameters: {trial.params}")

optuna.visualization.plot_optimization_history(study)
optuna.visualization.plot_slice(study)
optuna.visualization.plot_contour(study, params=["C", "penalty"])
optuna.visualization.plot_param_importances(study)