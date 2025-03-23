import optuna
from sklearn.ensemble import RandomForestClassifier
from data import X_train, X_test, y_train, y_test
from sklearn.metrics import accuracy_score
import numpy as np

def objective_rf(trial):

    params = {
        'n_estimators':trial.suggest_int("n_estimators", 50, 300),
        'max_depth':trial.suggest_int("max_depth", 1, 32, log=True),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
        'bootstrap': trial.suggest_categorical('bootstrap', [True, False])
    }


    rf_model = RandomForestClassifier(
        **params,
        random_state=42,
        n_jobs=-1
    )

    rf_model.fit(X_train, y_train)


    preds = rf_model.predict(X_test)
    accuracy = accuracy_score(y_test, preds)
    return accuracy


study = optuna.create_study(direction="maximize")
study.optimize(objective_rf, n_trials=100)

trial = study.best_trial


print(f"Accuracy: {trial.value}")
print(f"Best hyperparameters: {trial.params}")

optuna.visualization.plot_optimization_history(study)
optuna.visualization.plot_slice(study)
optuna.visualization.plot_contour(study, params=["n_estimators", "max_depth", "min_samples_split", "min_samples_leaf", "max_features"])
optuna.visualization.plot_param_importances(study)
