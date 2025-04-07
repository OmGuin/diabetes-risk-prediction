import optuna
from sklearn.linear_model import LogisticRegression
from data import get_data
from sklearn.model_selection import cross_val_score
import numpy as np
import pickle

X_train, X_test, y_train, y_test = get_data()

def objective_lgr(trial):
    
    params = {
        'C': trial.suggest_float('C', 1e-4, 10.0, log=True),  # Regularization strength
        'penalty': trial.suggest_categorical('penalty', ['l1', 'l2']),
        'solver': 'liblinear'
    }
    
    lgr_model = LogisticRegression(**params, random_state=42, n_jobs=-1)
    lgr_model.fit(X_train, y_train)

    scores = cross_val_score(lgr_model, X_test, y_test, cv=5, scoring='accuracy')

    return np.mean(scores)



study = optuna.create_study(direction="maximize")
study.optimize(objective_lgr, n_trials=100)

trial = study.best_trial

with open("lgr_trial.pkl", "wb") as file:
    pickle.dump(trial, file)

print(f"Accuracy: {trial.value}")
print(f"Best hyperparameters: {trial.params}")

optuna.visualization.plot_optimization_history(study)
optuna.visualization.plot_slice(study)
optuna.visualization.plot_contour(study, params=["C", "penalty"])
optuna.visualization.plot_param_importances(study)