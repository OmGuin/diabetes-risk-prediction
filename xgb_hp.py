from xgboost import XGBClassifier
from data import get_data
from sklearn.model_selection import cross_val_score
import optuna
import numpy as np
import pickle


X_train, X_test, y_train, y_test = get_data()

def objective_xgb(trial):
    
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'tree_method': 'gpu_hist',
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=50),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'lambda': trial.suggest_float('lambda', 1e-8, 10.0, log=True),  # L2
        'alpha': trial.suggest_float('alpha', 1e-8, 10.0, log=True)    # L1 
    }
    
    xgb_model = XGBClassifier(**params, use_label_encoder=False)
    xgb_model.fit(X_train, y_train)

    scores = cross_val_score(xgb_model, X_test, y_test, cv=5, scoring='accuracy')

    return np.mean(scores)




study = optuna.create_study(direction="maximize")
study.optimize(objective_xgb, n_trials=100)

trial = study.best_trial
with open("xgb_trial.pkl", "wb") as file:
    pickle.dump(trial, file)

print(f"Accuracy: {trial.value}")
print(f"Best hyperparameters: {trial.params}")

optuna.visualization.plot_optimization_history(study)
optuna.visualization.plot_slice(study)
optuna.visualization.plot_contour(study, params=["n_estimators", "max_depth", "learning_rate", "subsample", "colsample_bytree", "lambda", "alpha"])
optuna.visualization.plot_param_importances(study)
