from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import optuna
from data import get_data
import pickle

X_train, X_test, y_train, y_test = get_data()

with open("xgb_trial.pkl","rb") as file:
    trial = pickle.load(file)


best_xgb_model = XGBClassifier(**trial.params, tree_method='gpu_hist', use_label_encoder=False)
best_xgb_model.fit(X_train, y_train)
final_preds = best_xgb_model.predict(X_test)
final_accuracy = accuracy_score(y_test, final_preds)
print(f"Accuracy: {final_accuracy:.3f}")


with open("xgb_model.pkl", "wb") as file:
    pickle.dump(best_xgb_model, file)