import optuna
from sklearn.ensemble import RandomForestClassifier
from data import X_train, X_test, y_train, y_test
from sklearn.model_selection import cross_val_score
import pickle
from sklearn.metrics import accuracy_score
from rf_hp import trial



best_rf_model = RandomForestClassifier(**trial.params)
best_rf_model.fit(X_train, y_train)
final_preds = best_rf_model.predict(X_test)
final_accuracy = accuracy_score(y_test, final_preds)
print(f"Accuracy: {final_accuracy:.3f}")
with open("rf_model.pkl", "wb") as file:
    pickle.dump(best_rf_model, file)