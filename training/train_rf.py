from sklearn.ensemble import RandomForestClassifier
import pickle
from sklearn.metrics import accuracy_score
from data import get_data
import pickle

X_train, X_test, y_train, y_test = get_data()

with open("rf_trial.pkl","rb") as file:
    trial = pickle.load(file)

best_rf_model = RandomForestClassifier(**trial.params)
best_rf_model.fit(X_train, y_train)
final_preds = best_rf_model.predict(X_test)
final_accuracy = accuracy_score(y_test, final_preds)
print(f"Accuracy: {final_accuracy:.3f}")


with open("rf_model.pkl", "wb") as file:
    pickle.dump(best_rf_model, file)