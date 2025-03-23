import optuna
from sklearn.linear_model import LogisticRegression
from data import X_train, X_test, y_train, y_test
from sklearn.model_selection import cross_val_score
import pickle
from sklearn.metrics import accuracy_score
from lgr_hp import trial
from data import X_train, X_test, y_train, y_test


best_lgr_model = LogisticRegression(**trial.params, random_state=42)
best_lgr_model.fit(X_train, y_train)
final_preds = best_lgr_model.predict(X_test)
final_accuracy = accuracy_score(y_test, final_preds)
print(final_accuracy)
with open("lgr_model.pkl", "wb") as file:
    pickle.dump(best_lgr_model, file)