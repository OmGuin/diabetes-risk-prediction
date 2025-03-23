import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import pickle
from data import X_test, y_test



def evaluate_model(y_true, y_pred, model_name):
    print(f"{model_name} Evaluation:")
    print(classification_report(y_true, y_pred))
    print("Confusion Matrix:")
    sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt='d', cmap='Blues')
    plt.title(f"{model_name} Confusion Matrix")
    plt.show()
    plt.savefig(model_name+'.png', dpi=300, bbox_inches='tight')


with open("lgr_model.pkl", 'rb') as file:
    lgr_model = pickle.load(file)
with open("xgb_model.pkl", 'rb') as file:
    xgb_model = pickle.load(file)
with open("rf_model.pkl", 'rb') as file:
    rf_model = pickle.load(file)
y_pred_lgr = lgr_model.predict(X_test)
y_pred_xgb = xgb_model.predict(X_test)
y_pred_rf = rf_model.predict(X_test)


evaluate_model(y_test, y_pred_lgr, "LogisticRegression")
evaluate_model(y_test, y_pred_rf, "Random Forest")
evaluate_model(y_test, y_pred_xgb, "XGBoost")