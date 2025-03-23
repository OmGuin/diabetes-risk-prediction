import pandas as pd
import pickle
from data import X


with open("xgb_model.pkl", 'rb') as file:
    xgb_model = pickle.load(file)
with open("rf_model.pkl", 'rb') as file:
    rf_model = pickle.load(file)

rf_importances = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=False)
print("Random Forest Feature Importances: ")
print(rf_importances)

xgb_importances = pd.Series(xgb_model.feature_importances_, index=X.columns).sort_values(ascending=False)
print("XGBoost Feature Importances:")
print(xgb_importances)