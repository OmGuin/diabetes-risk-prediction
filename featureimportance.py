import pandas as pd
import pickle
from data import get_data

X = get_data(True)


with open("rf_model.pkl", 'rb') as file:
    rf_model = pickle.load(file)
rf_importances = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=False)
print(f"Random Forest Feature Importances: {rf_importances}")



with open("xgb_model.pkl", 'rb') as file:
    xgb_model = pickle.load(file)

xgb_importances = pd.Series(xgb_model.feature_importances_, index=X.columns).sort_values(ascending=False)
print(f"XGBoost Feature Importances: {xgb_importances}")
