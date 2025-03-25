import pandas as pd
import pickle

def get_feature_importance(model, X):
    if(model == "RF"):
        with open("rf_model.pkl", 'rb') as file:
            rf_model = pickle.load(file)
        rf_importances = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=False)
        print(f"Random Forest Feature Importances: {rf_importances}")
    elif(model == "XGB"):
        with open("xgb_model.pkl", 'rb') as file:
            xgb_model = pickle.load(file)

        xgb_importances = pd.Series(xgb_model.feature_importances_, index=X.columns).sort_values(ascending=False)
        print(f"XGBoost Feature Importances: {xgb_importances}")
