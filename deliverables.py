import pandas as pd
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import json
import pickle
import optuna

from featureimportance import get_feature_importance
from data import get_data
from eval import evaluate_model, evaluate

X_train, X_test, y_train, y_test = get_data()

#TRAINED RANDOM FOREST MODEL
with open("rf_model.pkl", 'rb') as file:
    rf_model = pickle.load(file)

#BASELINE TRAINED LOGISTIC REGRESSION MODEL
with open("lgr_model.pkl", 'rb') as file:
    lgr_model = pickle.load(file)

#TRAINED XGBOOST MODEL
with open("xgb_model.pkl", 'rb') as file:
    xgb_model = pickle.load(file)



get_feature_importance("RF", X_train)
get_feature_importance("XGB", X_train)


evaluate(X_test, y_test, "RF")
evaluate(X_test, y_test, "XGB")




