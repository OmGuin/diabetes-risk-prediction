# Diabetes Risk Prediction
Simple ensemble approach for robust diabetes risk association and inference  
I use 3 different models (logistic regression, random forest, gradient boosting) to find the best possible prediction for diabetes risk  
Using hyperparameter tuning, I intend to ensure the best possible deliverables


## Fixed Hyperparameters

- Data Imputation: Median
- Standard Scaler Standardization

This project heavily relies on  [Optuna](https://github.com/optuna/optuna) for hyperparameter tuning


## Dataset
https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset


## Hyperparameters for each model

### Logistic Regression
- C: 1e-4 to 10.0
- Regularization: L1 or L2

### Random Forest
- Number of trees: 50 to 300
- Max depth: 1 to 32
- Min samples to split a node: 2 to 10
- Min samples to split a node: 1 to 10
- Max features: Sqrt, log, none
- Bootstrapping

### XGradient Boosting
- Learning rate: 0.01 to 0.3
- Max depth: 3 to 10
- Number of trees: 100 to 1000
- Subsample: 0.5 to 1.0
- Colsample bytree: 0.5 to 1.0
- L1 & L2: 1e-8 to 10.0 


## Methodology
1. data.py
2. Model HP tuning
3. Model training
4. deliverables.py
