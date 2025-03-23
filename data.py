import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv("")
imputer = SimpleImputer(strategy='median')
data_imputed = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

X = data_imputed.drop("risk", axis=1)
y = data_imputed['risk']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)




if __name__ == "__main__":
    print(data_imputed.describe())
    numerical_features = data.select_dtypes(include=['float64', 'int64']).columns
    
    data[numerical_features].hist(bins=15, figsize=(15, 10))
    plt.suptitle('Distribution of Numerical Features')
    plt.show()

    correlation_matrix = data[numerical_features].corr()

    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Correlation Heatmap')
    plt.show()

    sns.countplot(x='diabetes_risk', data=data, palette='Set2')  # Replace 'diabetes_risk' with your actual target column name
    plt.title('Distribution of Diabetes Risk')
    plt.show()

    for feature in numerical_features:
        plt.figure(figsize=(8, 5))
        sns.boxplot(x='diabetes_risk', y=feature, data=data, palette='Set2')  # Replace 'diabetes_risk' with your target column name
        plt.title(f'Boxplot of {feature} vs Diabetes Risk')
        plt.show()

    sns.pairplot(data, hue='diabetes_risk', vars=numerical_features, palette='Set2')  # Replace 'diabetes_risk' with your target column name
    plt.suptitle('Pairplot of Numerical Features')
    plt.show()

    sns.heatmap(data[numerical_features + ['diabetes_risk']].corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Feature Correlation with Diabetes Risk')
    plt.show()