from sklearn.datasets import fetch_california_housing,load_iris,load_boston,load_diabetes,load_wine
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import PowerTransformer
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
from sklearn.metrics import r2_score,accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler, QuantileTransformer, PowerTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing._data import PowerTransformer
import warnings
warnings.filterwarnings('ignore')
# 1. Load data
# Load the iris dataset
# Define a list of datasets to process
datasets = [
    {'name': 'iris', 'data': load_iris(return_X_y=True)},
    {'name': 'boston', 'data': load_boston(return_X_y=True)},
    {'name': 'diabetes', 'data': load_diabetes(return_X_y=True)},
    {'name': 'california_housing', 'data': fetch_california_housing(return_X_y=True)},
    {'name': 'wine', 'data': load_wine(return_X_y=True)}
]

# Loop over the datasets
for dataset in datasets:
    # Load the data
    x, y = dataset['data']
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(x, y, shuffle=True, train_size=0.8, random_state=337)
    
    # Create a list of scalers to iterate over
    scalers = [StandardScaler(), MinMaxScaler(), MaxAbsScaler(), RobustScaler(), QuantileTransformer(n_quantiles=10), PowerTransformer(method='yeo-johnson')]
    
    # Iterate over the scalers and fit/transform the training data
    for scaler in scalers:
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train a model on the scaled data and evaluate its performance
        model = RandomForestRegressor()
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        r2 = r2_score(y_test, y_pred)
        print(f"{dataset['name']}, {scaler.__class__.__name__} scaler R2 score: {r2:.3f}")


#회귀데이터 맹그러
#for 써서
#scaler 6개 써서 올민 