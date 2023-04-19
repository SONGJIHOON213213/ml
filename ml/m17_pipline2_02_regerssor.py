from sklearn.utils import all_estimators
from sklearn.datasets import load_iris, load_breast_cancer, load_digits, load_wine, fetch_california_housing, load_diabetes
import pandas as pd
import numpy as np
import warnings
from sklearn.linear_model import QuantileRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler, MaxAbsScaler
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict
warnings.filterwarnings(action='ignore')
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.datasets import load_iris, load_breast_cancer, load_digits, load_wine, fetch_california_housing, load_diabetes
data_list = [load_diabetes]

data_list = [load_iris, load_breast_cancer, load_digits, load_wine, fetch_california_housing, load_diabetes]

scaler_list = [StandardScaler()]

# RandomForestRegressor, GaussianProcessRegressor, MLPRegressor, LinearRegression, QuantileRegressor
algorithms_regressor = [('RandomForestRegressor', RandomForestRegressor)]

from sklearn.pipeline import make_pipeline

algorithms_regressor = [('RandomForestRegressor', RandomForestRegressor)]

for i in range(len(data_list)):
    x, y = data_list[i](return_X_y=True)
    for j in scaler_list:
        scaler = j
        x_scaled = scaler.fit_transform(x)
        for name, algorithm in algorithms_regressor:
            try:
                model = algorithm()
                if name == 'RandomForestRegressor':
                    model.set_params(n_estimators=100, max_depth=15, min_samples_split=2, min_samples_leaf=1)
                elif name == 'GaussianProcessRegressor':
                    model.set_params(max_iter_predict=100)
                elif name == 'MLPRegressor':
                    model.set_params(hidden_layer_sizes=(100,), activation='relu', solver='adam', max_iter=1000)
                elif name == 'LinearRegression':
                    pass # LinearRegression은 하이퍼파라미터를 설정할 수 없습니다. 
                pipeline = make_pipeline(scaler, model)
                results = cross_val_score(pipeline, x, y, cv=5, scoring='r2')
                max_score = np.mean(results)
                max_name = name
                print('\n', type(scaler).__name__, ' - ', data_list[i].__name__, 'max_score :', max_name, max_score)
            except:
                continue