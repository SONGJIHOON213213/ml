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

path_ddarung = 'c:/study/_data/ddarung/'
path_kaggle =  'c:/study/_data/kaggle_bike/'

ddarung_train = pd.read_csv(path_ddarung + 'train.csv', index_col=0).dropna()
kaggle_train = pd.read_csv(path_kaggle + 'train.csv', index_col=0).dropna()

data_list = [ddarung_train, kaggle_train]

algorithms_regressor = all_estimators(type_filter='regressor')

max_score=0
max_name=''

scaler_list = [RobustScaler(), StandardScaler(), MinMaxScaler(), MaxAbsScaler()]
n_split = 10
kf = KFold(n_splits=n_split, shuffle=True, random_state=123)

for i, data in enumerate(data_list):
    x = data.drop(['count'], axis=1)
    y = data['count']
    for scaler in scaler_list:
        x_scaled = scaler.fit_transform(x)
        for name, algorithm in algorithms_regressor:
            try:
                model = algorithm()
                if name == "GaussianProcessRegressor":
                    model = GaussianProcessRegressor(alpha=1000)
                elif name == "QuantileRegressor":
                    model = QuantileRegressor(alpha=1000)
                results = cross_val_score(model, x_scaled, y)
                if max_score < np.mean(results):
                    max_score = np.mean(results)
                    max_name = name
            except:
                continue
        print('\n', type(scaler).__name__, ' - ', data_list[i].name, 'max_score :', max_name, max_score)