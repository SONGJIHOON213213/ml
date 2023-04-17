from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.tree import DecisionTreeRegressor
import numpy as np
from sklearn.datasets import fetch_covtype, load_diabetes,fetch_california_housing,load_digits
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
import warnings
warnings.filterwarnings(action='ignore')
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.datasets import load_diabetes, fetch_covtype, fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.svm import LinearSVR
from sklearn.ensemble import RandomForestRegressor

data_list = [load_diabetes(return_X_y=True), fetch_covtype(return_X_y=True), fetch_california_housing(return_X_y=True), load_digits(return_X_y=True)]
model_list = [LinearSVR(), RandomForestRegressor()]

for data, data_name in zip(data_list, ['diabetes', 'covtype', 'california_housing','digits']):
    x, y = data
    if data_name == 'digits':
        x = x.reshape(x.shape[0], -1) # 데이터셋의 특성값을 1차원으로 펼치기
    print(f"{data_name}:")
    for model, model_name in zip(model_list, ['LinearSVR', 'RandomForestRegressor']):
        print(f"  {model_name}:")
        best_r2 = -np.inf
        best_scaler_name = ''
        for scaler in [RobustScaler()]:
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            r2_scores = []
            for train_idx, test_idx in kf.split(x):
                x_train, y_train = x[train_idx], y[train_idx]
                x_test, y_test = x[test_idx], y[test_idx]
                
                if data_name == 'digits':
                    x_train = x_train.reshape(x_train.shape[0], -1) # 데이터셋의 특성값을 1차원으로 펼치기
                    x_test = x_test.reshape(x_test.shape[0], -1)
                
                try:
                    x_train_scaled = scaler.fit_transform(x_train, y_train) if scaler.__class__.__name__ == "RobustScaler" else scaler.fit_transform(x_train)
                    x_test_scaled = scaler.transform(x_test)

                    model.fit(x_train_scaled, y_train)
                    y_predict = model.predict(x_test_scaled)
                    r2 = r2_score(y_test, y_predict)
                    r2_scores.append(r2)

                except ValueError as e:
                    print(f"    {scaler.__class__.__name__} scaled {model_name} failed: {e}")
                    break

            if r2_scores:
                curr_best_r2 = round(np.max(r2_scores), 4)
                if curr_best_r2 > best_r2:
                    best_r2 = curr_best_r2
                    best_scaler_name = scaler.__class__.__name__

        print(f"    {best_scaler_name} scaled best R2 score: {best_r2}")