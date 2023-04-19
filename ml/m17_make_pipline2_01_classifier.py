import numpy as np
from sklearn.datasets import load_iris, load_breast_cancer, load_digits, load_wine, fetch_covtype, fetch_california_housing, load_diabetes
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, StratifiedKFold, RandomizedSearchCV, HalvingGridSearchCV, HalvingRandomSearchCV
from sklearn.metrics import accuracy_score, r2_score
import time
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
import warnings
warnings.filterwarnings(action='ignore')
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
data_list = [load_iris, load_breast_cancer, load_digits, load_wine, load_diabetes]

algorithms_classifier = [('SVC', SVC), ('RandomForestClassifier', RandomForestClassifier)]

scaler_list = [StandardScaler()]

for i in range(len(data_list)):
    if i < 4:
        x, y = data_list[i](return_X_y=True)
        for j in scaler_list:
            scaler = j
            x_scaled = scaler.fit_transform(x)
            for name, algorithm in algorithms_classifier:
                try:
                    if name == 'SVC':
                        model = make_pipeline(scaler, algorithm(C=1, kernel='rbf'))
                    else:
                        model = make_pipeline(scaler, algorithm(n_estimators=100, max_depth=15, min_samples_split=2, min_samples_leaf=1))
                    if name == 'GaussianProcessClassifier':
                        model.set_params(max_iter_predict=100)
                    elif name == 'MLPClassifier':
                        model.set_params(MLPClassifier(hidden_layer_sizes=(100,), activation='relu', solver='adam', max_iter=1000))
                    elif name == 'QuadraticDiscriminantAnalysis':
                        model.set_params(reg_param=0.5)
                    results = cross_val_score(model, x_scaled, y, cv=5)
                    max_score = np.mean(results)
                    max_name = name
                    print('\n', type(scaler).__name__, ' - ', data_list[i].__name__, 'max_score :', max_name, max_score)
                except:
                    continue