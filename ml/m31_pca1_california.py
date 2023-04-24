import numpy as np
import pandas as pd
from sklearn.datasets import _california_housing
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

from sklearn.datasets import fetch_california_housing
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

#1. 데이터
dataset = fetch_california_housing()
x = dataset['data']
y = dataset['target']

def run_model(x, y, label=''):
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=1234, shuffle=True)
    model = RandomForestRegressor(n_estimators=200, max_depth=20)
    model.fit(x_train, y_train)
    if label != '':
        print(f'{label} 결과')
    print(f'model score : {model.score(x_test, y_test)}')

run_model(x, y, 'PCA이전')

pca = PCA(n_components=7)
print(x.shape)
x = pca.fit_transform(x)
print(x.shape)

run_model(x, y, 'PCA이후')  

# PCA이전 결과
# model score : 0.8049914023204918
# (20640, 8)
# (20640, 7)
# PCA이후 결과
# model score : 0.7682166369529343
# PS C:\study\keras> 