import numpy as np
import pandas as pd
from sklearn.datasets import fetch_covtype, load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
dataset = load_breast_cancer()
x = dataset['data']
y = dataset['target']

# define a function to run the model and print the results
def model(x, y, label=''):
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=1234, shuffle=True)
    model = RandomForestRegressor(n_estimators=200, max_depth=20)
    model.fit(x_train, y_train)
    if label:
        print(label + ' 결과')
    print('model score: ' + str(model.score(x_test, y_test)))

# run the model with the original dataset
model(x, y, 'PCA 전')

# apply PCA and run the model again
pca = PCA(n_components=7)
x = pca.fit_transform(x)
print(x.shape)

model(x, y, 'PCA 후')

# PCA이전 결과
# model score : 0.7308122610294117
# (569, 30)
# (569, 7)
# PCA이후 결과
# model score : 0.7057154963235294