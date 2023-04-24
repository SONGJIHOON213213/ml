import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import random
# 0. seed initialization
seed=4
random.seed(seed)
np.random.seed(seed)

def run_model(x,y,label:str=''):
    x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8,random_state=seed,shuffle=True)
    from sklearn.ensemble import RandomForestRegressor
    model=RandomForestRegressor(n_estimators=200,max_depth=20)
    model.fit(x_train,y_train)
    if label!='':
        print(f'{label} 결과')
    print(f'model score : {model.score(x_test,y_test)}')
    

dataset=load_breast_cancer()

x=dataset['data']
y=dataset['target']

run_model(x,y,'PCA이전')

pca = PCA(n_components=7)
print(x.shape)
x=pca.fit_transform(x)
print(x.shape)

run_model(x,y,'PCA이후') 

# PCA이전 결과
# model score : 0.7308122610294117
# (569, 30)
# (569, 7)
# PCA이후 결과
# model score : 0.7057154963235294