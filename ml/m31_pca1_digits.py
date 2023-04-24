import numpy as np
import pandas as pd
from sklearn.datasets import load_iris,load_digits
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import random
# 0. seed initialization
def run_model(x,y,label:str=''):
    x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8,random_state=1234,shuffle=True)
    from sklearn.ensemble import RandomForestRegressor
    model=RandomForestRegressor(n_estimators=200,max_depth=20)
    model.fit(x_train,y_train)
    if label!='':
        print(f'{label} 결과')
    print(f'model score : {model.score(x_test,y_test)}')
    

dataset=load_digits()

x=dataset['data']
y=dataset['target']

run_model(x,y,'PCA이전')

pca = PCA(n_components=7)
print(x.shape)
x=pca.fit_transform(x)
print(x.shape)

run_model(x,y,'PCA이후') 

# PCA이전 결과
# model score : 0.8736778995354393
# (1797, 64)
# (1797, 7)
# PCA이후 결과
# model score : 0.8380268462977148