import numpy as np
from tensorflow.keras.layers import Dense
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold,cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC,LinearSVR
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV,HalvingGridSearchCV
import time 
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
#1.데이터
x , y = load_boston(return_X_y=True)

x_train,x_test,y_train,y_test = train_test_split(
    x, y, shuffle=True, random_state=337, test_size=0.2)

#2.모델
model = HalvingGridSearchCV(
    LinearSVR(), 
    {
        'C':[0.001, 0.01, 0.1, 1, 10, 100], 
        'epsilon':[0.001, 0.01, 0.1, 1, 10, 100]
    },
    cv=5,
    verbose=1,
    refit=True,
    n_jobs=-1,
    factor=10,
    min_resources=50,
    max_resources=100
)

#3.컴파일,훈련
start_time =time.time()
model.fit(x_train,y_train)
end_time = time.time()

print("최적의 매개변수:",model.best_estimator_)

print("최적의 파라미터:",model.best_params_)

print('best_score_ :', model.best_score_)

print('model.score_ :', model.score(x_test,y_test))

y_predcit = model.predict(x_test)

print('MSE 최적튠:', mean_squared_error(y_test, y_predcit))

print("걸린시간:",round(end_time - start_time,2),'초')  



#회귀문제
#LinearSVR