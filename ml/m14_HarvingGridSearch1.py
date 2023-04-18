import numpy as np
from tensorflow.keras.layers import Dense
from sklearn.datasets import load_iris,load_digits
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold,cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV,HalvingGridSearchCV
import time 
import pandas as pd
#1.데이터
# x , y = load_iris(return_X_y=True)
x , y = load_digits(return_X_y=True)

x_train,x_test,y_train,y_test = train_test_split(
    x, y, shuffle=True, random_state=337, test_size=0.2, stratify=y)

n_split = 3
kfold = KFold(n_splits=n_split,shuffle=True,random_state=337)

parameters = [
    {'C':[1,10,100,1000], 'kernel':['linear'], 'degree':[3,4,5]},
    {'C':[1,10,100], 'kernel':['rbf', 'linear'], 'gamma':[0.001, 0.0001]},
    {'C':[1,10,100,1000], 'kernel':['sigmoid'], 'gamma':[0.01, 0.001, 0.0001], 'degree':[3, 4]},
    {'C':[0.1, 1], 'gamma':[1, 10]}
]
#2.모델
# model = GridSearchCV(SCV())
model = HalvingGridSearchCV(SVC(),parameters,
                     cv=5,
                    #  n_iter=10,
                     verbose=1,
                     refit=True,
                     n_jobs=-1
                    ,factor=10)
#3.컴파일,훈련
start_time =time.time()
model.fit(x_train,y_train)
end_time = time.time()

print("최적의 매개변수:",model.best_estimator_)

print("최적의 파라미터:",model.best_params_)

print('best_score_ :', model.best_score_)

print('model.score_ :', model.score(x_test,y_test))

y_predcit = model.predict(x_test)
# print('accuracy_score:',accuracy_score(y_test,y_predcit)) 

y_pred_best =model.best_estimator_.predict(x_test)
print('ACC 최적튠:',accuracy_score(y_test,y_pred_best))

print("걸린시간:",round(end_time - start_time,2),'초')

#######################################################
#컬럼이 하나 또는 한개의리스트 벡터형태로고한다.
# print(pd.DataFrame(model.cv_results_).sort_values('rank_test_score', ascending=True))#컬럼이 하나 또는 한개의리스트 벡터형태로고한다.
# print(pd.DataFrame(model.cv_results_).columns)#컬럼이 하나 또는 한개의리스트 벡터형태로고한다.


path = 'c:/temp/'
pd.DataFrame(model.cv_results_).sort_values('rank_test_score', ascending=True)\
    .to_csv(path + 'm14_HarvinGridSearch1.csv' ) 


# min_resources_: 143 최소 훈련데이터
# max_resources_: 1437    최대 훈련데이터
# factor: 3
# ----------
# iter: 0 
# n_candidates: 52         즉, 52개의 후보 모델 각각에 대해 5개의 fold를 생성하여 총 260번의 fitting을 수행하고 있습니다.  #전체 파라미터 갯수 /factor
# n_resources: 100                                                                                                   0번쨰 훈련데이터갯수 * factor                                                                   
# Fitting 5 folds for each of 52 candidates, totalling 260 fits
# ----------
# iter: 1
# n_candidates: 18             18개의 후보모델  5개 fold 생성 300번의피팅                                             #전체 파라미터 갯수 /factor
# n_resources: 300                                                                                                  # min_resources * factor
# Fitting 5 folds for each of 18 candidates, totalling 90 fits
# ----------
# iter: 2                  
# n_candidates: 6              6개 후보모델  5개 fold 900번의피팅    
# n_resources: 900                                                
# Fitting 5 folds for each of 6 candidates, totalling 30 fits 

#예를 들어, iter: 0에서 n_candidates가 52이고 n_resources가 100이라면, factor는 0.52가 됩니다. 이것은 각 시도마다 평균적으로 0.52개의 하이퍼파라미터 조합이 탐색되는 것을 의미합니다.
#펙터(factor)는 이 지표를 통해 평가한 모델의 성능 차이를 나타내는 값입니다. 

#ACC 최적튠: 0.9944444444444445