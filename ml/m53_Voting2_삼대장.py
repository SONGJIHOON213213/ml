import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing,fetch_covtype,load_boston,load_wine,load_diabetes
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.ensemble import VotingRegressor #투표

#3대장.
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor #연산할 필요 없는 것들을 빼버림, 잘나오는 곳 한쪽으로만 감.
from catboost import CatBoostRegressor


#1. 데이터
datasets = [load_boston,fetch_california_housing, fetch_covtype,load_diabetes]

for data in datasets:
    dataset_name = data.__name__
    x, y = data(return_X_y=True)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, shuffle=True, train_size=0.8, random_state=1030
    )

scaler = StandardScaler()
x_train =  scaler.fit_transform(x_train)
x_test =  scaler.transform(x_test)

#2. 모델
xgb = XGBRegressor()
lg = LGBMRegressor()
cat = CatBoostRegressor(verbose=0) #verbose 디폴트 1 

model = VotingRegressor(
    estimators=[('XGB', xgb), ('LG', lg), ('CAT', cat)],
                #voting='soft', #디폴트는 하드, 성능은 소프트가 더 좋음.
)

#3. 훈련
model.fit(x_train,y_train)

#4. 평가, 예측
y_pred = model.predict(x_test)
print('model.score : ', model.score(x_test,y_test))
print("Voting.acc : ", r2_score(y_test,y_pred))

#Hard Voting
# model.score :  0.9473684210526315
# acc :  0.9473684210526315 

#Soft Voting
# model.score :  0.9649122807017544
# acc :  0.9649122807017544

Regressors = [xgb, lg, cat]
li = []
for model2 in Regressors:
    model2.fit(x_train, y_train)
    
    # 모델 예측
    y_predict = model2.predict(x_test)
    
    # 모델 성능 평가
    score2 = r2_score(y_test,y_predict)
    
    class_name = model2.__class__.__name__ 
    print("{0} R2 : {1:4f}".format(class_name, score2))
    li.append(score2)

# 리스트 출력
print(li)