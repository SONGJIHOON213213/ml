import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score 
import pickle

# 1. 데이터
x, y = load_breast_cancer(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=337, train_size=0.8, stratify=y)

scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2.모델 피클 불러오기
import pickle
path = 'c:/study/_save/prickle_test/'
model = pickle.load(open(path + 'm43_pcikle1_save.dat', 'rb'))


# 4. 평가, 예측
print(f'결과 : {model.score(x_test,y_test)}')   

y_predict = model.predict(x_test)
acc = accuracy_score(y_test,y_predict)
print("accuracy_score : ", acc)




# parameters = {'n_estimators': 1000,
#               'learning_rate': 0.3,
#               'max_depth': 2,
#               'gamma': 0,
#               'min_child_weight': 1,
#               'subsample': 0.7,
#               'colsample_bytree': 1,
#               'colsample_bylevel': 1,
#               'colsample_bynode': 1,
#               'reg_alpha': 0,
#               'reg_lambda': 0.01,
#               'random_state': 337}

