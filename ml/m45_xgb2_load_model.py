import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer,load_iris,load_wine,load_boston,load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score 


# 1. 데이터
x, y = load_breast_cancer(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=337, train_size=0.8, stratify=y)

scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

path = 'c:/study/_save/prickle_test/'

model = XGBClassifier()
model.load_model(path + 'm45_xgb1_save_model.dat')

# 4. 평가, 예측
print(f'결과 : {model.score(x_test,y_test)}')   

y_predict = model.predict(x_test)
acc = accuracy_score(y_test,y_predict)
print("accuracy_score : ", acc)


