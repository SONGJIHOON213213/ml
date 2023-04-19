#과제1 파이썬 기초문법 매서드or매소드 함수차이 3줄이상 
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

#1. 데이터
x,y = load_iris(return_X_y=True)

x_train,x_test,y_train,y_test = train_test_split(x,y,train_size = 0.8,shuffle=True,random_state=1234)

# scaler = MinMaxScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)

#2. 모델
model = Pipeline([("std",StandardScaler()),("lr", LinearRegression())])

#3.훈련,컴파일
model.fit(x_train,y_train)

#4.평가,예측
result = model.score(x_test,y_test)
print("model.score:", result) 

y_predict = model.predict(x_test)
mse = mean_squared_error(y_test,y_predict)
print("MSE:", mse)