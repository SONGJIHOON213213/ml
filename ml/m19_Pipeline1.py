#과제1 파이썬 기초문법 매서드or매소드 함수차이 3줄이상 
import numpy as np
from sklearn.datasets import load_iris,load_boston,load_breast_cancer
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import pipeline,Pipeline
from sklearn.svm import SVC
#1. 데이터
x,y = load_iris(return_X_y=True)

x_train,x_test,y_train,y_test = train_test_split(x,y,train_size = 0.8,shuffle=True,random_state=1234
                                                                                             )

# scaler = MinMaxScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)

#2. 모델
model = Pipeline([("std",StandardScaler()),("svc", SVC())])



#3.훈련,컴파일
model.fit(x_train,y_train)

#4.평가,예측
result = model.score(x_test,y_test)
print("model.score:", result) 
y_predict = model.predict(x_test)
acc = accuracy_score(y_test,y_predict) 

#model.score: 1.0