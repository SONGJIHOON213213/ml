from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import numpy as np
from sklearn.datasets import load_iris, load_breast_cancer, load_diabetes, load_digits, fetch_covtype, load_wine
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
import warnings
from sklearn.metrics import accuracy_score

x_data = [[0,0],[0,1],[1,0],[1,1]]
y_data = [0,0,0,1]

#2.모델
model = LinearSVC()

#3.훈련
model.fit(x_data,y_data)

#4평가,예측
y_predict = model.predict(x_data)

results = model.score(x_data,y_data)
print("model.score : ",results)

acc = accuracy_score(y_data,y_predict)
print('accuracy_score :',acc)