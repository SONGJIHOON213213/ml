import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
import warnings
from sklearn.datasets import load_iris, load_boston,load_breast_cancer,load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from xgboost import XGBClassifier, XGBRegressor
#피쳐 컬럼,열,특성
#1. 데이터
# x,y = load_iris(return_X_y=True)

# x_train,x_test,y_train,y_test = train_test_split(x,y,train_size = 0.8,shuffle=True,random_state=1234
#                                                                                              )

# scaler = MinMaxScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)

# #2. 모델
# model = RandomForestClassifier()
# model = XGBClassifier
# #3.훈련,컴파일
# model.fit(x_train,y_train)

# #4.평가,예측
# result = model.score(x_test,y_test)
# print("model.score:", result) 
# y_predict = model.predict(x_test)
# acc = accuracy_score(y_test,y_predict)  


warnings.filterwarnings('ignore')

datasets = [(load_iris(), XGBClassifier()), (load_boston(), XGBRegressor()),(load_breast_cancer(), XGBClassifier()),(load_digits(), XGBClassifier())]

for data, model in datasets:
    X, y = data.data, data.target
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    if isinstance(model, XGBClassifier):
        accuracy = accuracy_score(y_test, y_pred)
        print(f"{data.DESCR.splitlines()[0]} Accuracy: {accuracy}")
    elif isinstance(model, XGBRegressor):
        r2 = r2_score(y_test, y_pred)
        print(f"{data.DESCR.splitlines()[0]} R^2 Score: {r2}") 