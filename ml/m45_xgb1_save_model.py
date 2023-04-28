import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer,load_iris,load_wine,load_boston,load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score 


# 1. 데이터
x, y = load_breast_cancer(return_X_y=True)
x1, y1 = load_iris(return_X_y=True)
x2, y2 = load_wine(return_X_y=True)
x3, y4 = load_boston(return_X_y=True)
x5, y6 = load_digits(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=337, train_size=0.8, stratify=y)

scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 2. 모델
model = XGBClassifier()

model.fit(x_train, y_train,
          eval_set=[(x_train, y_train), (x_test, y_test)],
          verbose=1,
          eval_metric=['error'],
          early_stopping_rounds=10)

results = model.score(x_test,y_test)

# 4. 평가, 예측
print(f'결과 : {model.score(x_test,y_test)}')   

y_predict = model.predict(x_test)
acc = accuracy_score(y_test,y_predict)
print("accuracy_score : ", acc)


path = 'c:/study/_save/prickle_test/'
path2 = 'c:/study/_save/prickle_test/'
path3 = 'c:/study/_save/prickle_test/'
path4 = 'c:/study/_save/prickle_test/'
path5 = 'c:/study/_save/prickle_test/'

model.save_model(path + 'm45_xgb1_save_model.dat' )
model.save_model(path2 + 'm46_xgb1_save_model.dat' )
model.save_model(path3 + 'm47_xgb1_save_model.dat' )
model.save_model(path4 + 'm48_xgb1_save_model.dat' )
model.save_model(path5 + 'm49_xgb1_save_model.dat' )

