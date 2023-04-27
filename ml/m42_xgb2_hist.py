import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from xgboost import XGBClassifier

# 1. 데이터
x, y = load_breast_cancer(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=337, train_size=0.8, stratify=y)

scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

parameters = {'n_estimators': 1000,
              'learning_rate': 0.3,
              'max_depth': 2,
              'gamma': 0,
              'min_child_weight': 1,
              'subsample': 0.7,
              'colsample_bytree': 1,
              'colsample_bylevel': 1,
              'colsample_bynode': 1,
              'reg_alpha': 0,
              'reg_lambda': 0.01,
              'random_state': 337}

# 2. 모델
model = XGBClassifier(**parameters)

model.fit(x_train, y_train,
          eval_set=[(x_train, y_train), (x_test, y_test)],
          verbose=1,
          eval_metric=['error'],
          early_stopping_rounds=10)

# 4. 평가, 예측
print(f'결과 : {model.score(x_test,y_test)}')  

print("========================================")
hist = model.evals_result()

# 5. 그래프
epochs = len(hist['validation_0']['error'])
x_axis = range(0, epochs)

fig, ax = plt.subplots()
ax.plot(x_axis, hist['validation_0']['error'], label='Train')
ax.plot(x_axis, hist['validation_1']['error'], label='Test')
ax.legend()
plt.ylabel('Error')
plt.title('XGBoost Error')
plt.show()