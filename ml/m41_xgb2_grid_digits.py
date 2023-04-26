import numpy as np
from sklearn.datasets import load_breast_cancer, load_digits
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler
from xgboost import XGBClassifier, XGBRegressor

# 1. 데이터
x, y = load_digits(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=337, train_size=0.8, stratify=y)

scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

n_splits = 5
kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=337)


parameters = {'n_estimators' : [100],
              'learning_rate' : [0.1],
              'max_depth' : [2],
              'gamma' : [0],
              'min_child_weight' : [1],
              'subsample' : [0.7],
              'colsample_bytree' : [1],
              'colsample_bylevel' : [1],
              'colsample_bynode' : [1],
              'reg_alpha' : [0],
              'reg_lambda' : [0.01]}

# 2. 모델
xgb = XGBRegressor(random_state=337)
model = GridSearchCV(xgb, parameters, cv=kf, n_jobs=-1)

# 3. 훈련
model.fit(x_train, y_train)

# 4. 평가, 예측
print(f'best param :  {model.best_params_}\n best score : {model.best_score_}\n result : {model.score(x_test,y_test)}') 