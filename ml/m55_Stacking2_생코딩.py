import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_digits,load_iris,load_wine,load_boston
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import StackingClassifier, StackingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,BaggingClassifier
from sklearn.ensemble import VotingClassifier #투표
from xgboost import XGBClassifier
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor,LGBMClassifier #연산할 필요 없는 것들을 빼버림, 잘나오는 곳 한쪽으로만 감.
from catboost import CatBoostRegressor,CatBoostClassifier
from sklearn.metrics import r2_score

datasets = [load_iris, load_breast_cancer, load_digits, load_wine,load_boston]

for data in datasets:
    dataset_name = data.__name__
    x, y = data(return_X_y=True)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, shuffle=True, train_size=0.8, random_state=1030
    )

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    # Stacking 모델 구성
    estimators = [
        ('xgb', XGBRegressor()),
        ('lgb', LGBMRegressor()),
        ('cat', CatBoostRegressor(verbose=0))
    ]
    stacking_model = StackingRegressor(
        estimators=estimators,
        final_estimator=CatBoostRegressor(verbose=0)
    )
    stacking_model.fit(x_train, y_train)
    y_pred = stacking_model.predict(x_test)

    print(f'{dataset_name} 데이터')
    print('Stacking R2 score :', r2_score(y_test, y_pred))
    print()

    # final_estimator=DecisionTreeClassifier()
    # final_estimator=LogisticRegression()
    # final_estimator=RandomForestClassfier()
    # final_estimator=VotingClassifier()
     # 또는 'hard'
