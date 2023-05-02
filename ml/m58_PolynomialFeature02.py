import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_digits,load_iris,load_wine,load_boston
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,BaggingClassifier
from sklearn.ensemble import VotingClassifier #투표
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
datasets = [load_iris, load_breast_cancer, load_digits, load_wine, load_boston]

for data in datasets:
    dataset_name = data.__name__
    x, y = data(return_X_y=True)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, shuffle=True, train_size=0.8, random_state=1030
    )

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    
    # 다항 회귀 모델 구성
    poly = PolynomialFeatures(degree=2, include_bias=False)
    x_train_poly = poly.fit_transform(x_train)
    x_test_poly = poly.transform(x_test)

    model = RandomForestClassifier()
    model.fit(x_train_poly, y_train)
    y_pred = model.predict(x_test_poly)

    print(f'{dataset_name} 데이터')
    print('PolynomialFeatures R2 score:', r2_score(y_test, y_pred))
    print()
