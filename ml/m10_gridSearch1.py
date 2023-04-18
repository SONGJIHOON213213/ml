import numpy as np
from tensorflow.keras.layers import Dense
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold,cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

#1.데이터
x , y = load_iris(return_X_y=True)

x_train,x_test,y_train,y_test = train_test_split(
    x, y, shuffle=True, random_state=337, test_size=0.2, stratify=y)

gamma = [0.001, 0.01, 0.1, 1, 10, 100]
C = [0.001, 0.01, 0.1, 1, 10, 100] 
max_score = 0

best_parameters = {'gamma': None, 'C': None}

for i in gamma:
    for j in C:
        # 모델 정의
        model = SVC(gamma=i, C=j)

        # 모델 훈련
        model.fit(x_train, y_train)

        # 모델 평가
        score = model.score(x_test, y_test)

        # 최고 점수 갱신
        if max_score < score:
            max_score = score
            best_parameters = {'gamma': i, 'C': j}

print("acc:", max_score)
print("best parameters:", best_parameters) 
