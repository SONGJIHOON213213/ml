#[실습]y클래스를 3개까지 줄이고 그것을 smote해서 
#성능 비교
import numpy as np
import pandas as pd 
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,f1_score
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
# 데이터 불러오기
datasets = load_wine()
x = datasets.data
y = datasets['target']

# y 클래스를 3개에서 2개로 줄이기
x = x[y != 0]
y = y[y != 0] - 1  # 1, 2로 클래스 매핑

# 데이터셋 분할
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.75, shuffle=True, random_state=3377, stratify=y
) 

# SMOTE 적용 전, 0과 1의 개수 확인
print(pd.Series(y_train).value_counts())

# SMOTE 적용 후, 0과 1의 개수 확인
smote = SMOTE(random_state=337, k_neighbors=8)
x_train, y_train = smote.fit_resample(x_train.copy(), y_train.copy())
print(pd.Series(y_train).value_counts())

# 모델 학습 및 예측
model = RandomForestClassifier(random_state=3377)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

# 성능 평가
score = model.score(x_test, y_test)
print('model.score: ', score) 
print('accuracy_score : ',accuracy_score(y_test, y_pred))
print('f1_score(macro) : ', f1_score(y_test, y_pred, average='macro'))
print('f1_score(micro) : ', f1_score(y_test, y_pred, average='micro'))