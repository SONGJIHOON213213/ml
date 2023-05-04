import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

ddarung_path = 'c:/study/_data/ddarung/'

ddarung = pd.read_csv(ddarung_path + 'train.csv', index_col=0).dropna()

# 데이터를 살펴보고 시각화
print(ddarung.head())

# # 박스 플롯
ddarung.plot.box()
plt.show()

# 정보 및 통계 요약
ddarung.info()
print(ddarung.describe())

# 히스토그램
ddarung.hist(bins=50)
plt.show()

# 특정 열에 대한 박스 플롯
# 이 부분에서는 예시로 첫 번째 열을 사용합니다.
# 필요한 열 이름으로 바꿔주세요.
first_col_name = ddarung.columns[0]
ddarung[first_col_name].plot.box()
plt.show()

# 특정 열에 대한 히스토그램
ddarung[first_col_name].hist(bins=50)
plt.show()

# 데이터 분할
# 가정: 'target'이라는 열이 목표 변수임
y = ddarung[['hour_bef_visibility', 'hour_bef_precipitation']]
x = ddarung.drop(['count'], axis=1)

x['hour'] = np.log1p(x['hour'])

y = np.log1p(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, shuffle=True, random_state=1234)

y_train_log = np.log1p(y_train)
y_test_log = np.log1p(y_test)

# 모델 생성
model = RandomForestRegressor(random_state=1234)

# 모델 훈련
model.fit(x_train, y_train_log)

# 모델 평가
score = model.score(x_test, y_test_log)
print('score:', score)

# R2 스코어
print('R2 score:', r2_score(y_test, model.predict(x_test)))