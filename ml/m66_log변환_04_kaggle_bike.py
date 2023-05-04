import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

kaggle_bike_path = 'c:/study/_data/kaggle_bike/'

kaggle_bike = pd.read_csv(kaggle_bike_path + 'train.csv', index_col=0).dropna()

# 데이터를 살펴보고 시각화
print(kaggle_bike.head())

# # 박스 플롯
kaggle_bike.plot.box()
plt.show()

# 정보 및 통계 요약
kaggle_bike.info()
print(kaggle_bike.describe())

# 히스토그램
kaggle_bike.hist(bins=50)
plt.show()

# 특정 열에 대한 박스 플롯
# 이 부분에서는 예시로 첫 번째 열을 사용합니다.
# 필요한 열 이름으로 바꿔주세요.
first_col_name = kaggle_bike.columns[0]
kaggle_bike[first_col_name].plot.box()
plt.show()

# 특정 열에 대한 히스토그램
kaggle_bike[first_col_name].hist(bins=50)
plt.show()

# 데이터 분할
# 가정: 'target'이라는 열이 목표 변수임
y = kaggle_bike[['atemp', 'atemp']]
x = kaggle_bike.drop(['count'], axis=1)

x['atemp'] = np.log1p(x['atemp'])

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