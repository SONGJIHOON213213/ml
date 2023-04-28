from xgboost import XGBRegressor,XGBClassifier
from sklearn.feature_selection import SelectFromModel
import warnings
warnings.filterwarnings('ignore')
# 와인 quality 맞추는 파일 만들기
from tensorflow.python.keras.models import Sequential, Model, load_model
import numpy as np
from tensorflow.python.keras.layers import Dense, Input, Dropout
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import RobustScaler, MaxAbsScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
# 데이터 불러오기
path = 'c:/study/_data/wine/'  
train_csv = pd.read_csv(path + 'train.csv', index_col=0) 
test_csv = pd.read_csv(path + 'test.csv', index_col=0)

# Label Encoding
enc = LabelEncoder()
enc.fit(train_csv['type'])
train_csv['type'] = enc.transform(train_csv['type'])
test_csv['type'] = enc.transform(test_csv['type'])


# 이상치 처리
def remove_outliers(data):
    q1 = data.quantile(0.25)
    q3 = data.quantile(0.75)
    iqr = q3 - q1
    condition = (data < (q1 - 1.5 * iqr)) | (data > (q3 + 1.5 * iqr))
    data[condition] = np.nan
    data = data.fillna(method='ffill').fillna(method='bfill')
    return data.astype(np.float32)

train_csv = remove_outliers(train_csv)

# type열 삭제
train_csv = train_csv.drop(['type'], axis=1)
test_csv = test_csv.drop(['type'], axis=1)

# Label Encoding & One-hot Encoding
ohe = OneHotEncoder()
y = train_csv['quality'].values
y = y.reshape(-1, 1)
y = ohe.fit_transform(y).toarray()

# 데이터 전처리
x = train_csv.drop(['quality'], axis=1)
scaler = MinMaxScaler()
scaler.fit(x)
x = scaler.transform(x)

# test_csv를 DataFrame으로 변환
test_csv_df = pd.DataFrame(test_csv, columns=train_csv.columns[1:])

# test_csv 전처리
test_csv_df = scaler.transform(test_csv_df)

# 모델 학습 및 결과 출력
parameters = {'n_estimators': 1000, 'max_depth': 15, 'random_state': 749}
model = RandomForestClassifier(**parameters)

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=337, train_size=0.8)
model.fit(x_train, np.argmax(y_train, axis=1))

print('All features result:', model.score(x_test, np.argmax(y_test, axis=1)))
y_predict = model.predict(x_test)
acc = accuracy_score(np.argmax(y_test, axis=1), y_predict)
print('All features accuracy:', acc)

y_submit = model.predict(test_csv_df)
submission = pd.read_csv(path + 'sample_submission.csv', index_col=0)
y_submit += 3
submission['quality'] = y_submit

submission.to_csv('c:/study/_data/wine/434submission.csv') 

import seaborn as sns
import matplotlib.pyplot as plt

# quality를 제외한 12개의 feature 컬럼들에 대한 히스토그램 그리기
fig, axs = plt.subplots(ncols=6, nrows=2, figsize=(20, 10))
index = 0
axs = axs.flatten()

for i, col in enumerate(train_csv.columns):
    if col == 'quality' or col == 'type':
        continue
    sns.histplot(train_csv[col], ax=axs[i], kde=True)
    index += 1
plt.tight_layout()
plt.show()

# 12개의 feature 컬럼들에 대한 히트맵 그리기
plt.figure(figsize=(12,10))
sns.heatmap(train_csv.drop(['quality', 'type'], axis=1).corr(), annot=True, cmap='coolwarm')
plt.show()
