from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import r2_score, mean_squared_error
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
# 데이터 불러오기

#1. 데이터
path = 'c:/study/_data/wine/'  
train_csv = pd.read_csv(path + 'train.csv', 
                        index_col=0) 


print(train_csv)
print(train_csv.shape) #출력결과 (10886, 11)

test_csv = pd.read_csv(path + 'test.csv',
                       index_col=0) 
              
print(test_csv)       
print(test_csv.shape)  
# Label Encoding & One-hot Encoding
enc = LabelEncoder()
enc.fit(train_csv['type'])
train_csv['type'] = enc.transform(train_csv['type'])
test_csv['type'] = enc.transform(test_csv['type'])

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

from sklearn.metrics import accuracy_score
def accuracy(y_true, y_pred):
    y_pred = np.argmax(y_pred, axis=1)
    return accuracy_score(y_true, y_pred)
    
# 전체 특성으로 모델 학습 및 결과 출력
parameters = {'n_estimators' : 800,
              'learning_rate' : 0.1,
              'max_depth' : 17, 
               'gamma' : 0,
              'min_child_weight' : 0,
              'subsample' : 0.2,
              'colsample_bytree' : 0.5,
              'colsample_bylevel' : 0,
              'colsample_bynode' : 0,
              'reg_alpha' : 1,
              'reg_lambda' : 1,
              'random_state' : 749,
              }
model = XGBClassifier(**parameters)
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=337, train_size=0.8)

model.fit(x_train, y_train, eval_set=[(x_train, y_train), (x_test, y_test)],
          early_stopping_rounds=10, verbose=0, eval_metric='logloss')

print('All features result :', model.score(x_test, y_test))
y_predict = model.predict(x_test)
acc = accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_predict, axis=1))
print('All features accuracy :', acc)

y_submit = model.predict(test_csv)
y_submit = np.argmax(y_submit, axis =1)
submission = pd.read_csv(path + 'sample_submission.csv', index_col=0)

y_submit += 3
submission['quality'] = y_submit

submission.to_csv('c:/study/_data/wine/434submission.csv')