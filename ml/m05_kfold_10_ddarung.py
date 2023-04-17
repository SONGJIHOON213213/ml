from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras.layers import Dense, Input, Dropout
from sklearn.model_selection import train_test_split,cross_val_score,KFold
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler, MaxAbsScaler
import pandas as pd
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.svm import LinearSVC
import warnings
warnings.filterwarnings(action='ignore')
# 1. 데이터
# 1.1 경로, 가져오기
path = 'c:/study/_data/ddarung/'
path_save = 'c:/study/_data/save/ddarung/'

n_splits = 5
kfold = KFold()
# kfold = KFold(n_splits=n_splits, shuffle=True,random_state=123)

#2.모델구성
model = LinearSVC()

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)

# 1.2 확인사항 5가지
print(train_csv.shape, test_csv.shape) #(1459, 10) (715, 9)
print(train_csv.columns, test_csv.columns)
print(train_csv.info(), test_csv.info())
print(train_csv.describe(), test_csv.describe())
print(type(train_csv), type(test_csv))

# # 1.3 결측지 제거
print(train_csv.isnull().sum())
train_csv = train_csv.dropna()
print(train_csv.isnull().sum())

# 1.4 x, y 분리
x = train_csv.drop(['count'], axis=1)
y = train_csv['count']

# 1.5 train, test 분리
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=995, shuffle=True)

scaler = MinMaxScaler() # 0.0 711.0 #정규화란, 모든 값을 0~1 사이의 값으로 바꾸는 것이다
x = scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test) 
test_csv = scaler.transform(test_csv) 
print(np.min(x), np.max(x))


#3, 4. 컴파일,훈련,평가.예측
scores =cross_val_score(model,x,y, cv=kfold)


print('ACC: ', scores,'\n croos_val_score 평균 : ', round(np.mean(scores),4))
# 4.1 내보내기 순서 다르면 데이터 값이 안들어감
y_submit = model.predict(test_csv)
submission = pd.read_csv(path + 'submission.csv', index_col=0)
submission['count'] = y_submit
submission.to_csv(path + 'submission42.csv')