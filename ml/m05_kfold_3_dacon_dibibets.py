import numpy as np
from tensorflow.python.keras.models import Sequential, Model 
from tensorflow.python.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
from tensorflow.python.keras.callbacks import EarlyStopping
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split,cross_val_score,KFold
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings(action='ignore')
#1.데이터
x,y, = load_breast_cancer(return_X_y=True)
# x_train,x_test,y_train,y_test = train_test_split(
#     x,y,shuffle=True, random_state=123,test_size=0,2,
    
# )

n_splits = 5
kfold = KFold()
# kfold = KFold(n_splits=n_splits, shuffle=True,random_state=123)

#2.모델구성
model = LinearSVC()


#3, 4. 컴파일,훈련,평가.예측
scores =cross_val_score(model,x,y, cv=kfold)
print('ACC: ', scores,'\n croos_val_score 평균 : ', round(np.mean(scores),4)) 

path = 'c:/study/_data/dacon_diabets/'

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv  = pd.read_csv(path + 'test.csv', index_col=0)

# 1.2 확인사항 5가지
print(train_csv.shape, test_csv.shape) #(652, 9) (116, 8)
print(train_csv.columns, test_csv.columns)
print(train_csv.info(), test_csv.info())
print(train_csv.describe(), test_csv.describe())
print(type(train_csv), type(test_csv))

# 1.3 결측지 제거
train_csv = train_csv.dropna()

# 1.4 x, y 분리
x = train_csv.drop(['Outcome'], axis=1)
y = train_csv['Outcome']


# # 1.5 train, test 분리 과적합방지
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=1234, shuffle=True)

# scaler = MinMaxScaler() # 0.0 711.0 #정규화란, 모든 값을 0~1 사이의 값으로 바꾸는 것이다
scaler = StandardScaler() #정답은 (49-50) / 1 = -1이다. 여기서 표준편차란 평균으로부터 얼마나 떨어져있는지를 구한 것이다. 
# scaler = MaxAbsScaler #최대절대값과 0이 각각 1, 0이 되도록 스케일링
# scaler = RobustScaler #중앙값(median)과 IQR(interquartile range) 사용. 아웃라이어의 영향을 최소화
x = scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test) 
test_csv = scaler.transform(test_csv) 
print(np.min(x), np.max(x))



# # 2. 모델구성
input1 = Input(shape=(8,)) #input-> desen1 ->dense 2->desne3 -> output1-> 모델순서
dense1 = Dense(30)(input1)
dense2 = Dense(20)(dense1)
dense3 = Dense(10)(dense2)
output1 = Dense(1)(dense3)
model = Model(inputs = input1, outputs = output1)

#3, 4. 컴파일,훈련,평가.예측
scores =cross_val_score(model,x,y, cv=kfold)
print('ACC: ', scores,'\n croos_val_score 평균 : ', round(np.mean(scores),4)) 



# # 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)
y_predict = np.round(model.predict(x_test))
# print("==============================")
# print(y_test[:5])
# print(y_predict[:5])
# print(np.round(y_predict[:5]))
# print("=============================")
from sklearn.metrics import r2_score, accuracy_score
acc = accuracy_score(y_test, y_predict)
print('acc:', acc)

y_submit = np.round(model.predict(test_csv))
submission = pd.read_csv(path + 'sample_submission.csv', index_col = 0)
submission['Outcome'] = y_submit
submission.to_csv(path + 'sample_submission36.csv')
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('r2 : ', r2) 

#3, 4. 컴파일,훈련,평가.예측
scores =cross_val_score(model,x,y, cv=kfold)
print('ACC: ', scores,'\n croos_val_score 평균 : ', round(np.mean(scores),4)) 