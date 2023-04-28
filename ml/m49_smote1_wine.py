import numpy as np
import pandas as pd 
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,f1_score
from imblearn.over_sampling import SMOTE
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

print(x.shape,y.shape) 
print(np.unique(y,return_counts=True))
print(pd.Series(y).value_counts().sort_index())

x_train,x_test,y_train,y_test=train_test_split(
    x,y,train_size=0.75,shuffle=True,random_state=3377,stratify=y
) 

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(random_state=3377)

#3.훈련
model.fit(x_train,y_train)

#4.평가,예측
y_predict = model.predict(x_test)
score = model.score(x_test,y_test)
print('Before SMOTE')
print('model.score: ', score) 
print('accuracy_score : ',accuracy_score(y_test,y_predict))
print('f1_score(macro) : ', f1_score(y_test,y_predict,average='macro'))
print('f1_score(micro) : ', f1_score(y_test,y_predict,average='micro'))

print("=====================================SMOTE 적용후========")
smote = SMOTE(random_state=337,k_neighbors=8)
x_train,y_train = smote.fit_resample(x_train.copy(),y_train.copy())

#3.훈련
model.fit(x_train,y_train)

#4.평가,예측
y_predict = model.predict(x_test)
score = model.score(x_test,y_test)
print('After SMOTE')
print('model.score: ', score) 
print('accuracy_score : ',accuracy_score(y_test,y_predict))
print('f1_score(macro) : ', f1_score(y_test,y_predict,average='macro'))
print('f1_score(micro) : ', f1_score(y_test,y_predict,average='micro'))
print(pd.Series(y_train).value_counts().sort_index())