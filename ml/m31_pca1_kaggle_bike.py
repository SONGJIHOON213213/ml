import numpy as np
import pandas as pd
from sklearn.datasets import load_iris,load_diabetes
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import random
path = 'c:/study/_data/kaggle_bike/'   #점 하나 현재폴더의밑에 점하나는 스터디
train_csv = pd.read_csv(path + 'train.csv', 
                        index_col=0) 

print(train_csv)
print(train_csv.shape) #출력결과 (10886, 11)

test_csv = pd.read_csv(path + 'test.csv',
                       index_col=0) 
                      
print(test_csv)        #캐쥬얼 레지스트 삭제
print(test_csv.shape)  #출력결과 ((6493, 8))
##########################################


print(train_csv.columns) 
# #Index(['season', 'holiday', 'workingday', 'weather', 'temp', 'atemp',
#        'humidity', 'windspeed', 'casual', 'registered', 'count'],
#       dtype='object')
# #Index(['season', 'holiday', 'workingday', 'weather', 'temp', 'atemp',
#        'humidity', 'windspeed',]
#       dtype='object')
print(train_csv.info) 

print(type(train_csv)) 

################################
#결측치 처리 1 .제거
# pirnt(train_csv.insul11())
print(train_csv.isnull().sum())
train_csv = train_csv.dropna() ####결측치 제거#####
print(train_csv.isnull().sum()) #(11)
print(train_csv.info())
print(train_csv.shape)
############################## train_csv 데이터에서 x와y를 분리
x = train_csv.drop(['count','casual','registered'], axis=1) #2개 이상 리스트 
print(x)
y = train_csv['count']
print(y)
###############################train_csv 데이터에서 x와y를 분리
x_train, x_test, y_train, y_test = train_test_split(
x, y, shuffle=True, train_size=0.7, random_state=1234
)
print(x_train.shape, x_test.shape)
print(y_train.shape, y_test.shape)

def run_model(x,y,label:str=''):
    x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8,random_state=1234,shuffle=True)
    from sklearn.ensemble import RandomForestRegressor
    model=RandomForestRegressor(n_estimators=200,max_depth=20)
    model.fit(x_train,y_train)
    if label!='':
        print(f'{label} 결과')
    print(f'model score : {model.score(x_test,y_test)}')
    

dataset=load_diabetes()

x=dataset['data']
y=dataset['target']

run_model(x,y,'PCA이전')

pca = PCA(n_components=7)
print(x.shape)
x=pca.fit_transform(x)
print(x.shape)
 
run_model(x,y,'PCA이후') 

# PCA이전 결과
# model score : 0.4454254468968639
# (442, 10)
# (442, 7)
# PCA이후 결과
# model score : 0.42255132572488974