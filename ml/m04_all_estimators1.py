import numpy as np 
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import r2_score 
import warnings 
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.utils import all_estimators
import sklearn as sk
print(sk.__version__)
warnings.filterwarnings("ignore")

#1.데이터
datasets = fetch_california_housing(return_X_y=True)

X_train,x_test,y_train,y_test = train_test_split(
    x,y, shuffle=True,random_state=123, test_size=0.2    
)

scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test  = scaler.transform(x_test)


#2.모델
model = RandomForestRegressor(n_jobs=4)#코어4개다쓰겟다라는 의미
allAllgorithms = all_estimators(type_filter='regressor')
# allAllgorithms = all_estimators(type_filter='regressor')

#3.훈련
model.fit(x_train,y_train)


         
#4. 평가,예측 
r2 = model.score(x_test,y_test)
print("r2",r2)
y_predict = model.predict(x_test)
print(type(y_test))
print(y_predict)
r2_score = r2_score(y_test,y_predict)
print("r2_score:",r2_score) 

for (name, algorithm) in allAllgorithms:
    try: 
        model = algorithm()
        except:
     print(name,'은(는) 에러뜬 놈!!!')
print("============================")        
print('최고모델 :', max_name, max_r2)