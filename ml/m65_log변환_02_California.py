from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs
import numpy as np
from sklearn.preprocessing import QuantileTransformer, PowerTransformer, StandardScaler
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from sklearn.metrics import r2_score
#1.데이터셋
datasets = fetch_california_housing()
df = pd.DataFrame(datasets.data,columns=datasets.feature_names)
df['target'] = datasets.target
print(df)

df.boxplot()
plt.show()

df.info()
print(df.describe())

df['Population'].plot.box()
plt.show()

df['Population'].hist(bins=50)
df['target'].hist(bins=50)
plt.show()

y = df['target']
x = df.drop(['target'],axis=1)
###################################X Population 로그변환##################
x['Population'] = np.log(x['Population'])


x_train,x_test,y_train,y_test = train_test_split(
    x,y,shuffle=True,train_size=0.8,random_state=337,    
)

y_train_log = np.log1p(y_train)
y_test_log = np.log1p(y_test)
#2.모델
model = RandomForestRegressor(random_state=337)

#3.컴파일,훈련
model.fit(x_train,y_train_log)

#4.평가,예측
score = model.score(x_test,y_test_log)

print("로그 -> 지수 :",r2_score(y_test, np.expm1(model.predict(x_test)))) #로그변환한다음에 지수로변환해야됨

print('score :',score)


#로그 변환 전 :
#x[pop]만 로그변환 :
#y만 로그변환 :