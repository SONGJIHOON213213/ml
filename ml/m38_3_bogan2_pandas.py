import numpy as np
import pandas as pd 

data = pd.DataFrame([[2,np.nan,6,8,10],
                     [2,4,np.nan,8],
                     [2,4,6,8,10],
                     [np.nan,4,8,np.nan]]
                    ).transpose()
print(data) 
data.columns = ['x1','x2','x3','x4'] 


print(data.isnull()) 
print(data.isnull().sum()) 
print(data.info()) 

#1.결측치삭제
print("#####################결측치삭제######################")
# print(data['x1'].dropna())
print(data.dropna())
print("#####################결측치삭제######################")
print(data.dropna(axis=0))
print("#####################결측치삭제######################")
print(data.dropna(axis=1))

#2-1.특정값 - 평균
print("#####################결측치삭제######################")
means = data.mean()
print('평균: ',means) 

data2 = data.fillna(means)
print(data2) 
#2-2.특정값 - 평균
print("#####################결측치삭제######################")
median = data.median() 
print('중위값: ',median)
data3 = data.fillna(median)


#2-2.특정값 - ffill,bfill
print("#####################결측치처리,ffill,bfill######################")
data4 = data.fillna(method='ffill') 

data5 = data.fillna(method='bfill') 



#2-4 특정값 - 임의값으로 채우기 
print("#####################결측치처리######################")
data6 = data.fillna(value=777777)


data['x1'] = data['x1'].mean()

#1. x1컬럼에 평균값을 넣고 

#2. x2 컬럼에 중위값을 넣고

#3. x4 컬럼에 ffill한후 / 제일 위에 남은행에 77777로 채우기 



mean = data['x1'].mean()
data['x1'] = data['x1'].fillna(mean)

median = data['x2'].median()
data['x2'].fillna(median)

data['x4']=data['x4'].fillna(method='ffill').fillna(value=777777)
print(mean)
print(median) 
print(data)

#1. x1컬럼에 평균값을 넣고 

#2. x2 컬럼에 중위값을 넣고

#3. x4 컬럼에 ffill한후 / 제일 위에 남은행에 77777로 채우기 

#     x1   x2    x3        x4
# 0  6.5  2.0   2.0  777777.0
# 1  6.5  4.0   4.0       4.0
# 2  6.5  NaN   6.0       8.0
# 3  6.5  8.0   8.0       8.0
# 4  6.5  NaN  10.0       8.0