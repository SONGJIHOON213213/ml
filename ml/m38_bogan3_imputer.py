import numpy as np
import pandas as pd 


data = pd.DataFrame([[2,np.nan,6,8,10],
                     [2,4,np.nan,8],
                     [2,4,6,8,10],
                     [np.nan,4,8,np.nan]]
                    ).transpose()
data.columns = ['x1','x2','x3','x4']
print(data)
# from sklearn.experimental import enable_iterative_imputer
from sklearn.impute  import SimpleImputer,KNNImputer # 결측치 책임 돌리기
from sklearn.impute  import IterativeImputer
from xgboost import XGBClassifier, XGBRegressor
 # 결측치 책임 돌리기

# imputer = SimpleImputer()  #디폴트 평균!!
# imputer = SimpleImputer(strategy='mean') #평균!!!
# imputer = SimpleImputer(strategy='median') #중위!!!
# imputer = SimpleImputer(strategy='most_frequent') #최빈값!!! //갯수가 같을경우 가장 작은값
# imputer = SimpleImputer(strategy='constant')
# imputer = SimpleImputer(strategy='constant', fill_value=7777)
# imputer = KNNImputer()
imputer = IterativeImputer(estimator=XGBRegressor())
data2 = imputer.fit_transform(data)
print(data2) 

# 0   2.0  2.0   2.0  NaN
# 1   NaN  4.0   4.0  4.0
# 2   6.0  NaN   6.0  8.0
# 3   8.0  8.0   8.0  NaN
# 4  10.0  NaN  10.0  NaN
# [[ 2.          2.          2.          6.        ]
#  [ 6.5         4.          4.          4.        ]
#  [ 6.          4.66666667  6.          8.        ]
#  [ 8.          8.          8.          6.        ]
#  [10.          4.66666667 10.          6.        ]]