# import numpy as np
# import pandas as pd 
# import pandas as pd
# import numpy as np
# from xgboost import XGBRegressor
# from sklearn.impute import IterativeImputer
# from sklearn.impute import SimpleImputer, KNNImputer
# from sklearn.model_selection import train_test_split
# from sklearn.experimental import enable_iterative_imputer

# data = pd.DataFrame([[2,np.nan,6,8,10],
#                      [2,4,np.nan,8],
#                      [2,4,6,8,10],
#                      [np.nan,4,8,np.nan]]
#                     ).transpose()
# data.columns = ['x1','x2','x3','x4']
# print(data)
# # from sklearn.experimental import enable_iterative_imputer
# from sklearn.impute  import SimpleImputer,KNNImputer # 결측치 책임 돌리기
# from sklearn.impute  import IterativeImputer
# from xgboost import XGBClassifier, XGBRegressor
#  # 결측치 책임 돌리기

# # imputer = SimpleImputer()  #디폴트 평균!!
# # imputer = SimpleImputer(strategy='mean') #평균!!!
# # imputer = SimpleImputer(strategy='median') #중위!!!
# # imputer = SimpleImputer(strategy='most_frequent') #최빈값!!! //갯수가 같을경우 가장 작은값
# # imputer = SimpleImputer(strategy='constant')
# # imputer = SimpleImputer(strategy='constant', fill_value=7777)
# # imputer = KNNImputer()
# imputer = IterativeImputer(estimator=XGBRegressor())
# data2 = imputer.fit_transform(data)
# print(data2) 

# # 0   2.0  2.0   2.0  NaN
# # 1   NaN  4.0   4.0  4.0
# # 2   6.0  NaN   6.0  8.0
# # 3   8.0  8.0   8.0  NaN
# # 4  10.0  NaN  10.0  NaN
# # [[ 2.          2.          2.          6.        ]
# #  [ 6.5         4.          4.          4.        ]
# #  [ 6.          4.66666667  6.          8.        ]
# #  [ 8.          8.          8.          6.        ]
# #  [10.          4.66666667 10.          6.        ]] 

# 위 코드는 결측치를 대체하기 위한 imputation 기법을 사용하는 코드입니다. 결측치는 머신러닝 모델을 학습시키기 
# 위해 필요한 데이터의 일부분이 누락되어 있기 때문에, 대체하기 위한 방법이 필요합니다.
# 위 코드에서는 SimpleImputer, KNNImputer, IterativeImputer 세 가지 방법을 사용합니다. 각 방법의 차이는 다음과 같습니다.
# SimpleImputer: 평균, 중위수, 최빈수 등 단순한 통계치를 사용하여 대체합니다.
# KNNImputer: 가장 가까운 이웃 데이터들의 값을 사용하여 대체합니다.
# IterativeImputer: 머신러닝 모델을 사용하여 결측치를 예측하고, 예측된 값으로 대체합니다.
# 위 코드에서는 IterativeImputer를 사용하며, XGBRegressor를 예측 모델로 사용합니다. 
# 이 모델은 XGBoost라는 머신러닝 알고리즘을 기반으로 하며, 회귀 문제에 적용할 수 있습니다. 
# 이 모델을 사용하여 결측치를 예측하고 대체하게 됩니다.