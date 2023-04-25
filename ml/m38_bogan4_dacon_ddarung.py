import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split

# 1.1 경로, 가져오기
path = 'c:/study/_data/ddarung/'
df = pd.read_csv(path + 'train.csv', index_col=0)
train_csv = pd.read_csv(path + 'train.csv', index_col=0)

# 1.3 결측치 처리
num_cols = df.select_dtypes(include=np.number).columns
df[num_cols] = df[num_cols].fillna(df[num_cols].median())

# 1.4 라벨인코딩( 으로 object 결측지 제거 )
le = LabelEncoder()
for i in train_csv.columns:
    if train_csv[i].dtype == 'object':
        train_csv[i] = le.fit_transform(train_csv[i])


print(len(train_csv.columns))
print(train_csv.info())
train_csv = train_csv.dropna()
print(train_csv.shape) 


#  #   Column                  Non-Null Count  Dtype
# ---  ------                  --------------  -----
#  0   hour                    1459 non-null   int64
#  1   hour_bef_temperature    1457 non-null   float64
#  2   hour_bef_precipitation  1457 non-null   float64
#  3   hour_bef_windspeed      1450 non-null   float64
#  4   hour_bef_humidity       1457 non-null   float64
#  5   hour_bef_visibility     1457 non-null   float64
#  6   hour_bef_ozone          1383 non-null   float64
#  7   hour_bef_pm10           1369 non-null   float64
#  8   hour_bef_pm2.5          1342 non-null   float64
#  9   count                   1459 non-null   float64
# dtypes: float64(9), int64(1)
# memory usage: 125.4 KB
# None
# (1328, 10)