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
path = 'c:/study/_data/houseprice/'
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

