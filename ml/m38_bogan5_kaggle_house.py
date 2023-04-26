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

# <class 'pandas.core.frame.DataFrame'>
# Int64Index: 1460 entries, 1 to 1460
# Data columns (total 80 columns):
 #   Column         Non-Null Count  Dtype
# ---  ------         --------------  -----
#  0   MSSubClass     1460 non-null   int64
#  1   MSZoning       1460 non-null   int32
#  2   LotFrontage    1201 non-null   float64
#  3   LotArea        1460 non-null   int64
#  4   Street         1460 non-null   int32
#  5   Alley          1460 non-null   int32
#  6   LotShape       1460 non-null   int32
#  7   LandContour    1460 non-null   int32
#  8   Utilities      1460 non-null   int32
#  9   LotConfig      1460 non-null   int32
#  10  LandSlope      1460 non-null   int32
#  11  Neighborhood   1460 non-null   int32
#  12  Condition1     1460 non-null   int32
#  13  Condition2     1460 non-null   int32
#  14  BldgType       1460 non-null   int32
#  15  HouseStyle     1460 non-null   int32
#  16  OverallQual    1460 non-null   int64
#  17  OverallCond    1460 non-null   int64
#  18  YearBuilt      1460 non-null   int64
#  19  YearRemodAdd   1460 non-null   int64
#  20  RoofStyle      1460 non-null   int32
#  21  RoofMatl       1460 non-null   int32
#  22  Exterior1st    1460 non-null   int32
#  23  Exterior2nd    1460 non-null   int32
#  24  MasVnrType     1460 non-null   int32
#  25  MasVnrArea     1452 non-null   float64
#  26  ExterQual      1460 non-null   int32
#  27  ExterCond      1460 non-null   int32
#  28  Foundation     1460 non-null   int32
#  29  BsmtQual       1460 non-null   int32
#  30  BsmtCond       1460 non-null   int32
#  31  BsmtExposure   1460 non-null   int32
#  32  BsmtFinType1   1460 non-null   int32
#  33  BsmtFinSF1     1460 non-null   int64
#  34  BsmtFinType2   1460 non-null   int32
#  35  BsmtFinSF2     1460 non-null   int64
#  36  BsmtUnfSF      1460 non-null   int64
#  37  TotalBsmtSF    1460 non-null   int64
#  38  Heating        1460 non-null   int32
#  39  HeatingQC      1460 non-null   int32
#  40  CentralAir     1460 non-null   int32
#  41  Electrical     1460 non-null   int32
#  42  1stFlrSF       1460 non-null   int64
#  43  2ndFlrSF       1460 non-null   int64
#  44  LowQualFinSF   1460 non-null   int64
#  45  GrLivArea      1460 non-null   int64
#  46  BsmtFullBath   1460 non-null   int64
#  47  BsmtHalfBath   1460 non-null   int64
#  48  FullBath       1460 non-null   int64
#  49  HalfBath       1460 non-null   int64
#  50  BedroomAbvGr   1460 non-null   int64
#  51  KitchenAbvGr   1460 non-null   int64
#  52  KitchenQual    1460 non-null   int32
#  53  TotRmsAbvGrd   1460 non-null   int64
#  54  Functional     1460 non-null   int32
#  55  Fireplaces     1460 non-null   int64
#  56  FireplaceQu    1460 non-null   int32
#  57  GarageType     1460 non-null   int32
#  58  GarageYrBlt    1379 non-null   float64
#  59  GarageFinish   1460 non-null   int32
#  60  GarageCars     1460 non-null   int64
#  61  GarageArea     1460 non-null   int64
#  62  GarageQual     1460 non-null   int32
#  63  GarageCond     1460 non-null   int32
#  64  PavedDrive     1460 non-null   int32
#  65  WoodDeckSF     1460 non-null   int64
#  66  OpenPorchSF    1460 non-null   int64
#  67  EnclosedPorch  1460 non-null   int64
#  68  3SsnPorch      1460 non-null   int64
#  69  ScreenPorch    1460 non-null   int64
#  70  PoolArea       1460 non-null   int64
#  71  PoolQC         1460 non-null   int32
#  72  Fence          1460 non-null   int32
#  73  MiscFeature    1460 non-null   int32
#  74  MiscVal        1460 non-null   int64
#  75  MoSold         1460 non-null   int64
#  76  YrSold         1460 non-null   int64
#  77  SaleType       1460 non-null   int32
#  78  SaleCondition  1460 non-null   int32
#  79  SalePrice      1460 non-null   int64
# dtypes: float64(3), int32(43), int64(34)
# memory usage: 678.7 KB
# None
# (1121, 80)