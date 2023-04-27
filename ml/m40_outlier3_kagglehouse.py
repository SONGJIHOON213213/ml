import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

#  np.array 다차원 배열(multi-dimensional array)을 포함하여 더 복잡한 데이터를 처리
def split_xy(data):
    imputer = IterativeImputer(RandomForestRegressor())
    data = pd.DataFrame(imputer.fit_transform(data))
    data_x, data_y = np.array(data.drop(data.shape[1]-1, axis=1)), data[data.shape[1]-1]
    return data_x, data_y

# 이상치를 찾아서 b = [] 여기에 저장
def outliers(data_out):
    b = []
    for i in range(data_out.shape[1]):
        quartile_1, q2, quartile_3 = np.percentile(data_out[:, i], [25, 50, 75])
        iqr = quartile_3 - quartile_1
        lower_bound, upper_bound = quartile_1 - (iqr * 1.5), quartile_3 + (iqr * 1.5)
        b.append(np.where((data_out[:, i] > upper_bound) | (data_out[:, i] < lower_bound))[0])
    return b

    
#fit_transform는 IterativeImputer 클래스의 인스턴스를 만들 때 사용된 훈련 데이터를 기반으로 결측치를 대체합니다. 
def GGamji(x, y):
    imputer = IterativeImputer(RandomForestRegressor())
    scaler = MinMaxScaler()
    x = imputer.fit_transform(x)
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, shuffle=True, random_state=123)
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    model = RandomForestRegressor()
    model.fit(x_train, y_train)
    result = model.score(x_test, y_test)
    print(model.__class__.__name__, 'result : ', result)

path = 'c:/study/_data/houseprice/'
houseprice = pd.read_csv(path + 'train.csv', index_col=0)
houseprice_x, houseprice_y = split_xy(houseprice)

outliers_loc = outliers(houseprice_x)
for i in range(houseprice_x.shape[1]):
    houseprice_x[outliers_loc[i], i] = np.nan

GGamji (houseprice_x, houseprice_y) 

# 데이터를 불러옵니다. 이 경우, pd.read_csv() 함수를 사용하여 CSV 파일을 불러옵니다.

# split_xy() 함수를 사용하여 입력 데이터를 x와 y로 분리합니다.

# outliers() 함수를 사용하여 x에서 이상치를 찾아서 해당 위치의 값을 결측치로 처리합니다.

# ggamji() 함수를 사용하여 모델을 학습하고 평가합니다.

# 각 함수 내부에서 필요한 모듈 및 라이브러리를 import합니다.

# split_xy() 함수 내부에서는 IterativeImputer() 함수를 사용하여 결측치를 대체한 뒤, 
# pd.DataFrame() 함수를 사용하여 데이터프레임으로 변환합니다. 그 후, np.array() 함수를 사용하여 x와 y로 분리합니다.

# outliers() 함수 내부에서는 이상치를 찾아 해당 위치의 값을 결측치로 처리합니다. 해당 함수는 
# 이상치를 찾기 위해 np.percentile() 함수를 사용하여 25%, 50%, 75% 분위값을 계산하고, IQR 값을 구한 뒤, 
# lower/upper bound 값을 계산합니다. 그 후, np.where() 함수를 사용하여 이상치 위치를 찾습니다.

# GGamji() 함수 내부에서는 IterativeImputer() 함수를 사용하여 결측치를 대체하고, MinMaxScaler() 
# 함수를 사용하여 데이터를 정규화합니다. 그 후, train_test_split() 
# 함수를 사용하여 데이터를 학습용 데이터와 검증용 데이터로 분리하고, 
# RandomForestRegressor() 함수를 사용하여 모델을 학습합니다. 마지막으로, 모델의 성능을 평가하기 위해 score() 함수를 사용합니다. 


# 이상치란, 데이터의 분포에서 일반적인 값들로부터 크게 벗어난 값들을 의미합니다. 
# 이상치가 있는 데이터를 그대로 모델링에 사용하게 되면 모델의 예측력이 떨어질 수 있습니다. 
# 이상치를 찾아서 해당 위치의 값을 결측치로 처리하는 것은 이상치를 제거하는 방법 중 하나입니다. 
# 결측치로 처리된 위치는 모델링에서 제외되기 때문에 모델의 예측력이 향상될 수 있습니다. 
# 이후, 결측치를 보간하는 등의 방법으로 처리할 수 있습니다.

###percentile###########
# 전체 데이터에서 해당 백분위수보다 낮은 값을 갖는 데이터의 비율을 나타내는 지표입니다

#####IterativeImputer#####
# 네, 맞습니다. IterativeImputer는 다른 머신러닝 알고리즘을 사용하여 
# 결측값을 예측하고 해당 값을 대체하는 방식으로 결측치를 처리하는 방법 중 하나입니다. 
# RandomForestRegressor를 사용하면, 랜덤포레스트 알고리즘을 이용하여 결측값 예측이 이루어집니다. 
# 이때 IterativeImputer는 다른 결측값이 없는 컬럼의 값들을 기반으로 다른 컬럼의 결측값을 예측하고, 
# 예측한 값을 이용하여 다른 결측값을 예측하는 방식으로 결측값을 처리합니다. 

###회귀,분류###
# 회귀는 연속적인 값을 예측하는 문제를 다룹니다. 
# 즉, 입력 데이터와 출력 데이터 간의 관계를 모델링하고, 
# 입력 데이터가 주어졌을 때 출력 데이터 값을 예측하는 것입니다. 
# 예를 들어, 주택의 크기나 방 개수 등의 정보를 바탕으로 주택 가격을 예측하는 것이 회귀 문제입니다.

# 분류는 이산적인 값을 예측하는 문제를 다룹니다. 
# 즉, 입력 데이터가 어느 카테고리에 속하는지를 예측하는 것입니다. 
# 예를 들어, 이메일이 스팸인지 아닌지를 예측하는 것이 분류 문제입니다. 


# IQR은 데이터의 중간 50% 범위를 기반으로 계산되므로, IQR을 계산하기 위해서는 데이터의 
# 25% 분위와 75% 분위 값을 알아야 합니다. 따라서, 
# IQR을 계산하기 위한 기준으로 25% 분위값을 사용하는 것이 일반적입니다. 



# 25% 분위값은 IQR을 계산하기 위한 기준값으로 사용되는 값이지만, 이상치 탐지를 위해서는 다른 분위값도 사용될 수 있습니다. 
# 예를 들어, 
# 1사분위와 3사분위를 이용하여 IQR을 구하고, 이 범위를 벗어나는 데이터를 이상치로 판단할 수도 있습니다