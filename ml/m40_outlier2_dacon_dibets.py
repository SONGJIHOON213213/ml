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

def split_xy(data):
    imputer = IterativeImputer(RandomForestRegressor())
    data = pd.DataFrame(imputer.fit_transform(data))
    data_x, data_y = np.array(data.drop(data.shape[1]-1, axis=1)), data[data.shape[1]-1]
    return data_x, data_y


def outliers(data_out):
    b = []
    for i in range(data_out.shape[1]):
        quartile_1, q2, quartile_3 = np.percentile(data_out[:, i], [25, 50, 75])
        iqr = quartile_3 - quartile_1
        lower_bound, upper_bound = quartile_1 - (iqr * 1.5), quartile_3 + (iqr * 1.5)
        b.append(np.where((data_out[:, i] > upper_bound) | (data_out[:, i] < lower_bound))[0])
    return b


def Runmodel(x, y):
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
    Runmodel(diabetes_x, diabetes_y)

path = 'c:/study/_data/dacon_diabets/'
diabetes = pd.read_csv(path + 'train.csv', index_col=0)
diabetes_x, diabetes_y = split_xy(diabetes)

outliers_loc = outliers(diabetes_x)
for i in range(diabetes_x.shape[1]):
    diabetes_x[outliers_loc[i], i] = np.nan

Runmodel(diabetes_x, diabetes_y)