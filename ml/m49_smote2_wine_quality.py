import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.model_selection import GridSearchCV, train_test_split
from xgboost import XGBClassifier, XGBRegressor
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler, LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn. ensemble import RandomForestClassifier
import datetime
import warnings
warnings.filterwarnings('ignore')
date = datetime.datetime.now()
date = date.strftime('%m%d_%H%M%S')

le = LabelEncoder()
imputer = IterativeImputer(XGBRegressor())
scaler = MinMaxScaler()
path = 'c:/study/_data/wine/'
path_save = 'c:/study/_save/wine/'

wine = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
sample = pd.read_csv(path + 'sample_submission.csv', index_col=0)
wine['type'] = le.fit_transform(wine['type'])
test_csv['type'] = le.fit_transform(test_csv['type'])
test_csv = imputer.fit_transform(test_csv)
x = imputer.fit_transform(wine.drop(['quality'], axis=1))
y = wine['quality']

# y의 클래스를 7개에서 5~3개로 줄여서 성능 비교
for i in y:
    # if i == 3:
    #     i = 5
    # elif i == 4:
    #     i = 5
    if i == 8:
        i = 7
    elif i == 9:
        i = 7

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, shuffle=True, random_state=123)
x_train, x_test = scaler.fit_transform(x_train), scaler.transform(x_test)

model = RandomForestClassifier()

model.fit(x_train, y_train)

results = model.score(x_test, y_test)
print('result : ', results)