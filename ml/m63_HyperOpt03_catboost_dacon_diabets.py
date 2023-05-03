import pandas as pd
from sklearn.datasets import load_breast_cancer, load_diabetes, load_iris
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler 
from xgboost import XGBClassifier, XGBRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from bayes_opt import BayesianOptimization
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score
import numpy as np
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import BaggingClassifier
import time
import warnings
warnings.filterwarnings('ignore')

#1 데이터
# 1.1 경로, 가져오기
path = 'c:/study/_data/dacon_diabets/'

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv  = pd.read_csv(path + 'test.csv', index_col=0)

# 1.2 확인사항 5가지
print(train_csv.shape, test_csv.shape) #(652, 9) (116, 8)
print(train_csv.columns, test_csv.columns)
print(train_csv.info(), test_csv.info())
print(train_csv.describe(), test_csv.describe())
print(type(train_csv), type(test_csv))

# 1.3 결측지 제거
train_csv = train_csv.dropna()

# 1.4 x, y 분리
x = train_csv.drop(['Outcome'], axis=1)
y = train_csv['Outcome']

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=1234, shuffle=True)
# 1.3 결측치 처리
num_cols = x_train.select_dtypes(include=np.number).columns
x_train[num_cols] = x_train[num_cols].fillna(x_train[num_cols].median())
x_test[num_cols] = x_test[num_cols].fillna(x_train[num_cols].median())

# 1.4 라벨인코딩
le = LabelEncoder()
for i in x_train.columns:
    if x_train[i].dtype == 'object':
        x_train[i] = le.fit_transform(x_train[i])
        x_test[i] = le.transform(x_test[i])

print(len(x_train.columns))
print(x_train.info())
print(x_test.info())

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.7, shuffle = True, random_state = 1234)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


#hyperopt----------------------------------------------------------#
from hyperopt import hp
from hyperopt import fmin, tpe, Trials, STATUS_OK

search_space = {
    'learning_rate' : hp.uniform('learning_rate', 0.01, 1),        
    'depth' : hp.quniform('depth',3, 16, 1),              
    'one_hot_max_size' : hp.quniform('one_hot_max_size',24, 64, 1),          
    'min_data_in_leaf' : hp.quniform('min_data_in_leaf', 10, 200, 2), 
    'bagging_temperature' : hp.uniform('bagging_temperature', 0.5, 1),
    'random_strength' : hp.uniform('random_strength', 0.5, 1),         
    'l2_leaf_reg' : hp.uniform('l2_leaf_reg', 0.01, 30)
}

 
def cat_hamsu(search_space):
    params = {
        'iterations' : 10,
        'learning_rate' : search_space['learning_rate'],
        'depth' : int(search_space['depth']),
        'l2_leaf_reg' : search_space['l2_leaf_reg'],
        'bagging_temperature' : search_space['bagging_temperature'],
        'random_strength' : search_space['random_strength'],
        'one_hot_max_size' : int(search_space['one_hot_max_size']),
        'min_data_in_leaf' : int(search_space['min_data_in_leaf']),
        'task_type' : 'CPU',
        'logging_level' : 'Silent',
        
    }

    model = CatBoostClassifier(**params)
    model.fit(x_train, y_train)
            #   eval_set=[(x_train, y_train), (x_test, y_test)],
            #   eval_metric='AUC',
            #   verbose=0,
            #   early_stopping_rounds=50
            #   )
    y_predict = model.predict(x_test)
    return_value = -(accuracy_score(y_test, y_predict))

    return return_value

trial_val = Trials()   


best = fmin(
    fn= cat_hamsu,                            
    space= search_space,                        
    algo=tpe.suggest,                           
    max_evals=50,                               
    trials=trial_val,                           
    rstate = np.random.default_rng(seed=10)   
)


print("best:", best)


results = [aaa['loss'] for aaa in trial_val.results]   
df = pd.DataFrame({
        'learning_rate' : trial_val.vals['learning_rate'],
        'depth' : trial_val.vals['depth'],
        'l2_leaf_reg' : trial_val.vals['l2_leaf_reg'],
        'bagging_temperature' : trial_val.vals['bagging_temperature'],
        'random_strength' : trial_val.vals['random_strength'],
        'one_hot_max_size' : trial_val.vals['one_hot_max_size'],
        'min_data_in_leaf' : trial_val.vals['min_data_in_leaf'],
        'results': results
                   })

print(df)


min_row = df.loc[df['results'] == df['results'].min()]
print("최소 행",'\n' , min_row)

min_results = df.loc[df['results'] == df['results'].min(), 'results']
print(min_results.values)  