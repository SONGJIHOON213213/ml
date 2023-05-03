from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
from lightgbm import LGBMRegressor
import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')
import time

#1 데이터
x, y = load_diabetes(return_X_y = True)

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size = 0.7,
                                                    shuffle = True,
                                                    random_state = 1234)

scaler = StandardScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

lgbm_search_space = { # hp.quniform 정수형 값 간격, hp.uniform 실수형
                     'learning_rate' : hp.uniform('n_estimators', 0.3, 0.7),
                     'max_depth' : hp.quniform('max_depth', 3, 16, 1),
                     'num_leaves' : hp.quniform('num_leaves', 24, 64, 1),
                     'min_child_samples' : hp.quniform('minchild_sanple', 10, 200, 1),
                     'min_child_weight' : hp.quniform('minchild_weight', 1, 50, 1),
                     'subsample' : hp.uniform('subsample', 0.5, 1), # 0 ~ 1 사이
                     'colsample_bytree' : hp.uniform('colsample_bytree', 0.5, 1),
                     'max_bin' : hp.uniform('max_bin', 10, 500),
                     'reg_lambda' : hp.uniform('reg_lambda', -0.001, 10), # 
                     'reg_alpha' : hp.uniform('reg_alpha', 0.01, 50)}

# hp.quniform(label, low, high, q) : 하이퍼파라미터 이름, 최소값, 최대값, 간격
# hp.uniform(label, low, high, ) : 하이퍼파라미터 이름, 최소값, 최대값 / 정규분포 간격
# hp.randint(label, upper) : 0부터 최대값 upper까지 random한 정수값
# hp.loguniform(label, low, high) : exp(uniform[low, high]) 값 변환 / 정규분포

def lgb_hamsu(lgbm_search_space):
    params = { # 무조건 정수형
              'learning_rate' : lgbm_search_space['learning_rate'],
              'max_depth' : int(lgbm_search_space['max_depth']), # 무조건 정수형
              'num_leaves' : int(lgbm_search_space['num_leaves']), # 무조건 정수형
              'min_child_samples' : int(lgbm_search_space['min_child_samples']), # 무조건 정수형
              'min_child_weight' : int(lgbm_search_space['min_child_weight']), # 무조건 정수형
              'subsample' : lgbm_search_space['subsample'], # 0 ~ 1 사이 min() 1보다 작은값, max() 0보다 큰값
              'colsample_bytree' : lgbm_search_space['colsample_bytree'],
              'max_bin' : max(int(lgbm_search_space['max_bin']), 10), # max_bin와 10을 비교해서 가장 높은 값을 뽑아준다. # 무조건 정수형
              'reg_lambda' : max(lgbm_search_space['reg_lambda'], 0), # 무조건 양수만 나온다.
              'reg_alpha' : lgbm_search_space['reg_alpha']}
    
    model = LGBMRegressor(**params)
    
    model.fit(x_train, y_train,
              eval_set = [(x_train, y_train), (x_test, y_test)],
              eval_metric = 'rmse',
              verbose = 0,
              early_stopping_rounds = 50)
    
    y_predict = model.predict(x_test)
    results = mean_squared_error(y_test, y_predict)
    return results

trial_val = Trials() # hist를 보기위해

best = fmin(
    fn = lgb_hamsu,
    space = lgbm_search_space,
    algo = tpe.suggest,
    max_evals = 50,
    trials = trial_val,
    rstate = np.random.default_rng(seed = 10)
)
# print(best)

abc = pd.DataFrame([best])
print(abc)

# 인덱스를 0부터 시작하는 정수 인덱스로 변경
abc.reset_index(drop=True, inplace=True)

# 데이터프레임 출력 
print(abc)