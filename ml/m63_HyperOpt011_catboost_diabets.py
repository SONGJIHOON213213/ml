from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from catboost import CatBoostRegressor
from hyperopt import hp, fmin, Trials, STATUS_OK, tpe
import numpy as np
import time


# 데이터 로드 및 전처리
x, y = load_diabetes(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=337, train_size=0.8
)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


# CatBoost 모델 학습을 위한 하이퍼파라미터 공간 정의
catboost_space = {
    'learning_rate': hp.uniform('learning_rate', 0.001, 1),
    'max_depth': hp.quniform('max_depth', 3, 16, 1),
    'subsample': hp.uniform('subsample', 0.5, 1),
    'reg_lambda': hp.uniform('reg_lambda', 1, 10),
    'min_data_in_leaf': hp.quniform('min_data_in_leaf', 1, 100, 1),
    'max_bin': hp.quniform('max_bin', 100, 500, 1),
}


# CatBoost 모델 학습 함수 정의
def catboost_hamsu(params):
    model = CatBoostRegressor(
        n_estimators=1000,
        learning_rate=params['learning_rate'],
        max_depth=int(params['max_depth']),
        subsample=params['subsample'],
        reg_lambda=params['reg_lambda'],
        min_data_in_leaf=int(params['min_data_in_leaf']),
        max_bin=int(params['max_bin']),
        random_state=337
    )
    model.fit(
        x_train, y_train,
        eval_set=[(x_test, y_test)],
        use_best_model=True,
        verbose=False,
    )
    y_predict = model.predict(x_test)
    results = r2_score(y_test, y_predict)
    return {'loss': -results, 'status': STATUS_OK}


# 하이퍼파라미터 튜닝 및 최적값 탐색
trials = Trials()

start_time = time.time()
best = fmin(fn=catboost_hamsu,
            space=catboost_space,
            algo=tpe.suggest,
            max_evals=100,
            trials=trials,
            verbose=1,
            rstate=np.random.RandomState(337).randint(0, 1000000))
end_time = time.time()

# 결과 출력
print("Time taken:", end_time - start_time)
print("Best hyperparameters:", best)
print("Best R^2 score:", -trials.best_trial['result']['loss'])