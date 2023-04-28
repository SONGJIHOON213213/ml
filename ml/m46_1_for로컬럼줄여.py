from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import r2_score, mean_squared_error
from xgboost import XGBRegressor
from sklearn.feature_selection import SelectFromModel
import numpy as np
import warnings
warnings.filterwarnings('ignore')
# 데이터 로드 및 스케일링
x, y = load_diabetes(return_X_y=True)
scaler = RobustScaler()
x = scaler.fit_transform(x)

# 전체 특성으로 모델 학습 및 결과 출력
parameters = {'n_estimators' : 1000,
              'learning_rate' : 0.3,
              'max_depth' : 2,
              'gamma' : 0,
              'min_child_weight' : 0,
              'subsample' : 0.2,
              'colsample_bytree' : 0.5,
              'colsample_bylevel' : 0,
              'colsample_bynode' : 0,
              'reg_alpha' : 0,
              'reg_lambda' : 1,
              'random_state' : 337,
              }
model = XGBRegressor(**parameters)
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=337, train_size=0.8)
model.fit(x_train, y_train, eval_set=[(x_train, y_train), (x_test, y_test)], early_stopping_rounds=10, verbose=0, eval_metric='rmse')
print('All features result :', model.score(x_test, y_test))
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('All features r2 :', r2)
mse = mean_squared_error(y_test, y_predict)
print('All features rmse :', np.sqrt(mse))

# 특성 선택을 위한 SelectFromModel 초기화
selection = SelectFromModel(model, threshold='mean', prefit=False)

# 5번째 컬럼을 제외한 데이터 준비
x_exclude_1_5 = np.delete(x, [0,9], axis=1)

# 특정 컬럼을 제외한 특성 선택을 수행하고 결과 출력
for i in range(x.shape[1]):
    if i not in [0,1,2,3,4,5,6,7]: # 1번째와 5번째 컬럼을 제외한 경우
        selection.fit(np.delete(x, i, axis=1), y)
        selection_x = selection.transform(np.delete(x, i, axis=1))
        x_train_sel, x_test_sel, y_train, y_test = train_test_split(selection_x, y, random_state=337, train_size=0.8)
        x_train_sel, x_test_sel = scaler.fit_transform(x_train_sel), scaler.transform(x_test_sel)
        selection_model = XGBRegressor(early_stopping_rounds=10, **parameters)
        selection_model.fit(x_train_sel, y_train, eval_set=[(x_train_sel, y_train), (x_test_sel, y_test)], verbose=0, eval_metric='rmse')
        y_predict = selection_model.predict(x_test_sel)
        score = r2_score(y_test, y_predict)
        print(f'특정 컬럼 삭제 {i+1}: R2 score = {score}')
