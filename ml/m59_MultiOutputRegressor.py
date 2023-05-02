import numpy as np
import pandas as pd
from sklearn.datasets import load_linnerud
from sklearn.linear_model import Lasso,Ridge 
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.multioutput import MultiOutputRegressor
# MultioutputRegressor
x, y = load_linnerud(return_X_y=True)

# Use only the first target variable (Weight)
# y = y[:, 0]

# print(x)
# print(y)
# print(x.shape, y.shape)

model = Ridge()
model.fit(x, y)
y_pred = model.predict(x)
print(model.__class__.__name__,"스코어 : ",(y, y_pred))
print(model.predict([[2, 110, 43]]))

model = XGBRegressor()
model.fit(x, y)
y_pred = model.predict(x)
print(model.__class__.__name__,"스코어 : ", round(mean_absolute_error(y, y_pred), 4))
print(model.predict([[2, 110, 43]]))

model = MultiOutputRegressor(LGBMRegressor())
model.fit(x, y)
y_pred = model.predict(x)
print(model.__class__.__name__,"스코어 : ", round(mean_absolute_error(y, y_pred), 4))
print(model.predict([[2, 110, 43]]))

model = CatBoostRegressor(loss_function='MultiRMSE', verbose = 0)
model.fit(x, y)
y_pred = model.predict(x)
print(model.__class__.__name__,"스코어 : ", round(mean_absolute_error(y, y_pred), 4))
print(model.predict([[2, 110, 43]]))