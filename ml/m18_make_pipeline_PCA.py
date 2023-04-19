from sklearn.utils import all_estimators
from sklearn.datasets import load_iris, load_breast_cancer, load_digits, load_wine, fetch_california_housing, load_diabetes
import pandas as pd
import numpy as np
import warnings
from sklearn.linear_model import QuantileRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler,MaxAbsScaler
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict
warnings.filterwarnings(action='ignore')
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.experimental import all_estimators
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
x,y = load_digits(return_X_y=True)
print(x.shape)
print(np.unique(y,return_counts=True))

pca = PCA(n_components=8)
x =  pca.fit_transform()
print(x.shape)

x_train,x_test,y_train,y_test = train_test_split(
    x,y,train_size =0.8,shuffle =True,random_state=337
)
#2.모델
model = make_pipeline(PCA())

#3.훈련,컴파일
model.fit(x_train,y_train)

#4.평가,예측
result = model.score(x_test,y_test)
print("model.score:", result) 
y_predict = model.predict(x_test)
acc = accuracy_score(y_test,y_predict) 
print('accuracy_score:' , acc) 
print(model, ":",model.feature_importances_)
