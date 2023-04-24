import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
import warnings
from sklearn.datasets import load_iris, load_boston,load_breast_cancer,load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from xgboost import XGBClassifier, XGBRegressor
import matplotlib.pyplot as plt

# 1. 데이터
iris = load_iris()
X, y = iris.data, iris.target
feature_names = iris.feature_names

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 2. 모델
model = RandomForestClassifier()
model_drop = RandomForestClassifier()

# 3. 훈련
model.fit(X_train_scaled, y_train)
model.fit(X_train_scaled, y_train)
model.fit(X_train_scaled, y_train)
model.fit(X_train_scaled, y_train)

#4.평가,예측
result = model.score(x_test,y_test)
print("model.score:", result) 

y_predict = model.predict(x_test)
mse = mean_squared_error(y_test,y_predict)
print("MSE:", mse)
    
def plot_feature_importance(model):
    n_feature = len(feature_names)
    plt.barh(np.arange(n_feature), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_feature), feature_names)
    plt.xlabel('Feature Importances')
    plt.ylabel('Feature')
    plt.ylim(-1, n_feature)
    plt.title(model)
    
plt.subplots(2,2,1)    
plot_feature_importance(model)
plt.subplots(2,2,2)    
plot_feature_importance(mode2)
plt.subplots(2,2,3)    
plot_feature_importance(mode3)
plt.subplots(2,2,4)    
plot_feature_importance(mode4)
plt.subplots(2,2,5)    
plot_feature_importance(mode5)
plt.show()