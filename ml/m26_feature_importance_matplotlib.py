import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor
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
model = RandomForestRegressor()
model_drop = RandomForestRegressor()

# 3. 훈련
model.fit(X_train_scaled, y_train)

# Feature Importance
for i in range(len(model.feature_importances_)):
    print(f"Feature: {i}, Score: {model.feature_importances_[i]:.5f}")
    
# 4. 평가
r2 = r2_score(y_test, model.predict(X_test_scaled))
print(f"R2 Score: {r2:.5f}")

# Feature Drop
X_train_drop = np.delete(X_train_scaled, [0, 1], axis=1)
X_test_drop = np.delete(X_test_scaled, [0, 1], axis=1)

# 5. 훈련
model_drop.fit(X_train_drop, y_train)

# Feature Importance
for i in range(len(model_drop.feature_importances_)):
    print(f"Feature: {i}, Score: {model_drop.feature_importances_[i]:.5f}") 
    
def plot_feature_importance(model):
    n_feature = len(feature_names)
    plt.barh(np.arange(n_feature), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_feature), feature_names)
    plt.xlabel('Feature Importances')
    plt.ylabel('Feature')
    plt.ylim(-1, n_feature)
    plt.title(model)
    
plot_feature_importance(model)
plt.show()