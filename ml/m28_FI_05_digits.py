import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
import warnings
from sklearn.datasets import load_iris, load_boston, load_breast_cancer, load_digits
from sklearn.metrics import accuracy_score, r2_score
from xgboost import XGBClassifier, XGBRegressor
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.metrics import mean_squared_error


# 1. 데이터
digits = load_digits()
X, y = digits.data, digits.target
feature_names = digits.feature_names

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = [DecisionTreeClassifier(), RandomForestClassifier(), GradientBoostingClassifier(), XGBClassifier()]

def plot_feature_importance(model):
    n_feature = len(feature_names)
    plt.barh(np.arange(n_feature), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_feature), feature_names)
    plt.xlabel('Feature Importances')
    plt.ylabel('Feature')
    plt.ylim(-1, n_feature)
    plt.title(type(model).__name__)

for i, model in enumerate(models):
    model.fit(X_train_scaled, y_train)
    y_predict = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_predict)
    acc = accuracy_score(y_test, y_predict)
    print(f"{model.__class__.__name__} - 기존 ACC: {acc:.5f}")

    plt.subplot(2, 2, i+1)
    plot_feature_importance(model)

plt.show()

# Feature Drop
X_train_drop = np.delete(X_train_scaled, [0, 1], axis=1)
X_test_drop = np.delete(X_test_scaled, [0, 1], axis=1)

for i, model in enumerate(models):
    model.fit(X_train_drop, y_train)
    y_predict = model.predict(X_test_drop)
    mse = mean_squared_error(y_test, y_predict)
    acc = accuracy_score(y_test, y_predict)
    print(f"{model.__class__.__name__} - 컬럼 삭제후 ACC: {acc:.5f}")

plt.show()