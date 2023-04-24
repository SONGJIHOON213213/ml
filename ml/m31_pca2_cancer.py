import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# 데이터 로드
datasets = load_breast_cancer()
x = datasets['data']
y = datasets.target 

# 기본 결과 출력
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=1234
)
model = RandomForestRegressor(random_state=1234)
model.fit(x_train, y_train)
default_result = model.score(x_test, y_test)
print(f"기본 결과: {default_result:.2f}")

# 차원 축소 후 결과 출력
for n_components in range(1, 3):
    pca = PCA(n_components=n_components)
    x_pca = pca.fit_transform(x)

    x_train, x_test, y_train, y_test = train_test_split(
        x_pca, y, test_size=0.3, random_state=1234
    )

    model = RandomForestRegressor(random_state=1234)
    model.fit(x_train, y_train)
    result = model.score(x_test, y_test)
    
    print(f"차원{n_components}개 축소: {result:.2f}")