import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_wine
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris,load_diabetes
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import random

# 1.데이터
# 1.1 경로, 가져오기
path = 'c:/study/_data/dacon_diabets/'

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)

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

lda = LinearDiscriminantAnalysis(n_components=1)
lda.fit_transform(x, y)
print(x.shape) 

pca = PCA(n_components=2)
x_pca = pca.fit_transform(x)

print(f"PCA 후 데이터 형태: {x_pca.shape}")

lda = LinearDiscriminantAnalysis(n_components=1)
x_lda = lda.fit_transform(x_pca, y)

print(f"LDA 후 데이터 형태: {x_lda.shape}")  

# PCA 후 데이터 형태: (652, 2)
# LDA 후 데이터 형태: (652, 1)