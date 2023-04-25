# Linear Discriminet Analysis
# 상민인가 회귀에서 된다고 했다!!!
# 성호는 y에 라운드 떄렷어!!!
# 결론 회귀안된다.
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris,load_breast_cancer,load_diabetes
from tensorflow.keras.datasets import cifar100
from sklearn.datasets import fetch_california_housing
#1.데이터 

x,y = fetch_california_housing(return_X_y=True)#회귀는안됨
x,y = load_diabetes(return_X_y=True)
print(y)
print(len(np.round(y)))

lda = LinearDiscriminantAnalysis()

x_lda = lda.fit_transform(x)
print(x_lda.shape)



