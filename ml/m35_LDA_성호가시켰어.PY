# Linear Discriminet Analysis
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris,load_breast_cancer,load_diabetes,load_digits
from tensorflow.keras.datasets import cifar100
x,y = load_iris(return_X_y=True)
x,y = load_breast_cancer(return_X_y=True)
x,y = load_diabetes(return_X_y=True)

(x_train,y_train),(x_test,y_test) = cifar100.load_data()
print(x_train.shape)

x_train =  x_train.reshape(50000,32*32*3)

pca = PCA(n_components=99)
x_train = pca.fit_transform(x_train)


print(x.shape)

lda = LinearDiscriminantAnalysis(n_components=101)
# lda = LinearDiscriminantAnalysis(n_components=3)
lda.fit_transform(x)
print(x.shape)

import matplotlib.pyplot as plt
from sklearn.datasets import load_iris 
## iris 데이터넷 로드
iris = load_iris()
##데이터셋 꽃잎길이와 폭정보 추출
x = iris.data[:,2:] #petal legnth,width
y = iris.target 




