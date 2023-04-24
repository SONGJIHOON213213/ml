import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

#1. 데이터
datasets = load_breast_cancer()
print(datasets.feature_names)
x = datasets['data']
y = datasets.target 

pca = PCA(n_components=30)
x = pca.fit_transform(x)
print(x)
print(x.shape) 

pca_EVR = pca.explained_variance_ratio_
print(pca_EVR) 
print(sum(pca_EVR)) 


pca_cumsum = np.cumsum(pca_EVR)

import matplotlib.pyplot as plt 
plt.plot(pca_cumsum) 
plt.gird()
plt.show()