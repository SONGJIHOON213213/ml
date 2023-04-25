# Linear Discriminet Analysis

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris,load_breast_cancer,load_diabetes,load_digits

x,y = load_iris(return_X_y=True)
x,y = load_breast_cancer(return_X_y=True)
x,y = load_diabetes(return_X_y=True)

pca = PCA(n_components=3)
x = pca.fit_transform(x)

print(x.shape)

lda = LinearDiscriminantAnalysis(n_components=3)
lda.fit_transform(x)
print(x.shape)

import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
