import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer

x, y = load_breast_cancer(return_X_y=True)

pca = PCA(n_components=1)
x_pca = pca.fit_transform(x)

print(f"PCA 후 데이터 형태: {x_pca.shape}")

lda = LinearDiscriminantAnalysis(n_components=1)
x_lda = lda.fit_transform(x_pca, y)

print(f"LDA 후 데이터 형태: {x_lda.shape}")  

# PCA 후 데이터 형태: (569, 1)
# LDA 후 데이터 형태: (569, 1)