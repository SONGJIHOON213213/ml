import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_digits,load_iris,load_wine,load_boston
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression

x = np.arange(8).reshape(4,2)
# [[0 1]
# [2 3]
# [4 5]
# [6 7]]

print(x)

pf = PolynomialFeatures(degree=2)
x_pf = pf.fit_transform(x)

print(x_pf)
print(x_pf.shape)
# [[0 1]
#  [2 3]
#  [4 5]
#  [6 7]]
# [[ 1.  0.  1.  0.  0.  1.]
#  [ 1.  2.  3.  4.  6.  9.]
#  [ 1.  4.  5. 16. 20. 25.]
#  [ 1.  6.  7. 36. 42. 49.]]
# (4, 6)
print("====================================2=====================================")

x = np.arange(8).reshape(4,2)
# [[0 1]
# [2 3]
# [4 5]
# [6 7]]

print(x)

pf = PolynomialFeatures(degree=3)
x_pf = pf.fit_transform(x)

print(x_pf)
print(x_pf.shape)
# [[0 1]
#  [2 3]
#  [4 5]
#  [6 7]]
# [[  1.   0.   1.   0.   0.   1.   0.   0.   0.   1.]
#  [  1.   2.   3.   4.   6.   9.   8.  12.  18.  27.]
#  [  1.   4.   5.  16.  20.  25.  64.  80. 100. 125.]
#  [  1.   6.   7.  36.  42.  49. 216. 252. 294. 343.]]
# (4, 10)



print("==================================3=======================================")
# datasets = [load_iris, load_breast_cancer, load_digits, load_wine, load_boston]

# for data in datasets:
#     dataset_name = data.__name__
#     x, y = data(return_X_y=True)

#     x_train, x_test, y_train, y_test = train_test_split(
#         x, y, shuffle=True, train_size=0.8, random_state=1030
#     )

#     scaler = StandardScaler()
#     x_train = scaler.fit_transform(x_train)
#     x_test = scaler.transform(x_test)
    
#     # 다항 회귀 모델 구성
#     poly = PolynomialFeatures(degree=2, include_bias=False)
#     x_train_poly = poly.fit_transform(x_train)
#     x_test_poly = poly.transform(x_test)

#     model = LinearRegression()
#     model.fit(x_train_poly, y_train)
#     y_pred = model.predict(x_test_poly)

#     print(f'{dataset_name} 데이터')
#     print('PolynomialFeatures R2 score:', r2_score(y_test, y_pred))
#     print()


print("#################################################")

x = np.arange(8).reshape(4,3)
# [[0 1]
# [2 3]
# [4 5]
# [6 7]]

print(x)

pf = PolynomialFeatures(degree=2)
x_pf = pf.fit_transform(x)

print(x_pf)
print(x_pf.shape)  
