from tensorflow.keras.datasets import mnist
import numpy as np 
from sklearn.decomposition import PCA 

# Load MNIST dataset
(x_train, _), (x_test , _) = mnist.load_data()


x = np.concatenate((x_train, x_test), axis=0)


x = x.reshape(x.shape[0], -1)

from tensorflow.keras.datasets import mnist
import numpy as np 
from sklearn.decomposition import PCA 


(x_train, _), (x_test , _) = mnist.load_data()

x = np.concatenate((x_train, x_test), axis=0)


x = x.reshape(x.shape[0], -1)


pca = PCA()
pca.fit(x)


variance_ratio = pca.explained_variance_ratio_


GGamji = np.cumsum(variance_ratio)

n_components_095 = np.argmax(GGamji>= 0.95) + 1
n_components_099 = np.argmax(GGamji >= 0.99) + 1
n_components_0999 = np.argmax(GGamji>= 0.999) + 1
n_components_1 = np.argmax(GGamji >= 1) + 1

print("0.95개는 몇개:", n_components_099)
print("0.99개는 몇개:", n_components_099)
print("0.999개는 몇개:", n_components_0999)
print("1.0개는 몇개:", n_components_1)  

# PCA는 데이터의 차원을 축소하면서도 가능한 많은 정보(분산)를 유지하기 위한 방법입니다. 따라서 
# PCA를 사용할 때, 분산을 보존하는 것이 매우 중요합니다. 분산을 보존하지 않으면, 데이터의 중요한 특징을 잃을 수 있기 때문입니다.

# 예를 들어, MNIST 데이터 세트의 경우, 원래 이미지는 28x28 픽셀이므로 784차원의 
# 벡터로 표현됩니다. 그러나 이러한 고차원 데이터를 그대로 사용하면 데이터 분석 및 
# 예측 모델링에 많은 문제가 발생할 수 있습니다. PCA를 사용하여 데이터를 저차원으로 
# 축소하면, 분산을 보존하면서도 적은 수의 주요 특징을 유지할 수 있습니다. 
# 따라서 PCA를 사용하면 데이터 분석 및 예측 모델링에 더욱 적합한 데이터를 얻을 수 있습니다 

# pca = PCA(n_components=784)
# x = pca.fit_transform(x) 
# pca_EVR = pca.explained_variance_ratio_
# cumsum = np.cumsum(pca_EVR)
# print(cumsum)

# print(np.argmax(cumsum >= 1.0))
