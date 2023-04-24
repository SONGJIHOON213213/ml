from tensorflow.keras.datasets import mnist
import numpy as np 
from sklearn.decomposition import PCA 
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.utils import to_categorical

#1.데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x = np.concatenate((x_train, x_test), axis=0)
y = np.concatenate((y_train, y_test), axis=0)

#차원의 크기를 자동으로 계산
x = x.reshape(x.shape[0], -1)


pca = PCA(n_components=784)
x = pca.fit_transform(x)


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

#2.모델

#CNN모델
cnn_model = Sequential()
cnn_model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
cnn_model.add(MaxPooling2D((2, 2)))
cnn_model.add(Flatten())
cnn_model.add(Dense(10, activation='softmax'))
cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#DNN모델
dnn_model = Sequential()
dnn_model.add(Dense(128, activation='relu', input_shape=(784,)))
dnn_model.add(Dense(10, activation='softmax'))
dnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 모델에서 입력 데이터의 shape을 (28, 28, 1)로 지정했는데, 이는 3차원 
# 하지만 fit() 메서드에 입력되는 데이터는 (batch_size, height, width, channels)와 
# 은 4차원 shape을 가져함. 따라서, fit() 메서드에 입력하기 위해 데이터를 4차원으로 reshape해줘야댐.

#3.훈련
cnn_model.fit(x_train.reshape(-1, 28, 28, 1), y_train, epochs=10, batch_size=32, validation_data=(x_test.reshape(-1, 28, 28, 1), y_test))
cnn_loss, cnn_acc = cnn_model.evaluate(x_test.reshape(-1, 28, 28, 1), y_test)


dnn_model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test))
dnn_loss, dnn_acc = dnn_model.evaluate(x_test, y_test)

#4.평가,예측
print('PCA 0.95: ', pca.n_components_)
print('PCA 0.99: ', np.argmax(np.cumsum(pca.explained_variance_ratio_) >= 0.99) + 1)
print('PCA 0.999: ', np.argmax(np.cumsum(pca.explained_variance_ratio_) >= 0.999) + 1)
print('PCA 1.0: ', np.argmax(np.cumsum(pca.explained_variance_ratio_) >= 1) + 1)
print('CNN 값: ', cnn_acc)
print('DNN 값: ', dnn_acc)

#요약
# 첫째, 차원 축소를 통해 데이터의 크기를 줄일 수 있습니다. 데이터가 크고 복잡한 경우, 
# 처리 시간이 매우 오래 걸릴 수 있습니다. 따라서, 차원 축소를 통해 데이터의 크기를 줄이면, 연산 속도를 높일 수 있습니다.

# 둘째, 차원 축소를 통해 데이터의 특징을 추출하고, 중요한 정보를 유지할 수 있습니다. 
# 예를 들어, 이미지 데이터의 경우, 각 픽셀은 하나의 특징입니다. 
# 하지만, 모든 픽셀을 모두 사용해야 할 필요는 없습니다. 

# 차원 축소를 통해 픽셀에서 중요한 정보를 추출할 수 있습니다. 이러한 방식으로, 
# 차원 축소는 데이터의 특징을 추출하고, 중요한 정보를 유지하는 데 도움을 줄 수 있습니다.
#우선 PCA를 통해 차원 축소를 수행한 후에, 그 결과를 이용해 CNN 모델과 
# DNN 모델을 각각 훈련시키는 것입니다. 이를 통해 차원 축소로 인한 성능 저하를 최소화하면서도 모델의 복잡도를 줄임

# PCA 0.95:  784
# PCA 0.99:  331
# PCA 0.999:  486
# PCA 1.0:  713
# CNN Accuracy:  0.9049999713897705
# DNN Accuracy:  0.948285698890686 