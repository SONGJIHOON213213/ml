import tensorflow as tf
import numpy as np
import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error, r2_score

# 1 데이터
tf.compat.v1.set_random_seed(1234)

x, y = load_breast_cancer(return_X_y=True)
y = y.reshape(-1, 1)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=1234, shuffle=True)
n_features = x_train.shape[1]

x_p = tf.compat.v1.placeholder(tf.float32, shape=[None, n_features])
y_p = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

w = tf.compat.v1.Variable(tf.compat.v1.random_normal([n_features, 1], name='weight'))
b = tf.compat.v1.Variable(tf.compat.v1.zeros([1], name='bias'))

logits = tf.matmul(x_p, w) + b
hypothsis = tf.nn.sigmoid(logits)

# 3-1 컴파일
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=y_p))

# optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.000001)
optimizer = tf.train.AdamOptimizer(learning_rate=0.00000001)

train = optimizer.minimize(loss)

# 3-2 훈련
sess = tf.compat.v1.Session()

sess.run(tf.compat.v1.global_variables_initializer())

epochs = 11

for s in range(epochs):
    _, loss_val = sess.run([train, loss], feed_dict={x_p: x_train, y_p: y_train})

    if s % 20 == 0:  # Print loss every 200 steps
        print(f'step : {s}, loss : {loss}')

y_predict = sess.run(hypothsis, feed_dict={x_p: x_test})
y_predict = np.round(y_predict).reshape(-1, 1)
y_predict_binary = np.where(y_predict > 0.5, 1, 0)
acc = accuracy_score(y_test, y_predict)
r2 = r2_score(y_test, y_predict)
mse = mean_squared_error(y_test, y_predict_binary)
# 4 평가
print(f'accuracy : {acc}, r2_score : {r2}')
print(f"Mean Squared Error: {mse:.4f}")