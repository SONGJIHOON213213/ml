import tensorflow as tf
import numpy as np
import pandas as pd

from sklearn.datasets import load_diabetes, fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# 1 데이터
tf.compat.v1.set_random_seed(1234)

ddarung_path = 'c:/_study/_data/_dacon_ddarung/'
kaggle_bike_path = 'c:/_study/_data/_kaggle_bike/'

ddarung = pd.read_csv(ddarung_path + 'train.csv', index_col = 0).dropna()
kaggle_bike = pd.read_csv(kaggle_bike_path + 'train.csv', index_col = 0).dropna()

x1 = ddarung.drop(['count'], axis = 1)
y1 = ddarung['count']

x2 = kaggle_bike.drop(['count', 'casual', 'registered'], axis = 1)
y2 = kaggle_bike['count']

data_list = [load_diabetes,
             fetch_california_housing,
             (x1, y1),
             (x2, y2)]

for d in range(len(data_list)):
    if d < 2:
        x, y = data_list[d](return_X_y = True)
        y = y.reshape(-1, 1) # (442, 1)
    else:
        x, y = data_list[d]
        y = y.values.reshape(-1, 1)
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, random_state = 1234, shuffle = True)
    n_features = x_train.shape[1]
    
    x_p = tf.compat.v1.placeholder(tf.float32, shape = [None, n_features])
    y_p = tf.compat.v1.placeholder(tf.float32, shape = [None, 1])

    w = tf.compat.v1.Variable(tf.compat.v1.random_normal([n_features, 1], name = 'weight'))
    b = tf.compat.v1.Variable(tf.compat.v1.zeros([1], name = 'bias'))

    # 2 모델
    hypothsis = tf.compat.v1.matmul(x_p, w) + b

    # 3-1 컴파일
    loss = tf.reduce_mean(tf.square(hypothsis - y_p)) # mse

    # optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.000001)
    optimizer = tf.train.AdamOptimizer(learning_rate = 0.00000001)

    train = optimizer.minimize(loss)

    # 3-2 훈련
    sess = tf.compat.v1.Session()

    sess.run(tf.compat.v1.global_variables_initializer())

    epochs = 11
    
    for s in range(epochs):
        _, loss_val = sess.run([train, loss], feed_dict = {x_p : x_train, y_p : y_train})
        
        if s % 20 == 0:  # Print loss every 200 steps
            print(f'step : {s}, loss : {loss}')

        y_predict = sess.run(hypothsis, feed_dict = {x_p : x_test})

        # 4 평가
    r2 = r2_score(y_test, y_predict)
    print(f'데이터 : {d}, r2_score : {r2}')

# hy = x * w + b
# x = (5,3) * w + b = (5,1)

# (5, 3) * (3,1) = 5,1
# sess.run(tf.compat.v1.global_variables_initializer())





# [array([[-100.08837],
#        [-166.6625 ],
#        [-107.20363],
#        [-104.32795],
#        [ -53.53002]], dtype=float32), array([[-0.9852853 ],
#        [-0.7733676 ],
#        [ 0.21306361]], dtype=float32), array([-2.569937], dtype=float32)]
####################3.컴파일 훈련
#[실습]

#8500
# R2 score: 0.40495455



#분류
#iris
#cancer
#dacon_diabet
#wine
#fetch_covtype
#digits
#dacon_wine

##이진
#diabets
#_california
#dacon_ddarung
#_kaggle_bike

