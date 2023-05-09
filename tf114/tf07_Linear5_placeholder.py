import tensorflow as tf
tf.set_random_seed(337)

# 1. 데이터
# x =  [1, 2, 3, 4, 5]
# y =  [2, 4, 6, 8, 10]
x = tf.placeholder(tf.float32,shape=[None])
y = tf.placeholder(tf.float32,shape=[None])
w = tf.Variable(tf.random_normal([1]), dtype=tf.float32)
b = tf.Variable(tf.random_normal([1]), dtype=tf.float32)

# 2. 모델 정의
hypothesis = x * w + b
# 3-1. 손실 함수 정의
loss = tf.reduce_mean(tf.square(hypothesis - y))
# 3-2. 최적화 함수 정의
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(loss)

with tf.compat.v1.Session() as sess:
    # 4. 훈련
    sess.run(tf.global_variables_initializer())
    
    epochs = 2001
    for step in range(epochs):
        _, loss_val,w_val, b_val = sess.run([train, loss],feed_dict={x:[1,2,3,4,5], y:[2,4,6,8,10]})
        if step % 20 == 0:
            # print(step, sess.run(loss), sess.run(b), sess.run(b))
            print(step,loss_val, w_val, b_val)