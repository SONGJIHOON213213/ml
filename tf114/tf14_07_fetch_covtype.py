import tensorflow as tf
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

# 1. 데이터
x, y = fetch_covtype(return_X_y=True)
y = y.reshape(-1, 1)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, shuffle=True, random_state=337
)

# 2. 모델
x = tf.compat.v1.placeholder(tf.float32, shape=[None, x_train.shape[1]])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])
w = tf.compat.v1.Variable(tf.compat.v1.random_normal([x_train.shape[1], 1]), name='weights')
b = tf.compat.v1.Variable(tf.compat.v1.zeros([1]), name='bias')

hypothesis = tf.compat.v1.matmul(x, w) + b

# 3. 컴파일
loss = tf.reduce_mean(tf.square(hypothesis - y))
optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.001)
train = optimizer.minimize(loss)

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

# 모델 학습
epochs = 10
batch_size = 5

for epoch in range(epochs):
    for step in range(x_train.shape[0] // batch_size):
        start = step * batch_size
        end = (step + 1) * batch_size
        x_batch = x_train[start:end, :]
        y_batch = y_train[start:end, :]
        cost_val, hy_val, _ = sess.run([loss, hypothesis, train], feed_dict={x: x_batch, y: y_batch})
        if step % 100 == 0:
            print(f"Epoch: {epoch+1}, Step: {step}, Cost: {cost_val}, Prediction:\n{hy_val}")

# R2 score 계산
mean_y = tf.reduce_mean(y)
ss_tot = tf.reduce_sum(tf.square(y - mean_y))
ss_res = tf.reduce_sum(tf.square(y - hypothesis))
r_squared = 1 - (ss_res / ss_tot)

print("R2 score:", sess.run(r_squared, feed_dict={x: x_test, y: y_test}))
print("MAE (TensorFlow):", sess.run(mean_absolute_error, feed_dict={x: x_test, y: y_test}))