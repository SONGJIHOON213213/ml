import tensorflow as tf
tf.compat.v1.set_random_seed(338)


#1.데이터
x_data = [[1,2],[2,3],[3,1],[4,3],[5,3],[6,2]]
y_data = [[0],[0],[0],[1],[1],[1]]  

x = tf.compat.v1.placeholder(tf.float32)
y = tf.compat.v1.placeholder(tf.float32)

w = tf.compat.v1.Variable(tf.compat.v1.random_normal([1]))
b = tf.compat.v1.Variable(tf.compat.v1.random_normal([1]))

# 2. 모델
hypothesis = x * w  + b

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.00001)

# 3. 훈련
cost = tf.reduce_mean(tf.square(hypothesis - y))
train = optimizer.minimize(cost)

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
epochs = 1000
batch_size = len(x_data)

for epoch in range(epochs):
    for step in range(batch_size):
        x_batch = x_data[step]
        y_batch = y_data[step]
        cost_val, hy_val, _ = sess.run([cost, hypothesis, train],
                                       feed_dict={x: x_batch,y: y_batch})
        if step % 100 == 0:
            print(f"Epoch: {epoch+1}, Step: {step}, Cost: {cost_val}, Prediction:\n{hy_val}")
    
# R2 score 계산
mean_y = tf.reduce_mean(y)
ss_tot = tf.reduce_sum(tf.square(y - mean_y))
ss_res = tf.reduce_sum(tf.square(y - hypothesis))
r_squared = 1 - (ss_res / ss_tot)

print("R2 score:", sess.run(r_squared, feed_dict={x: x_data, y: y_data}))

sess.close()

