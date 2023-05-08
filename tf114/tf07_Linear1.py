import tensorflow  as tf
tf.set_random_seed(37)

#1.데이터
x = [1,2,3]
y = [1,2,3]

w = tf.Variable(111,dtype=tf.float32)
b = tf.Variable(0,dtype=tf.float32)

#2.모델 구성
hypothesis = x * w + b

#3-1.컴파일 훈련
loss = tf.reduce_mean(tf.square(hypothesis - y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(loss) #경사상법 방식으로 loss를최적화시킨다.
#model.compile(loss='mse', optimizer='sgd')
#3-2 훈련.
sess = tf.compat.v1.Session()
sess.run(tf.global_variables_initializer())

epochs = 2001
for step in range(epochs):
    sess.run(train)
    if step %20 == 0:   #sess_run 보고싶으면 사용
        print(step,sess.run(loss),sess.run(b))
sess.close()
 