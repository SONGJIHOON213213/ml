import tensorflow as tf
tf.compat.v1.set_random_seed(123)
import warnings
warnings.filterwarnings
# 1. 데이터
x_data = [[73, 51, 65.],
          [92, 98, 11.],
          [89, 31, 33.],
          [99,33,100.],
          [17,66,79.]]
y_data = [[152],[185],[180],[205],[142]]

x = tf.compat.v1.placeholder(tf.float32)
y = tf.compat.v1.placeholder(tf.float32)

w = tf.compat.v1.Variable(tf.compat.v1.random_normal([1]),name = 'weight')
b = tf.compat.v1.Variable(tf.compat.v1.random_normal([1]),name = 'bias')

# 2. 모델
hypothesis = x * w + b

# sess.run(tf.compat.v1.global_variables_initializer())

# 3. 훈련

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())


print(sess.run([hypothesis,w ],feed_dict={x:x_data,w:w}))


