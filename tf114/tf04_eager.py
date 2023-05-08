import tensorflow as tf
print(tf.__version__)


#즉시실행모드!!
print(tf.executing_eagerly())   #False 

tf.compat.v1.disable_eager_execution()#즉시 실행모드 끄겠다.
print(tf.executing_eagerly())   #True

aaa = tf.constant('hello world')

sess = tf.compat.v1.Session()
# print(sess.run(aaa))

