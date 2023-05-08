#현재버전이 1.0이면 그냥출력
#현재버전이 2.0이면 즉시실행모드 끄고 출력
#if문써서 1번소스 변경
import tensorflow as tf

# 현재 TensorFlow 버전 출력
print("TensorFlow version:", tf.__version__)

# TensorFlow 버전이 1.0인 경우
if tf.__version__.startswith('1.'):
    with tf.Session() as sess:
        print(sess.run(tf.constant("True")))

# TensorFlow 버전이 2.0인 경우
else:
    tf.compat.v1.disable_eager_execution()
    with tf.compat.v1.Session() as sess:
        print(sess.run(tf.constant("False")))