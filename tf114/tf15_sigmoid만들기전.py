import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import mean_squared_error, r2_score

# 1. 데이터
x_data = [[1,2],[2,3],[3,1],[4,3],[5,3],[6,2]]
y_data = [[0],[0],[0],[1],[1],[1]]  

x = tf.compat.v1.placeholder(tf.float32, shape=[None, 2])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

w = tf.compat.v1.Variable(tf.compat.v1.random_normal([2, 1]))
b = tf.compat.v1.Variable(tf.compat.v1.random_normal([1]))

logits = tf.matmul(x, w) + b
hypothsis = tf.nn.sigmoid(logits)

# 3-1. 컴파일
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=y))
# loss = tf.reduce_mean(y*tf.log(hypothsis)+ (1-y)*tf.log(1-hypothsis))

# 두 번째 식은 직접 Cross Entropy를 계산하는 식으로, 
# sigmoid 함수를 미리 계산하여 예측값을 생성한 후에 Cross Entropy를 계산합니다. 
# 이러한 방식은 sigmoid 함수에서의 수치적인 불안정성이 발생할 수 있습니다.
# 따라서 첫 번째 식을 사용하는 것이 더욱 안정적이며, 
# 론적으로 두 식이 동일한 값을 출력하지만, 
# 구현상의 이유로 첫 번째 식을 사용하는 것을 권장합니다.


optimizer = tf.train.AdamOptimizer(learning_rate=0.0012)
train = optimizer.minimize(loss)

# 3-2. 훈련
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

epochs = 1000

for s in range(epochs):
    _, loss_val = sess.run([train, loss], feed_dict={x: x_data, y: y_data})
    if (s+1) % 1000 == 0:
        print(f"Step {s+1}, Loss: {loss_val:.4f}")

# 4. 평가
y_predict = sess.run(hypothsis, feed_dict={x: x_data})
y_predict_binary = np.where(y_predict > 0.5, 1, 0)

acc = accuracy_score(y_data, y_predict_binary)
print(f"Accuracy: {acc:.4f}")

f1 = f1_score(y_data, y_predict_binary)
print(f"F1 score: {f1:.4f}")

mse = mean_squared_error(y_data, y_predict_binary)
r2 = r2_score(y_data, y_predict)
print(f"Mean Squared Error: {mse:.4f}")
print('r2 스코어 :', r2)