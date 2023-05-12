import tensorflow as tf
import matplotlib.pyplot as plt


# x_train = [1,2,3]  #[1]
# y_train = [1,2,3]  #[2]
                   #웨이트값 계산

x_train = [1]  
y_train = [2]                 
x = tf.compat.v1.placeholder(tf.float32)
y = tf.compat.v1.placeholder(tf.float32)


w = tf.compat.v1.Variable([10],dtype=tf.float32,name='weight') 

# 2. 모델 구성
hypothesis = x * w

# 3-1. 컴파일
loss = tf.reduce_mean(tf.square(hypothesis - y))
###########################옵티마이저####################################
lr = 0.1

gradient = tf.reduce_mean((x * w - y) * x)

descant = w - lr * gradient
update = w.assign(descant) #w = w - lr * gradient 


w_history = []
loss_history = []

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

for step in range(21):
    
    _,loss_v,w_v = sess.run([update,loss,w],feed_dict={x:x_train,y:y_train})
    print(step,'\t',loss_v,'\t',w_v)
    w_history.append(w_v)
    loss_history.append(loss_v)
sess.close()

print("====================w history==========================")
print(w_history)
print("====================Hypothesis history==========================")
print(loss_history) 

# 그래프로 시각화
plt.plot(w_history, loss_history)
plt.ylabel('weights')
plt.xlabel('loss')
plt.show()

############체인룰######################

