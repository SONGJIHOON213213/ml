import tensorflow as tf

node1 = tf.constant(3.0)
node2 = tf.constant(6.0)

addition_node = tf.add(node1, node2) 
multiplication_node = tf.multiply(node1, node2)  
subtraction_node = tf.subtract(node1, node2)  
division_node = tf.divide(node2, node1)  

with tf.Session() as sess:
    addition_result = sess.run(addition_node)
    multiplication_result = sess.run(multiplication_node)
    subtraction_result = sess.run(subtraction_node)
    division_result = sess.run(division_node)

print("덧셈 결과: ", addition_result)
print("곱셈 결과: ", multiplication_result)
print("뺄셈 결과: ", subtraction_result)
print("나눗셈 결과: ", division_result)
