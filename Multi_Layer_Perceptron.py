# -*- coding:utf-8 -*-
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/",one_hot =True)
import tensorflow as tf
sess = tf.InteractiveSession()
#定义前馈传播的部分
#定义网络的权重和偏置
in_units = 784
h1_units = 300
w1 = tf.Variable(tf.truncated_normal([in_units,h1_units], stddev = 0.1))
b1 = tf.Variable(tf.zeros([h1_units]))
w2 = tf.Variable(tf.zeros([h1_units,10]))
b2 = tf.Variable(tf.zeros([10]))
#定义输入和Dropout的比率
x = tf.placeholder(tf.float32, [None, in_units])
keep_prob = tf.placeholder(tf.float32)
#定义模型结构
hidden1 = tf.nn.relu(tf.matmul(x,w1) + b1)
hidden1_drop = tf.nn.dropout(hidden1,keep_prob)
y=tf.nn.softmax(tf.matmul(hidden1_drop, w2) + b2)

#定义损失函数和选择优化器
y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ *tf.log(y), reduction_indices = [1]))
train_step = tf.train.AdagradOptimizer(0.3).minimize(cross_entropy)

####定义准确率的操作
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

####训练过程
####将数据写入excel
import xlwt
# 创建一个workbook，设置编码
workbook = xlwt.Workbook(encoding = 'utf-8')
worksheet = workbook.add_sheet('test')

tf.global_variables_initializer().run()
for i in range(10000):
	batch_xs, batch_ys = mnist.train.next_batch(1000)
	train_accuracy = accuracy.eval(feed_dict={x: batch_xs, y_: batch_ys,keep_prob:0.75})
	sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys,keep_prob:0.75})
	train_accuracy = ("%.6f" % train_accuracy)	
	worksheet.write(i,0, label = train_accuracy)
	print (train_accuracy)

#测试		
workbook.save('CNNs_test.xls')
print(sess.run(accuracy, feed_dict={x: mnist.test.images,
                                    y_: mnist.test.labels, keep_prob:1.0}))
									
									
####结果：0.9734									