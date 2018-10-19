# -*- coding:utf-8 -*-
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
##########结果（0.9236）无隐含层的网络
# -*- coding:utf-8 -*-
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
sess = tf.InteractiveSession()
dir='/home/kaka/Documents/input_data'
# 1.Import data
mnist = input_data.read_data_sets(dir, one_hot=True)

#Print the shape of mist
print (mnist.train.images.shape,mnist.train.labels.shape)
print(mnist.test.images.shape, mnist.train.labels.shape)
print(mnist.validation.images.shape, mnist.validation.labels.shape)

######前向传播的过程
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.matmul(x, W) + b

####定义损失函数和优化器
y_ = tf.placeholder(tf.float32, [None, 10])

# The raw formulation of cross-entropy,
#
#   tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.nn.softmax(y)),
#                                 reduction_indices=[1]))
#
# can be numerically unstable.
#
# So here we use tf.nn.softmax_cross_entropy_with_logits on the raw
# outputs of 'y', and then average across the batch.
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

#定义准确率的操作
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
	train_accuracy = accuracy.eval(feed_dict={x: batch_xs, y_: batch_ys})
	train_accuracy.__class__
	
	sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
	train_accuracy = ("%.6f" % train_accuracy)	
	worksheet.write(i,0, label = train_accuracy)
	#print ("%.6f" % train_accuracy)
# Test trained model
workbook.save('CNNs_test.xls')
print(sess.run(accuracy, feed_dict={x: mnist.test.images,
                                    y_: mnist.test.labels}))