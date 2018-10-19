__author__ = 'Caofang'
#####################################
##    采用3*3的卷积核和2*2的池化核 ##   
##             计算耗时            ## 
#####################################
# -*- coding:utf-8 -*-
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
slim = tf.contrib.slim
trunc_normal = lambda stddev: tf.truncated_normal_initializer(0,0,stddev)

#################
#使用tf.contrib.slim辅助设计网络
def inception_v3_arg_scope(weight_decay)= 0.00004，
							stddev=0.1.
							batch_norm_var_collection='moving_vars'):
	
	batch_norm_params = {
	
		'decay' : 0.9997,
		'epsilon' : 0.001,
		'ipdates_collections' : tf.GraphKeys.UPDATE_OPS,
		'variables_variance' : [batch_norm_var_collection],
	}
}

with slim.arg_scope([slim.conv2d, slim.fully_connected],
					weights_regularizer = slim.l2_regularizer(weight_decay)):
	with slim.arg_scope(
		[slim.conv2d],
		weight_initializer = tf.truncated_normal_initializer(stddev = stddev),
		activation_fn = tf.nn.relu,
		normalizer_fn = slim.batch_norm,
		normalizer_params = batch_norm_params) as sc:
	return sc
#定义生成inceptionv3网络的卷积部分
def inception_v3_base(inputs, scope=None):
	
	
	end_points = {}
	with tf.variable_scope(scope, 'InceptionV3', [inputs]):
		with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
								stride = 1, padding='VALID'):
			net = slim.conv2d(inputs, 32, [3,3], stride=2, scope = 'Con2d_1a_3*3')
			net = slim.conv2d(net, 32, [3,3], scope='Conv2d_2a_3*3')
			net = slim.conv2d(net, 64, [3,3], padding = 'SAME', 
								scope = 'Con2d_2b_3*3')
			net = slim.max_pool2d(net, [3,3], stride=2, scope='MaxPool_3a_3*3')
			net = slim.conv2d(net, 80, [1,1], scope='Conv2d_3b_1*1')
			net = slim.conv2d(net, 192, [3,3], scope='Conv2d_4a_3*3')
			net = slim.max_pool2d(net, [3,3], stride=2, scope='MaxPool_5a_3*3')
			
	#三个连续的inception模块组
	
		with slim.variable_scope([slim.con2d,slim.max_pool2d，slim.avg_pool2d]
							,stride=1, padding='SAME'):
		##第一个模块组（Mixed_5b,Mixed_5c,Mixed_5d）
		###Mixed_5b
			with tf.variable_scope('Mixed_5b'):
				with tf.variable_scope('Branch_0'):
					branch_0 = slim.conv2d(net, 64, [1,1], scope='Conv2d_0a_1*1')
				with tf.variable_scope('Branch_1'):
					branch_1 = slim.conv2d(net, 48, [1,1], scope='Conv2d_0a_1*1')
					branch_1 = slim.conv2d(branch_1, 64, [5,5], scope='Conv2d_0b_5*5')
				with tf.variable_scope('Branch_2'):
					branch_2 = slim.conv2d(net, 64, [1,1], scope='Conv2d_0a_1*1')
					branch_2 = slim.conv2d(branch_2, 96, [3,3], scope='Conv2d_0b_3*3')
					branch_2 = slim.conv2d(branch_2, 96, [3,3], scope='Conv2d_0c_3*3')
				with tf.variable_scope('Branch_3'):
					branch_3 = slim.avg_pool2d(net, [3,3], scope='AvgPool_0a_3*3')
					branch_3 = slim.conv2d(branch_3, 32, [1,1], scope='Conv2d_0b_1*1')
				net = tf.conact([branch_0, branch_1, branch_2, branch_3], 3)
		###Mixed_5c
			with tf.variable_scope('Mixed_5c'):
				with tf.variable_scope('Branch_0'):
					branch_0 = slim.conv2d(net, 64, [1,1], scope='Conv2d_0a_1*1')
				with tf.variable_scope('Branch_1'):
					branch_1 = slim.conv2d(net, 48, [1,1], scope='Conv2d_0a_1*1')
					branch_1 = slim.conv2d(branch_1, 64, [5,5], scope='Conv2d_0b_5*5')
				with tf.variable_scope('Branch_2'):
					branch_2 = slim.conv2d(net, 64, [1,1], scope='Conv2d_0a_1*1')
					branch_2 = slim.conv2d(branch_2, 96, [3,3], scope='Conv2d_0b_3*3')
					branch_2 = slim.conv2d(branch_2, 96, [3,3], scope='Conv2d_0c_3*3')
				with tf.variable_scope('Branch_3'):
					branch_3 = slim.avg_pool2d(net, [3,3], scope='AvgPool_0a_3*3')
					branch_3 = slim.conv2d(branch_3, 64, [1,1], scope='Conv2d_0b_1*1')
				net = tf.conact([branch_0, branch_1, branch_2, branch_3], 3)	
		###Mixed_5d
			with tf.variable_scope('Mixed_5d'):
				with tf.variable_scope('Branch_0'):
					branch_0 = slim.conv2d(net, 64, [1,1], scope='Conv2d_0a_1*1')
				with tf.variable_scope('Branch_1'):
					branch_1 = slim.conv2d(net, 48, [1,1], scope='Conv2d_0a_1*1')
					branch_1 = slim.conv2d(branch_1, 64, [5,5], scope='Conv2d_0b_5*5')
				with tf.variable_scope('Branch_2'):
					branch_2 = slim.conv2d(net, 64, [1,1], scope='Conv2d_0a_1*1')
					branch_2 = slim.conv2d(branch_2, 96, [3,3], scope='Conv2d_0b_3*3')
					branch_2 = slim.conv2d(branch_2, 96, [3,3], scope='Conv2d_0c_3*3')
				with tf.variable_scope('Branch_3'):
					branch_3 = slim.avg_pool2d(net, [3,3], scope='AvgPool_0a_3*3')
					branch_3 = slim.conv2d(branch_3, 32, [1,1], scope='Conv2d_0b_1*1')
				net = tf.conact([branch_0, branch_1, branch_2, branch_3], 3)
				
		##第二个模块组（Mixed_6a,Mixed_6b,Mixed_6c,Mixed_6d,Mixed_6e）
		###Mixed_6a
			with tf.variable_scope('Mixed_6a'):
				with tf.variable_scope('Branch_0'):
					branch_0 = slim.conv2d(net, 384, [3,3], stride=2, 
											paadding = 'VALID',scope='Conv2d_0a_1*1')
				with tf.variable_scope('Branch_1'):
					branch_1 = slim.conv2d(net, 64, [1,1], scope='Conv2d_0a_1*1')
					branch_1 = slim.conv2d(branch_1, 96, [3,3], scope='Conv2d_0b_5*5')
					branch_1 = slim.conv2d(Branch_1, 96, [3,3], stride=2, 
											paadding = 'VALID',scope='Conv2d_0a_1*1')
				with tf.variable_scope('Branch_2'):
					branch_2 = slim.Max_pool2d(net, [3,3], stride=2, 
											paadding = 'VALID',scope='Maxpool_1a_3*3')
				net = tf.conact([branch_0, branch_1, branch_2, branch_3], 3)
		###Mixed_6b
			with tf.variable_scope('Mixed_6b'):
				with tf.variable_scope('Branch_0'):
					branch_0 = slim.conv2d(net, 192, [1,1], scope='Conv2d_0a_1*1')
				with tf.variable_scope('Branch_1'):
					branch_1 = slim.conv2d(net, 128, [1,1], scope='Conv2d_0a_1*1')
					branch_1 = slim.conv2d(branch_1, 128, [1,7], scope='Conv2d_0b_1*7')
					branch_1 = slim.conv2d(branch_1, 192, [7,1], scope='Conv2d_0c_7*1')
				with tf.variable_scope('Branch_2'):
					branch_2 = slim.conv2d(net, 128, [1,1], scope='Conv2d_0a_1*1')
					branch_2 = slim.conv2d(branch_2, 128, [7,1], scope='Conv2d_0b_7*1')
					branch_2 = slim.conv2d(branch_2, 128, [1,7], scope='Conv2d_0c_1*7')
					branch_2 = slim.conv2d(branch_2, 128, [7,1], scope='Conv2d_0d_1*7')
					branch_2 = slim.conv2d(branch_2, 192, [1,7], scope='Conv2d_0e_1*7')
				with tf.variable_scope('Branch_3'):
					branch_3 = slim.avg_pool2d(net, [3,3], scope='AvgPool_0a_3*3')
					branch_3 = slim.con2d(branch_3, 192, [1,1], scope='Conv2d_0b_1*1')
				net = tf.conact([branch_0, branch_1, branch_2, branch_3], 3)
		###Mixed_6c
			with tf.variable_scope('Mixed_6c'):
				with tf.variable_scope('Branch_0'):
					branch_0 = slim.conv2d(net, 192, [1,1], scope='Conv2d_0a_1*1')
				with tf.variable_scope('Branch_1'):
					branch_1 = slim.conv2d(net, 160, [1,1], scope='Conv2d_0a_1*1')
					branch_1 = slim.conv2d(branch_1, 160, [1,7], scope='Conv2d_0b_1*7')
					branch_1 = slim.conv2d(branch_1, 192, [7,1], scope='Conv2d_0c_7*1')
				with tf.variable_scope('Branch_2'):
					branch_2 = slim.conv2d(net, 160, [1,1], scope='Conv2d_0a_1*1')
					branch_2 = slim.conv2d(branch_2, 160, [7,1], scope='Conv2d_0b_7*1')
					branch_2 = slim.conv2d(branch_2, 160, [1,7], scope='Conv2d_0c_1*7')
					branch_2 = slim.conv2d(branch_2, 160, [7,1], scope='Conv2d_0d_1*7')
					branch_2 = slim.conv2d(branch_2, 192, [1,7], scope='Conv2d_0e_1*7')
				with tf.variable_scope('Branch_3'):
					branch_3 = slim.avg_pool2d(net, [3,3], scope='AvgPool_0a_3*3')
					branch_3 = slim.con2d(branch_3, 192, [1,1], scope='Conv2d_0b_1*1')
				net = tf.conact([branch_0, branch_1, branch_2, branch_3], 3)		
		###Mixed_6d
			with tf.variable_scope('Mixed_6c'):
				with tf.variable_scope('Branch_0'):
					branch_0 = slim.conv2d(net, 192, [1,1], scope='Conv2d_0a_1*1')
				with tf.variable_scope('Branch_1'):
					branch_1 = slim.conv2d(net, 160, [1,1], scope='Conv2d_0a_1*1')
					branch_1 = slim.conv2d(branch_1, 160, [1,7], scope='Conv2d_0b_1*7')
					branch_1 = slim.conv2d(branch_1, 192, [7,1], scope='Conv2d_0c_7*1')
				with tf.variable_scope('Branch_2'):
					branch_2 = slim.conv2d(net, 160, [1,1], scope='Conv2d_0a_1*1')
					branch_2 = slim.conv2d(branch_2, 160, [7,1], scope='Conv2d_0b_7*1')
					branch_2 = slim.conv2d(branch_2, 160, [1,7], scope='Conv2d_0c_1*7')
					branch_2 = slim.conv2d(branch_2, 160, [7,1], scope='Conv2d_0d_1*7')
					branch_2 = slim.conv2d(branch_2, 192, [1,7], scope='Conv2d_0e_1*7')
				with tf.variable_scope('Branch_3'):
					branch_3 = slim.avg_pool2d(net, [3,3], scope='AvgPool_0a_3*3')
					branch_3 = slim.con2d(branch_3, 192, [1,1], scope='Conv2d_0b_1*1')
				net = tf.conact([branch_0, branch_1, branch_2, branch_3], 3)
		###Mixed_6d
			with tf.variable_scope('Mixed_6c'):
				with tf.variable_scope('Branch_0'):
					branch_0 = slim.conv2d(net, 192, [1,1], scope='Conv2d_0a_1*1')
				with tf.variable_scope('Branch_1'):
					branch_1 = slim.conv2d(net, 192, [1,1], scope='Conv2d_0a_1*1')
					branch_1 = slim.conv2d(branch_1, 192, [1,7], scope='Conv2d_0b_1*7')
					branch_1 = slim.conv2d(branch_1, 192, [7,1], scope='Conv2d_0c_7*1')
				with tf.variable_scope('Branch_2'):
					branch_2 = slim.conv2d(net, 160, [1,1], scope='Conv2d_0a_1*1')
					branch_2 = slim.conv2d(branch_2, 192, [7,1], scope='Conv2d_0b_7*1')
					branch_2 = slim.conv2d(branch_2, 192, [1,7], scope='Conv2d_0c_1*7')
					branch_2 = slim.conv2d(branch_2, 192, [7,1], scope='Conv2d_0d_1*7')
					branch_2 = slim.conv2d(branch_2, 192, [1,7], scope='Conv2d_0e_1*7')
				with tf.variable_scope('Branch_3'):
					branch_3 = slim.avg_pool2d(net, [3,3], scope='AvgPool_0a_3*3')
					branch_3 = slim.con2d(branch_3, 192, [1,1], scope='Conv2d_0b_1*1')
				net = tf.conact([branch_0, branch_1, branch_2, branch_3], 3)
			end_points['Mixed_6e'] = net
		##第三个模块组（Mixed_7a,Mixed_7b,Mixed_7c）
		###Mixed_7a	
			with tf.variable_scope('Mixed_7a'):
				with tf.variable_scope('Branch_0'):
					branch_0 = slim.conv2d(net, 192, [1,1], scope='Conv2d_0a_1*1')
					branch_0 = slim.conv2d(branch_0, 320, [3,3], stride=2, 
											paadding = 'VALID', scope='Conv2d_0a_1*1')
				with tf.variable_scope('Branch_1'):
					branch_1 = slim.conv2d(net, 192, [1,1], scope='Conv2d_0a_1*1')
					branch_1 = slim.conv2d(branch_1, 192, [1,7], scope='Conv2d_0b_1*7')
					branch_1 = slim.conv2d(branch_1, 192, [7,1], scope='Conv2d_0c_7*1')
					branch_1 = slim.conv2d(branch_1, 192, [7,1], scope='Conv2d_0c_7*1')
					branch_1 = slim.conv2d(branch_1, 192, [7,1], stride=2, 
											paadding = 'VALID' scope='Conv2d_0c_7*1')
				with tf.variable_scope('Branch_2'):
					branch_2 = slim.max_pool2d(net, [3,3], stride = 2, padding = 'VAILD',
												scope='Conv2d_0a_1*1')
				net = tf.conact([branch_0, branch_1, branch_2], 3)
			###Mixed_7b
			with tf.variable_scope('Mixed_7b'):
				with tf.variable_scope('Branch_0'):
					branch_0 = slim.conv2d(net, 320, [1,1], scope='Conv2d_0a_1*1')
				with tf.variable_scope('Branch_1'):
					branch_1 = slim.conv2d(net, 384, [1,1], scope='Conv2d_0a_1*1')
					branch_1 = tf.conact ([
							slim.conv2d(branch_1, 384, [1,3], scope='Conv2d_0b_1*3')
							slim.conv2d(branch_1, 384, [3,1], scope='Conv2d_0c_3*1')],3)
				with tf.variable_scope('Branch_2'):
					branch_2 = slim.conv2d(net, 448, [1,1], scope='Conv2d_0a_1*1')
					branch_2 = slim.conv2d(branch_2, 384, [3,3], scope='Conv2d_0a_3*3')
					branch_2 = tf.conact ([
							slim.conv2d(branch_2, 384, [1,3], scope='Conv2d_0b_1*3')
							slim.conv2d(branch_2, 384, [3,1], scope='Conv2d_0c_3*1')],3)
				with tf.variable_scope('Branch_3'):
					branch_3 = slim.avg_pool2d(net, [3,3], scope='AvgPool_0a_3*3')	
					branch_3 = slim.conv2d(branch_3, 192, [1,1], scope='Conv2d_0a_1*1')
				net = tf.conact([branch_0, branch_1, branch_2], 3)
			###Mixed_7b
			with tf.variable_scope('Mixed_7b'):
				with tf.variable_scope('Branch_0'):
					branch_0 = slim.conv2d(net, 320, [1,1], scope='Conv2d_0a_1*1')
				with tf.variable_scope('Branch_1'):
					branch_1 = slim.conv2d(net, 384, [1,1], scope='Conv2d_0a_1*1')
					branch_1 = tf.conact ([
							slim.conv2d(branch_1, 384, [1,3], scope='Conv2d_0b_1*3')
							slim.conv2d(branch_1, 384, [3,1], scope='Conv2d_0c_3*1')],3)
				with tf.variable_scope('Branch_2'):
					branch_2 = slim.conv2d(net, 448, [1,1], scope='Conv2d_0a_1*1')
					branch_2 = slim.conv2d(branch_2, 384, [3,3], scope='Conv2d_0a_3*3')
					branch_2 = tf.conact ([
							slim.conv2d(branch_2, 384, [1,3], scope='Conv2d_0b_1*3')
							slim.conv2d(branch_2, 384, [3,1], scope='Conv2d_0c_3*1')],3)
				with tf.variable_scope('Branch_3'):
					branch_3 = slim.avg_pool2d(net, [3,3], scope='AvgPool_0a_3*3')	
					branch_3 = slim.conv2d(branch_3, 192, [1,1], scope='Conv2d_0a_1*1')
				net = tf.conact([branch_0, branch_1, branch_2], 3)
			return net, end_points

#实现inception v3网络的全局平均池化
def inception_v3(inputs,
				num_classes = 1000,
				is_trsining=True,
				dropout_keep_prob = 0.8,
				prediction_fn = slim.softmax,
				spatial_squeeze = True,
				reuse=None,
				scope='InceptionV3'):
	with tf.variable_scope(scope,'')