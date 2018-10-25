# -*- coding: utf-8 -*-
# authr: Caofang
# Time : 2018/10/25 22:12
# @File : ResNet.py

import collections
import tensorflow as tf
slim = tf.contrib.slim

def subsample(inputs, factor, scope=None):
    if factor == 1:
        return inputs
    else:
        return slim.max_pool2d(inputs, [1,1], stride=factor, scope=scope)



