# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 22:34:21 2021

@author: ITU
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


def run_cnn():
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    learning_rate = 0.0001
    epochs = 10
    batch_size = 50

    x = tf.placeholder(tf.float32, [None, 784])
   
    x_shaped = tf.reshape(x, [-1, 28, 28, 1])

    y = tf.placeholder(tf.float32, [None, 10])

    def create_new_conv_layer(input_data, num_input_channels, num_filters, filter_shape, pool_shape, name):
        conv_filt_shape = [filter_shape[0], filter_shape[1], num_input_channels, num_filters]
        
        weights = tf.Variable(tf.truncated_normal(conv_filt_shape, stddev = 0.03), name = name+'_W')
        
        bias = tf.Variable(tf.truncated_normal([num_filters]), name=name+'_b')
        
        out_layer = tf.nn.conv2d(input_data, weights, [1,1,1,1], padding='SAME')
        
        out_layer += bias
        
        out_layer = tf.nn.relu(out_layer)
        
        ksize = [1, pool_shape[0], pool_shape[1], 1]
        strides = [1,2,2,1]
        out_layer = tf.nn.max_pool(out_layer, ksize = ksize, strides=strides, padding = 'SAME')
        return out_layer

    layer1 = create_new_conv_layer(x_shaped, 1, 32, [5, 5], [2, 2], name='layer1')
    layer2 = create_new_conv_layer(layer1, 32, 64, [5, 5], [2, 2], name='layer2')

    flattened = tf.reshape(layer2, [-1, 7 * 7 * 64])
    
    wd1 = tf.Variable(tf.truncated_normal([7 * 7 * 64, 1000], stddev = 0.03), name = 'wd1')
    bd1 = tf.Variable(tf.truncated_normal([1000], stddev = 0.01), name = 'bd1')
    dense_layer1 = tf.matmul(flattened, wd1) + bd1
    dense_layer1= tf.nn.relu(dense_layer1)
    
    wd2 = tf.Variable(tf.truncated_normal([1000, 10], stddev = 0.03), name = 'wd2')
    bd2 = tf.Variable(tf.truncated_normal([10], stddev = 0.01), name = 'bd2')
    dense_layer2 = tf.matmul(dense_layer1, wd2) + bd2
    y_ = tf.nn.softmax(dense_layer2)
    