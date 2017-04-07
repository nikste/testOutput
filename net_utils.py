import tensorflow as tf

'''
convenience functions for creating networks
'''

def weight_variable(shape, name='notnamed'):
  initial = tf.truncated_normal(shape, stddev=0.1, name=name)
  return tf.Variable(initial)

def bias_variable(shape, name='notnamed'):
  initial = tf.constant(0.1, shape=shape, name=name)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x, padding='SAME'):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding=padding)

def convolutional(x, shape, name=""):
    W_conv = weight_variable(shape, name=name+"_weight")
    b_conv = bias_variable([shape[-1]], name=name+"_bias")
    return tf.nn.relu(conv2d(x, W_conv) + b_conv, name=name)