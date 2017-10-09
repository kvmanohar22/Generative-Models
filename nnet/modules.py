import numpy as np 

import tensorflow as tf
from tensorflow.contrib.layers.python.layers import batch_norm


def weight_init(shape, name=None):
	"""
	Weights Initialization
	"""

	if name is None:
		name='W'
	
	W = tf.get_variable(name=name, shape=shape, 
		initializer=tf.contrib.layers.xavier_initializer())
	return W 


def bias_init(shape, name=None):
	"""
	Bias Initialization
	"""

	if name is None:
		name='b'

	b = tf.get_variable(name=name, shape=shape, 
		initializer=tf.constant_initializer(0.2))
	return b


def conv2d(input, is_training, kernel, stride=1, 
	name=None, use_batch_norm=False, reuse=None):
	
	"""
	2D convolution layer with relu activation
	"""

	if name is None:
		name='2d_convolution'

	with tf.variable_scope(name):
		W = weight_init(kernel, 'W')
		b = bias_init(kernel[3], 'b')	

		strides=[1, stride, stride, 1]
		activation = tf.nn.conv2d(input=input, filter=W, strides=strides, padding='SAME')
		output = activation + b

		if use_batch_norm:
			return tf.nn.relu(batch_norm(output, is_training=is_training))
		else:
			return tf.nn.relu(output)


def deconv(input, is_training, kernel, stride=1, 
	name=None, reuse=None):
	
	"""
	2D convolution layer with relu activation
	"""

	if name is None:
		name='de_convolution'

	with tf.variable_scope(name):
		W = weight_init(kernel, 'W')
		b = bias_init(kernel[3], 'b')	

		strides=[1, stride, stride, 1]
		activation = tf.nn.conv2d_transpose(input=input, filter=W, strides=strides)
		output = activation + b


def max_pool(input, kernel=3, stride=2, name=None):
	"""	
	Max-pool
	"""

	if name is None: 
		name='max_pool'

	ksize = [1, kernel, kernel, 1]
	strides = [1, stride, stride, 1]
	output = tf.nn.max_pool(input, ksize=ksize, strides=strides, padding='SAME')

	return output


def fully_connected_linear(input, output, name=None, reuse=None):
	"""
	Fully-connected linear activations
	"""

	if name is None:
		name='fully_connected_linear'

	with tf.variable_scope(name):
		shape = input.get_shape()
		input_units = int(shape[1])

		W = weight_init([input_units, output], 'W')
		b = bias_init([output], 'b')			

		output = tf.add(tf.matmul(input, W), b)
		return output

 
def fully_connected(input, output, is_training, activation=tf.nn.relu, name=None, 
	use_batch_norm=False, reuse=None):
	
	"""
	Fully-connected layer with induced non-linearity of 'relu'
	"""

	if name is None:
		name='fully_connected'

	with tf.variable_scope(name):
		linear_output = fully_connected_linear(input=input, output=output)

		if activation is None:
			return linear_output
		else:
			if use_batch_norm:
				output = activation(batch_norm(linear_output, is_training=is_training))
				return output
			else:
				return activation(linear_output)


def dropout_layer(input, keep_prob=0.5, name=None):
	"""
	Dropout layer
	"""
	if name is None:
		name='Dropout'

	output = tf.nn.dropout(input, keep_prob=keep_prob)
	return output
