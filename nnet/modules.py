import numpy as np 

import tensorflow as tf
from tensorflow.contrib.layers.python.layers import batch_norm


def weight_init(shape, name=None, initializer=tf.contrib.layers.xavier_initializer()):
	"""
	Weights Initialization
	"""

	if name is None:
		name='W'
	
	W = tf.get_variable(name=name, shape=shape, 
		initializer=initializer)

	tf.summary.histogram(name, W)
	return W 


def bias_init(shape, name=None, constant=0.0):
	"""
	Bias Initialization
	"""

	if name is None:
		name='b'

	b = tf.get_variable(name=name, shape=shape, 
		initializer=tf.constant_initializer(constant))

	tf.summary.histogram(name, b)
	return b


def conv2d(input, kernel, stride=1, name=None, alpha=0.0,
	is_training=False, use_batch_norm=False, reuse=False, 
	initializer=tf.contrib.layers.xavier_initializer(),
	bias_constant=0.0, use_leak=False, activation=tf.nn.relu):
	
	"""
	2D convolution layer with relu activation
	"""

	if name is None:
		name='2d_convolution'

	with tf.variable_scope(name, reuse=reuse):
		W = weight_init(kernel, 'W', initializer)
		b = bias_init(kernel[3], 'b', bias_constant)

		strides=[1, stride, stride, 1]
		output = tf.nn.conv2d(input=input, filter=W, strides=strides, padding='SAME')
		output = output + b

		if activation is None:
			return output

		if use_batch_norm:
			if use_leak:
				return leaky_relu(batch_norm(output, is_training=is_training), alpha)
			else:
				return activation(batch_norm(output, is_training=is_training))
		else:
			if use_leak:
				return leaky_relu(output, alpha)
			else:
				return activation(output)


def deconv(input, kernel, output_shape, stride=1, name=None,
	activation=None, use_batch_norm=False, is_training=False,
	reuse=False, initializer=tf.contrib.layers.xavier_initializer(),
	bias_constant=0.0, use_leak=False, alpha=0.0):
	
	"""
	2D convolution layer with relu activation
	"""

	if name is None:
		name='de_convolution'

	with tf.variable_scope(name, reuse):
		W = weight_init(kernel, 'W', initializer)
		b = bias_init(kernel[2], 'b', bias_constant)

		strides=[1, stride, stride, 1]
		output = tf.nn.conv2d_transpose(value=input, filter=W, output_shape=output_shape, strides=strides)
		output = output + b

		if activation is None:
			return output

		if use_batch_norm:
			if use_leak:
				return leaky_relu(batch_norm(output, is_training=is_training), alpha)
			else:
				return activation(batch_norm(output, is_training=is_training))
		else:
			if use_leak:
				return leaky_relu(output, alpha)
			else:
				return activation(output)


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


def fully_connected_linear(input, output, name=None, reuse=False,
	bias_constant=0.0, initializer=tf.contrib.layers.xavier_initializer()):
	"""
	Fully-connected linear activations
	"""

	if name is None:
		name='fully_connected_linear'

	with tf.variable_scope(name, reuse):
		shape = input.get_shape()
		input_units = int(shape[1])

		W = weight_init([input_units, output], 'W', initializer)
		b = bias_init([output], 'b', bias_constant)

		output = tf.add(tf.matmul(input, W), b)
		return output

 
def fully_connected(input, output, is_training, activation=tf.nn.relu, 
	name=None, use_batch_norm=False, reuse=False, use_leak=False, alpha=0.2,
	initializer=tf.contrib.layers.xavier_initializer(), bias_constant=0.0):
	
	"""
	Fully-connected layer with induced non-linearity of 'relu'
	"""

	if name is None:
		name='fully_connected'

	output = fully_connected_linear(input=input, output=output, name=name, reuse=reuse,
		initializer=initializer, bias_constant=bias_constant)

	if activation is None:
		return output
	else:
		if use_batch_norm:
			if use_leak:
				return leaky_relu(batch_norm(output, is_training=is_training), alpha)
			else:
				return activation(batch_norm(output, is_training=is_training))
		else:
			if use_leak:
				return leaky_relu(output, alpha)
			else:
				return activation(output)


def dropout_layer(input, keep_prob=0.5, name=None):
	"""
	Dropout layer
	"""
	if name is None:
		name='Dropout'

	output = tf.nn.dropout(input, keep_prob=keep_prob)
	return output


def leaky_relu(input, alpha=0.2, name="lrelu"):
	"""
	Leaky ReLU
	"""
	return tf.maximum(input, alpha * input)