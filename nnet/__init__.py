import tensorflow as tf 
import numpy as np


import modules as model
from options import Options


class Discriminator(object):
	"""
	Discriminator part of GANs
	"""

	def __init__(self, image, opts, reuse=False, is_training=False):
		self.image = image
		self.reuse = reuse
		self.is_training = is_training
		self.opts = opts
		self.dims = opts.d_dims
		self.network()

	def network(self):
		conv1 = model.conv2d(self.image, [5,5,3,self.dims], 2, "conv1", is_training, False, self.reuse)
		conv2 = model.conv2d(conv1, [5,5,self.dims,self.dims*2], 2, "conv2", is_training, True, self.reuse)
		conv3 = model.conv2d(conv2, [5,5,self.dims*2,self.dims*4], 2, "conv3", is_training, True, self.reuse)
		full4 = model.fully_connected(tf.reshape(conv3, [self.opts.batch_size, -1]), 1, is_training, None, "full4", False, self.reuse)
		return tf.nn.softmax(full4), full4


class Generator(object):
	"""
	Generator part of GANs
	"""

	def __init__(self, z, opts, is_training):
		self.z = z
		self.opts = opts
		self.dims = opts.g_dims
		self.is_training = is_training
		self.network()

	def network(self):
		fulll1 = model.fully_connected(self.z, self.dims*4, is_training, "full1", False, self.reuse)
		dconv2 = model.deconv(tf.reshape(fulll1, [-1, 4, 4, self.dims*4]), [-1, 8, 8, self.dims*4], 2, "dconv2", is_training, False)
		dconv3 = model.deconv(dconv2, [-1, 16, 16, self.dims*2], 2, "dconv3", is_training, False)
		dconv4 = model.deconv(dconv3, [-1, 32, 32, self.dims], 2, "dconv4", is_training, False)
		return tf.nn.tanh(dconv4)


class Encoder(object):
	"""
	Encoder part of Autoencoder
	"""

	def __init__(self, image):
		self.image = image


class Decoder(object):
	"""
	Decoder part of Autoencoder
	"""

	def __init__(self, latent_vector):
		self.latent_vector = latent_vector


class Train(object):
	"""
	Main Class which drives the training process
	"""
	def __init__(self, opts):
		self.sess = tf.Session()
		self.opts = opts
		self.img = opts.image_shape
		self.imgs = tf.placeholder(tf.float32, [None, self.img, self.img, 3], name="True_images")
		self.z = tf.placeholder(tf.float32, [None, self.opts.latent_shape], name="noise")
		self.train()

	def train(self):
		