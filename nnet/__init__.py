import tensorflow as tf 
import numpy as np


class Discriminator(object):
	"""
	Discriminator part of GANs
	"""

	def __init__(self, image):
		self.image = image


class Generator(object):
	"""
	Generator part of GANs
	"""

	def __init__(self, z):
		self.z = z


class Encoder(object):
	"""
	Encoder of an Autoencoder
	"""

	def __init__(self, image):
		self.image = image


class Decoder(object):
	"""
	Decoder of an Autoencoder
	"""

	def __init__(self, latent_vector):
		self.latent_vector = latent_vector

