import tensorflow as tf 
import numpy as np
import time

import modules as model
from options import Options
from utils import Dataset

class GAN(object):

	def __init__(self, image, opts, reuse=False, is_training=False):
		self.image = image
		self.reuse = reuse
		self.is_training = is_training
		self.opts = opts
		self.dims = opts.d_dims
		self.pred = self.Discriminator()
		self.generated_imgs = self.Generator()

	def Discriminator(self):
		"""
		Discriminator part of GAN
		"""

		with tf.variable_scope("discriminator"):
			conv1 = model.conv2d(self.image, [5,5,3,self.dims], 2, "conv1", is_training, False, self.reuse)
			conv2 = model.conv2d(conv1, [3,3,self.dims,self.dims*2], 2, "conv2", is_training, True, self.reuse)
			conv3 = model.conv2d(conv2, [3,3,self.dims*2,self.dims*4], 2, "conv3", is_training, True, self.reuse)
			full4 = model.fully_connected(tf.reshape(conv3, [self.opts.batch_size, -1]), 1, is_training, None, "full4", False, self.reuse)
		return tf.nn.softmax(full4), full4

	def Generator(self):
		"""
		Generator part of GAN
		"""

		with tf.variable_scope("generator"):
			fulll1 = model.fully_connected(self.z, self.dims*4*4*4, is_training, tf.nn.relu, "full1", False, self.reuse)
			dconv2 = model.deconv(tf.reshape(fulll1, [-1, 4, 4, self.dims*4]), [8, 8, self.dims*2, self.dims*4], 2, "dconv2", is_training, False)
			dconv3 = model.deconv(dconv2, [16, 16, self.dims, self.dims*2], 2, "dconv3", is_training, False)
			dconv4 = model.deconv(dconv3, [32, 32, self.dims, 3], 2, "dconv4", is_training, False)
		return tf.nn.tanh(dconv4)

	def loss(self):
		pass

	def train(self):
		pass


class VAE(object):
	"""
	Variatinoal Autoencoder
	"""

	def __init__(self, opts, is_training):
		self.images = tf.placeholder(tf.float32, [None, 32, 32, 3], "images") # 32x32x3
		self.lr = tf.placeholder(tf.float32, [], "learning_rate")
		self.is_training = is_training
		self.opts = opts
		self.mean, self.std = self.encoder()

		unit_gauss = tf.random_normal([self.opts.batch_size, self.opts.encoder_vec_size])
		self.z = self.mean + (self.std * unit_gauss)

		self.generated_imgs = self.decoder(self.z)
		self.l1, self.l2 = self.loss()
		self.l = self.l1+self.l2
		self.sess = tf.Session()
		self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.l)
		self.init = tf.global_variables_initializer()
		self.saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)
		tf.summary.scalar('Encoder loss', self.l1)
		tf.summary.scalar('Decoder loss', self.l2)
		tf.summary.scalar('Total loss', self.l)
		self.summaries = tf.summary.merge_all()
		self.writer = tf.summary.FileWriter(self.opts.summaries+str('/train/'), self.sess.graph)

	def encoder(self):
		"""
		Encoder to generate the `latent vector`
		"""

		dims = self.opts.g_dims
		with tf.variable_scope("encoder"):
			conv1 = model.conv2d(self.images, [3, 3, 3, dims], 2, "conv1") # 16x16x64
			conv2 = model.conv2d(conv1, [3, 3, dims, dims * 2], 2, "conv2") # 8x8x128
			conv3 = model.conv2d(conv2, [3, 3, dims * 2, dims * 4], 2, "conv3") # 4x4x256
			full4 = model.fully_connected(tf.reshape(conv3, [-1, 4*4*256]), self.opts.encoder_vec_size * 2, self.is_training, None, "full4") # 40
			encoder_mean = full4[:, :self.opts.encoder_vec_size] # 20
			encoder_stds = full4[:, self.opts.encoder_vec_size:] # 20
		return encoder_mean, encoder_stds

	def decoder(self, z):
		"""
		Generate images from the `latent vector`
		"""

		dims = self.opts.g_dims
		with tf.variable_scope("decoder"):
			full1 = model.fully_connected(z, 4*4*256, self.is_training, tf.nn.relu, "full1") # 4x4x256
			dconv2 = model.deconv(tf.reshape(full1, [-1, 4, 4, 256]), [3,3,128,256], [self.opts.batch_size, 8, 8, 128], 2, "dconv2", tf.nn.relu) # 8x8x128
			dconv3 = model.deconv(dconv2, [3,3,64,128], [self.opts.batch_size, 16, 16, 64], 2, "dconv3", tf.nn.relu) # 16x16x64
			dconv4 = model.deconv(dconv3, [3,3,3,64], [self.opts.batch_size, 32, 32, 3], 2, "dconv4", tf.nn.sigmoid) # 32x32x3
			return tf.reshape(dconv4, [-1, 32*32*3])

	def loss(self):
		encoder_loss = 0.5 * tf.reduce_sum(tf.square(self.mean)+tf.square(self.std)-tf.log(tf.square(self.std))-1, 1)
		img_flat = tf.reshape(self.images, [-1, 32*32*3])
		decoder_loss = -tf.reduce_sum(img_flat * tf.log(self.generated_imgs) + (1-img_flat)*tf.log(1-self.generated_imgs), 1)

		return tf.reduce_mean(encoder_loss), tf.reduce_mean(decoder_loss)

	def train(self):
		utils = Dataset()
		lr = self.opts.base_lr
		self.sess.run(self.init)
		for iteration in xrange(1, self.opts.MAX_iterations):
			batch_num = 0
			for batch_begin, batch_end in zip(xrange(0, self.opts.train_size, self.opts.batch_size), \
				xrange(self.opts.batch_size, self.opts.train_size, self.opts.batch_size)):
				begin_time = time.time()
				batch_imgs = utils.load_batch(batch_begin, batch_end)
				_, l1, l2, summary = self.sess.run([self.optimizer , self.l1, self.l2, self.summaries], feed_dict={self.images:batch_imgs, self.lr:lr})

				batch_num += 1
				self.writer.add_summary(summary, iteration * (self.opts.train_size/self.opts.batch_size) + batch_num)
				if batch_num % self.opts.display == 0:
					log  = '-'*20
					log += '\nIteration: {}/{}'.format(iteration, self.opts.MAX_iterations)
					log += ' Batch Number: {}/{}'.format(batch_num, self.opts.train_size/self.opts.batch_size)
					log += ' Batch Time: {}\n'.format(time.time() - begin_time)
					log += ' Learning Rate: {}\n'.format(lr)
					log += ' Encoder Loss: {}\n'.format(l1)
					log += ' Decoder Loss: {}\n'.format(l2)
					print log
				if iteration % self.opts.lr_decay == 0 and batch_num == 1:
					lr = lr * self.opts.lr_decay_factor
				if iteration % self.opts.ckpt_frq == 0 and batch_num == 1:
					self.saver.save(self.sess, self.opts.ckpt_dir+"/{}_{}_{}".format(iteration, lr, l1+l2))
				if iteration % self.opts.generate_frq == 0 and batch_num == 1:
					generate_imgs = utils.test_images
					imgs = self.sess.run(self.generated_imgs, feed_dict={self.images:generate_imgs, self.lr:lr})
					imgs = np.reshape(imgs, (64, 3, 32, 32)).transpose(0, 2, 3, 1)
					utils.save_batch_images(imgs, [8, 8], str(iteration)+".jpg")

	def test(self, image):
		latest_ckpt = tf.train.latest_checkpoint(self.opts.ckpt_dir)
		tf.saver.restore(self.sess, latest_ckpt)
		

