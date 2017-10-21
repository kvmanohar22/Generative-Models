from options import Options
from six.moves import cPickle
import numpy as np
import os
from skimage import io
from scipy.misc import imsave

from tensorflow.examples.tutorials.mnist import input_data

class Dataset(object):
	def __init__(self, opts):
		self.opts = opts

		if self.opts.dataset == 'CIFAR':
			imgs_file = os.path.join(os.path.join(self.opts.root_dir, self.opts.dataset_dir), 'data_batch_2')
			with open(imgs_file, 'rb') as f:
				dict = cPickle.load(f)
			imgs = dict['data']
			self.images = np.reshape(imgs, (10000, 3, 32, 32)).transpose(0, 2, 3, 1).astype('uint8')
			self.test_images = np.zeros((self.opts.test_size, 32, 32, 3))
			self.images = self.images.astype('float')
			if self.opts.model == "vae":
				self.images /= 255.0
			else:
				self.images = self.images / 127.5 - 1.
			np.random.shuffle(self.images)
			self.test_images = self.images[:self.opts.test_size, :, :, :]
		else:
			data_dir = os.path.join(self.opts.root_dir, self.opts.dataset_dir)
			mnist = input_data.read_data_sets(data_dir, one_hot=True)
			images = mnist.train.images
			self.labels = mnist.train.labels
			self.images = np.reshape(images, (55000, 28, 28, 1)).astype('float')
			self.test_images = np.zeros((self.opts.test_size, 28, 28, 1))
			np.random.shuffle(self.images)
			self.test_images = self.images[:self.opts.test_size, :, :]
		self.save_batch_images(self.test_images, [8, 8], "target.jpg")

	def load_batch(self, start_idx, end_idx):
		if self.opts.use_labels:
			# print 'here'
			return self.images[start_idx:end_idx], self.labels[start_idx:end_idx]
		else:
			return self.images[start_idx:end_idx]

	def save_batch_images(self, images, grid, img_file, normalized=False):
		h = images.shape[1]
		w = images.shape[2]
		# if normalized and self.opts.model != 'gan':
		# 	images = images * 255.0
		# elif normalized and self.opts.model == 'gan':
		# 	images = images * 127.5 + 127.5
		num = images.shape[0]
		if self.opts.dataset == "CIFAR":
			imgs = np.zeros((h * grid[0], h * grid[1], self.opts.channels))
		else:
			imgs = np.zeros((h * grid[0], h * grid[1]))

		for idx, image in enumerate(images):
			i = idx % grid[1]
			j = idx / grid[1]
			if self.opts.dataset == "CIFAR":
				imgs[i*w:w*(i+1), j*h:(j+1)*h, :] = image
			else:
				imgs[i*w:w*(i+1), j*h:(j+1)*h] = np.reshape(image, (28, 28))
		img_file_path = os.path.join(self.opts.root_dir, self.opts.sample_dir, img_file)
		imsave(img_file_path, imgs)
		# with open(self.opts.root_dir+self.opts.sample_dir+img_file.split('.')[0]+'-pkl', 'wb') as f:
		# 	cPickle.dump(imgs, f)
