from options import Options
from six.moves import cPickle
import numpy as np
from skimage import io
from scipy.misc import imsave

class Dataset(object):
	def __init__(self):
		self.opts = Options()
		imgs_file = self.opts.root_path+'/data_batch_1'
		with open(imgs_file, 'rb') as f:
			dict = cPickle.load(f)
		imgs = dict['data']
		self.images = np.reshape(imgs, (10000, 3, 32, 32)).transpose(0, 2, 3, 1).astype('uint8') / 127.5 - 1.
		self.test_images = np.zeros((64, 32, 32, 3))
		np.random.shuffle(self.images)
		self.test_images = self.images[:64, :, :, :]
		self.save_batch_images(self.test_images, [8, 8], "target.jpg")

	def load_batch(self, start_idx, end_idx):
		return self.images[start_idx:end_idx]

	def save_batch_images(self, images, grid, img_file):
		h = images.shape[1]
		w = images.shape[2]
		num = images.shape[0]
		imgs = np.zeros((h * grid[0], h * grid[1], 3))

		for idx, image in enumerate(images):
			i = idx % grid[1]
			j = idx / grid[1]
			imgs[i*w:w*(i+1), j*h:(j+1)*h, :] = image
		imsave(self.opts.base_img_dir+img_file, imgs)
		with open(self.opts.base_img_dir+img_file.split('.')[0]+'-pkl', 'wb') as f:
			cPickle.dump(imgs, f)
