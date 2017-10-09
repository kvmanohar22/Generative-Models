class Options(object):
	"""
	Holds all the necessary hyper-parameters and other variables
	"""
	self.batch_size = 1
	self.g_dims = 64
	self.d_dims = 64
	self.image_shape = 32
	self.latent_shape = 100

	self.root_path = ""
	self.mean_img = self.root_path+"mean_img.png"