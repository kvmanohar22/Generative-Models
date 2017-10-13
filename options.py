class Options(object):
	"""
	Holds all the necessary hyper-parameters and other variables
	"""
	def __init__(self):
		self.batch_size = 64
		self.g_dims = 64
		self.d_dims = 64
		self.image_shape = 32
		self.latent_shape = 100
		self.MAX_iterations = 15000
		self.encoder_vec_size = 20
		self.display = 12
		self.base_lr = 0.0001
		self.root_path = "/home/kv/GAN/data"
		self.lr_decay = 100
		self.lr_decay_factor = 0.5
		self.ckpt_dir = "/home/kv/GAN/ckpt"
		self.ckpt_frq = 25
		self.train_size = 10000
		self.summaries = '/home/kv/GAN/Summary'
		self.generate_frq = 20
		self.base_img_dir = '/home/kv/GAN/Images/'
