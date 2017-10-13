import io
import os
import sys
import time
import argparse
from time import gmtime, strftime

import nnet
import utils

import nnet
from options import Options 

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def main():
	print 'Beginning Run\n'
	net = nnet.VAE(Options(), True)
	print 'Training the network...\n'
	net.train()
	print 'Done training the network...\n'
		

if __name__ == '__main__':
	try:
		main()
	except Exception as E:
		print E