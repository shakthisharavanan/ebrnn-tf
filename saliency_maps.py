"""
Generate saliency maps using ebrnn method

Author: Shakthi Duraimurugan

"""


import sys
import os
from os import path
import time
from time import sleep

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pylab as plt

from tqdm import tqdm, trange, tqdm_notebook, tnrange
import glob
import time
import subprocess as sp
import pdb

slim_dir = "/mnt/workspace/models/research/slim/"
sys.path.insert(0, slim_dir)
from nets import vgg
from preprocessing import vgg_preprocessing

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'



class cnn_lstm(object):
	def __init__(self, image_size, batch_size, image_mean, checkpoints_dir, n_frames, n_hidden, n_classes):
		# Set Paths
		self.image_size = image_size
		self.batch_size = batch_size
		self.image_mean = image_mean

		# Define the VGG16 Network
		slim = tf.contrib.slim
		g1 = tf.Graph()
		with g1.as_default():
			self.input_batch = tf.placeholder(dtype=tf.uint8, shape=(None,240,320,3))
			resized_images = tf.image.resize_images(self.input_batch, [self.image_size,self.image_size])
			channels = tf.split(axis=3, num_or_size_splits=3, value=resized_images)
			for i in range(3):
				channels[i] -= self.image_mean[i]
			normalized_images = tf.concat(axis=3, values=channels)

			with slim.arg_scope(vgg.vgg_arg_scope()):
				outputs, end_points = vgg.vgg_16(normalized_images,num_classes=1000, is_training=False)
			
			self.final_conv = end_points['vgg_16/conv5/conv5_3']
			self.fc7 = end_points['vgg_16/fc7']
			self.probablities = tf.nn.softmax(outputs)

			self.init_fn = slim.assign_from_checkpoint_fn(os.path.join(checkpoints_dir, 'vgg_16.ckpt'),slim.get_model_variables('vgg_16'))
			self.cnn_sess = tf.Session()

		# Define LSTM Network
		g2 = tf.Graph()
		with g2.as_default():
			self.x = tf.placeholder(tf.float32, (None, n_frames, 4096))
			self.sequence_length = tf.placeholder(tf.int32, shape=None)
			lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden, activation = tf.nn.relu)
			seq_out, states = tf.nn.dynamic_rnn(cell = lstm_cell, sequence_length = self.sequence_length, inputs = self.x, dtype = tf.float32)
			w = tf.get_variable(name = "lstm_weights", shape = [n_hidden, n_classes], 
					initializer = tf.contrib.layers.xavier_initializer(), trainable = True)
			b = tf.get_variable(name = "lstm_bias", shape = [n_classes], 
					initializer = tf.contrib.layers.xavier_initializer(), trainable = True)
			final_out = seq_out[:, -1, :] # last time step's output
			self.out = tf.matmul(final_out, w) + b
			self.lstm_sess = tf.Session()
			self.lstm_saver = tf.train.Saver()



def read_video(video_path):
	video_frames = []
	command = [ 'ffmpeg', '-i', video_path, '-f', 'image2pipe', '-pix_fmt', 'rgb24', '-vcodec', 'rawvideo', '-']
	pipe = sp.Popen(command, stdout = sp.PIPE, stderr = open(os.devnull, 'w'), bufsize=10**8)
	while True:
	    raw_image = pipe.stdout.read(240*320*3)
	    if len(raw_image) != 240*320*3:
	        break;
	    # transform the byte read into a numpy array
	    image =  np.fromstring(raw_image, dtype='uint8')
	    image = image.reshape((240,320,3))
	    video_frames.append(image)
	    # throw away the data in the pipe's buffer.
	    pipe.stdout.flush()
	video_frames = np.asarray(video_frames, dtype=np.uint8)
	return video_frames

def trim_video(video_frames, sequence_length = 30):
	offset = video_frames.shape[0]//sequence_length
	trimmed_video_frames = (video_frames[::offset])[:sequence_length]
	return trimmed_video_frames


if __name__ == "__main__":

	# Set Paths
	checkpoints_dir = "/mnt/workspace/models/checkpoints/"
	video_path = "./v_GolfSwing_g01_c01.avi"
	label_dir = "/mnt/workspace/datasets/ucf101/ucf24/labels/"
	labels = [x.replace(label_dir,"") for x in sorted(glob.glob(label_dir+"*"))] # ['Basketball', 'BasketballDunk', 'Biking', 'CliffDiving', 'CricketBowling', 'Diving', 'Fencing', 'FloorGymnastics', 'GolfSwing', 'HorseRiding', 'IceDancing', 'LongJump', 'PoleVault', 'RopeClimbing', 'SalsaSpin', 'SkateBoarding', 'Skiing', 'Skijet', 'SoccerJuggling', 'Surfing', 'TennisSwing', 'TrampolineJumping', 'VolleyballSpiking', 'WalkingWithDog']


	# Set some paraeters
	image_size = vgg.vgg_16.default_image_size
	batch_size = 16
	image_mean = [123.68, 116.779, 103.939]  
	sampling_rate = 30 # hz
	n_frames = 30 # No. of frames
	n_hidden = 1024

	# Read video and trim to 30 frames
	video_frames = read_video(video_path)
	trimmed_video_frames = trim_video(video_frames)

	# Create the cnn-lstm model
	model = cnn_lstm(image_size, batch_size, image_mean, checkpoints_dir, n_frames, n_hidden, 24)

	# Load vgg16 weights and extract features
	video_features = []
	slim = tf.contrib.slim
	model.init_fn(model.cnn_sess)
	for start, end in zip(range(0, n_frames, batch_size), range(batch_size, n_frames + batch_size, batch_size)):
		fc7 = model.cnn_sess.run(model.fc7, feed_dict = {model.input_batch: trimmed_video_frames[start:end]})
		video_features = video_features + list(fc7)
	video_features = np.array(video_features)[:,0,0,:] # from (30, 1, 1, 4096) to (30, 4096)
	video_features = np.expand_dims(video_features, axis = 0) # (1, 30, 4096)

	# Load lstm model weights and get predictions
	model.lstm_saver.restore(model.lstm_sess, tf.train.latest_checkpoint('./saved_models/'))
	pred = model.lstm_sess.run(model.out, feed_dict = {model.x: video_features, model.sequence_length: n_frames})
	index = np.argmax(pred)
	print("Prediction: {0}".format(labels[index]))


	""" Now perform EBRNN steps """
	# Get gradient of LSTM Layer
	dy_dx = model.lstm_sess.run(tf.gradients(model.out[:, index], model.x), feed_dict = {model.x: video_features, model.sequence_length: n_frames}) #shape (1, 30, 4096)


	pass




