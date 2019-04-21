"""
Load and test LSTM Model
"""

"""
Train LSTM with vgg16 features

Author: Shakthi Duraimurugan

"""

import os
import sys
from os import path
import time
from time import sleep

import tensorflow as tf
import numpy as np
# import cv2
# import matplotlib.pyplot as plt
import pylab as plt
from sklearn.utils import shuffle

from tqdm import tqdm, trange, tqdm_notebook, tnrange
import glob
import time
import pandas as pd
import h5py
import pickle as pkl
import subprocess as sp
import pdb

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

slim_dir = "/mnt/workspace/models/research/slim/"
checkpoints_dir = "/mnt/workspace/models/checkpoints/"
sys.path.insert(0, slim_dir)
from nets import vgg
from preprocessing import vgg_preprocessing


def read_video(x, y, sequence_length):
	"""
	Read features and convert to 30 frames for every video
	"""
	features_batch = []
	for i in range(len(x)):
		features = np.load(x[i])[:,0,0,:]
		offset = features.shape[0]//sequence_length
		# if(features.shape[0]<40):
		# 	print(x[i])
		trimmed_features = (features[::offset])[:sequence_length]
		features_batch.append(trimmed_features)

	return features_batch

def lstm_model(x, batch_size, sequence_length, n_hidden, n_classes):
	lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden, activation = tf.nn.relu)
	seq_out, states = tf.nn.dynamic_rnn(cell = lstm_cell, sequence_length = sequence_length, inputs = x, dtype = tf.float32)
	w = tf.get_variable(name = "lstm_weights", shape = [n_hidden, n_classes], 
			initializer = tf.contrib.layers.xavier_initializer(), trainable = True)
	b = tf.get_variable(name = "lstm_bias", shape = [n_classes], 
			initializer = tf.contrib.layers.xavier_initializer(), trainable = True)
	final_out = seq_out[:, -1, :] # last time step's output
	out = tf.matmul(final_out, w) + b
	return out


if __name__ == '__main__':

	# Set Paths
	dataset_dir = "/mnt/workspace/datasets/UCF-101/"
	label_dir = "/mnt/workspace/datasets/ucf101/ucf24/labels/"
	checkpoints_dir = "/mnt/workspace/models/checkpoints/"
	extracted_features_dir = "/mnt/workspace/ebrnn-tf/fc7_features/"
	labels = [x.replace(label_dir,"") for x in sorted(glob.glob(label_dir+"*"))] # ['Basketball', 'BasketballDunk', 'Biking', 'CliffDiving', 'CricketBowling', 'Diving', 'Fencing', 'FloorGymnastics', 'GolfSwing', 'HorseRiding', 'IceDancing', 'LongJump', 'PoleVault', 'RopeClimbing', 'SalsaSpin', 'SkateBoarding', 'Skiing', 'Skijet', 'SoccerJuggling', 'Surfing', 'TennisSwing', 'TrampolineJumping', 'VolleyballSpiking', 'WalkingWithDog']
	# pdb.set_trace()

	# Define some hyperparameters
	batch_size = 64
	sample_rate = 30 # Hz
	n_frames = 30 # No. of frames
	epochs = 10
	n_hidden = 1024
	lr = 1e-5 # 1e-3 for tanh lstm activation
	# lr = 1e-3

	all_videos = []
	all_labels = []
	for i in range(len(labels)):
		videos = sorted(glob.glob(extracted_features_dir+labels[i]+"/*"))
		all_videos += videos
		all_labels += ([i] * len(videos)) # set labels as indices
	# pdb.set_trace()

	# Create test train split
	all_videos, all_labels = shuffle(all_videos, all_labels)
	train_ratio = 0.8
	train_len = int(train_ratio * len(all_videos))
	train_videos = all_videos[:train_len]
	train_labels = all_labels[:train_len]
	test_videos = all_videos[train_len:]
	test_labels = all_labels[train_len:]

	n_train = len(train_videos)
	n_test = len(test_videos)
	# labels = sorted(set(all_labels)) # ['Basketball', 'BasketballDunk', 'Biking', 'CliffDiving', 'CricketBowling', 'Diving', 'Fencing', 'FloorGymnastics', 'GolfSwing', 'HorseRiding', 'IceDancing', 'LongJump', 'PoleVault', 'RopeClimbing', 'SalsaSpin', 'SkateBoarding', 'Skiing', 'Skijet', 'SoccerJuggling', 'Surfing', 'TennisSwing', 'TrampolineJumping', 'VolleyballSpiking', 'WalkingWithDog']

	# Define placeholders
	x = tf.placeholder(tf.float32, (1, n_frames, 4096))
	y = tf.placeholder(tf.uint8, (None))
	y_one_hot = tf.one_hot(y, len(labels))
	sequence_length = tf.placeholder(tf.int32, shape=None)
	logits = lstm_model(x, batch_size, sequence_length, n_hidden, len(labels))
	cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_one_hot, logits=logits)
	loss_operation = tf.reduce_mean(cross_entropy)
	optimizer = tf.train.AdamOptimizer(learning_rate = lr)
	training_operation = optimizer.minimize(loss_operation)

	# Gradient clipping code
	# gvs = optimizer.compute_gradients(loss_operation)
	# capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
	# training_operation = optimizer.apply_gradients(capped_gvs)

	# gradients, variables = zip(*optimizer.compute_gradients(loss_operation))
	# gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
	# training_operation = optimizer.apply_gradients(zip(gradients, variables))

	correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y_one_hot, 1))
	accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	saver = tf.train.Saver()

	# pdb.set_trace()

	# Training procedure
	with tf.Session() as sess:
		saver.restore(sess, tf.train.latest_checkpoint('./saved_models/'))
		# sess.run(tf.global_variables_initializer())

		# Load a video to test
		batch_x = test_videos[0:1]
		batch_y = test_labels[0:1]
		print(batch_y)
		features_x = read_video(batch_x, batch_y, n_frames)
		pred = sess.run(logits, feed_dict = {x: features_x, y: batch_y, sequence_length: n_frames})
		pdb.set_trace()
		index = np.argmax(pred)
		print("Prediction: {0}".format(labels[index]))

		# Find dy/dx
		dy_dx = sess.run(tf.gradients(logits[:, index], x), feed_dict = {x: features_x, y: batch_y, sequence_length: n_frames}) #shape (1, 30, 4096)

		#Normalize for each 4096 neurons
		layer_norm = dy_dx[0]/(dy_dx[0].sum(axis = 1)[:, None])

		# # Do Excitation backprop

		# weights = tf.trainable_variables()
		# weights_val = sess.run(weights)

		# # Get lstm weights
		# wi, wc, wf, wo = np.split(lstm_weights_val, 4, axis = 1)

		# wxi = wi[:4096, :]
		# whi = wi[4096:, :]

		# wxC = wC[:4096, :]
		# whC = wC[4096:, :]

		# wxf = wf[:4096, :]
		# whf = wf[4096:, :]

		# wxo = wo[:4096, :]
		# who = wo[4096:, :]

	# # Do Excitation backprop
	# image_size = vgg.vgg_16.default_image_size

	# with tf.Graph().as_default():
	# 	# x = tf.placeholder(dtype = tf.float32, shape = (image_size, image_size, 3))
	# 	# normalized_image = vgg_preprocessing.preprocess_image(x, image_size, image_size, is_training=False)
	# 	# normalized_images = tf.expand_dims(normalized_image, 0)
	# 	# with slim.arg_scope(vgg.vgg_arg_scope()):
	# 	# 	output, endpoints = vgg.vgg_16(normalized_images, num_classes = 1000, is_training = False)
	# 	# 	probabilities = tf.nn.softmax(output)
	# 	init_fn = slim.assign_from_checkpoint_fn(os.path.join(checkpoints_dir, 'vgg_16.ckpt'), slim.get_model_variables('vgg_16'))
	# 	# Run in a session
	# 	with tf.Session() as sess:
	# 		init_fn(sess)
	# 		probability, layers = sess.run([probabilities, endpoints], feed_dict = {x: image})
	# 		layer_names, layer_activations = zip(*list(layers.items()))
	# 		# pdb.set_trace()
	# 		probability = probability[0, 0:]
	# 		sorted_inds = [i[0] for i in sorted(enumerate(-probability), key=lambda x:x[1])]

	# 		# pdb.set_trace()

	# 		for i in range(10):
	# 			index = sorted_inds[i]
	# 			print('Probability %0.2f%% => [%s]' % (probability[index] * 100, labels[index]))
	# 		# plt.imshow(image)
	# 		# plt.annotate('{0}: {1: 0.2f}%'.format(labels[sorted_inds[0]], probability[sorted_inds[0]]*100), xy = (0.02, 0.95), xycoords = 'axes fraction')
	# 		# plt.show()

	# 		weights = tf.trainable_variables()
	# 		weights_val = sess.run(weights)

	# 		# Set MWP as a dict
	# 		P = {}

	# 		# Set one hot vector for the winning class
	# 		p = np.zeros((1000,1))
	# 		p[sorted_inds[0], 0] = 1
	# 		P['fc8'] = np.copy(p) # 1000 X 1


	# 		""" For fc7 MWP """
	# 		# Get fc8 weights
	# 		fc8_weights = np.copy((weights_val[-2])[0,0]) # 4096 X 1000

	# 		# Get fc7 activations
	# 		fc7_activations = np.copy((layer_activations[-2])[0,0]).T # 4096 X 1

	# 		# Calculate MWP of fc7 using Eq 10 in paper
	# 		fc8_weights = fc8_weights.clip(min = 0) # threshold weights at 0
	# 		m = np.dot(fc8_weights.T, fc7_activations) # 1000 x 1
	# 		n = P['fc8'] / m # 1000 x 1
	# 		o = np.dot(fc8_weights, n) # 4096 x 1
	# 		P['fc7'] = fc7_activations * o # 4096 x 1



	# 		""" For fc6 MWP """
	# 		# Get fc7 weights
	# 		fc7_weights = np.copy((weights_val[-4])[0,0]) # 4096 X 4096

	# 		# Get fc6 activations
	# 		fc6_activations = np.copy((layer_activations[-3])[0,0]).T # 4096 X 1

	# 		# Calculate MWP of fc6 using Eq 10 in paper
	# 		fc7_weights = fc7_weights.clip(min = 0) # threshold weights at 0
	# 		m = np.dot(fc7_weights.T, fc6_activations) # 4096 x 1
	# 		n = P['fc7'] / m # 4096 x 1
	# 		o = np.dot(fc7_weights, n) # 4096 x 1
	# 		P['fc6'] = fc6_activations * o # 4096 * 1



	# 		""" For pool5 MWP """
	# 		# Get fc6 weights
	# 		fc6_weights = np.copy(weights_val[-6]) # (7, 7, 512, 4096)
	# 		fc6_weights_reshaped = fc6_weights.reshape(-1, 4096) # (25088, 4096)

	# 		# Get pool5 activations
	# 		pool5_activations = np.copy(layer_activations[-4]).reshape(-1, 1) # (25088, 1)

	# 		# Calculate MWP of pool5 using Eq 10 in paper
	# 		fc6_weights_reshaped = fc6_weights_reshaped.clip(min = 0) # threshold weights at 0
	# 		m = np.dot(fc6_weights_reshaped.T, pool5_activations) # 4096 x 1
	# 		n = P['fc6'] / m # 4096 x 1
	# 		o = np.dot(fc6_weights_reshaped, n) # 25088 x 1
	# 		P['pool5'] = pool5_activations * o # 25088 x 1
	# 		P['pool5'] = P['pool5'].reshape(7, 7, 512)

	# 		heatmap = np.sum(P['pool5'], axis = 2)
	# 		heatmap_resized = transform.resize(heatmap, (image_size, image_size), order = 3, mode = 'constant')
	# 		plt.imshow(image)
	# 		plt.imshow(heatmap_resized, cmap = 'jet', alpha = 0.7)
	# 		plt.show()






