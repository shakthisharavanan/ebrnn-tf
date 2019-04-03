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
	x = tf.placeholder(tf.float32, (None, n_frames, 4096))
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
		sess.run(tf.global_variables_initializer())
		print("Training")
		with tqdm(total = epochs) as outer_pbar:
			outer_pbar.set_description("Epochs")
			for i in range(epochs):
				previous_test_accuracy = 0
				train_videos, train_labels = shuffle(train_videos, train_labels)
				with tqdm(total = n_train) as inner_pbar:
					inner_pbar.set_description("Batch")
					for start, end in zip(range(0, n_train, batch_size), range(batch_size, n_train + batch_size, batch_size)):
						# Read features for each mini batch and train lstm
						batch_x = train_videos[start:end]
						batch_y = train_labels[start:end]
						features_x = read_video(batch_x, batch_y, n_frames)
						seq_len = np.zeros(batch_size, np.int32)
						seq_len[:] = n_frames
						sess.run(training_operation, feed_dict = {x: features_x, y: batch_y, sequence_length: n_frames})
						inner_pbar.update(batch_size)

					# Check test accuracy
					total_accuracy = 0
					for start, end in zip(range(0, n_test, batch_size), range(batch_size, n_test + batch_size, batch_size)):
						batch_x = test_videos[start:end]
						batch_y = test_labels[start:end]
						features_x = read_video(batch_x, batch_y, n_frames)
						accuracy = sess.run(accuracy_operation, feed_dict={x: features_x, y: batch_y, sequence_length: n_frames})
						total_accuracy += (accuracy * len(batch_x))
					current_test_accuracy = total_accuracy / n_test
					print("Test accuracy = {:.3f}".format(current_test_accuracy))

					# Save the current and the best model
					saver.save(sess, './saved_models/latest_model')
					if(current_test_accuracy>=previous_test_accuracy):
						saver.save(sess, './saved_models/best_model')
					previous_test_accuracy = current_test_accuracy

				outer_pbar.update(1)



