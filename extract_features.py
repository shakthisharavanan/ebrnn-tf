"""
Code to extract Imagenet features from UCF videos using pretrained VGG-16

Author: Shakthi Duraimurugan

"""

import sys
import os
from os import path
import time
from time import sleep

import tensorflow as tf
import numpy as np
# import cv2
# import matplotlib.pyplot as plt
import pylab as plt

from tqdm import tqdm, trange, tqdm_notebook, tnrange
import glob
import time
import pandas as pd
import h5py
import pickle as pkl
import subprocess as sp
import pdb

slim_dir = "/mnt/workspace/models/research/slim/"
sys.path.insert(0, slim_dir)
from nets import vgg
from preprocessing import vgg_preprocessing

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class feature_extractor(object):
	def __init__(self, image_size, batch_size, image_mean):
		# Set Paths
		self.dataset_dir = "/mnt/workspace/datasets/UCF-101/"
		self.label_dir = "/mnt/workspace/datasets/ucf101/ucf24/labels/"
		self.checkpoints_dir = "/mnt/workspace/models/checkpoints/"
		self.extracted_features_dir = "/mnt/workspace/ebrnn-tf/fc7_features/"

		self.labels = [x.replace(self.label_dir,"") for x in sorted(glob.glob(self.label_dir+"*"))]

		self.image_size = image_size
		self.batch_size = batch_size
		self.image_mean = image_mean

		# Define the Network
		slim = tf.contrib.slim
		with tf.Graph().as_default():
			self.input_batch = tf.placeholder(dtype=tf.uint8, shape=(batch_size,240,320,3))
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
			# print(self.probablities)
			# print(self.final_conv)
			# print(self.fc7)

			init_fn = slim.assign_from_checkpoint_fn(os.path.join(self.checkpoints_dir, 'vgg_16.ckpt'),slim.get_model_variables('vgg_16'))
			self.sess = tf.Session()
			init_fn(self.sess)


	def read_video(self, video_path):
		command = [ 'ffmpeg', '-i', video_path, '-f', 'image2pipe', '-pix_fmt', 'rgb24', '-vcodec', 'rawvideo', '-']
		pipe = sp.Popen(command, stdout = sp.PIPE, stderr = open(os.devnull, 'w'), bufsize=10**8)
		# self.video_frames = []
		while True:
		    # read 420*360*3 bytes (= 1 frame)
		    raw_image = pipe.stdout.read(240*320*3)
		    if len(raw_image) != 240*320*3:
		        break;
		    # transform the byte read into a numpy array
		    image =  np.fromstring(raw_image, dtype='uint8')
		    image = image.reshape((240,320,3))
		    self.video_frames.append(image)
		    # throw away the data in the pipe's buffer.
		    pipe.stdout.flush()
		self.video_frames = np.asarray(self.video_frames, dtype=np.uint8)
		# return video_frames

	def get_imagenet_features(self, label, video_path):
		# Read frames from the video
		self.video_frames = []
		self.read_video(video_path)
		n = self.video_frames.shape[0]

		if not(os.path.exists(self.extracted_features_dir + label)):
			os.mkdir(self.extracted_features_dir + label)

		filename = self.extracted_features_dir + label + "/" + video_path.split('/')[-1].split('.')[0]

		# Iterate through batches and extract features
		full_video_features = []
		for start, end in zip(range(0, n, self.batch_size), range(self.batch_size, n + self.batch_size, self.batch_size)):
			current_batch = np.zeros((self.batch_size, 240, 320, 3), dtype = np.uint8)
			current_batch[:min(end, n) - start] = self.video_frames[start:end]

			# final_conv = self.sess.run(self.final_conv, feed_dict = {self.input_batch: current_batch})
			fc7 = self.sess.run(self.fc7, feed_dict = {self.input_batch: current_batch})
			full_video_features = full_video_features + list(fc7)

		features = np.asarray(full_video_features[:n], np.float32) 
		np.save(filename, features)
		# pdb.set_trace()
		pass


if __name__ == "__main__":
	# Set some paraeters
	image_size = vgg.vgg_16.default_image_size
	batch_size = 16
	# image_mean = [103.939, 116.779, 123.68]
	image_mean = [123.68, 116.779, 103.939]  
	sampling_rate = 30 # hz

	extractor = feature_extractor(image_size, batch_size, image_mean)
	with tqdm(total = len(extractor.labels)) as outer_pbar:
		for label in extractor.labels:
			outer_pbar.set_description("Total Progress")
			outer_pbar.update(1)
			videos = sorted(glob.glob(extractor.dataset_dir+label+"/*"))
			with tqdm(total = len(videos)) as inner_pbar:
				for video_path in videos:
					inner_pbar.set_description(label)
					inner_pbar.update(1)
					extractor.get_imagenet_features(label, video_path)




