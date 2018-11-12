"""
Code to extract imagenet features from UCF sports 101 dataset

Author: Shakthi Duraimurugan
"""

import sys
import os
from os import path

import tensorflow as tf
import numpy as np

from tqdm import tqdm, trange
import glob
import time
import pandas as pd
import h5py
import pickle as pkl
import subprocess

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

slim_dir = "/mnt/workspace/models/research/slim/"
sys.path.insert(0, slim_dir)
from nets import vgg
image_size = vgg.vgg_16.default_image_size
# print(image_size)

dataset_path = "/mnt/workspace/datasets/ucf101/ucf24/"
batch_size = 16

slim = tf.contrib.slim

with tf.Graph().as_default():
	input_batch = tf.placeholder(dtype=tf.uint8, shape=(batch_size,320,240,3))
	# print(input_batch)
	resized_images = tf.image.resize_images(tf.image.convert_image_dtype(input_batch, dtype=tf.float32),
						[image_size,image_size]) # resize to default vgg size
	# print(resized_images)
	normalized_images = tf.multiply(tf.subtract(resized_images, 0.5), 2.0) #normalise from {0,1} to {-1,1}
	# print(normalized_images)
	with slim.arg_scope(vgg.vgg_arg_scope()):
		outputs, end_points = vgg.vgg_16(normalized_images,
										num_classes=1001, is_training=False)
		final_conv = end_points['vgg_16/conv5/conv5_3']
		fc7 = end_points['vgg_16/fc7']
		print(fc7)
		# for item in end_points:
		# 	print(item)
			# for key,val in item.items():
				# print(key,val)
