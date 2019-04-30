"""
Code to test on UCF 24 bounding box
"""
import numpy as np
import pickle
import pdb
import os
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import glob



def trim(bbox_frames, sequence_length = 30):
	offset = len(bbox_frames)//sequence_length
	# print(bbox_frames.shape[0], offset)
	# pdb.set_trace()
	trimmed_bbox_frames = (bbox_frames[::offset])[:sequence_length]
	return trimmed_bbox_frames


def check_hit(gt_file, pred):

	# Read the txt file
	f = open(gt_file, 'r')
	all_bbox_gt = f.read().split('\n')
	num_bbox = len(all_bbox_gt) - 1
	for i in range(num_bbox):
		_, xmin, ymin, xmax, ymax = all_bbox_gt[i].split(' ')
		x, y = pred
		if (x>float(xmin) and x<float(xmax) and y>float(ymin) and y<float(ymax)):
			return True

		# pdb.set_trace()
	return False


# ground_truth = pickle.load(open("pyannot.pkl", "rb"))
hits = {}
misses = {}

hits['total'] = 0
misses['total'] = 0

ground_truth_trimmed = {}



f = open('testlist01.txt', 'r')
test_list =  f.read().split('\n')
f.close()

pred_dir = './heatmaps/'
gt_dir = "/mnt/workspace/datasets/ucf101/ucf24/labels/"
extracted_features_dir = './fc7_features/'
images_dir = "/mnt/workspace/datasets/ucf101/ucf24/rgb-images/"

labels = [x.replace(images_dir,"") for x in sorted(glob.glob(images_dir+"*"))] # ['Basketball', 'BasketballDunk', 'Biking', 'CliffDiving', 'CricketBowling', 'Diving', 'Fencing', 'FloorGymnastics', 'GolfSwing', 'HorseRiding', 'IceDancing', 'LongJump', 'PoleVault', 'RopeClimbing', 'SalsaSpin', 'SkateBoarding', 'Skiing', 'Skijet', 'SoccerJuggling', 'Surfing', 'TennisSwing', 'TrampolineJumping', 'VolleyballSpiking', 'WalkingWithDog']
for label in labels:
	hits[label] = 0
	misses[label] = 0
# pdb.set_trace()


for file in test_list:
	curr_label = file.split('/')[0]
	pred_filename = pred_dir + file + '.npy'
	gt_folder = gt_dir + file
	images_folder = images_dir + file + '/*'

	num_frames = len(glob.glob(images_folder))
	gt_all_frame = ['%05d' % i + ".txt" for i in range(num_frames+1)[1:]]
	gt_trimmed = trim(gt_all_frame) # Get trimmed gt files

	available_gt = os.listdir(gt_folder)
	# gt_matched = list(set(gt_trimmed).intersection(available_gt))


	# Load predicted for all 30 frames
	pred = np.load(pred_filename)

	# Check hit or miss for each frame
	for i in range(len(gt_trimmed)):
		if (gt_trimmed[i] in available_gt):

			# Get hit or miss
			hit_flag = check_hit(gt_folder + "/" + gt_trimmed[i], pred[i])
			if (hit_flag):
				hits['total'] += 1
				hits[curr_label] += 1
			else:
				misses['total'] += 1
				misses[curr_label] += 1

accuracy = {}
for key in hits.keys():
	accuracy[key] = hits[key] / (hits[key] + misses[key])

# Save the accuracy dictionary
with open('accuracy.pickle', 'wb') as handle:
	pickle.dump(accuracy, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('hits.pickle', 'wb') as handle:
	pickle.dump(hits, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('misses.pickle', 'wb') as handle:
	pickle.dump(misses, handle, protocol=pickle.HIGHEST_PROTOCOL)

pdb.set_trace()


	# # label_filenames = sorted(glob.glob(labels_filename))
	# # for label in labels:
	# # 	f = open(label, 'r')
	# # 	num_bbox = len(f.read().split('\n')) - 1
	# # 	f.close()
	# # 	# bbox_gt = f.read().split('\n')[0].split(' ')[0]
	# # 	# print(bbox_gt)
	# # 	if(len_gt != 2):
	# # 		print(file)
	# """ To check length"""
	# # fc7_file = extracted_features_dir + file + '.npy'
	# # labels_length = len(sorted(glob.glob(labels_filename)))
	# # fc7 = np.load(fc7_file)
	# # fc7_len = fc7.shape[0]
	# # if fc7_len != labels_length:
	# # 	print(labels_length, fc7_len)

	# # if not(os.path.exists(filename)):
	# # 	print(filename)

	# # # Trim Ground truth to 30 frames
	# # if (ground_truth[file]['annotations'][0]['boxes'].shape[0] != ground_truth[file]['numf']):
	# # 	continue
	# # 	# counter += 1
	# # 	# print(file)
	# # bbox_gt = trim(ground_truth[file]['annotations'][0]['boxes'])
	# # bbox_pred = np.load(filename)
	# # pdb.set_trace()


pdb.set_trace() 

pass 