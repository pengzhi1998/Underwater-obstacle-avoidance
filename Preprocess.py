import numpy as np
import cv2
import rospy
import random
import copy
import pickle
np.set_printoptions(threshold=np.inf)
from collections import deque
from pathlib import Path
file_num = 15000

def ismember(A, B): # check whether B is a member of A, A is a list, B is a number
    return [ np.sum(a == B) for a in A ]

# data_rgb = pickle.load(open("data_rgb.p", "rb"))
# data_depth = pickle.load(open("data_depth.p", "rb"))

# def rgb_data_color_aug(rgb_images): # doesn't seem like to be used by any file
# 	rgb_images = np.asarray(rgb_images).astype(float) # does not use new space just transfer the initial array to ndarray
# 	noise = np.random.rand(4) * 0.4 + 0.8 # random number in [0, 1.2] which is a noise on the initial image
# 	print noise
# 	rgb_images = rgb_images * noise[0]
# 	for x in xrange(3):
# 		rgb_images[:, :, :, x] = rgb_images[:, :, :, x] * noise[x + 1]
# 	rgb_images[rgb_images>255.] = 255.
# 	rgb_images = np.uint8(rgb_images)
# 	return rgb_images

def crop_img(img):
	size = np.shape(img)
	if len(size) == 2:
		# cropped_img = img[height_start : height_end, width_start : width_end]
		cropped_img = img[45 : 435, 60 : 580]
	elif len(size) == 3:
		cropped_img = img[45 : 435, 60 : 580, :]
	return cropped_img

D_training = deque() # contain all the training sets which include both the depth-image as well as rgb-image
D_testing = deque() # contain all the training sets which include both the depth-image as well as rgb-image
D_validating = deque()
testing = 0.3
cnt = 0
path_rgb = './rgb/'
path_depth = './depth/'
inpaint_flag = False


# depth_dim = (160, 128)
depth_dim = (304, 228)
rgb_dim = (304, 228)

# for rgb_image, depth_image in zip(data_rgb, data_depth):
# 	rgb_image = rgb_image
# 	depth_image = depth_image
# 	rgb_image_resized = cv2.resize(rgb_image, rgb_dim, interpolation = cv2.INTER_AREA)
# 	depth_image_resized = cv2.resize(depth_image, depth_dim, interpolation = cv2.INTER_AREA)
# 	depth_image_resized = np.reshape(depth_image_resized, (depth_dim[1], depth_dim[0], 1)) # add a dimension (third dimension)
# 	mask = copy.deepcopy(depth_image_resized)
# 	mask = np.isnan(mask)
# 	mask = 1.0 - (mask + np.zeros((128, 160, 1))) # the purpose of this is transfering all the pixels' value which is not equal to 0 to 1
# 	mask = np.array(mask, dtype=np.float32)
# 	depth_image_resized[np.isnan(depth_image_resized)] = 0
# 	print np.sum(mask), np.min(depth_image_resized), np.max(depth_image_resized)
# 	if random.uniform(0, 1.0) > .08: # if the random number is bigger than 0.1, then classify the image as training set, or the test set
# 		D_training.append((rgb_image_resized, depth_image_resized, mask))
# 	else:
# 		D_testing.append((rgb_image_resized, depth_image_resized, mask))
for i in xrange(15000, file_num + 19):
	print 'img', i
	depth_file = Path(path_depth+'depth'+str(i)+'.png')
	rgb_file = Path(path_rgb+'rgb'+str(i)+'.png')
	if depth_file.is_file() and rgb_file.is_file():
		cv_depth_img = cv2.imread(path_depth+'depth'+str(i)+'.png', cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
		print np.shape(cv_depth_img), np.max(cv_depth_img), np.min(cv_depth_img)
		# cv_depth_img = crop_img(cv_depth_img)
		print np.max(cv_depth_img)
		cv_depth_img = np.array(cv_depth_img, dtype=np.float32)
		print np.max(cv_depth_img)
		cv_depth_img = cv_depth_img / 1000.0
		print np.max(cv_depth_img)
		cv_depth_img = cv2.resize(cv_depth_img, depth_dim, interpolation = cv2.INTER_AREA)
		print cv_depth_img.shape, np.max(cv_depth_img), np.min(cv_depth_img)

		mask = copy.deepcopy(cv_depth_img)
		mask[mask == 0.] = 1.
		mask[mask != 1.] = 0.
		mask = 1. - mask # the purpose of this is transfering all the pixels' value which is not equal to 0 to 1
		mask = np.reshape(mask, (depth_dim[1], depth_dim[0], 1))
		mask = np.array(mask, np.float32)

		cv_rgb_img = cv2.imread(path_rgb+'rgb'+str(i)+'.png')
		# cv_rgb_img = crop_img(cv_rgb_img)
		cv_rgb_img = cv2.resize(cv_rgb_img, rgb_dim, interpolation = cv2.INTER_AREA) # ignore the third dimension, but it just keeps the value of 3
		cv_depth_img = np.reshape(cv_depth_img, (depth_dim[1], depth_dim[0], 1)) # add a dimension (third dimension)
		print cv_rgb_img.shape

		rand = random.uniform(0., 1.)
		if rand > 1: # if the random number is bigger than 0.1, then classify the image as training set, or the test set
			D_training.append((cv_rgb_img, cv_depth_img, mask))
		elif rand > 1:
			D_validating.append((cv_rgb_img, cv_depth_img, mask))
		else:
			D_testing.append((cv_rgb_img, cv_depth_img, mask))


# print 'rgb size:', cv_rgb_img.shape
print 'depth size:', cv_depth_img.shape
print 'mask size:', mask.shape
pickle.dump(D_training, open("rgb_depth_images_training_real.p", "wb")) # write the images into this .p file in binary form
pickle.dump(D_testing, open("rgb_depth_images_testing_real.p", "wb"))
pickle.dump(D_testing, open("rgb_depth_images_validating_real.p", "wb")) # some totally new datasets just for test
print 'D_training:',  len(D_training), '| D_testing:', len(D_testing), '| D_validating:', len(D_validating)
