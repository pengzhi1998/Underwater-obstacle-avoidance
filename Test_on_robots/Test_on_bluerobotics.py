from __future__ import print_function
from RealWorld_bluerobotics import RealWorld
import torch
import torch.nn as nn
import random
import numpy as np
import time
import rospy
from fcrn import FCRN # this file could be changed
import cv2
batch_size = 1
from torch.autograd import Variable
dtype = torch.cuda.FloatTensor

ACTIONS = 7 # number of valid actions
SPEED = 2 # DoF of speed
GAMMA = 0.99 # decay rate of past observations
OBSERVE = 10. # timesteps to observe before training
EXPLORE = 20000. # frames over which to anneal epsilon
FINAL_EPSILON = 0.0001 # final value of epsilon
INITIAL_EPSILON = 0.001 # starting value of epsilon
REPLAY_MEMORY = 50000 # number of previous transitions to remember
BATCH = 8 # size of minibatch
MAX_EPISODE = 20000
DEPTH_IMAGE_WIDTH = 160
DEPTH_IMAGE_HEIGHT = 128
RGB_IMAGE_HEIGHT = 228
RGB_IMAGE_WIDTH = 304
CHANNEL = 3
TAU = 0.001 # Rate to update target network toward primary network
H_SIZE = 8*10*64
IMAGE_HIST = 4

class QNetwork(nn.Module):
    def __init__(self):
        super(QNetwork, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, (10, 14), (8, 8), padding = (1, 4))
        self.conv2 = nn.Conv2d(32, 64, (4, 4), 2, padding = (1, 1))
        self.conv3 = nn.Conv2d(64, 64, (3, 3), 1, padding = (1, 1))

        self.fc1_adv = nn.Linear(8*10*64, 512)
        self.fc1_val = nn.Linear(8*10*64, 512)

        self.fc2_adv = nn.Linear(512, 7)
        self.fc2_val = nn.Linear(512, 1)

        self.relu = nn.ReLU()

    def forward(self, x): # remember to initialize
        # batch_size = x.size(0)
        # print x.shape
        x = self.relu(self.conv1(x))
        # print x.shape
        x = self.relu(self.conv2(x))
        # print x.shape
        x = self.relu(self.conv3(x))
        # print x.shape
        x = x.view(x.size(0), -1)
        # print x.shape

        adv = self.relu(self.fc1_adv(x))
        val = self.relu(self.fc1_val(x))

        adv = self.fc2_adv(adv)
        val = self.fc2_val(val).expand(x.size(0), 7) # shape = [batch_size, 7]
        # print adv.mean(1).unsqueeze(1).shape
        x = val + adv - adv.mean(1).unsqueeze(1).expand(x.size(0), 7) # shape = [batch_size, 7]
        # print x.shape
        return x

def TestNetwork():
	# define and load depth_prediction network
	depth_net = FCRN(batch_size)
	resume_file_depth = 'checkpoint.pth.tar'
	checkpoint = torch.load(resume_file_depth)
	depth_net.load_state_dict(checkpoint['state_dict'])
	depth_net.cuda()

	# define and load q_learning network
	online_net = QNetwork()
	resume_file_online = 'online_with_noise.pth.tar'
	checkpoint_online = torch.load(resume_file_online)
	online_net.load_state_dict(checkpoint_online['state_dict'])
	online_net.cuda()
	rospy.sleep(1.)

	# Initialize the World and variables
	env = RealWorld()
	print('Environment initialized')
	episode = 0

	# start training
	rate = rospy.Rate(3)

	with torch.no_grad():
		while not rospy.is_shutdown():
			episode += 1
			t = 0

			rgb_img_t1 = env.GetRGBImageObservation()
			rgb_img_t1 = rgb_img_t1[np.newaxis, :]
			rgb_img_t1 = torch.from_numpy(rgb_img_t1)
			rgb_img_t1 = rgb_img_t1.permute(0, 3, 1, 2)
			rgb_img_t1 = Variable(rgb_img_t1.type(dtype))
			depth_img_t1 = depth_net(rgb_img_t1) - 0.2
			depth_img_t1 = torch.squeeze(depth_img_t1, 1)
			depth_imgs_t1 = torch.stack((depth_img_t1, depth_img_t1, depth_img_t1, depth_img_t1), dim=1)

			while not rospy.is_shutdown():
				rgb_img_t1 = env.GetRGBImageObservation()
				# cv2.imwrite('rgb_depth.png', rgb_img_t1)
				rgb_img_t1 = rgb_img_t1[np.newaxis, :]
				rgb_img_t1 = torch.from_numpy(rgb_img_t1)
				rgb_img_t1 = rgb_img_t1.permute(0, 3, 1, 2)
				rgb_img_t1 = Variable(rgb_img_t1.type(dtype))
				depth_img_t1 = depth_net(rgb_img_t1) - 0.2
				depth_imgs_t1 = torch.cat((depth_img_t1, depth_imgs_t1[:, :(IMAGE_HIST - 1), :, :]), 1)
				depth_imgs_t1_cuda = Variable(depth_imgs_t1.type(dtype))
				predicted_depth = depth_img_t1[0].data.squeeze().cpu().numpy().astype(np.float32)
				cv2.imwrite('predicted_depth.png', predicted_depth * 50)


				Q_value_list = online_net(depth_imgs_t1_cuda)
				Q_value_list = Q_value_list[0]
				Q_value, action = torch.max(Q_value_list, 0)
				env.Control(action)
				t += 1
				rate.sleep()

def main():
	TestNetwork()

if __name__ == "__main__":
	main()
