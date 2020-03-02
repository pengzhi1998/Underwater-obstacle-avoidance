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

depth_net = FCRN(batch_size)
resume_file_depth = 'checkpoint.pth.tar'
checkpoint = torch.load(resume_file_depth)
depth_net.load_state_dict(checkpoint['state_dict'])
depth_net.cuda()

# img = cv2.imread("299902454.png") * 2
img = cv2.imread("./test_images/599.png")
print img.shape
img = cv2.resize(img, (640, 480), interpolation=cv2.INTER_NEAREST)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# img = cv2.imread("rgb_depth.png")
# img[:, :, 2] = img[:, :, 2] * 0.2
# img[:, :, 1] = img[:, :, 1] * 0.85
# img[:, :, 0] = img[:, :, 0] * 0.88
# print img.shape
# cv2.imwrite("rgb_depth_pre.png", img)
img = torch.from_numpy(img[np.newaxis, :])
img = img.permute(0, 3, 1, 2)
img = Variable(img.type(dtype))
depth_img = depth_net(img)
depth_img = depth_img.permute(0, 2, 3, 1)
print depth_img.shape
depth_img = depth_img[0].data.squeeze().cpu().numpy().astype(np.float32)
# depth_img = depth_img[0].data.cpu().numpy()
cv2.imwrite("under_water.png", depth_img*50)
print depth_img
