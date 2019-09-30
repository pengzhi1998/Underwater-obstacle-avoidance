import torch
from GazeboWorld import GazeboWorld
import torch.nn as nn
import numpy as np
import rospy
DEPTH_IMAGE_WIDTH = 160
DEPTH_IMAGE_HEIGHT = 128
IMAGE_HIST = 4
from torch.autograd import Variable
dtype = torch.cuda.FloatTensor

env = GazeboWorld()

class DDDQN(nn.Module):
    def __init__(self):
        super(DDDQN, self).__init__()
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

test_net = DDDQN()
resume_file = 'online_with_noise.pth.tar'
checkpoint = torch.load(resume_file)
test_net.load_state_dict(checkpoint['state_dict'])
test_net.cuda()

def test():
    while True:
        env.ResetWorld()
        depth_img_t1 = env.GetDepthImageObservation()
        depth_imgs_t1 = np.stack((depth_img_t1, depth_img_t1, depth_img_t1, depth_img_t1), axis=0)
        rate = rospy.Rate(3)
        rospy.sleep(1.)
        test_net.eval()
        with torch.no_grad():
            testing_loss = 0
            reset = False
            t = 0
            while not rospy.is_shutdown():
                depth_img_t1 = env.GetDepthImageObservation()
                print depth_img_t1.max()
                depth_img_t1 = np.reshape(depth_img_t1, (1, DEPTH_IMAGE_HEIGHT, DEPTH_IMAGE_WIDTH))
                depth_imgs_t1 = np.append(depth_img_t1, depth_imgs_t1[:(IMAGE_HIST - 1), :, :], axis=0)
                reward_t, terminal, reset, total_reward = env.GetRewardAndTerminate(t)
                depth_imgs_t1_cuda = depth_imgs_t1[np.newaxis, :]
                depth_imgs_t1_cuda = torch.from_numpy(depth_imgs_t1_cuda)
                depth_imgs_t1_cuda = Variable(depth_imgs_t1_cuda.type(dtype))
                Q_value_list = test_net(depth_imgs_t1_cuda)
                print Q_value_list
                Q_value_list = Q_value_list[0]
                Q_value, action = torch.max(Q_value_list, 0)
                env.Control(action)
                t += 1
                rate.sleep()

def main():
    test()

if __name__ == "__main__":
    main()
