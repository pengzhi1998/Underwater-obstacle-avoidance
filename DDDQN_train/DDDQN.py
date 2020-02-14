import torch
import torch.nn as nn
import numpy as np
from GazeboWorld import GazeboWorld
import rospy
import os
import time
import random
import matplotlib.pyplot as plt
import time
from visdom import Visdom
viz = Visdom()

fig = plt.figure()
from collections import deque
from torch.autograd import Variable
dtype = torch.cuda.FloatTensor

np.set_printoptions(threshold=np.inf)

GAME = 'GazeboWorld'
ACTIONS = 7 # number of valid actions
SPEED = 2 # DoF of speed
GAMMA = 0.97 # decay rate of past observations
OBSERVE = 10. # timesteps to observe before training
EXPLORE = 20000. # frames over which to anneal epsilon
FINAL_EPSILON = 0.001 # final value of epsilon
INITIAL_EPSILON = .5 # starting value of epsilon
REPLAY_MEMORY = 13500 # number of previous transitions to remember
BATCH = 8 # size of minibatch
MAX_EPISODE = 50000
MAX_T = 200
DEPTH_IMAGE_WIDTH = 160
DEPTH_IMAGE_HEIGHT = 128
RGB_IMAGE_HEIGHT = 228
RGB_IMAGE_WIDTH = 304
CHANNEL = 3
TARGET_UPDATE = 100 # every 1500 steps, we need to update the target network with the parameters in online network
H_SIZE = 8*10*64
IMAGE_HIST = 4

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

def train():
    learning_rate = 1.0e-5

    online_net = DDDQN()
    target_net = DDDQN()

    if os.path.isfile('../../stored_networks/online_with_noise.pth.tar') and os.path.isfile('../../stored_networks/target_with_noise.pth.tar'):
        resume_file_online = '../../stored_networks/online_with_noise.pth.tar'
        checkpoint_online = torch.load(resume_file_online)
        online_net.load_state_dict(checkpoint_online['state_dict'])
        resume_file_target = '../../stored_networks/target_with_noise.pth.tar'
        checkpoint_target = torch.load(resume_file_target)
        target_net.load_state_dict(checkpoint_target['state_dict'])

    online_net = online_net.cuda()
    target_net = target_net.cuda()
    # optimizer = torch.optim.Adam(online_net.parameters(), lr=learning_rate)
    loss_func = nn.MSELoss().cuda()

    rospy.sleep(1.)

    env = GazeboWorld()
    print 'Environment initialized'

    D = deque()
    terminal = False

    episode = 0
    epsilon = INITIAL_EPSILON
    Step = 0 # the step counter to indicate whether update the network, very similar to the parameter "t", but
             # but "t" is an inside loop parameter, which means every episode the "t" will be redefined
    rate = rospy.Rate(3)
    # loop_time = time.time()
    # last_loop_time = loop_time # ?
    ten_episode_reward = 0
    while episode < MAX_EPISODE and not rospy.is_shutdown():
        env.ResetWorld()

        depth_img_t1 = env.GetDepthImageObservation()
        depth_imgs_t1 = np.stack((depth_img_t1, depth_img_t1, depth_img_t1, depth_img_t1), axis=0)

        optimizer = torch.optim.Adam(online_net.parameters(), lr=learning_rate)
        online_net.train()
        # some settings
        t = 0
        r_epi = 0.
        terminal = False
        reset = False
        # loop_time_buf = []
        action_index = 0
        loss_sum = 0

        while not reset and not rospy.is_shutdown():
            depth_img_t1 = env.GetDepthImageObservation()
            depth_img_t1 = np.reshape(depth_img_t1, (1, DEPTH_IMAGE_HEIGHT, DEPTH_IMAGE_WIDTH))
            depth_imgs_t1 = np.append(depth_img_t1, depth_imgs_t1[:(IMAGE_HIST - 1), :, :], axis=0)
            reward_t, terminal, reset, total_reward = env.GetRewardAndTerminate(t)
            if reset == True:
                ten_episode_reward += total_reward # to compute the average reward over 50 episodes
            if t > 0:
                # depth_imgs_t is the state images for former time, depth_imgs_t1 is the state images for latter time
                D.append((depth_imgs_t, a_t, reward_t, depth_imgs_t1, terminal))
                if len(D) > REPLAY_MEMORY:
                    D.popleft()
            depth_imgs_t = depth_imgs_t1
            depth_imgs_t1_cuda = depth_imgs_t1[np.newaxis, :]
            depth_imgs_t1_cuda = torch.from_numpy(depth_imgs_t1_cuda)
            depth_imgs_t1_cuda = Variable(depth_imgs_t1_cuda.type(dtype))
            print D.__len__()

            a = online_net(depth_imgs_t1_cuda) # the shape of a is [1, 11] which has two dimensions, so we need a[0] to have a dimensionality reduction
            # readout_t = a[0].cpu().detach().numpy()
            print a
            readout_t = a[0]
            a_t = np.zeros([ACTIONS]) # ([0., 0., 0., 0., 0., 0., 0.])

            # there aren't enough data in the buffer, so if the episode is not big enough, we just collect them
            if episode <= 10:
                action_index = random.randrange(ACTIONS) # any number from 0-6
                a_t[action_index] = 1 # let the specified action value to be 1
            else: # the data is enough, so let's begin to train the network take some reasonable steps
                rdnum = random.random()
                # if rdnum <= epsilon: # choose a random action
                if rdnum <= epsilon:
                    print "-------------Random Action---------------"
                    action_index = random.randrange(ACTIONS)
                    a_t[action_index] = 1
                else:
                    max_q_value, action_index = torch.max(readout_t, 0)
                    a_t[action_index] = 1
                # control the agent
            env.Control(action_index)
            # print "action"
            # at the same time, we need to train the network
            if episode > OBSERVE:
                minibatch = random.sample(D, BATCH)
                y_batch = []
                # get the batch variables
                depth_imgs_t_batch = torch.FloatTensor([d[0] for d in minibatch]) # the former batch
                depth_imgs_t_batch = Variable(depth_imgs_t_batch.type(dtype))
                a_batch = torch.FloatTensor([d[1] for d in minibatch])
                a_batch = Variable(a_batch.type(dtype))
                r_batch = torch.FloatTensor([d[2] for d in minibatch])
                r_batch = Variable(r_batch.type(dtype))
                depth_imgs_t1_batch = torch.FloatTensor([d[3] for d in minibatch]) # the latter batch
                depth_imgs_t1_batch = Variable(depth_imgs_t1_batch.type(dtype))

                # print depth_imgs_t1_batch.shape
                Q1 = online_net(depth_imgs_t1_batch)
                Q2 = target_net(depth_imgs_t1_batch)
                for i in range(0, len(minibatch)):
                    terminal_batch = minibatch[i][4]
                    if terminal_batch:
                        y_batch.append(r_batch[i])
                    else:
                        y_batch.append(r_batch[i] + GAMMA * Q2[i, torch.argmax(Q1[i])])
                # the culculation with Q(s', a')
                y_batch = torch.FloatTensor(y_batch)
                y_batch = Variable(y_batch.type(dtype))
                # Q(s, a)
                Q_current = online_net(depth_imgs_t_batch)
                Q_predicted_value = torch.sum(torch.mul(Q_current, a_batch), 1)

                loss = loss_func(y_batch, Q_predicted_value)
                # back propagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_sum += loss.item()

            # update the network every 100 steps
            if (Step + 1) % TARGET_UPDATE == 0:
                target_net.load_state_dict(online_net.state_dict())
            Step += 1

            r_epi = r_epi + reward_t
            t += 1
            rate.sleep()

            # scale down epsilon
            if epsilon > FINAL_EPSILON and episode > OBSERVE:
                epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        # save the network every 500 episodes
        if (episode + 1) % 150 == 0:
            torch.save({
                'episode': episode + 1,
                'state_dict': online_net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epsilon': episode
            }, '../../stored_networks/online_with_noise.pth.tar')
            learning_rate = learning_rate * 0.96
        if (episode + 51) % 150 == 0:
            torch.save({
                'episode': episode + 1,
                'state_dict': online_net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epsilon': episode
            }, '../../stored_networks/target_with_noise.pth.tar')

        print "episode:", episode, ", loss:", loss_sum / t, ", total reward for this episode:", total_reward
        if (episode + 1) % 10 == 0:
            average_reward = ten_episode_reward / 10 # type: int
            print "the average reward:", average_reward
            viz.line(
                Y = np.expand_dims(np.array(average_reward), axis = 0),
                X = np.expand_dims(np.array(episode), axis = 0),
                win = 'reward',
                update='append'
            )
            ten_episode_reward = 0

        episode += 1

def main():
    train()

if __name__ == "__main__":
    main()
