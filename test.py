import torch
import torch.utils.data
from fcrn import FCRN
import pickle
from torch.autograd import Variable
import torch.nn as nn
dtype = torch.cuda.FloatTensor
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import copy

batch_size = 16
resume_from_file = True

model = FCRN(batch_size)
model = model.cuda()
loss_fn = torch.nn.MSELoss().cuda()

resume_file = 'checkpoint.pth.tar'

if resume_from_file:
    if os.path.isfile(resume_file):
        print("=> loading checkpoint '{}'".format(resume_file))
        checkpoint = torch.load(resume_file)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(resume_file, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(resume_file))

# testing_data = pickle.load(open('rgb_depth_images_testing_real.p', "rb"))
testing_data = pickle.load(open('rgb_depth_images_validating_real.p', "rb"))
    # train_loader = torch.utils.data.DataLoader(training_data, batch_size=16, shuffle=True, drop_last=True)
test_loader = torch.utils.data.DataLoader(testing_data, batch_size=16, shuffle=False, drop_last=True)

model.eval()
idx = 0
path = 'test_data/'
mean_total = 0.
with torch.no_grad():
    loss_func = nn.MSELoss().cuda()
    testing_loss = 0
    count = 0
    total_mean_difference = 0
    for rgb_image, depth_image, mask in test_loader:
        points_1 = np.sum(mask[7].numpy())
        points = np.sum(mask.numpy())

        input_var = rgb_image.permute(0, 3, 1, 2)
        input_var = Variable(input_var.type(dtype)) # shape of input_var is [32, 228, 304, 3], but the input of torch network is [N, C_in, H, W], need some conversion
                # print input_var.shape
        depth_var = depth_image.permute(0, 3, 1, 2)
        depth_var = Variable(depth_var.type(dtype))

                # input_rgb_image = input_var[0].data.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                # input_gt_depth_image = depth_var[0][0].data.cpu().numpy().astype(np.float32)
                # pred_depth_image = output[0].data.squeeze().cpu().numpy().astype(np.float32)
                #
                # input_gt_depth_image /= np.max(input_gt_depth_image)
                # pred_depth_image /= np.max(pred_depth_image)
                #
                # plt.imsave('input_rgb_epoch_{}.png'.format(epoch + 1), input_rgb_image)
                # plt.imsave('gt_depth_epoch_{}.png'.format(epoch + 1), input_gt_depth_image, cmap="viridis")
                # plt.imsave('pred_depth_epoch_{}.png'.format(epoch + 1), pred_depth_image, cmap="viridis")
        mask_var = mask.permute(0, 3, 1, 2)
        mask_var = Variable(mask_var.type(dtype))

        output = model(input_var)


        input_gt_depth_image = depth_var[7].data.squeeze().cpu().numpy().astype(np.float32)
        pred_depth_image = output[7].data.squeeze().cpu().numpy().astype(np.float32)

        # input_gt_depth_image /= np.nanmax(input_gt_depth_image)
        #
        # pred_depth_image /= np.max(pred_depth_image)
        test_var = copy.deepcopy(depth_var)
        test_var[depth_var==0] = np.nan

        input_gt_depth_image[input_gt_depth_image==0.] = np.nan
        difference_1 = np.subtract(input_gt_depth_image, pred_depth_image)
        difference_1[np.isnan(difference_1)] = 0.
        difference_1 = np.abs(difference_1)
        input_gt_depth_image[np.isnan(input_gt_depth_image)] = 0.
        print "max difference:", count+1, np.sum(difference_1) / points_1, np.nanmin(input_gt_depth_image), np.nanmin(pred_depth_image)

        difference = torch.sub(test_var, output)
        difference[torch.isnan(difference)] = 0.
        mean = torch.mean(difference)
        print "mean:", mean
        difference = torch.sum(torch.abs(difference)) / points
        total_mean_difference = total_mean_difference + difference
        print count+1, ":", difference
        idx = idx + 1
        # plt.imsave('Test_gt_depth_{:05d}.png'.format(idx), input_gt_depth_image)
        # plt.imsave('Test_pred_depth_{:05d}.png'.format(idx), pred_depth_image)
        cv2.imwrite(path + 'depth' + str(idx) + '.png', input_gt_depth_image * 50)
        cv2.imwrite(path + 'rgb' + str(idx) + '.png', pred_depth_image * 50)
        # depth_var[np.isnan(depth_var.data.squeeze().cpu().numpy())] = 0.
        # loss = loss_func(output, mask_var)
        loss = loss_func(torch.mul(output, mask_var), torch.mul(depth_var, mask_var)) / points * 327680
        testing_loss += loss.item()
        count += 1
        mean_total += mean

    print total_mean_difference / 47, mean_total / 47
    end_loss = testing_loss/count
    print 'the loss of test:', end_loss
