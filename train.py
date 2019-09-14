import torch
import torch.utils.data
import os
import pickle
dtype = torch.cuda.FloatTensor
from torch.autograd import Variable
from fcrn import FCRN
import torch.nn as nn
import numpy as np
import random
import cv2
import matplotlib.pyplot as plt
from weights import load_weights
weights_file = "NYU_ResNet-UpProj.npy"
path = 'test_data/'
resume_file = 'checkpoint_without_mask.pth.tar'
checkpoint = torch.load(resume_file)
start_epoch = checkpoint['epoch']


def main():
    batch_size = 16
    learning_rate = 1.0e-6
    monentum = 0.9 # might not be used
    weight_decay = 0.0005
    num_epoches = 6000
    best_val_err = 1.0e3

    # read in the processed images and put them in the dataloader
    training_data = pickle.load(open('rgb_depth_images_training_real.p', "rb"))
    testing_data = pickle.load(open('rgb_depth_images_validating_real.p', "rb"))
    train_loader = torch.utils.data.DataLoader(training_data, batch_size=16, shuffle=True, drop_last=True)
    test_loader = torch.utils.data.DataLoader(testing_data, batch_size=16, shuffle=True, drop_last=True)

    model = FCRN(batch_size)
    model.load_state_dict(checkpoint['state_dict'])
    # model.load_state_dict(load_weights(model, weights_file, dtype))
    model = model.cuda()

    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_func = nn.MSELoss().cuda()

    # start_epoch = 0
    for epoch in range(num_epoches): # it is a difference in python version2 and 3, pay attension
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        model.train()
        running_loss = 0
        count = 0
        # epoch_loss = 0

        for rgb_image, depth_image, mask in train_loader:
            # print rgb_image.shape, mask.shape, depth_image.shape, np.nanmax(depth_image), np.nanmin(depth_image)
            points = np.sum(mask.numpy())
            if random.random() < 0.5: # with 50% probability to flip the rgb image and the depth image
                rgb_image = torch.from_numpy(np.flip(rgb_image, axis=1).copy())
                depth_image = torch.from_numpy(np.flip(depth_image, axis=1).copy())
                mask = torch.from_numpy(np.flip(mask, axis=1).copy())

            if random.random() < 0.5: # with 50% probability to flip the rgb image and the depth image
                rgb_image = torch.from_numpy(np.flip(rgb_image, axis=2).copy())
                depth_image = torch.from_numpy(np.flip(depth_image, axis=2).copy())
                mask = torch.from_numpy(np.flip(mask, axis=2).copy())

            input_var = rgb_image.permute(0, 3, 1, 2)
            input_var = Variable(input_var.type(dtype)) # shape of input_var is [32, 228, 304, 3], but the input of torch network is [N, C_in, H, W], need some conversion

            # mask = mask.cuda()
            depth_var = depth_image.permute(0, 3, 1, 2)
            depth_var = Variable(depth_var.type(dtype)) # shape of depth_var is [32, 128, 160, 1]
            # shape = depth_var.size() # shape of input_var is [32, 228, 304, 3]

            mask_var = mask.permute(0, 3, 1, 2)
            mask_var = Variable(mask_var.type(dtype))

            output = model(input_var)

            # d_show = torch.abs(torch.sub(torch.mul(output, mask), torch.mul(depth_var, mask)))

            loss = loss_func(output, depth_var)
            # print "1"
            # loss = loss_func(torch.mul(output, mask_var), torch.mul(depth_var, mask_var)) / points * 320000
            running_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # print "2"
            count += 1
        print 'epoch:', epoch
        epoch_loss = running_loss / count
        print 'epoch loss:', epoch_loss

        model.eval()
        with torch.no_grad():
            testing_loss = 0
            count = 0
            for rgb_image, depth_image, mask in test_loader:
                points = np.sum(mask.numpy())
                input_var = rgb_image.permute(0, 3, 1, 2)
                input_var = Variable(input_var.type(dtype)) # shape of input_var is [32, 228, 304, 3], but the input of torch network is [N, C_in, H, W], need some conversion
                # print input_var.shape
                depth_var = depth_image.permute(0, 3, 1, 2)
                depth_var = Variable(depth_var.type(dtype))

                mask_var = mask.permute(0, 3, 1, 2)
                mask_var = Variable(mask_var.type(dtype))

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

                output = model(input_var)
                # loss = loss_func(torch.mul(output, mask_var), torch.mul(depth_var, mask_var)) / points * 320000
                loss = loss_func(output, depth_var)
                testing_loss += loss.item()
                count += 1

                input_gt_depth_image = depth_var[0].data.squeeze().cpu().numpy().astype(np.float32)
                pred_depth_image = output[0].data.squeeze().cpu().numpy().astype(np.float32)
                if (epoch + 1) % 10 == 0:
                    cv2.imwrite(path + "ground_truth" + str(epoch + 1) + ".png", input_gt_depth_image * 50)
                    cv2.imwrite(path + "predicted" + str(epoch + 1) + ".png", pred_depth_image * 50)

            end_loss = testing_loss/count
            print 'the loss of test:', end_loss

            if end_loss < best_val_err:
                best_val_err = end_loss
                torch.save({
                'epoch': start_epoch + epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, 'checkpoint_without_mask.pth.tar')
            # else:
            #     learning_rate = learning_rate * 0.8 # if the error is bigger than the best one, it means the
            #                 # the learning rate might be too big, and the learning just slips the best point

        if epoch < 300:
            if epoch % 10 == 0:
                learning_rate = learning_rate * 0.6
        else:
            learning_rate = learning_rate * 0.99

if __name__ == '__main__':
    main()








