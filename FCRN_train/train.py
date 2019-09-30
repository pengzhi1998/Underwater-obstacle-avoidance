# import torch
# import torch.utils.data
# import torchvision
# # from loader import *
# import numpy as np
# import os
# from fcrn import FCRN
# from torch.autograd import Variable
# from weights import load_weights
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plot
# import pickle
# dtype = torch.cuda.FloatTensor
# weights_file = "NYU_ResNet-UpProj.npy"
# import random
# import cv2
#
# path = 'predict/'
#
# def load_split():
#     current_directoty = os.getcwd()
#     train_lists_path = current_directoty + '/trainIdxs.txt'
#     test_lists_path = current_directoty + '/testIdxs.txt'
#
#     train_f = open(train_lists_path)
#     test_f = open(test_lists_path)
#
#     train_lists = []
#     test_lists = []
#
#     train_lists_line = train_f.readline()
#     while train_lists_line:
#         train_lists.append(int(train_lists_line) - 1)
#         train_lists_line = train_f.readline()
#     train_f.close()
#
#     test_lists_line = test_f.readline()
#     while test_lists_line:
#         test_lists.append(int(test_lists_line) - 1)
#         test_lists_line = test_f.readline()
#     test_f.close()
#
#     val_start_idx = int(len(train_lists) * 0.8)
#
#     val_lists = train_lists[val_start_idx:-1]
#     train_lists = train_lists[0:val_start_idx]
#
#     return train_lists, val_lists, test_lists
#
#
# def main():
#     batch_size = 16
#     data_path = 'nyu_depth_v2_labeled.mat'
#     learning_rate = 1.0e-5
#     monentum = 0.9
#     weight_decay = 0.0005
#     num_epochs = 100
#     resume_from_file = False
#
#     # 1.Load data
#     # train_lists, val_lists, test_lists = load_split()
#     print("Loading data......")
#     train_data = pickle.load(open('rgb_depth_images_training_real.p', "rb"))
#     val_data = pickle.load(open('rgb_depth_images_validating_real.p', "rb"))
#     train_loader = torch.utils.data.DataLoader(train_data, batch_size=16, shuffle=True, drop_last=True)
#     val_loader = torch.utils.data.DataLoader(val_data, batch_size=16, shuffle=True, drop_last=True)
#     # train_loader = torch.utils.data.DataLoader(NyuDepthLoader(data_path, train_lists),
#     #                                            batch_size=batch_size, shuffle=True, drop_last=True)
#     # val_loader = torch.utils.data.DataLoader(NyuDepthLoader(data_path, val_lists),
#     #                                            batch_size=batch_size, shuffle=True, drop_last=True)
#     # train_loader = torch.utils.data.DataLoader(training_data,
#     #                                            batch_size=batch_size, shuffle=True, drop_last=True)
#     # val_loader = torch.utils.data.DataLoader(testing_data,
#     #                                            batch_size=batch_size, shuffle=True, drop_last=True)
#     # test_loader = torch.utils.data.DataLoader(NyuDepthLoader(data_path, test_lists),
#     #                                          batch_size=batch_size, shuffle=True, drop_last=True)
#     # print(train_loader)
#     # 2.Load model
#     print("Loading model......")
#     model = FCRN(batch_size)
#
#     model.load_state_dict(load_weights(model, weights_file, dtype))
#     print "model_dict updated."
#     model = model.cuda()
#
#     # 3.Loss
#     loss_fn = torch.nn.MSELoss().cuda()
#     print("loss_fn set.")
#
#     # 5.Train
#     best_val_err = 1.0e3
#
#     # validate
#     model.eval()
#     num_correct, num_samples = 0, 0
#     loss_local = 0
#     with torch.no_grad():
#         for rgb_image, depth_image, mask in val_loader:
#             points = np.sum(mask.numpy())
#
#             if random.random() < 0.5: # with 50% probability to flip the rgb image and the depth image
#                 rgb_image = torch.from_numpy(np.flip(rgb_image, axis=1).copy())
#                 depth_image = torch.from_numpy(np.flip(depth_image, axis=1).copy())
#                 mask = torch.from_numpy(np.flip(mask, axis=1).copy())
#
#             if random.random() < 0.5: # with 50% probability to flip the rgb image and the depth image
#                 rgb_image = torch.from_numpy(np.flip(rgb_image, axis=2).copy())
#                 depth_image = torch.from_numpy(np.flip(depth_image, axis=2).copy())
#                 mask = torch.from_numpy(np.flip(mask, axis=2).copy())
#
#             input_var = rgb_image.permute(0, 3, 1, 2)
#             input_var = Variable(input_var.type(dtype))
#             depth_var = depth_image.permute(0, 3, 1, 2)
#             depth_var = Variable(depth_var.type(dtype))
#             mask_var = mask.permute(0, 3, 1, 2)
#             mask_var = Variable(mask_var.type(dtype))
#
#             output = model(input_var)
#
#             # input_rgb_image = input_var[0].data.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
#             # input_gt_depth_image = depth_var[0][0].data.cpu().numpy().astype(np.float32)
#             # pred_depth_image = output[0].data.squeeze().cpu().numpy().astype(np.float32)
#             #
#             # # input_gt_depth_image /= np.max(input_gt_depth_image)
#             # # pred_depth_image /= np.max(pred_depth_image)
#             #
#             # plot.imsave('input_rgb_epoch_0.png', input_rgb_image)
#             # plot.imsave('gt_depth_epoch_0.png', input_gt_depth_image, cmap="viridis")
#             # plot.imsave('pred_depth_epoch_0.png', pred_depth_image, cmap="viridis")
#
#             # depth_var = depth_var[:, 0, :, :]
#             # loss_fn_local = torch.nn.MSELoss()
#
#             # loss_local += loss_fn(output, depth_var)
#             loss_local = loss_fn(torch.mul(output, mask_var), torch.mul(depth_var, mask_var)) / points * 1108992
#
#             num_samples += 1
#
#     err = float(loss_local) / num_samples
#     print('val_error before train:', err)
#
#     start_epoch = 0
#
#     resume_file = 'checkpoint.pth.tar'
#     if resume_from_file:
#         if os.path.isfile(resume_file):
#             print("=> loading checkpoint '{}'".format(resume_file))
#             checkpoint = torch.load(resume_file)
#             start_epoch = checkpoint['epoch']
#             model.load_state_dict(checkpoint['state_dict'])
#             print("=> loaded checkpoint '{}' (epoch {})"
#                   .format(resume_file, checkpoint['epoch']))
#         else:
#             print("=> no checkpoint found at '{}'".format(resume_file))
#
#     for epoch in range(num_epochs):
#
#         # 4.Optim
#         optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#         # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=monentum)
#         # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=monentum, weight_decay=weight_decay)
#         print("optimizer set.")
#
#         print('Starting train epoch %d / %d' % (start_epoch + epoch + 1, num_epochs))
#         model.train()
#         running_loss = 0
#         count = 0
#         epoch_loss = 0
#
#         #for i, (input, depth) in enumerate(train_loader):
#         for input, depth, mask in train_loader:
#             # points = np.sum(mask)
#             # if random.random() < 0.5: # with 50% probability to flip the rgb image and the depth image
#             #     rgb_image = torch.from_numpy(np.flip(rgb_image, axis=1).copy())
#             #     depth_image = torch.from_numpy(np.flip(depth_image, axis=1).copy())
#             #     mask = torch.from_numpy(np.flip(mask, axis=1).copy())
#             #
#             # if random.random() < 0.5: # with 50% probability to flip the rgb image and the depth image
#             #     rgb_image = torch.from_numpy(np.flip(rgb_image, axis=2).copy())
#             #     depth_image = torch.from_numpy(np.flip(depth_image, axis=2).copy())
#             #     mask = torch.from_numpy(np.flip(mask, axis=2).copy())
#             # input_var = Variable(rgb_image.type(dtype))
#             # depth_var = Variable(depth_image.type(dtype))
#             # mask = Variable(mask.type(dtype))
#
#             points = np.sum(mask.numpy())
#
#             if random.random() < 0.5: # with 50% probability to flip the rgb image and the depth image
#                 rgb_image = torch.from_numpy(np.flip(rgb_image, axis=1).copy())
#                 depth_image = torch.from_numpy(np.flip(depth_image, axis=1).copy())
#                 mask = torch.from_numpy(np.flip(mask, axis=1).copy())
#
#             if random.random() < 0.5: # with 50% probability to flip the rgb image and the depth image
#                 rgb_image = torch.from_numpy(np.flip(rgb_image, axis=2).copy())
#                 depth_image = torch.from_numpy(np.flip(depth_image, axis=2).copy())
#                 mask = torch.from_numpy(np.flip(mask, axis=2).copy())
#
#             input_var = rgb_image.permute(0, 3, 1, 2)
#             input_var = Variable(input_var.type(dtype))
#             depth_var = depth_image.permute(0, 3, 1, 2)
#             depth_var = Variable(depth_var.type(dtype))
#             mask_var = mask.permute(0, 3, 1, 2)
#             mask_var = Variable(mask_var.type(dtype))
#
#             output = model(input_var)
#             loss = loss_fn(torch.mul(output, mask_var), torch.mul(depth_var, mask_var)) / points * 1108992
#             print('loss:', loss.item())
#             count += 1
#             running_loss += loss.data.cpu().numpy()
#
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#
#         epoch_loss = running_loss / count
#         print('epoch loss:', epoch_loss)
#
#         # validate
#         model.eval()
#         num_correct, num_samples = 0, 0
#         loss_local = 0
#         with torch.no_grad():
#             for input, depth, mask in val_loader:
#                 # input_var = Variable(input.type(dtype))
#                 # depth_var = Variable(depth.type(dtype))
#                 #
#                 # output = model(input_var)
#                 #
#                 # input_rgb_image = input_var[0].data.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
#                 # input_gt_depth_image = depth_var[0][0].data.cpu().numpy().astype(np.float32)
#                 # pred_depth_image = output[0].data.squeeze().cpu().numpy().astype(np.float32)
#                 #
#                 # input_gt_depth_image /= np.max(input_gt_depth_image)
#                 # pred_depth_image /= np.max(pred_depth_image)
#
#                 points = np.sum(mask.numpy())
#
#                 if random.random() < 0.5: # with 50% probability to flip the rgb image and the depth image
#                     rgb_image = torch.from_numpy(np.flip(rgb_image, axis=1).copy())
#                     depth_image = torch.from_numpy(np.flip(depth_image, axis=1).copy())
#                     mask = torch.from_numpy(np.flip(mask, axis=1).copy())
#
#                 if random.random() < 0.5: # with 50% probability to flip the rgb image and the depth image
#                     rgb_image = torch.from_numpy(np.flip(rgb_image, axis=2).copy())
#                     depth_image = torch.from_numpy(np.flip(depth_image, axis=2).copy())
#                     mask = torch.from_numpy(np.flip(mask, axis=2).copy())
#
#                 input_var = rgb_image.permute(0, 3, 1, 2)
#                 input_var = Variable(input_var.type(dtype))
#                 depth_var = depth_image.permute(0, 3, 1, 2)
#                 depth_var = Variable(depth_var.type(dtype))
#                 mask_var = mask.permute(0, 3, 1, 2)
#                 mask_var = Variable(mask_var.type(dtype))
#
#                 # plot.imsave('input_rgb_epoch_{}.png'.format(start_epoch + epoch + 1), input_rgb_image)
#                 # plot.imsave('gt_depth_epoch_{}.png'.format(start_epoch + epoch + 1), input_gt_depth_image, cmap="viridis")
#                 # plot.imsave('pred_depth_epoch_{}.png'.format(start_epoch + epoch + 1), pred_depth_image, cmap="viridis")
#
#                 # depth_var = depth_var[:, 0, :, :]
#                 # loss_fn_local = torch.nn.MSELoss()
#                 # input_rgb_image = input_var[0].squeeze().cpu().numpy().astype(np.uint8)
#                 input_gt_depth_image = depth_var[0].data.squeeze().cpu().numpy().astype(np.float32)
#                 pred_depth_image = output[0].data.squeeze().cpu().numpy().astype(np.float32)
#                 # cv2.imwrite(path + 'input_rgb' + str(epoch+1) + '.png', input_rgb_image)
#                 cv2.imwrite(path + 'input_depth' + str(epoch+1) + '.png', input_gt_depth_image * 50)
#                 cv2.imwrite(path + 'output_depth' + str(epoch+1) + '.png', pred_depth_image * 50)
#                 output = model(input_var)
#                 loss_local = loss_fn(torch.mul(output, mask_var), torch.mul(depth_var, mask_var)) / points * 1108992
#                 # loss_local += loss_fn(output, depth_var)
#
#                 num_samples += 1
#
#         err = float(loss_local) / num_samples
#         print('val_error:', err)
#
#         if err < best_val_err:
#             best_val_err = err
#             torch.save({
#                 'epoch': start_epoch + epoch + 1,
#                 'state_dict': model.state_dict(),
#                 'optimizer': optimizer.state_dict(),
#             }, 'checkpoint.pth.tar')
#
#         if epoch % 10 == 0:
#             learning_rate = learning_rate * 0.6
#
#
# if __name__ == '__main__':
#     main()

import torch
import torch.utils.data
import torchvision
from loader import *
import os
from fcrn import FCRN
from torch.autograd import Variable
from weights import load_weights
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plot

dtype = torch.cuda.FloatTensor
weights_file = "NYU_ResNet-UpProj.npy"


def load_split():
    current_directoty = os.getcwd()
    train_lists_path = current_directoty + '/trainIdxs.txt'
    test_lists_path = current_directoty + '/testIdxs.txt'

    train_f = open(train_lists_path)
    test_f = open(test_lists_path)

    train_lists = []
    test_lists = []

    train_lists_line = train_f.readline()
    while train_lists_line:
        train_lists.append(int(train_lists_line) - 1)
        train_lists_line = train_f.readline()
    train_f.close()

    test_lists_line = test_f.readline()
    while test_lists_line:
        test_lists.append(int(test_lists_line) - 1)
        test_lists_line = test_f.readline()
    test_f.close()

    val_start_idx = int(len(train_lists) * 0.8)

    val_lists = train_lists[val_start_idx:-1]
    train_lists = train_lists[0:val_start_idx]

    return train_lists, val_lists, test_lists


def main():
    batch_size = 16
    # data_path = 'nyu_depth_v2_labeled.mat'
    data_path = 'test.mat'
    learning_rate = 1.0e-5
    monentum = 0.9
    weight_decay = 0.0005
    num_epochs = 50
    resume_from_file = False

    # 1.Load data
    train_lists, val_lists, test_lists = load_split()
    print("Loading data......")
    train_loader = torch.utils.data.DataLoader(NyuDepthLoader(data_path, train_lists),
                                               batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(NyuDepthLoader(data_path, val_lists),
                                               batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = torch.utils.data.DataLoader(NyuDepthLoader(data_path, test_lists),
                                             batch_size=batch_size, shuffle=True, drop_last=True)
    print(train_loader)
    # 2.Load model
    print("Loading model......")
    model = FCRN(batch_size)
    model.load_state_dict(load_weights(model, weights_file, dtype))
    """
    print('\nresnet50 keys:\n')
    for key, value in resnet50_pretrained_dict.items():
        print(key, value.size())
    """
    #model_dict = model.state_dict()
    """
    print('\nmodel keys:\n')
    for key, value in model_dict.items():
        print(key, value.size())
    print("resnet50.dict loaded.")
    """
    # load pretrained weights
    #resnet50_pretrained_dict = {k: v for k, v in resnet50_pretrained_dict.items() if k in model_dict}
    print("resnet50_pretrained_dict loaded.")
    """
    print('\nresnet50_pretrained keys:\n')
    for key, value in resnet50_pretrained_dict.items():
        print(key, value.size())
    """
    #model_dict.update(resnet50_pretrained_dict)
    print("model_dict updated.")
    """
    print('\nupdated model dict keys:\n')
    for key, value in model_dict.items():
        print(key, value.size())
    """
    #model.load_state_dict(model_dict)
    print("model_dict loaded.")
    model = model.cuda()

    # 3.Loss
    loss_fn = torch.nn.MSELoss().cuda()
    print("loss_fn set.")

    # 5.Train
    best_val_err = 1.0e3

    # validate
    model.eval()
    num_correct, num_samples = 0, 0
    loss_local = 0
    with torch.no_grad():
        for input, depth in val_loader:
            input_var = Variable(input.type(dtype))
            depth_var = Variable(depth.type(dtype))

            output = model(input_var)

            input_rgb_image = input_var[0].data.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
            input_gt_depth_image = depth_var[0][0].data.cpu().numpy().astype(np.float32)
            pred_depth_image = output[0].data.squeeze().cpu().numpy().astype(np.float32)

            input_gt_depth_image /= np.max(input_gt_depth_image)
            pred_depth_image /= np.max(pred_depth_image)

            plot.imsave('input_rgb_epoch_0.png', input_rgb_image)
            plot.imsave('gt_depth_epoch_0.png', input_gt_depth_image, cmap="viridis")
            plot.imsave('pred_depth_epoch_0.png', pred_depth_image, cmap="viridis")

            # depth_var = depth_var[:, 0, :, :]
            # loss_fn_local = torch.nn.MSELoss()

            loss_local += loss_fn(output, depth_var)

            num_samples += 1

    err = float(loss_local) / num_samples
    print('val_error before train:', err)

    start_epoch = 0

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

    for epoch in range(num_epochs):

        # 4.Optim
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=monentum)
        # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=monentum, weight_decay=weight_decay)
        print("optimizer set.")

        print('Starting train epoch %d / %d' % (start_epoch + epoch + 1, num_epochs))
        model.train()
        running_loss = 0
        count = 0
        epoch_loss = 0

        #for i, (input, depth) in enumerate(train_loader):
        for input, depth in train_loader:
            # input, depth = data
            #input_var = input.cuda()
            #depth_var = depth.cuda()
            input_var = Variable(input.type(dtype))
            depth_var = Variable(depth.type(dtype))

            output = model(input_var)
            loss = loss_fn(output, depth_var)
            print('loss:', loss.data.cpu())
            count += 1
            running_loss += loss.data.cpu().numpy()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        epoch_loss = running_loss / count
        print('epoch loss:', epoch_loss)

        # validate
        model.eval()
        num_correct, num_samples = 0, 0
        loss_local = 0
        with torch.no_grad():
            for input, depth in val_loader:
                input_var = Variable(input.type(dtype))
                depth_var = Variable(depth.type(dtype))

                output = model(input_var)

                input_rgb_image = input_var[0].data.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                input_gt_depth_image = depth_var[0][0].data.cpu().numpy().astype(np.float32)
                pred_depth_image = output[0].data.squeeze().cpu().numpy().astype(np.float32)

                input_gt_depth_image /= np.max(input_gt_depth_image)
                pred_depth_image /= np.max(pred_depth_image)

                plot.imsave('input_rgb_epoch_{}.png'.format(start_epoch + epoch + 1), input_rgb_image)
                plot.imsave('gt_depth_epoch_{}.png'.format(start_epoch + epoch + 1), input_gt_depth_image, cmap="viridis")
                plot.imsave('pred_depth_epoch_{}.png'.format(start_epoch + epoch + 1), pred_depth_image, cmap="viridis")

                # depth_var = depth_var[:, 0, :, :]
                # loss_fn_local = torch.nn.MSELoss()

                loss_local += loss_fn(output, depth_var)

                num_samples += 1

        err = float(loss_local) / num_samples
        print('val_error:', err)

        if err < best_val_err:
            best_val_err = err
            torch.save({
                'epoch': start_epoch + epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, 'checkpoint.pth.tar')

        if epoch % 10 == 0:
            learning_rate = learning_rate * 0.6


if __name__ == '__main__':
    main()
