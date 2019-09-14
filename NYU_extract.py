import numpy as np
from scipy import misc
import h5py
import os
import cv2

dataFile  = 'nyu_depth_v2_labeled.mat'
data = h5py.File(dataFile)

image = np.array(data['images'])
depth = np.array(data['depths'])
print(image.shape)
image = np.transpose(image,(0,3,2,1))
print(image.shape)
depth = np.transpose(depth,(0,2,1))
depth = depth[:, :, :, np.newaxis]
print(depth.shape)
path_rgb = './rgb'
if not os.path.isdir(path_rgb):
    os.makedirs(path_rgb)
path_depth = './depth'
if not os.path.isdir(path_depth):
    os.makedirs(path_depth)

# for i in range(image.shape[0]):
#     print(i)
#     index = str(i)
#     image_index_path = path_rgb + '/rgb' + index + '.png'
#     out_img = image[i, :, :, :]
#     misc.imsave(image_index_path, out_img)
#     print out_img.shape

for i in range(depth.shape[0]):
    print(i)
    index = str(i)
    depth_index_path = path_depth + '/depth' + index + '.png'
    out_depth = depth[i, :, :, :]
    out_depth = np.array(out_depth, np.float32)
    out_depth = out_depth * 1000
    out_depth = np.array(out_depth, np.uint16)
    cv2.imwrite(depth_index_path, out_depth)
    print out_depth.shape
