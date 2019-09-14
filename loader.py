import os
import numpy as np
import h5py
from PIL import Image
import torch
import torch.utils.data as data
import torchvision.transforms as transforms


class NyuDepthLoader(data.Dataset):
    def __init__(self, data_path, lists):
        self.data_path = data_path
        self.lists = lists

        self.nyu = h5py.File(self.data_path)

        self.imgs = self.nyu['images']
        self.dpts = self.nyu['depths']

