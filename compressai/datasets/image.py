from pathlib import Path
import imageio
import pandas as pd
import torch
# import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from compressai.registry import register_dataset
from compressai.datasets import build_nlp
import os
import random
import os.path
import math
import cv2
import numpy as np
from compressai.datasets import np_transforms
import skimage.color as color
imageio.plugins.freeimage.download()
HDR_EXTENSIONS = [
    '.hdr',
]

def show(x, title=None, cbar=False, figsize=None):
    import matplotlib.pyplot as plt
    plt.figure(figsize=figsize)
    plt.imshow(x, interpolation='nearest', cmap='gray')
    if title:
        plt.title(title)
    if cbar:
        plt.colorbar()
    plt.show()


def is_hdr_file(filename):
    return any(filename.endswith(extension) for extension in HDR_EXTENSIONS)


def imageio_hdr_loader_loader(path):
    # print(str(path))
    img = imageio.imread(path)
    return img # cv2.imread(path, flags=cv2.IMREAD_ANYDEPTH)   # np.araay

def cv2_hdr_loader_loader(path):
    return cv2.imread(path, flags=cv2.IMREAD_ANYDEPTH)

@register_dataset("ImageFolder")
class ImageFolder(Dataset):
    """ ImageFolder can be used to load images where there are no labels."""

    def __init__(self,csv_file,
                 hdr_root,
                 patch_size=512,
                 train=True,
                 test=False
                 ):

        hdr_images = []
        self.test = test

        for filename in os.listdir(hdr_root):
            if is_hdr_file(filename):
                hdr_images.append('{}'.format(filename))

        self.hdr_root = hdr_root
        self.hdr_imgs = hdr_images
        self.data = pd.read_csv(csv_file, sep=' ', header=None)
        print('length of hdr_imgs: ', len(self.hdr_imgs))
        self.hdr_loader_imgio = imageio_hdr_loader_loader
        self.hdr_loader_cv2 = cv2_hdr_loader_loader
        self.patch_size = patch_size
        self.train = train
        self.transform = np_transforms.Compose([
            np_transforms.ToTensor()
        ])

    @staticmethod
    def _random_flip(img, mode):
        return cv2.flip(img, mode)

    @staticmethod
    def _random_rot(img, rot_coin):
        if rot_coin == 0:
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        if rot_coin == 1:
            img = cv2.rotate(img, cv2.ROTATE_180)
        if rot_coin == 2:
            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

        return img

    def __getitem__(self, index):
        filename = self.hdr_imgs[index]
        filename2 = os.path.join(self.hdr_root, filename)

        hdr = self.hdr_loader_imgio(filename2)
        re_I = hdr # [0:448, 0:448, :]
        hdr_h = color.rgb2hsv(re_I)
        hdr_l = self.transform(hdr_h[:, :, 2][:, :, np.newaxis])

        ####################################
        # Load image for HDR reconstruction
        ####################################
        hdr1 = self.transform(re_I)
        hdr1 = torch.log10(hdr1).clamp_(-8, 8)
        reference = self.transform(re_I)
        ####################################

        if self.test:
            s_min = 1e-8
            ##############
            s_max = 1e4  # 4 5 6 7
            # s_max = random.choice([1e3, 1e4, 1e5])

        else:
            s_min = 1e-8
            # s_max = 5e4
            s_max = random.choice([1e4, 1e5, 1e6, 1e7, 1e8])

        gI = ((hdr_l - hdr_l.min()) / (hdr_l.max() - hdr_l.min()))
        gI = (1 - s_min) * gI + s_min
        hdr = torch.log10(gI)

        hdr_h = self.transform(hdr_h)
        hdr_hs = hdr_h[0:2, :, :]
        hs = hdr_hs.squeeze()
        in_hsv = torch.cat([hs, hdr], dim=0)
        ref_hsv = torch.cat([hs, gI], dim=0)

        sample = {'nlp_I': in_hsv, 'hdr_name': filename, 'hdr_l': ref_hsv, 's_max': s_max, 'hdr': hdr1, 'reference': reference}

        return sample

    def __len__(self):
        return len(self.hdr_imgs)

