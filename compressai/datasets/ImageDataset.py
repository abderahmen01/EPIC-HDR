import os
import functools
import numpy as np
import pandas as pd
import imageio
from torch.utils.data import Dataset
import random
from compressai.registry import register_dataset
from compressai.datasets import build_nlp
import colorsys
import torch
import matplotlib.pyplot as plt
import skimage.color as color
import cv2
from PIL import Image
import scipy.io  as scio
import matplotlib
from skimage import io, transform
from compressai.datasets import np_transforms
IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.exr', '.hdr']
# IMG_EXTENSIONS = ['.exr', '.hdr']

def show(x, title=None, cbar=False, figsize=None):
    import matplotlib.pyplot as plt
    plt.figure(figsize=figsize)
    plt.imshow(x, interpolation='nearest', cmap='gray')
    if title:
        plt.title(title)
    if cbar:
        plt.colorbar()
    plt.show()

def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.
    Args:
        filename (string): path to a file
        extensions (iterable of strings): extensions to consider (lowercase)
    Returns:
        bool: True if the filename ends with one of given extensions
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)


def image_loader(image_name):
    if has_file_allowed_extension(image_name, IMG_EXTENSIONS):
        img = imageio.imread(image_name)
    return img


def get_default_img_loader():
    return functools.partial(image_loader)

# @register_dataset("ImageDataset")
class ImageDataset(Dataset):
    def __init__(self, csv_file,
                 img_dir,
                 transform=None,
                 test=False,
                 get_loader=get_default_img_loader):
        self.data = pd.read_csv(csv_file, sep=' ', header=None)
        self.img_dir = img_dir
        self.test = test
        self.loader = get_loader()
        self.transform = transform


    def __getitem__(self, index):

        image_name = os.path.join(self.img_dir, self.data.iloc[index, 0])
        #image_name = '/home/h428ti/桌面/HDR_images_rename/0074.hdr'
        I = self.loader(image_name)
        re_I = self.Crop(I,448)
        hdr_h = color.rgb2hsv(re_I)

        if self.transform is not None:
            hdr_l = self.transform(hdr_h[:, :, 2][:, :, np.newaxis])

        if self.test:
            s_min = 5.0
            # s_max = self.data.iloc[index, 1]
            s_max =1e8
            # hdr_h = torch.from_numpy(hdr_h)
            hdr_h = self.transform(hdr_h)
        else:
            s_min = 5.0
            # s_max = 1e5
            s_max = random.choice([1e6, 1e7, 1e8])
            hdr_h = self.transform(hdr_h)

        hdr_l = hdr_l.unsqueeze(0)
        gI = ((hdr_l - hdr_l.min()) / (hdr_l.max() - hdr_l.min()))
        gI = (s_max - s_min) * gI + s_min
        nlp = build_nlp.nlpclass()
        nlp = nlp.nlp(gI)

        for i in range(nlp.__len__()):
            nlp[i] = nlp[i].squeeze().unsqueeze(0)
        gI = gI.squeeze().unsqueeze(0)

        sample = {'nlp_I': nlp, 'simulate_L': gI, 'hdr_name': self.data.iloc[index, 0], 'hdr_h': hdr_h}
        return sample

    def __len__(self):
        return len(self.data.index)
    def to_tensor(self,np_image):
        return torch.FloatTensor(np_image.transpose((2, 0, 1)).copy())
    def Resize(self,np_image):
        n_t = np_transforms.Scale(512)
        return n_t(np_image)

    def Crop(self,np_image,size):
        n_t = np_transforms.Compose([np_transforms.RandomCrop(size),
        np_transforms.RandomHorizontalFlip()])
        return n_t(np_image)