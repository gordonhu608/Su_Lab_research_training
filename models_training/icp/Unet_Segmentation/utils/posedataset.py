from __future__ import print_function
import torch.utils.data as data
import os
import os.path
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import pickle

def load_pickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def get_split_files(split_name, split_dir, training_data_dir):
    with open(os.path.join(split_dir, f"{split_name}.txt"), 'r') as f:
        prefix = [os.path.join(training_data_dir, line.strip()) for line in f if line.strip()]
        rgb = [p + "_color_kinect.png" for p in prefix]
        depth = [p + "_depth_kinect.png" for p in prefix]
        label = [p + "_label_kinect.png" for p in prefix]
        meta = [p + "_meta.pkl" for p in prefix]
    return rgb, depth, label, meta

def to_homog(points):
    # N: number of points
    ones = np.ones((points.shape[0],1))
    points_homog = np.concatenate([points, ones], axis=-1)
    return points_homog

def preprocess(pil_img, scale, is_mask):
    w, h = pil_img.size
    newW, newH = int(scale * w), int(scale * h)

    pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
    img_ndarray = np.asarray(pil_img)

    if img_ndarray.ndim == 2 and not is_mask:
        img_ndarray = img_ndarray[np.newaxis, ...]
    elif not is_mask:
        img_ndarray = img_ndarray.transpose((2, 0, 1))

    if not is_mask:
        img_ndarray = img_ndarray / 255

    return img_ndarray

class UnetDataset(data.Dataset):
    def __init__(self,
                 root,
                 scale,
                 split='train',
                 data_augmentation=True,
                 class_choice=None):
        self.scale = scale
        self.root = root
        self.seg_classes = {}
        self.data_augmentation = data_augmentation

        self.training_data_dir = root +  "/training_data/v2.2"
        self.split_dir = root + "/training_data/splits/v2"
        
        self.rgb_files, _, self.label_files, _ = \
            get_split_files(split, self.split_dir, self.training_data_dir)

        #self.rgb_files = self.rgb_files[:10]
        #self.label_files = self.label_files[:10]

        #self.transform = transforms.CenterCrop((450, 800))

    def __getitem__(self, index):
        rgb = Image.open(self.rgb_files[index])
        label = Image.open(self.label_files[index])
        #rgb = self.transform(rgb)
        #label = self.transform(label)

        assert rgb.size ==label.size, \
            'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'
        img = preprocess(rgb, self.scale, is_mask=False)
        mask = preprocess(label, self.scale, is_mask=True)

        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).long().contiguous()
        }
        

    def __len__(self):
        return len(self.rgb_files)


class UnettestDataset(data.Dataset):
    def __init__(self,
                 root,
                 scale):

        self.scale = scale
        self.root = root

        self.prefix = np.unique([x.split('_')[0] for x in os.listdir(self.root)])
        self.rgb_files = [self.root + p + "_color_kinect.png" for p in self.prefix]
        #self.transform = transforms.CenterCrop((450, 800))

    def __getitem__(self, index):

        prefix = self.prefix[index]
        rgb = Image.open(self.rgb_files[index])
        #rgb = self.transform(rgb)

        img = preprocess(rgb, self.scale, is_mask=False)

        return torch.as_tensor(img.copy()).float().contiguous(), prefix

    def __len__(self):
        return len(self.rgb_files)
