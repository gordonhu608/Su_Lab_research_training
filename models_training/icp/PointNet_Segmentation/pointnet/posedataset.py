import torch.utils.data as data
import os
import os.path
import torch
import numpy as np
import sys
import json
from PIL import Image
import pickle
import torchvision.transforms as transforms

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

class pointnetDataset(data.Dataset):
    def __init__(self,
                 root,
                 npoints=2500,
                 split='train',
                 data_augmentation=True):
        self.npoints = npoints
        self.root = root
        self.seg_classes = {}
        self.data_augmentation = data_augmentation

        self.training_data_dir = root +  "/training_data/v2.2"
        self.split_dir = root + "/training_data/splits/v2"
        
        self.rgb_files, self.depth_files, self.label_files, self.meta_files = \
            get_split_files(split, self.split_dir, self.training_data_dir)
        
        self.crop = transforms.CenterCrop((450, 800))

    def __getitem__(self, index):
        #rgb = np.array(Image.open(self.rgb_files[index])) / 255   # convert 0-255 to 0-1
        depth = self.crop(Image.open(self.depth_files[index]))
        label = self.crop(Image.open(self.label_files[index]))

        depth = np.array(depth) / 1000   # convert from mm to m
        label = np.array(label)
        meta = load_pickle(self.meta_files[index])

        intrinsic = meta['intrinsic']
        extrinsic = meta['extrinsic']
        z = depth
        v, u = np.indices(z.shape)
        uv1 = np.stack([u + 0.5, v + 0.5, np.ones_like(z)], axis=-1)
        points_viewer = uv1 @ np.linalg.inv(intrinsic).T * z[..., None]  
        viewer_points = to_homog(points_viewer.reshape(-1, 3))
        world_points = viewer_points @ np.linalg.inv(extrinsic).T
        point_set = world_points[:, :3].astype(np.float32)
        seg = label.reshape(-1).astype(np.int64)

        choice = np.random.choice(len(seg), self.npoints, replace=True)
        #resample
        point_set = point_set[choice, :]

        point_set = point_set - np.expand_dims(np.mean(point_set, axis = 0), 0) # center
        dist = np.max(np.sqrt(np.sum(point_set ** 2, axis = 1)),0)
        point_set = point_set / dist #scale

        if self.data_augmentation:
            theta = np.random.uniform(0,np.pi*2)
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
            point_set[:,[0,2]] = point_set[:,[0,2]].dot(rotation_matrix) # random rotation
            point_set += np.random.normal(0, 0.02, size=point_set.shape) # random jitter

        seg = seg[choice]
        point_set = torch.from_numpy(point_set)
        seg = torch.from_numpy(seg)

        return point_set, seg

    def __len__(self):
        return len(self.depth_files)


class pointtestDataset(data.Dataset):
    def __init__(self,
                 root,
                 npoints=2500,
                 data_augmentation=False):
        self.npoints = npoints
        self.root = root
        self.seg_classes = {}

        self.prefix = np.unique([x.split('_')[0] for x in os.listdir(self.root)])
        self.rgb_files = [self.root + p + "_color_kinect.png" for p in self.prefix]
        self.depth_files = [self.root + p + "_depth_kinect.png" for p in self.prefix]
        self.meta_files = [self.root + p + "_meta.pkl" for p in self.prefix]

    def __getitem__(self, index):
        #rgb = np.array(Image.open(self.rgb_files[index])) / 255   # convert 0-255 to 0-1
        depth = self.crop(Image.open(self.depth_files[index]))
        depth = np.array(depth) / 1000   # convert from mm to m
        meta = load_pickle(self.meta_files[index])
        prefix = self.prefix[index]

        intrinsic = meta['intrinsic']
        extrinsic = meta['extrinsic']
        z = depth
        v, u = np.indices(z.shape)
        uv1 = np.stack([u + 0.5, v + 0.5, np.ones_like(z)], axis=-1)
        points_viewer = uv1 @ np.linalg.inv(intrinsic).T * z[..., None]  
        viewer_points = to_homog(points_viewer.reshape(-1, 3))
        world_points = viewer_points @ np.linalg.inv(extrinsic).T
        point_set = world_points[:, :3].astype(np.float32)

        choice = np.random.choice(len(point_set), self.npoints, replace=True)
        #resample
        point_set = point_set[choice, :]

        center = np.expand_dims(np.mean(point_set, axis = 0), 0) # center
        point_set = point_set - center
        dist = np.max(np.sqrt(np.sum(point_set ** 2, axis = 1)),0)
        point_set = point_set / dist #scale

        point_set = torch.from_numpy(point_set)

        return point_set, prefix, center, dist

    def __len__(self):
        return len(self.depth_files)
