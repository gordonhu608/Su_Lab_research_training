from re import sub
from regex import subf
import torch.utils.data as data
import os
import os.path
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import os, sys

BOP_ROOT = 'D:/lm/train_pbr/000000'

def data_transforms(split):
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomCrop(32, padding=4), 
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    return data_transforms[split]

class BopDataset(data.Dataset):
    def __init__(self, root, split='train_pbr', transform='train'):
        self.bop_root = os.path.join(root, split)
        subfolder = os.listdir(self.bop_root)
        self.rgb_files = None
        self.processed_mask = []
        self.transform = transform
        for f in subfolder:
            rgb_folder = os.path.join(self.bop_root, f) + '/rgb'
            mask_folder = os.path.join(self.bop_root, f) + '/mask_visib/'
            rgb_prefix = os.listdir(rgb_folder)
            rgb_files = [os.path.join(rgb_folder, p) for p in rgb_prefix]
            self.rgb_files += rgb_files
            for prefix in rgb_prefix:
                mask_all = None
                for label in range(15):
                    mask_prefix = prefix[:-4] + '_' +str(label).zfill(6) +'.png'
                    mask_file = mask_folder + mask_prefix
                    mask = np.asarray(Image.open(mask_file), dtype=np.uint32)
                    mask = (mask == 255) * (label + 1)
                    mask_all += mask
                self.processed_mask.append(mask_all)

    def __getitem__(self, index):
        rgb = Image.open(self.rgb_files[index]).convert('RGB')
        w, h = rgb.size
        mask = self.processed_mask[index]
        rgb = rgb.transpose((2, 0, 1))
        rgb = torch.as_tensor(rgb.copy()).float().contiguous()

        masks =  torch.as_tensor(mask.copy()).long().contiguous()
            #masks = torch.as_tensor(mask, dtype=torch.uint8)
        img = data_transforms(self.transform)(rgb)
        target = {}
        target['masks'] = masks

        return img, target

    def __len__(self):
        return len(self.rgb_files)

