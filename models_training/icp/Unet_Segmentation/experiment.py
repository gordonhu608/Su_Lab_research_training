import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm.notebook import tqdm
from utils.dice_score import *
from unet import UNet
from utils.posedataset import UnetDataset, UnettestDataset
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from datetime import datetime
from utils.file_utils import *

#from torch.optim.lr_scheduler import ReduceLROnPlateau

ROOT_STATS_DIR = './experiment_data'

class Experiment(object):
    def __init__(self, name):
        config_data = {'batchSize': 2, 'workers': 4, 'nepoch': 10, 'lr': 1e-5,
          'root': 'D:/291dataset', 'val_percent': 0.1,'save_checkpoint': True,
                'img_scale': 0.8, 'amp': False}

        self.testroot = 'D:/291dataset/testing_data_final(2022)/testing_data_final/v2.2/'
        self.__name = name
        self.__experiment_dir = os.path.join(ROOT_STATS_DIR, self.__name)
        self.scale = config_data['img_scale']
        random.seed(123)
        torch.manual_seed(123)
        self.batchSize = config_data['batchSize']
        self.earlystop = 4
        self.__lr = config_data['lr']
        self.amp = config_data['amp']
        # Load Datasets
        dataset = UnetDataset(
            root=  config_data['root'], 
            scale= self.scale,
            data_augmentation=False)

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=config_data['batchSize'],
            shuffle=True,
            num_workers=int(config_data['workers']),
            pin_memory = True)

        val_dataset = UnetDataset(
            root= config_data['root'],
            scale=self.scale,
            split='val',
            data_augmentation=False)

        valdataloader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=config_data['batchSize'],
            shuffle=False,
            drop_last = True,
            num_workers=0,
            pin_memory = True)
        
        test_dataset = UnettestDataset(
            root= self.testroot,
            scale=self.scale)

        testdataloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory = True)

        self.__train_loader =dataloader
        self.__val_loader = valdataloader
        self.__test_loader = testdataloader
        self.num_classes = 82
        
        self.n_val = int(len(val_dataset))
        self.n_train = int(len(dataset))
        # Setup Experiment
        self.__epochs = config_data['nepoch']
        self.__current_epoch = 0
        self.__training_losses = []
        self.__val_losses = []
        self.global_step = 0
        self.__best_model = None  # Save your best model in this field and use this in test method.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Init Model
        blue = lambda x: '\033[94m' + x + '\033[0m'

        self.__model = UNet(n_channels=3, n_classes=self.num_classes, bilinear=True)
        self.grad_scaler = torch.cuda.amp.GradScaler(enabled=self.amp)
        self.__criterion = nn.CrossEntropyLoss()

        self.optimizer = optim.RMSprop(self.__model.parameters(), lr=self.__lr, weight_decay=1e-8, momentum=0.9)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'max', patience=2)  # goal: maximize Dice score
        self.num_batch = len(dataset) / config_data['batchSize']                                  
        self.__init_model()

        # Load Experiment Data if available
        self.__load_experiment()

    # Loads the experiment data if exists to resume training from last saved checkpoint.
    def __load_experiment(self):
        os.makedirs(ROOT_STATS_DIR, exist_ok=True)

        if os.path.exists(self.__experiment_dir):
            self.__training_losses = read_file_in_dir(self.__experiment_dir, 'training_losses.txt')
            self.__val_losses = read_file_in_dir(self.__experiment_dir, 'val_losses.txt')
            self.__current_epoch = len(self.__training_losses)

            model_dict = torch.load(os.path.join(self.__experiment_dir, 'latest_model.pt'))
            self.__model.load_state_dict(model_dict['model'])
            self.optimizer.load_state_dict(model_dict['optimizer'])

        else:
            os.makedirs(self.__experiment_dir)

    def __init_model(self):
        if torch.cuda.is_available():
            self.__model = self.__model.cuda().float()
            self.__criterion = self.__criterion.cuda()

    # Main method to run your experiment.
    def run(self):
        start_epoch = self.__current_epoch
        for epoch in range(start_epoch, self.__epochs):  # loop over the dataset multiple times
            print("start epoch:", epoch)
            start_time = datetime.now()
            self.__current_epoch = epoch
            train_loss= self.__trainandval()
            val_loss = self.__test()
            self.__record_stats(train_loss, val_loss)
            self.__log_epoch_stats(start_time)
            self.__save_model()
            if len(self.__val_losses) > self.earlystop:
                if self.__val_losses[-1] > self.__val_losses[-2] > self.__val_losses[-3] > self.__val_losses[-4]:
                    break 

    def __trainandval(self):
        self.__model.train()
        epoch_loss = 0
        val_score = 0

        with tqdm(total=self.n_train, desc=f'Epoch {self.__current_epoch}/{self.__epochs}', unit='img') as pbar:
            for batch in self.__train_loader:
                images = batch['image']
                true_masks = batch['mask']

                images = images.to(device=self.device, dtype=torch.float32)
                true_masks = true_masks.to(device=self.device, dtype=torch.long)

                with torch.cuda.amp.autocast(enabled=self.amp):
                    masks_pred = self.__model(images)
                    loss = self.__criterion(masks_pred, true_masks) \
                           + dice_loss(F.softmax(masks_pred, dim=1).float(),
                                       F.one_hot(true_masks, self.__model.n_classes).permute(0, 3, 1, 2).float(),
                                       multiclass=True)

                self.optimizer.zero_grad(set_to_none=True)
                self.grad_scaler.scale(loss).backward()
                self.grad_scaler.step(self.optimizer)
                self.grad_scaler.update()

                pbar.update(images.shape[0])
                self.global_step += 1
                epoch_loss += loss.item()
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                # Evaluation round
                division_step = (self.n_train // (10 * self.batchSize))
                if division_step > 0:
                    if (self.global_step) % division_step == 0:
                        val_score = self.__evaluate()
                        self.scheduler.step(val_score)
                        print(f'Val dice score: {val_score}')

        return epoch_loss / len(self.__train_loader)


    def __evaluate(self):
        self.__model.eval()
        num_val_batches = len(self.__val_loader)
        dice_score = 0

       
        for batch in tqdm(self.__val_loader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            image, mask_true = batch['image'], batch['mask']
        
            image = image.to(device=self.device, dtype=torch.float32)
            mask_true = mask_true.to(device=self.device, dtype=torch.long)
            mask_true = F.one_hot(mask_true, self.__model.n_classes).permute(0, 3, 1, 2).float()

            with torch.no_grad():
              
                mask_pred = self.__model(image)

                mask_pred = F.one_hot(mask_pred.argmax(dim=1), self.__model.n_classes).permute(0, 3, 1, 2).float()
                # compute the Dice score, ignoring background
                dice_score += multiclass_dice_coeff(mask_pred[:, 1:, ...], mask_true[:, 1:, ...], reduce_batch_first=False)

        self.__model.train()

        if num_val_batches == 0:
            return dice_score
        return dice_score / num_val_batches


    def __test(self):
        self.__model.eval()
        epoch_loss = 0
        with torch.no_grad():
            for batch in tqdm(self.__val_loader):
                images = batch['image']
                true_masks = batch['mask']

                images = images.to(device=self.device, dtype=torch.float32)
                true_masks = true_masks.to(device=self.device, dtype=torch.long)

                with torch.cuda.amp.autocast(enabled=self.amp):
                    masks_pred = self.__model(images)
                    loss = self.__criterion(masks_pred, true_masks) \
                        + dice_loss(F.softmax(masks_pred, dim=1).float(),
                                    F.one_hot(true_masks, self.__model.n_classes).permute(0, 3, 1, 2).float(),
                                    multiclass=True)

                epoch_loss += loss.item()
                    
        return epoch_loss / len(self.__val_loader)

    def generate(self):

        self.__model = self.__model.eval()
        os.makedirs('./generation', exist_ok=True)
        experiment_dir = './generation/' + self.__name
        os.makedirs(experiment_dir, exist_ok=True)
        with torch.no_grad():
            
            for data in tqdm(self.__test_loader):
                image, prefix = data
                image = image.to(device=self.device, dtype=torch.float32)
                
                pred = self.__model(image).squeeze()
                #print(prefix)
                pred = pred.cpu().data.numpy()
                output = np.argmax(pred, axis=0)
                im = Image.fromarray(output.astype(np.uint8))
               
                im = im.resize((1280, 720), resample=Image.NEAREST)
                #im.show()
                #return im.size
                name = prefix[0] +  "_label_kinect.png" 
                im.save(experiment_dir + '/' + name)

    def __save_model(self):
        root_model_path = os.path.join(self.__experiment_dir, 'latest_model.pt')
        model_dict = self.__model.state_dict()
        model_dict = {'model': model_dict, 'optimizer': self.optimizer.state_dict()}
        torch.save(model_dict, root_model_path)

    def __record_stats(self, train_loss, val_loss):
        self.__training_losses.append(train_loss)
        self.__val_losses.append(val_loss)

        self.plot_stats()

        write_to_file_in_dir(self.__experiment_dir, 'training_losses.txt', self.__training_losses)
        write_to_file_in_dir(self.__experiment_dir, 'val_losses.txt', self.__val_losses)

    def __log(self, log_str, file_name=None):
        print(log_str)
        log_to_file_in_dir(self.__experiment_dir, 'all.log', log_str)
        if file_name is not None:
            log_to_file_in_dir(self.__experiment_dir, file_name, log_str)

    def __log_epoch_stats(self, start_time):
        time_elapsed = datetime.now() - start_time
        time_to_completion = time_elapsed * (self.__epochs - self.__current_epoch - 1)
        train_loss = self.__training_losses[self.__current_epoch]
        val_loss = self.__val_losses[self.__current_epoch]
        summary_str = "Epoch: {}, Train Loss: {}, Val Loss: {},Took {}, ETA: {}\n"
        summary_str = summary_str.format(self.__current_epoch + 1, train_loss, val_loss, 
                                         str(time_elapsed),str(time_to_completion))
        self.__log(summary_str, 'epoch.log')

    def plot_stats(self):
        
        e = len(self.__training_losses)
        x_axis = np.arange(1, e + 1, 1)
        plt.figure()
        plt.plot(x_axis, self.__training_losses, label="Training Loss")
        plt.plot(x_axis, self.__val_losses, label="Validation Loss")
        plt.xlabel("Epochs")
        plt.legend(loc='best')
        plt.title(self.__name + " Stats Plot")
        plt.savefig(os.path.join(self.__experiment_dir, "stat_plot.png"))
        plt.show()
