import os
import random
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from pointnet.posedataset import pointnetDataset, pointtestDataset
from pointnet.model import PointNetSeg,feature_transform_regularizer
import torch.nn.functional as F
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import time
from utils.file_utils import *

import torch.nn as nn
#from torch.optim.lr_scheduler import ReduceLROnPlateau

ROOT_STATS_DIR = './experiment_data'

class Experiment(object):
    def __init__(self, name):
        config_data = {'batchSize': 32, 'workers': 0, 'nepoch': 10, 
         'model': "", 'dataset': "", 'npoints': 5000, 'root': 'D:/291dataset',
          'feature_transform': True }

        self.testroot = 'D:/291dataset/testing_data_final(2022)/testing_data_final/v2.2/'
        self.__name = name
        self.__experiment_dir = os.path.join(ROOT_STATS_DIR, self.__name)
        self.npoints = config_data['npoints']
        random.seed(123)
        torch.manual_seed(123)
        self.batchSize = config_data['batchSize']
        self.earlystop = 4
        #self.__lr = config_data['experiment']['learning_rate']
        # Load Datasets
        dataset = pointnetDataset(
            root=  config_data['root'], 
            npoints= self.npoints,
            data_augmentation=False)

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=config_data['batchSize'],
            shuffle=True,
            num_workers=int(config_data['workers']),
            pin_memory = True)

        val_dataset = pointnetDataset(
            root= config_data['root'],
            npoints=self.npoints,
            split='val',
            data_augmentation=False)

        valdataloader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=config_data['batchSize'],
            shuffle=True,
            num_workers=int(config_data['workers']),
            pin_memory = True)

        test_dataset = pointtestDataset(
            root= self.testroot,
            npoints=self.npoints)

        testdataloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=True,
            num_workers=int(config_data['workers']),
            pin_memory = True)

        self.__train_loader =dataloader
        self.__val_loader = valdataloader
        self.__test_loader = testdataloader
        self.num_classes = 82
        self.feature_transform = config_data['feature_transform']
     
        # Setup Experiment
        self.__epochs = config_data['nepoch']
        self.__current_epoch = 0
        self.__training_losses = []
        self.__val_losses = []
        self.__training_acces= []
        self.__val_acces = []
        self.__best_model = None  # Save your best model in this field and use this in test method.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Init Model

        self.__model = PointNetSeg(k=self.num_classes,feature_transform=self.feature_transform)

        self.__criterion = nn.CrossEntropyLoss()

        self.optimizer = optim.Adam(self.__model.parameters(), lr=0.001, betas=(0.9, 0.999))
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=20, gamma=0.5)

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
            self.scheduler.step()
            start_time = datetime.now()
            self.__current_epoch = epoch
            train_loss, val_loss, train_acc, val_acc = self.__trainandval()
            self.__record_stats(train_loss, val_loss, train_acc, val_acc)
            self.__log_epoch_stats(start_time)
            self.__save_model()
            if len(self.__val_losses) > self.earlystop:
                if self.__val_losses[-1] > self.__val_losses[-2] > self.__val_losses[-3] > self.__val_losses[-4]:
                    break 

    def __trainandval(self):
        training_loss = 0
        val_loss = 0
        epoch_train_acc = []
        epoch_val_acc = []

        counter = 0
        for i, data in tqdm(enumerate(self.__train_loader, 0)):
            points, target = data
            points = points.transpose(2, 1)
            points, target = points.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            self.__model = self.__model.train()
            pred, trans, trans_feat = self.__model(points)
            #pred = self.__model(points)
            pred = pred.view(-1, self.num_classes)

            target = target.view(-1, 1)[:, 0] 
            #print(pred.size(), target.size())
            loss = F.nll_loss(pred, target)
            #loss = self.__criterion(pred, target)
            if self.feature_transform:
                loss +=feature_transform_regularizer(trans_feat) * 0.001
            loss.backward()
            self.optimizer.step()
            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(target.data).cpu().sum()
            train_acc =  correct.item()/float(self.batchSize * self.npoints)

            training_loss += loss.item()
            epoch_train_acc.append(train_acc)

            
            if i % 10 == 0:
                with torch.no_grad():
                    counter += 1
                    j, data = next(enumerate(self.__val_loader, 0))
                    points, target = data
                    points = points.transpose(2, 1)
                    points, target = points.to(self.device), target.to(self.device)
                    self.__model = self.__model.eval()
                    pred, _, _ = self.__model(points)
                    #pred = self.__model(points)
                    pred = pred.view(-1, self.num_classes)
                    target = target.view(-1, 1)[:, 0] 
                    loss = F.nll_loss(pred, target)
                    #loss = self.__criterion(pred, target)
                    pred_choice = pred.data.max(1)[1]
                    correct = pred_choice.eq(target.data).cpu().sum()
                    val_acc =  correct.item()/float(self.batchSize * self.npoints)
                    val_loss += loss.item()
                    epoch_val_acc.append(val_acc)
        
        return training_loss / len(self.__train_loader), val_loss/counter, np.mean(epoch_train_acc), np.mean(epoch_val_acc)

    def test(self):
        self.__model = self.__model.eval()

        shape_ious = []
        with torch.no_grad():
            for i,data in tqdm(enumerate(self.__val_loader, 0)):
                points, target = data
                points = points.transpose(2, 1)
                points, target = points.cuda(), target.cuda()
                pred, _, _ = self.__model(points)
                pred_choice = pred.data.max(2)[1]

                pred_np = pred_choice.cpu().data.numpy()
                target_np = target.cpu().data.numpy() 

                for shape_idx in range(target_np.shape[0]):
                    parts = range(self.num_classes)#np.unique(target_np[shape_idx])
                    part_ious = []
                    for part in parts:
                        I = np.sum(np.logical_and(pred_np[shape_idx] == part, target_np[shape_idx] == part))
                        U = np.sum(np.logical_or(pred_np[shape_idx] == part, target_np[shape_idx] == part))
                        if U == 0:
                            iou = 1 #If the union of groundtruth and prediction points is empty, then count part IoU as 1
                        else:
                            iou = I / float(U)
                        part_ious.append(iou)
                    shape_ious.append(np.mean(part_ious))

            print("mIOU for class {}: {}".format('class_choice', np.mean(shape_ious)))

    def generate(self):
        self.__model = self.__model.eval()
        os.makedirs('./generation', exist_ok=True)
        experiment_dir = os.path.join('./generation', self.__name)

        with torch.no_grad():
            for i, data in tqdm(enumerate(self.__val_loader, 0)):
                #points, prefix = data
                points, target = data
                points = points.transpose(2, 1)
                points = points.cuda()
                pred, _, _ = self.__model(points)
                pred = pred.view(-1, self.num_classes)
                pred = pred.cpu().data.numpy()
                print(np.unique(np.argmax(pred, axis=1)))
                return pred,  target.cuda().cpu().data.numpy()   #np.argmax(pred, axis=1)

    def __save_model(self):
        root_model_path = os.path.join(self.__experiment_dir, 'latest_model.pt')
        model_dict = self.__model.state_dict()
        model_dict = {'model': model_dict, 'optimizer': self.optimizer.state_dict()}
        torch.save(model_dict, root_model_path)

    def __record_stats(self, train_loss, val_loss, train_acc, val__acc):
        self.__training_losses.append(train_loss)
        self.__val_losses.append(val_loss)
        self.__val_acces.append(val__acc)
        self.__training_acces.append(train_acc)

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
        train_acc = self.__training_acces[self.__current_epoch]
        val_acc = self.__val_acces[self.__current_epoch]
        summary_str = "Epoch: {}, Train Loss: {}, Val Loss: {},  Train acc: {}, Val acc: {},Took {}, ETA: {}\n"
        summary_str = summary_str.format(self.__current_epoch + 1, train_loss, val_loss, train_acc, val_acc, 
                                         str(time_elapsed),str(time_to_completion))
        self.__log(summary_str, 'epoch.log')

    def plot_stats(self):
        
        f = plt.figure(figsize=(12, 5))
        ax = f.add_subplot(121)
        ax2 = f.add_subplot(122)
        # plot training and validation accuracy
        ax.plot(self.__training_acces, label="Training accuracy")
        ax.plot(self.__val_acces, label="Validation accuracy")
        ax.legend(loc='upper left')
        # plot training and validation loss
        ax2.plot(self.__training_losses, label="Training loss")
        ax2.plot(self.__val_losses, label="Validation loss")
        ax2.legend(loc='upper right')
        plt.title(self.__name + " Stats Plot")
        plt.savefig(os.path.join(self.__experiment_dir, "stat_plot.png"))
        plt.show()
