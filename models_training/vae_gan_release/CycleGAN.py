import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.autograd as autograd
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from tqdm.notebook import tqdm 
import utils
from PIL import Image
from torch.autograd import Variable
import itertools
from torchvision.utils import save_image, make_grid

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
        )

    def forward(self, x):
        return x + self.block(x)

class Generator(nn.Module):
    def __init__(self, num_blocks):
        super(Generator, self).__init__()
        channels = 3
        out_features = 64

        model = [
            nn.ReflectionPad2d(channels),
            nn.Conv2d(channels, out_features, kernel_size=7),
            nn.InstanceNorm2d(out_features),
            nn.ReLU(inplace=True),
        ]
        in_features = out_features
        # Downsample  
        for _ in range(2):
            out_features *= 2
            model += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features
        
        for _ in range(num_blocks):
            model += [ResidualBlock(out_features)]
        # Upsample
        for _ in range(2):
            out_features //= 2
            model += [
                nn.Upsample(scale_factor=2),
                nn.Conv2d(in_features, out_features, 3, stride=1, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features

        model += [ 
            nn.ReflectionPad2d(channels),
            nn.Conv2d(out_features, channels, 7),
            nn.Tanh()
        ]

        self.model = nn.Sequential(*model)
    
    def forward(self, x):
        return self.model(x)


class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()
        c, h, w = input_shape
        self.output_shape = (1, h//2 **4, w//2 ** 4)

        def disblock(in_filters, out_filters, normalize=True):
            layers = [nn.Conv2d(in_filters, out_filters, 4, 2, 1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *disblock(c, 64, normalize=False),
            *disblock(64, 128),
            *disblock(128, 256),
            nn.ZeroPad2d((1, 0, 1, 0)), #left right top bottom
            nn.Conv2d(256, 1, 4, padding=1)
        )

    def forward(self, x):
        return self.model(x)

def train():
    device = torch.device('cuda')
    mnist, cmnist = load_q4_data()
    mnist, cmnist = torch.Tensor(mnist), torch.Tensor(cmnist)
    mnist = mnist.repeat(1, 3, 1, 1)
    train_ds = TensorDataset(mnist, cmnist)
    dataloader = DataLoader(train_ds, batch_size=1, shuffle=True, pin_memory=True)

    criterion_GAN = torch.nn.MSELoss().cuda()
    criterion_cycle = torch.nn.L1Loss().cuda()
    criterion_identity = torch.nn.L1Loss().cuda()

    input_shape = (3, 28, 28)
    opt = {'lr': 2e-4, 'b1': 0.5, 'b2':0.999, 'n_epochs': 3, 'epoch':0 , 'decay_epoch': 2, 
            'lambda_cyc': 10.0, 'lambda_id': 5.0}
    
    Tensor = torch.cuda.FloatTensor 
    # Initialize generator and discriminator
    G_XY = Generator(input_shape, 3).cuda()
    G_YX = Generator(input_shape, 3).cuda()
    D_X = Discriminator(input_shape).cuda()
    D_Y = Discriminator(input_shape).cuda()

    def weights_init_normal(m):
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
            if hasattr(m, "bias") and m.bias is not None:
                torch.nn.init.constant_(m.bias.data, 0.0)
        elif classname.find("BatchNorm2d") != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)

    G_XY.apply(weights_init_normal)
    G_YX.apply(weights_init_normal)
    D_X.apply(weights_init_normal)
    D_Y.apply(weights_init_normal)

    optimizer_G = torch.optim.Adam(
        itertools.chain(G_XY.parameters(), G_YX.parameters()), lr=opt.lr, betas=(opt.b1, opt.b2)
    )
    optimizer_D_X = torch.optim.Adam(D_X.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D_Y = torch.optim.Adam(D_Y.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
        
    class LambdaLR:
        def __init__(self, n_epochs, offset, decay_start_epoch):
            assert (n_epochs - decay_start_epoch) > 0, "Decay must start before the training session ends!"
            self.n_epochs = n_epochs
            self.offset = offset
            self.decay_start_epoch = decay_start_epoch

        def step(self, epoch):
            return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (self.n_epochs - self.decay_start_epoch)
    
    # Learning rate update schedulers
    lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(
        optimizer_G, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
    )
    lr_scheduler_D_X = torch.optim.lr_scheduler.LambdaLR(
        optimizer_D_X, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
    )
    lr_scheduler_D_Y = torch.optim.lr_scheduler.LambdaLR(
        optimizer_D_Y, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
    )

    loss_G_metric = []
    loss_D_metric = []
    for epoch in range(opt.epoch, opt.n_epochs):
        loss_D_epoch = []
        loss_G_epoch = []
        for i, (x, y) in enumerate(dataloader):

            # Set model input
            real_X = Variable(x.type(Tensor))
            real_Y = Variable(y.type(Tensor))

            # Adversarial ground truths
            valid = Variable(Tensor(np.ones((real_X.size(0), *D_X.output_shape))), requires_grad=False)
            fake = Variable(Tensor(np.zeros((real_X.size(0), *D_X.output_shape))), requires_grad=False)

            # ------------------
            #  Train Generators
            # ------------------

            G_XY.train()
            G_YX.train()

            optimizer_G.zero_grad()

            # Identity loss
            loss_id_X = criterion_identity(G_YX(real_X), real_X)
            loss_id_Y = criterion_identity(G_XY(real_Y), real_Y)

            loss_identity = (loss_id_X + loss_id_Y) / 2

            # GAN loss
            fake_Y = G_XY(real_X)
            loss_GAN_XY = criterion_GAN(D_Y(fake_Y), valid)
            fake_X = G_YX(real_Y)
            loss_GAN_YX = criterion_GAN(D_X(fake_X), valid)

            loss_GAN = (loss_GAN_XY + loss_GAN_YX) / 2

            # Cycle loss
            recov_X = G_YX(fake_Y)
            loss_cycle_X = criterion_cycle(recov_X, real_X)
            recov_Y = G_XY(fake_X)
            loss_cycle_Y = criterion_cycle(recov_Y, real_Y)

            loss_cycle = (loss_cycle_X + loss_cycle_Y) / 2

            # Total loss
            loss_G = loss_GAN + opt.lambda_cyc * loss_cycle + opt.lambda_id * loss_identity

            loss_G.backward()
            optimizer_G.step()
            loss_G_epoch.append(loss_G.item())
            # -----------------------
            #  Train Discriminator X
            # -----------------------

            optimizer_D_X.zero_grad()

            # Real loss
            loss_real = criterion_GAN(D_X(real_X), valid)
            loss_fake = criterion_GAN(D_X(fake_X.detach()), fake)
            # Total loss
            loss_D_X = (loss_real + loss_fake) / 2

            loss_D_X.backward()
            optimizer_D_X.step()

            # -----------------------
            #  Train Discriminator Y
            # -----------------------

            optimizer_D_Y.zero_grad()

            # Real loss
            loss_real = criterion_GAN(D_Y(real_Y), valid)
            loss_fake = criterion_GAN(D_Y(fake_Y.detach()), fake)
            # Total loss
            loss_D_Y = (loss_real + loss_fake) / 2

            loss_D_Y.backward()
            optimizer_D_Y.step()

            loss_D = (loss_D_X + loss_D_Y) / 2
            loss_D_epoch.append(loss_D.item())
        loss_G_metric.append(np.mean(loss_G_epoch))
        loss_D_metric.append(np.mean(loss_D_epoch))
        lr_scheduler_G.step()
        lr_scheduler_D_X.step()
        lr_scheduler_D_Y.step()

    plot_gan_training(loss_G_metric, 'CGAN-G Losses', 'results/cgan_g_losses.png')
    plot_gan_training(loss_D_metric, 'CGAN-D Losses', 'results/cgan_d_losses.png')

    valloader = DataLoader(train_ds, batch_size=1, shuffle=False, pin_memory=True)
    real_mnist = []
    trans_cmnist = []
    recov_mnist = []
    real_cmnist = []
    trans_mnist = []
    recov_cmnist = []

    for _ in range(20):
        x, y = next(iter(valloader))
        G_XY.eval()
        G_YX.eval()
        real_X = Variable(x.type(Tensor))
        fake_Y = G_XY(real_X)
        recov_X = G_YX(fake_Y)
        real_Y = Variable(y.type(Tensor))
        fake_X = G_YX(real_Y)
        recov_Y = G_XY(fake_X)

        real_mnist.append(x.squeeze(0).mean(0).unsqueeze(2).cpu().detach().numpy())
        trans_cmnist.append(fake_Y.squeeze(0).permute(1, 2, 0).cpu().detach().numpy())
        recov_mnist.append(recov_X.squeeze(0).mean(0).unsqueeze(2).cpu().detach().numpy())
        real_cmnist.append(y.squeeze(0).permute(1, 2, 0).cpu().detach().numpy())
        trans_mnist.append(fake_X.squeeze(0).mean(0).unsqueeze(2).cpu().detach().numpy())
        recov_cmnist.append(recov_Y.squeeze(0).permute(1, 2, 0).cpu().detach().numpy())

    return np.array(real_mnist), np.array(trans_cmnist), np.array(recov_mnist), \
                 np.array(real_cmnist), np.array(trans_mnist), np.array(recov_cmnist)
        
        

    



