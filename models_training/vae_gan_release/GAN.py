import torch
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import torch.autograd as autograd
from torch import nn
import matplotlib.pyplot as plt
from torch.utils.data import  DataLoader
from tqdm.notebook import tqdm 
import utils

device = torch.device('cuda')


class GenBlock(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=None, ksize=3, pad=1,
                 activation=nn.ReLU()):
        super(GenBlock, self).__init__()
        self.activation = activation
   
        hidden_channels = out_channels if hidden_channels is None else hidden_channels
        self.c1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=ksize, padding=pad)
        self.c2 = nn.Conv2d(hidden_channels, out_channels, kernel_size=ksize, padding=pad)

        self.b1 = nn.BatchNorm2d(in_channels)
        self.b2 = nn.BatchNorm2d(hidden_channels)
        self.c_sc = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

    def upsample_conv(self, x, conv):
        return conv(nn.UpsamplingNearest2d(scale_factor=2)(x))

    def residual(self, x):
        h = x
        h = self.b1(h)
        h = self.activation(h)
        h = self.upsample_conv(h, self.c1) 
        h = self.b2(h)
        h = self.activation(h)
        h = self.c2(h)
        return h

    def shortcut(self, x):
        x = self.upsample_conv(x, self.c_sc) 
        return x
       

    def forward(self, x):
        return self.residual(x) + self.shortcut(x)

class Generator(nn.Module):
    def __init__(self, activation=nn.ReLU(), n_classes=10):
        super(Generator, self).__init__()
        self.activation = activation
        self.n_classes = n_classes
        self.ch = 128
        self.l1 = nn.Linear(128, 4*4*256)
        self.block2 = GenBlock(256, self.ch, activation=activation)
        self.block3 = GenBlock(self.ch, self.ch, activation=activation)
        self.block4 = GenBlock(self.ch, self.ch, activation=activation)
        self.b5 = nn.BatchNorm2d(self.ch)
        self.c5 = nn.Conv2d(self.ch, 3, kernel_size=3, stride=1, padding=1)

    def forward(self, z):

        h = z
        h = self.l1(h).view(-1, 256, 4, 4)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        h = self.b5(h)
        h = self.activation(h)
        h = nn.Tanh()(self.c5(h))
        return h

def _downsample(x):
    # Downsample (Mean Avg Pooling with 2x2 kernel)
    return nn.AvgPool2d(kernel_size=2)(x)


class DisBlock(nn.Module):
    def __init__(self, in_channels, out_channels, ksize=3, pad=1, activation=nn.ReLU()):
        super(DisBlock, self).__init__()
        self.activation = activation

        self.c1 = nn.Conv2d(in_channels, out_channels, kernel_size=ksize, padding=pad)
        self.c2 = nn.Conv2d(out_channels, out_channels, kernel_size=ksize, padding=pad)
        self.c_sc = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

    def residual(self, x):
        h = x
        h = self.c1(h)
        h = self.activation(h)
        h = self.c2(h)
        h = _downsample(h)
        return h

    def shortcut(self, x):
        return self.c_sc(_downsample(x))

    def forward(self, x):
        return self.residual(x) + self.shortcut(x)
    

class Discriminator(nn.Module):
    def __init__(self, activation=nn.ReLU()):
        super(Discriminator, self).__init__()
        self.ch = 128
        self.activation = activation
        self.block1 = DisBlock(3, 256)
        self.block2 = DisBlock(256, self.ch)
        self.block3 = DisBlock(self.ch, self.ch)
        self.block4 = DisBlock(self.ch, self.ch)
        self.l5 = nn.Linear(self.ch, 1, bias=False)
    
    def forward(self, x):
        h = x
        h = self.block1(h)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        h = self.activation(h)
        # Global average pooling
        h = h.sum(2).sum(2)
        output = self.l5(h)

        return output

def train():
    train_data, _ = utils.load_pickled_data('cifar10.pkl')
    train_data = train_data.transpose((0, 3, 1, 2)) / 255.0
    train_loader = DataLoader(train_data, batch_size=256, shuffle=True, pin_memory=True)
    cuda = True if torch.cuda.is_available() else False
    lambda_gp = 10
    n_critic = 5
    # Initialize generator and discriminator
    generator = Generator()
    discriminator = Discriminator()

    generator.cuda()
    discriminator.cuda()
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=2e-4, betas=(0, 0.9))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0, 0.9))
    Tensor = torch.cuda.FloatTensor 
    # lr scheduler

    def compute_gradient_penalty(D, real_samples, fake_samples):
        """Calculates the gradient penalty loss for WGAN GP"""
        # Random weight term for interpolation between real and fake samples
        alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1)))
        # Get random interpolation between real and fake samples
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
        d_interpolates = D(interpolates)
        fake = Variable(Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)
        # Get gradient w.r.t. interpolates
        gradients = autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty
    
    loss_metric = []
    for epoch in tqdm(range(3)):
        for i, (imgs) in enumerate(train_loader):

            real_imgs = Variable(imgs.type(Tensor))
            optimizer_D.zero_grad()
            z = Variable(Tensor(np.random.normal(0, 1, (1024, 128))))
            fake_imgs = generator(z)
            real_validity = discriminator(real_imgs)
            fake_validity = discriminator(fake_imgs)

            gradient_penalty = compute_gradient_penalty(discriminator, real_imgs.data, fake_imgs.data)
            d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty

            d_loss.backward()
            optimizer_D.step()
            optimizer_G.zero_grad()
            loss_metric.append(d_loss.item())

            if i % n_critic == 0:

                fake_imgs = generator(z)
                # Loss measures generator's ability to fool the discriminator
                # Train on fake images
                fake_validity = discriminator(fake_imgs)
                g_loss = -torch.mean(fake_validity)

                g_loss.backward()
                optimizer_G.step()
                # save fake_imgs.data

    def sample(num):
        z = Variable(Tensor(np.random.normal(0, 1, (num, 128))))
        generated_imgs = generator(z)
        return generated_imgs.permute(0, 2, 3, 1)

    return np.array(loss_metric), np.array(sample(1000))

