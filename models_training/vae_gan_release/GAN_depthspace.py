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
class DepthToSpace(nn.Module):
    def __init__(self, block_size):
        super().__init__()
        self.block_size = block_size
        self.block_size_sq = block_size * block_size

    def forward(self, input):
        output = input.permute(0, 2, 3, 1).contiguous()
        batch_size, d_height, d_width, d_depth = output.size()
        s_depth = int(d_depth / self.block_size_sq)
        s_width = int(d_width * self.block_size)
        s_height = int(d_height * self.block_size)
        t_1 = output.view(batch_size, d_height, d_width, \
                    self.block_size_sq, s_depth)
        spl = t_1.split(self.block_size, 3)
        stack = [t_t.contiguous().view(batch_size, d_height, s_width, s_depth) \
                        for t_t in spl]
        output = torch.stack(stack, 0).transpose(0, 1).permute(0, 2, 1, 3, 4) \
            .contiguous().view(batch_size, s_height, s_width, s_depth)
        output = output.permute(0, 3, 1, 2).contiguous()
        return output 

class SpaceToDepth(nn.Module):
    def __init__(self, block_size):
        super().__init__()
        self.block_size = block_size
        self.block_size_sq = block_size * block_size
    
    def forward(self, input):
        output = input.permute(0, 2, 3, 1).contiguous()
        (batch_size, s_height, s_width, s_depth) = output.size()
        d_depth = s_depth * self.block_size_sq
        d_width = int(s_width / self.block_size)
        d_height = int(s_height / self.block_size)
        t_1 = output.split(self.block_size, 2)
        stack = [t_t.contiguous().view(batch_size, d_height, d_depth) for t_t in t_1]
        output = torch.stack(stack, 1)
        output = output.permute(0, 2, 1, 3).contiguous()
        output = output.permute(0, 3, 1, 2).contiguous()
        return output

class Upsample_Conv2d(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=(3, 3), stride=1, padding=1):
        super().__init__()
        self.depth_space = DepthToSpace(block_size=2)
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size, \
            stride=stride, padding=padding)
    
    def forward(self, x):
        x = torch.cat([x, x, x, x], dim=1)
        x = self.depth_space(x)
        x = self.conv(x)
        return x 

class Downsample_Conv2d(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=(3, 3), stride=1, padding=1):
        super().__init__()
        self.space_depth = SpaceToDepth(block_size=2)
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size, \
            stride=stride, padding=padding)
    
    def forward(self, x):
        x = self.space_depth(x)
        #print("sapce_depth",x.shape)
        chunked = torch.Tensor([i.cpu().detach().numpy() for i in x.chunk(4, dim=1)]).to(device)
        #print('chunk',chunked.shape)
        x = torch.sum(chunked, 0) / 4.0 
        #print("summed",x.shape)
        #x = torch.sum(x.chunk(4, dim=1)) / 4.0
        x = self.conv(x)
        return x 

class ResnetBlockUp(nn.Module):
    def __init__(self, in_dim, kernel_size=(3,3), n_filters=256):
        super().__init__()
        self.model = nn.Sequential(
            nn.BatchNorm2d(in_dim),
            nn.ReLU(),
            nn.Conv2d(in_dim, n_filters, kernel_size, padding=1),
            nn.BatchNorm2d(n_filters),
            nn.ReLU(),
        )
        self.up1 = Upsample_Conv2d(n_filters, n_filters, kernel_size, padding=1)
        self.up2 = Upsample_Conv2d(in_dim, n_filters, kernel_size=(1,1), padding=0)

    def forward(self, x):
        _x = x
        _x = self.model(_x)
        residual = self.up1(_x)
        shortcut = self.up2(x)

        return residual + shortcut

class Generator(nn.Module):
    def __init__(self, n_samples=1024, n_filters=128):
        super().__init__()
        self.n_samples = n_samples
        self.linear = nn.Linear(128, 4*4*256)
        self.model = nn.Sequential(
            ResnetBlockUp(in_dim=256, n_filters=n_filters),
            ResnetBlockUp(in_dim=n_filters, n_filters=n_filters),
            ResnetBlockUp(in_dim=n_filters, n_filters=n_filters),
            nn.BatchNorm2d(n_filters),
            nn.ReLU(),
            nn.Conv2d(n_filters, 3, kernel_size=(3, 3), padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        z = self.linear(z)
        z = z.view(-1, 256, 4, 4)
        z = self.model(z)

        return z 

class ResnetBlockDown(nn.Module):
    def __init__(self, in_dim, kernel_size=(3,3), n_filters=256):
        super().__init__()
        self.model = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_dim, n_filters, kernel_size, padding=1),
            nn.ReLU(),
        )
        self.down1 = Downsample_Conv2d(n_filters, n_filters, kernel_size, padding=1)
        self.down2 = Downsample_Conv2d(in_dim, n_filters, kernel_size=(1,1), padding=0)

    def forward(self, x):
        _x = x
        _x = self.model(_x)
        residual = self.down1(_x) 
        shortcut = self.down2(x)

        return residual + shortcut

class Discriminator(nn.Module):
    def __init__(self, n_samples=1024, n_filters=128):
        super().__init__()
        self.n_samples = n_samples
        self.model = nn.Sequential(
            nn.Conv2d(3, 256, kernel_size=(3, 3), padding=1),
            ResnetBlockDown(in_dim=256, n_filters=n_filters),
            ResnetBlockDown(in_dim=n_filters, n_filters=n_filters),
            ResnetBlockDown(in_dim=n_filters, n_filters=n_filters),
            nn.ReLU(),
            #nn.Conv2d(n_filters, 3, kernel_size=(3,3), padding=1), #48
            #nn.Tanh()
        )
        self.linear = nn.Linear(2048, 1) #2048, 1)

    def forward(self, x):
        x = self.model(x)
        #print(x.shape)
        x = torch.flatten(x, start_dim=1)
        #print(x.shape)
        out = self.linear(x)
        #print(out.shape)
        return out

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
        #print(interpolates.shape)
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

