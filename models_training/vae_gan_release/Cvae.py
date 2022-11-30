import torch
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import torch.optim as optim
from torch import nn
import matplotlib.pyplot as plt
from torch import distributions
from torch.utils.data import TensorDataset, DataLoader
import scipy.io
from tqdm.notebook import tqdm 
from utils import *

class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()
        in_channels = 3  + 1
        self.conv1 = nn.Conv2d(4, 32, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(32) 
        self.conv2 = nn.Conv2d(32, 64, 3, 2, 1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, 2, 1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, 3, 2, 1)
        self.bn4 = nn.BatchNorm2d(256)
        self.linear = nn.Linear(4 * 4 * 256, 2 * latent_dim)
        self.enc_mu = torch.nn.Linear(2 * latent_dim, latent_dim)
        self.enc_log_sigma = torch.nn.Linear(2 * latent_dim, latent_dim)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))

        x = torch.flatten(x, start_dim=1)
        x = self.linear(x)
        mu = self.enc_mu(x)
        log_sigma = self.enc_log_sigma(x)

        return [mu, log_sigma]

class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super(Decoder, self).__init__()
        self.linear = nn.Linear(latent_dim + 10, 4 * 4 * 128)
        self.conv1 = nn.ConvTranspose2d(128, 128, 4, 2, 1)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = nn.ConvTranspose2d(128, 64, 4, 2, 1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.ConvTranspose2d(64, 32, 4, 2, 1)
        self.bn3 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32, 3, 3, 1, 1)

    def forward(self, x):
        x = F.relu(self.linear(x))
        x = x.view(-1, 128, 4, 4)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        return self.conv4(x)

class VAE(nn.Module):
    def __init__(self, encoder, decoder):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.embed_class = nn.Linear(10, 32 * 32) #num class  img size
        self.embed_data = nn.Conv2d(3, 3, kernel_size=1)
    
    def forward(self, input, y):
        embedded_class = self.embed_class(y)
        embedded_class = embedded_class.view(-1, 32, 32).unsqueeze(1)
        embedded_input = self.embed_data(input)
        x = torch.cat([embedded_input, embedded_class], dim=1)
        mu, logvar = self.encoder(x)

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = eps * std + mu

        y = y.squeeze(1)
        z = torch.cat([z,y], dim=1)

        return [self.decoder(z), input, mu, logvar]

    def sample(self, num_samples, y, latent_dim=16):
        z = torch.randn(num_samples, latent_dim)
        z = z.to(torch.device('cuda'))
        y = y.squeeze(1)
        z = torch.cat([z, y], dim=1)

        samples = self.decoder(z)
        return samples
    
    def generate(self, x, y):
        return self.forward(x, y)[0]


def problem_cvae():
    device = torch.device('cuda')
    train = scipy.io.loadmat('train_32x32.mat')
    train_x = torch.Tensor(train['X']).float()
    train_y = torch.Tensor(train['y']).long() - 1
    train_y = F.one_hot(train_y, num_classes=10)
    train_y = train_y.float()
    train_x = train_x.permute(3, 2, 0, 1)
    train_ds = TensorDataset(train_x[:15000],train_y[:15000])
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, pin_memory=True)
    
    test_ds = TensorDataset(train_x[-5000:],train_y[-5000:])
    sample_ds =  TensorDataset(train_x[20000:21000], train_y[20000:21000])
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False, pin_memory=True)

    encoder = Encoder(16)
    decoder = Decoder(16)
    vae = VAE(encoder, decoder).to(device)
    optimizer = optim.Adam(vae.parameters(), lr=1e-3)
    loss_metric = []
    test_metric = []
    for epoch in tqdm(range(20)):
        vae.train()
        for data in train_loader:
            x, y= data
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            recon, x, mu, log_var = vae(x, y)
            recon_loss = F.mse_loss(recon, x)
            kl =  torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
            loss = recon_loss + kl 
            loss.backward()
            optimizer.step()
            metric = [loss.item(), recon_loss.item(), kl.item()]
            loss_metric.append(metric)
        vae.eval()
        with torch.no_grad():
            test_loss = []
            test_kl = []
            test_recon = []
            for data in test_loader:
                x, y = data
                x =x.to(device)
                y = y.to(device)
                recon, x, mu, log_var = vae(x, y)
                recon_loss = F.mse_loss(recon, x)
                kl =  torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
                loss = recon_loss + kl 
                test_loss.append(loss.item())
                test_kl.append(kl.item())
                test_recon.append(recon_loss.item())
            metric = [np.mean(test_loss), np.mean(test_recon), np.mean(test_kl)]
            test_metric.append(metric)
    vae.eval()
    sample_loader = DataLoader(sample_ds, batch_size=1, shuffle=False, pin_memory=True)
    with torch.no_grad():
        pairs = []
        first_50 = iter(sample_loader)
        for _ in range(50):
            x, y = next(first_50)
            x = x.to(device)
            y = y.to(device)
            pairs.append(x.squeeze(0).cpu().detach().numpy())
            pairs.append(vae.generate(x, y).squeeze(0).cpu().detach().numpy())
        interpolations = []
        first_10 = iter(sample_loader)
        for _ in range(10):
            x, y = next(first_10)
            x = x.to(device)
            y = y.to(device)
            new_x = vae.generate(x, y)
            np_x = x.squeeze(0).cpu().detach().numpy()
            np_new_x = new_x.squeeze(0).cpu().detach().numpy()
            for weight in range(10):
                weight = weight / 10
                interpolations.append(weight * np_x + (1-weight) * np_new_x )
    sample_one_y = F.one_hot(torch.from_numpy(np.ones(100)).long(), num_classes=10).float().to(device)
    #return np.array(loss_metric), np.array(test_metric), vae.sample(100, sample_one_y).cpu().detach().numpy(),\
     #       np.array(pairs),np.array(interpolations)
    return np.array(loss_metric), np.array(test_metric), np.transpose(vae.sample(100, sample_one_y).cpu().detach().numpy(), (0,2,3,1)),\
            np.transpose(np.array(pairs), (0, 2, 3,1)), np.transpose(np.array(interpolations), (0,2,3,1))

def plot_vae_training_plot(train_losses, test_losses, title, fname):
    elbo_train, recon_train, kl_train = train_losses[:, 0], train_losses[:, 1], train_losses[:, 2]
    elbo_test, recon_test, kl_test = test_losses[:, 0], test_losses[:, 1], test_losses[:, 2]
    plt.figure()
    n_epochs = len(test_losses) - 1
    x_train = np.linspace(0, n_epochs, len(train_losses))
    x_test = np.arange(n_epochs + 1)

    plt.plot(x_train, elbo_train, label='-elbo_train')
    plt.plot(x_train, recon_train, label='recon_loss_train')
    plt.plot(x_train, kl_train, label='kl_loss_train')
    plt.plot(x_test, elbo_test, label='-elbo_test')
    plt.plot(x_test, recon_test, label='recon_loss_test')
    plt.plot(x_test, kl_test, label='kl_loss_test')

    plt.legend()
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
def cvae_results(fn, dset_id=1):

    train_losses, test_losses, samples, reconstructions, interpolations = fn()
    samples, reconstructions, interpolations = samples.astype('float32'), reconstructions.astype('float32'), interpolations.astype('float32')
    print(f'Final -ELBO: {test_losses[-1, 0]:.4f}, Recon Loss: {test_losses[-1, 1]:.4f}, '
          f'KL Loss: {test_losses[-1, 2]:.4f}')
    plot_vae_training_plot(train_losses, test_losses, f'VAE Dataset {dset_id} Train Plot',
                           f'results/VAE_dset{dset_id}_train_plot.png')
    show_samples(samples, title=f'VAE Dataset {dset_id} Samples',
                 fname=f'results/VAE_dset{dset_id}_samples.png')
    show_samples(reconstructions, title=f'VAE Dataset {dset_id} Reconstructions',
                 fname=f'results/VAE_dset{dset_id}_reconstructions.png')
    show_samples(interpolations, title=f'VAE Dataset {dset_id} Interpolations',
                 fname=f'results/VAE_dset{dset_id}_interpolations.png')