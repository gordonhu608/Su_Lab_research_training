from doctest import script_from_examples
from random import sample
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

class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1, 1)
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
        sigma = torch.exp(log_sigma)

        return distributions.Normal(loc=mu, scale=sigma)

class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super(Decoder, self).__init__()
        self.linear = nn.Linear(latent_dim, 4 * 4 * 128)
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
       
        mu = torch.tanh(self.conv4(x))
        return distributions.Normal(mu, torch.ones_like(mu))

class VAE(nn.Module):
    def __init__(self, encoder, decoder):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, state):
        q_z = self.encoder(state)
        z = q_z.rsample()

        return self.decoder(z), q_z

    def sample(self, num_samples, latent_dim=16):
        z = torch.randn(num_samples, latent_dim)
        z = z.to(torch.device('cuda'))
        samples = self.decoder(z).sample()

        return samples
    
    def generate(self, x):
        return self.forward(x)[0].sample()

def train():    
    device = torch.device('cuda')
    train = scipy.io.loadmat('train_32x32.mat')
    train_x = torch.Tensor(train['X']).float()
    train_y = torch.Tensor(train['y']).long()

    train_ds = TensorDataset(train_x,train_y)
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, pin_memory=True)
    
    encoder = Encoder(16)
    decoder = Decoder(16)
    vae = VAE(encoder, decoder).to(device)
    optimizer = optim.Adam(vae.parameters(), lr=1e-3)
    for epoch in range(100):
        for data in train_loader:
            inputs, _ = data
            inputs = inputs.to(device)
            optimizer.zero_grad()
            p_x, q_z = vae(inputs)
            log_likelihood = p_x.log_prob(inputs).sum(-1).mean()
            kl = torch.distributions.kl_divergence(
                q_z, 
                torch.distributions.Normal(0, 1.)
            ).sum(-1).mean()
            loss = -(log_likelihood - kl)
            recon_loss = - log_likelihood
            loss.backward()
            optimizer.step()
            l = loss.item()
        print(epoch, l, recon_loss.item(), kl.item())  

def problem_vae(train_data, test_data, dset_id):
    """
    train_data: An (n_train, 32, 32, 3) uint8 numpy array of color images with values in {0, ..., 255}
    test_data: An (n_test, 32, 32, 3) uint8 numpy array of color images with values in {0, ..., 255}
    dset_id: An identifying number of which dataset is given (1 or 2). Most likely
               used to set different hyperparameters for different datasets

    Returns
    - a (# of training iterations, 3) numpy array of full negative ELBO, reconstruction loss E[-log p(x|z)],
      and KL term E[KL(q(z|x) | p(z))] evaluated every minibatch
    - a (# of epochs + 1, 3) numpy array of full negative ELBO, reconstruciton loss E[-p(x|z)],
      and KL term E[KL(q(z|x) | p(z))] evaluated once at initialization and after each epoch
    - a (100, 32, 32, 3) numpy array of 100 samples from your VAE with values in {0, ..., 255}
    - a (100, 32, 32, 3) numpy array of 50 real image / reconstruction pairs
      FROM THE TEST SET with values in {0, ..., 255}
    - a (100, 32, 32, 3) numpy array of 10 interpolations of length 10 between
      pairs of test images. The output should be those 100 images flattened into
      the specified shape with values in {0, ..., 255}
    """
    device = torch.device('cuda')
    train_data = torch.from_numpy(train_data).float()
    train_data = train_data.permute(0,3,1,2)
    train_loader = DataLoader(train_data, batch_size=128, shuffle=True, pin_memory=True)
    test_data = torch.from_numpy(test_data).float()
    test_data = test_data.permute(0,3,1,2)
    test_loader = DataLoader(test_data, batch_size=128, shuffle=False, pin_memory=True)
    encoder = Encoder(16)
    decoder = Decoder(16)
    vae = VAE(encoder, decoder).to(device)
    optimizer = optim.Adam(vae.parameters(), lr=1e-3)
    loss_metric = []
    test_metric = []
    for epoch in tqdm(range(20)):
        vae.train()
        for data in train_loader:
            inputs = data
            inputs = inputs.to(device)
            optimizer.zero_grad()
            p_x, q_z = vae(inputs)
            log_likelihood = p_x.log_prob(inputs).sum(-1).mean() #recons_loss =F.mse_loss(recons, input)
            kl = torch.distributions.kl_divergence(
                q_z, 
                torch.distributions.Normal(0, 1.)
            ).mean(0).sum()
            loss = -(log_likelihood - kl)
            recon_loss = - log_likelihood
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
                inputs = data.to(device)
                p_x, q_z = vae(inputs)
                log_likelihood = p_x.log_prob(inputs).sum(-1).mean()
                kl = torch.distributions.kl_divergence(
                    q_z, 
                    torch.distributions.Normal(0, 1.)
                ).mean(0).sum()
                loss = -(log_likelihood - kl)
                recon_loss = - log_likelihood
                test_loss.append(loss.item())
                test_kl.append(kl.item())
                test_recon.append(recon_loss.item())
            metric = [np.mean(test_loss), np.mean(test_recon), np.mean(test_kl)]
            test_metric.append(metric)
    vae.eval()
    sample_loader = DataLoader(test_data, shuffle=False, pin_memory=True)
    with torch.no_grad():
        pairs = []
        for _ in range(50):
            x = next(iter(sample_loader)).to(device)
            pairs.append(x.squeeze(0).cpu().detach().numpy())
            pairs.append(vae.generate(x).squeeze(0).cpu().detach().numpy())
        interpolations = []
        for _ in range(10):
            x = next(iter(sample_loader)).to(device)
            new_x = vae.generate(x)
            np_x = x.squeeze(0).cpu().detach().numpy()
            np_new_x = new_x.squeeze(0).cpu().detach().numpy()
            for weight in range(10):
                weight = weight / 10
                interpolations.append(weight * np_x + (1-weight) * np_new_x )
    return np.array(loss_metric), np.array(test_metric), np.transpose(vae.sample(100).cpu().detach().numpy(), (0,2,3,1)),\
            np.transpose(np.array(pairs), (0, 2, 3,1)), np.transpose(np.array(interpolations), (0,2,3,1))