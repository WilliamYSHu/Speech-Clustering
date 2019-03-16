"""VAE implementation based on Pytorch"""

from __future__ import print_function
from __future__ import division
import torch
import torch.utils.data as tud
import numpy as np
from torch import nn, optim
from torch.nn import functional as F


class MNIST(tud.Dataset):
    def __init__(self, X, transform=None):
        self.X = X
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        if self.transform:
            x = self.transform(x)
        return x

class ToTensor(object):
    def __call__(self, sample):
        m = torch.from_numpy(sample).type(torch.float)
        return m

class VNet(nn.Module):
    """
    VAE Network Architecture

    Parameters
    ----------
    dx: int, input dimension
    dh: int, latent dimension
    """
    def __init__(self, dx, dh):
        super(VNet, self).__init__()
        self.enc = nn.Sequential(nn.Linear(dx, 400), nn.ReLU(True))
        self.mu_enc = nn.Linear(400, dh)
        self.var_enc = nn.Linear(400, dh)
        self.dec = nn.Sequential(nn.Linear(dh, 400),
                                 nn.ReLU(True),
                                 nn.Linear(400, dx),
                                 nn.Sigmoid())

    def encode(self, x):
        h = self.enc(x)
        return self.mu_enc(h), self.var_enc(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        return self.dec(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

class VAE(object):
    """Variational AutoEncoder (VAE)

    Parameters
    ----------
    n_inputs: int, feature size of input data
    n_components: int, feature size of output
    lr: float, learning rate (default: 0.001)
    batch_size: int, batch size (default: 128)
    cuda: bool, whether to use GPU if available (default: True)
    path: string, path to save trained model (default: "vae.pth")
    kkl: float, float, weight on loss term -KL(q(z|x)||p(z)) (default: 1.0)
    kv: float, weight on variance term inside -KL(q(z|x)||p(z)) (default: 1.0)
    """
    def __init__(self, n_inputs, n_components, lr=1.0e-3, batch_size=64, cuda=True, path="vae.pth", kkl=1.0, kv=1.0):
        self.model = VNet(n_inputs, n_components)
        self.device = torch.device("cuda" if cuda and torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.batch_size = batch_size
        self.lr = lr
        self.path = path
        self.kkl = kkl
        self.kv = kv
        self.initialize()

    def fit(self, Xr, Xd, epochs):
        """Fit VAE from data Xr
        Parameters
        ----------
        :in:
        Xr: 2d array of shape (n_data, n_dim). Training data
        Xd: 2d array of shape (n_data, n_dim). Dev data, used for early stopping
        epochs: int, number of training epochs
        """
        train_loader = tud.DataLoader(MNIST(Xr, transform=ToTensor()),
            batch_size=self.batch_size, shuffle=True)
        dev_loader = tud.DataLoader(
            MNIST(Xd, transform=ToTensor()),
            batch_size=self.batch_size, shuffle=True)
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        best_dev_loss = np.inf
        for epoch in range(1, epochs + 1):
            train_loss = self._train(train_loader, optimizer)
            dev_loss, _ = self._evaluate(dev_loader)
            if dev_loss < best_dev_loss:
                torch.save(self.model, self.path)
            print('Epoch: %d, train loss: %.4f, dev loss: %.4f' % (
                epoch, train_loss, dev_loss))
        return

    def transform(self, X):
        """Transform X
        Parameters
        ----------
        :in:
        X: 2d array of shape (n_data, n_dim)
        :out:
        Z: 2d array of shape (n_data, n_components)
        """
        try:
            self.model = torch.load(self.path)
        except Exception as err:
            print("Error loading '%s'\n[ERROR]: %s\nUsing initial model!" % (self.path, err))
        test_loader = tud.DataLoader(MNIST(X, transform=ToTensor()), batch_size=self.batch_size, shuffle=False)
        _, Z = self._evaluate(test_loader)
        return Z

    def _train(self, train_loader, optimizer):
        self.model.train()
        train_loss = 0
        for batch_idx, data in enumerate(train_loader):
            data = data.to(self.device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = self.model(data)
            loss = self._loss_function(recon_batch, data, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
        return train_loss/(batch_idx+1)

    def _evaluate(self, loader):
        self.model.eval()
        loss = 0
        fs = []
        with torch.no_grad():
            for batch_idx, data in enumerate(loader):
                data = data.to(self.device)
                recon_batch, mu, logvar = self.model(data)
                loss += self._loss_function(recon_batch, data, mu, logvar).item()
                fs.append(mu)
        fs = torch.cat(fs).cpu().numpy()
        return loss/(batch_idx+1), fs

    def _loss_function(self, recon_x, x, mu, logvar):
        """VAE Loss
        Parameters
        ----------
        :in:
        recon_x: 2d tensor of shape (batch_size, n_dim), reconstructed input
        x: 2d tensor of shape (batch_size, n_dim), input data
        mu: 2d tensor of shape (batch_size, n_components), latent mean
        logvar: 2d tensor of shape (batch_size, n_components), latent log-variance
        :out:
        l: 1d tensor, VAE loss
        """
        n, d = mu.shape
        BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')/n
        KLD = -0.5*(d + self.kv*(logvar-logvar.exp()).sum()/n - mu.pow(2).sum()/n)
        l = BCE + self.kkl*KLD
        return l

    def initialize(self):
        """
        Model Initialization
        """
        def _init_weights(m):
            if type(m) == nn.Linear:
                nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0)

        self.model.apply(_init_weights)
        return
