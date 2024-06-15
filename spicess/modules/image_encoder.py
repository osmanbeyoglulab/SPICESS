import torch.nn as nn
import torch
import numpy as np
import timm

class CNN_VAE(nn.Module):
#adapted from https://github.com/uhlerlab/cross-modal-autoencoders
    def __init__(self, kernel, stride, padding, nc, 
            hidden1, hidden2, hidden3, hidden4, hidden5, fc1, fc2, cond_dim=0, dropout=0.1):
        super().__init__()

        self.nc = nc #num channels
        self.hidden5=hidden5
        self.fc1 = fc1
        self.fc2 = fc2 #latent_dim
        self.cond_dim = cond_dim
        self.dropout = dropout

        # self.encoder = nn.Sequential(
        #     # nn.MaxPool2d(2, stride=2),
        #     nn.Conv2d(nc, hidden1, kernel, stride, padding, bias=False),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Conv2d(hidden1, hidden2, kernel, stride, padding, bias=False),
        #     # nn.BatchNorm2d(hidden2),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Conv2d(hidden2, hidden3, kernel, stride, padding, bias=False),
        #     # nn.BatchNorm2d(hidden3),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Conv2d(hidden3, hidden4, kernel, stride, padding, bias=False),
        #     # nn.BatchNorm2d(hidden4),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Conv2d(hidden4, hidden5, kernel, stride, padding, bias=False),
        #     # nn.BatchNorm2d(hidden5),
        #     nn.LeakyReLU(0.2, inplace=True),
        # )

        self.encoder = timm.create_model(
            'resnet34', True, num_classes=0, global_pool="avg"
        )
        
        for p in self.encoder.parameters():
            p.requires_grad = True

        self.fc_mus = nn.Sequential(
            nn.Linear(fc1, fc2),
            # nn.BatchNorm1d(fc2),
            nn.Mish(),
        )
        
        self.fc_vars = nn.Sequential(
            nn.Linear(fc1, fc2),
            # nn.BatchNorm1d(fc2),
            nn.Mish(),
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(hidden5, hidden4, kernel, stride, padding, bias=False),
            # nn.BatchNorm2d(hidden4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(hidden4, hidden3, kernel, stride, padding, bias=False),
            # nn.BatchNorm2d(hidden3),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(hidden3, hidden2, kernel, stride, padding, bias=False),
            # nn.BatchNorm2d(hidden2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(hidden2, hidden1, kernel, stride, padding, bias=False),
            # nn.BatchNorm2d(hidden1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(hidden1, nc, kernel, stride, padding, bias=False),
            nn.Sigmoid(),
        )

        self.fcD1 = nn.Sequential(
            nn.Linear(fc2, fc1),
            # nn.BatchNorm1d(fc1),
            nn.Mish(inplace=True),
        )

    def encode(self, x):
        h = self.encoder(x)
        if torch.isnan(torch.sum(h)):
            print('convolution exploded')
        # h = h.view(-1, h.size()[1]*h.size()[2]*h.size()[3])

        
        return self.fc_mus(h), self.fc_vars(h)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        h = self.fcD1(z)
        h = h.view(-1, self.hidden5, int(np.sqrt(self.fc1/self.hidden5)), int(np.sqrt(self.fc1/self.hidden5)))
        return self.decoder(h)
    

    def forward(self, x, c=None):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        res = self.decode(z)
        return res, z, mu, logvar