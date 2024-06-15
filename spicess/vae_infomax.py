## Author: Koushul Ramjattun
from argparse import Namespace
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import random

"""Set all seeds to ensure reproducibility."""
np.random.seed(0)
torch.manual_seed(0)
random.seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from .modules.clip import ImageEncoder, ProjectionHead
from .modules.inner_product import InnerProductDecoder
from .modules.infomax import ContrastiveGraph, ContrastiveProjectionGraph
from .modules.image_encoder import CNN_VAE



pool = nn.MaxPool2d(2, stride=2)

class GraphEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(GraphEncoder, self).__init__()
        self.conv = GCNConv(in_channels, hidden_channels, cached=False)
        self.prelu = nn.PReLU(hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels, cached=False)
        self.prelu2 = nn.PReLU(hidden_channels)

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        x = self.prelu(x)
        x = self.conv2(x, edge_index)
        x = self.prelu2(x)
        return x
    
def corruption(x, edge_index):
    return x[torch.randperm(x.size(0))], edge_index

class InfoMaxVAE(nn.Module):
    def __init__(
        self,
        input_dim,
        dropout = 0,
        latent_dim = 32,
        encoder_dim = 128,
        use_hist = False,
        pretrained = None,
        freeze_encoder = True,
        kernel = 4, 
        stride = 2, 
        padding = 1, 
        hidden1 = 64, 
        hidden2 = 128, 
        hidden3 = 128,
        hidden4 = 128,
        hidden5 = 96,
        fc1 = 512,
        # fc1 = 1536,
        nc = 1,
        train_idx = None,
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.encoder_dim = encoder_dim
        self.dropout = dropout
        self.use_hist = use_hist
        self.train_idx = train_idx   

        
        if self.use_hist:
            
            self.image_encoder = CNN_VAE(
                kernel, 
                stride, 
                padding, 
                nc, 
                hidden1, 
                hidden2, 
                hidden3, 
                hidden4, 
                hidden5, 
                fc1,
                latent_dim,
            )
            
            if pretrained:
                self.image_encoder.load_state_dict(torch.load(pretrained))
            
                if freeze_encoder:
                    for param in self.image_encoder.encoder.parameters():
                        param.requires_grad = False #freeze
            
            self.image_encoder.train()
            
                
        self.encoders = nn.ModuleList([
            ContrastiveGraph(
                hidden_channels=encoder_dim, 
                encoder=GraphEncoder(input_dim[0], encoder_dim),
                summary=lambda z, *args, **kwargs: torch.sigmoid(z.mean(dim=0)),
                corruption=corruption),
            ContrastiveGraph(
                hidden_channels=encoder_dim, 
                encoder=GraphEncoder(input_dim[1], encoder_dim),
                summary=lambda z, *args, **kwargs: torch.sigmoid(z.mean(dim=0)),
                corruption=corruption),
        ])
        
        
        dim1, dim2 = input_dim
        
        
                
        self.fc_mus = nn.ModuleList([
            nn.Sequential(
                nn.Linear(encoder_dim, latent_dim),
                nn.BatchNorm1d(latent_dim),
                nn.Mish()
            ),
            nn.Sequential(
                nn.Linear(encoder_dim, latent_dim),
                nn.BatchNorm1d(latent_dim),
                nn.Mish()
            )
        ])
    
        self.fc_vars = nn.ModuleList([
            nn.Sequential(
                nn.Linear(encoder_dim, latent_dim),
                nn.BatchNorm1d(latent_dim),
                nn.Mish()
            ),
            nn.Sequential(
                nn.Linear(encoder_dim, latent_dim),
                nn.BatchNorm1d(latent_dim),
                nn.Mish()
            )
        ])
        

        self.decoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(latent_dim, dim1),
                nn.BatchNorm1d(dim1),
                nn.Mish(),
                nn.Dropout(dropout),
                nn.Linear(dim1, 2*dim1),
                nn.BatchNorm1d(2*dim1),
                nn.Mish(),
                nn.Dropout(dropout),
                nn.Linear(2*dim1, dim1),
                nn.Sigmoid()
            ), 
            nn.Sequential(
                nn.Linear(latent_dim, dim2),
                nn.BatchNorm1d(dim2),
                nn.Mish(),
                nn.Dropout(dropout),
                nn.Linear(dim2, 2*dim2),
                nn.BatchNorm1d(2*dim2),
                nn.Mish(),
                nn.Dropout(dropout),
                nn.Linear(2*dim2, dim2),
                nn.Sigmoid()
            ), 
        ])
        
        self.adjacency = InnerProductDecoder(
            dropout=dropout, 
            act=lambda x: x)
        
    def freeze_all(self):
        for param in self.parameters():
            param.requires_grad = False
    
    def unfreeze_all(self):
        for param in self.parameters():
            param.requires_grad = True
            
    def unfreeze_gene_vae(self):
        
        params = []
        params.extend(self.encoders[0].parameters())
        params.extend(self.fc_mus[0].parameters())
        params.extend(self.fc_vars[0].parameters())
        params.extend(self.decoders[0].parameters())
        
        for param in params:
            param.requires_grad = True            

    def encode(self, X, A):
        edge_index = A.nonzero().t().contiguous()
        gex_pos_z, gex_neg_z, gex_summary = self.encoders[0](X[0], edge_index)
        pex_pos_z, pex_neg_z, pex_summary = self.encoders[1](X[1], edge_index)                
        return [gex_pos_z, gex_neg_z, gex_summary, pex_pos_z, pex_neg_z, pex_summary]
    

    def refactor(self, X):
        index = range(2)
        zs = []
        mus = []
        logvars = []
        for x, i in zip(X, index):
            mu = self.fc_mus[i](x)
            logvar = self.fc_vars[i](x)
            std = torch.exp(logvar / 2)
            if not self.training:
                zs.append(mu)
            else:
                std = std + 1e-7
                q = torch.distributions.Normal(mu, std)
                zs.append(q.rsample())
            mus.append(mu)
            logvars.append(logvar)
        return zs, mus, logvars

    def combine(self, Zs):
        """This moves correspondent latent embeddings 
        in similar directions over the course of training 
        and is key in the formation of similar latent spaces."""         
        
        if self.use_hist:
            mZ = (1/3) * (Zs[0] + Zs[1] + Zs[2])
            return [mZ, mZ, mZ]
        
        mZ = 0.5*(Zs[0] + Zs[1])        
        return [mZ, mZ]
        


    def decode(self, X):
        return [self.decoders[i](X[i]) for i in range(2)]
    

    def forward(self, X, A, Y=None):
        output = Namespace()
        
        # assert self.use_hist or Y is not None, 'Image input is required'
        
        gex_pos_z, gex_neg_z, gex_summary, pex_pos_z, pex_neg_z, pex_summary = self.encode(X, A)
        
        if self.use_hist:
            # background = self.image_encoder.fc_cond(X[0])
            # image_mu, image_logvar = self.image_encoder.encode(Y, background)
            image_mu, image_logvar = self.image_encoder.encode(Y)
            
            image_z = self.image_encoder.reparameterize(image_mu, image_logvar)
        
        encoded = [gex_pos_z, pex_pos_z]
        zs, mus, logvars = self.refactor(encoded)
        if self.use_hist:
            zs.append(image_z)
            
        combined = self.combine(zs)
        
        X_hat = self.decode(combined)

        if self.use_hist:
            image_recons = self.image_encoder.decode(combined[0])
            
        output.adj_recon = self.adjacency(combined[0])      

        output.gex_z = zs[0] 
        output.pex_z = zs[1]
        output.gex_pos_z = gex_pos_z
        output.pex_pos_z = pex_pos_z
        output.gex_neg_z = gex_neg_z
        output.pex_neg_z = pex_neg_z
        output.gex_summary = gex_summary
        output.pex_summary = pex_summary
        output.gex_model_weight = self.encoders[0].weight
        output.pex_model_weight = self.encoders[1].weight
        output.gex_mu, output.pex_mu = mus
        output.gex_logvar, output.pex_logvar = logvars
        output.gex_c = combined[0]
        output.pex_c = combined[1]
        output.gex_recons, output.pex_recons = X_hat

        output.gex_input, output.pex_input = X
        
        if self.use_hist:
            output.img_mu = image_mu
            output.img_logvar = image_logvar
            output.img_recons = image_recons
            output.img_z = image_z
            output.img_input = Y
            output.img_c = combined[2]
            output.use_hist = self.use_hist 

        return output
    
    def enable_dropout(self):
        """ Function to enable the dropout layers during test-time """
        for m in self.modules():
            if m.__class__.__name__.startswith('Dropout'):
                m.train()
                
                
    @torch.no_grad()
    def get_embeddings(self, X, adj):
        self.eval()
        edge_index = adj.nonzero().t().contiguous()
        pos_z, _, _ = self.encoders[0](X, edge_index)
        z = self.fc_mus[0](pos_z)
        return z.cpu().numpy()
    
    @torch.no_grad()
    def impute(self, X, adj, enable_dropout=False, return_z=False):
        self.eval()
        if enable_dropout:
            self.enable_dropout()
        edge_index = adj.nonzero().t().contiguous()
        pos_z, _, _ = self.encoders[0](X, edge_index)
        z = self.fc_mus[0](pos_z)
        decoded = self.decoders[1](z)
        recovered_gex = self.decoders[0](z)
        
        
        if return_z:
            return decoded.cpu().numpy(), z.cpu().numpy(), recovered_gex.cpu().numpy()       
            
        return decoded.cpu().numpy()
        
    @torch.no_grad()
    def img2proteins(self, Y):                
        self.eval()
        self.image_encoder.eval()
                
        mu, logvar = self.image_encoder.encode(Y)
        z = self.image_encoder.reparameterize(mu, logvar)
        decoded = self.decoders[1](z)
            
        return decoded.cpu().numpy()