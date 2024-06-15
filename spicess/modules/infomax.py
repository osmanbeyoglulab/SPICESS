import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

import copy
from typing import Callable, Tuple
import torch
from torch import Tensor
from torch.nn import Module, Parameter
from torch_geometric.nn.inits import reset, uniform

EPS = 1e-15


class LogReg(nn.Module):
    def __init__(self, ft_in, nb_classes):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(ft_in, nb_classes)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq):
        ret = self.fc(seq)
        return ret


class Encoder(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, activation, k: int = 2):
        super(Encoder, self).__init__()
        self.base_model = GCNConv
        self.in_channels = in_channels
        assert k >= 2
        self.k = k
        self.conv = [self.base_model(in_channels, 2 * out_channels)]
        for _ in range(1, k-1):
            self.conv.append(self.base_model(2 * out_channels, 2 * out_channels))
        self.conv.append(self.base_model(2 * out_channels, out_channels))
        self.conv = nn.ModuleList(self.conv)

        self.activation = activation

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        for i in range(self.k):
            x = self.activation(self.conv[i](x, edge_index))
        return x


class ContrastiveProjectionGraph(torch.nn.Module):
    
    def __init__(self, in_channels, out_channels, num_hidden, latent_dim, tau=0.5, activation=F.elu, k=2):
        super(ContrastiveProjectionGraph, self).__init__()
        self.encoder: Encoder = Encoder(in_channels, out_channels, activation, k)
        self.tau: float = tau
        self.latent_dim: int = latent_dim
        self.fc1 = torch.nn.Linear(num_hidden, latent_dim)
        self.fc2 = torch.nn.Linear(latent_dim, num_hidden)

    def forward(self, x: torch.Tensor,
                edge_index: torch.Tensor) -> torch.Tensor:
        return self.encoder(x, edge_index)

    def projection(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z)

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor):
        f = lambda x: torch.exp(x / self.tau)
        refl_sim = f(self.sim(z1, z1))
        between_sim = f(self.sim(z1, z2))

        return -torch.log(
            between_sim.diag()
            / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))

    def batched_semi_loss(self, z1: torch.Tensor, z2: torch.Tensor, batch_size: int):
        # Space complexity: O(BN) (semi_loss: O(N^2))
        device = z1.device
        num_nodes = z1.size(0)
        num_batches = (num_nodes - 1) // batch_size + 1
        f = lambda x: torch.exp(x / self.tau)
        indices = torch.arange(0, num_nodes).to(device)
        losses = []

        for i in range(num_batches):
            mask = indices[i * batch_size:(i + 1) * batch_size]
            refl_sim = f(self.sim(z1[mask], z1))  # [B, N]
            between_sim = f(self.sim(z1[mask], z2))  # [B, N]

            losses.append(-torch.log(
                between_sim[:, i * batch_size:(i + 1) * batch_size].diag()
                / (refl_sim.sum(1) + between_sim.sum(1)
                   - refl_sim[:, i * batch_size:(i + 1) * batch_size].diag())))

        return torch.cat(losses)

    def loss(self, z1: torch.Tensor, z2: torch.Tensor, mean: bool = True, batch_size: int = 0):
        h1 = self.projection(z1)
        h2 = self.projection(z2)

        if batch_size == 0:
            l1 = self.semi_loss(h1, h2)
            l2 = self.semi_loss(h2, h1)
        else:
            l1 = self.batched_semi_loss(h1, h2, batch_size)
            l2 = self.batched_semi_loss(h2, h1, batch_size)

        ret = (l1 + l2) * 0.5
        ret = ret.mean() if mean else ret.sum()

        return ret
    
    def __repr__(self) -> str:
        return self.__class__.__name__ + ' (' + str(self.encoder.in_channels) + ' -> ' + str(self.latent_dim) + ')'
        


def drop_feature(x, drop_prob):
    drop_mask = torch.empty(
        (x.size(1), ),
        dtype=torch.float32,
        device=x.device).uniform_(0, 1) < drop_prob
    x = x.clone()
    x[:, drop_mask] = 0

    return x




class ContrastiveGraph(torch.nn.Module):

    def __init__(self, hidden_channels, encoder, summary, corruption):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.encoder = encoder
        self.summary = summary
        self.corruption = corruption
        self.weight = Parameter(torch.empty(hidden_channels, hidden_channels))

        self.reset_parameters()

    def reset_parameters(self):
        reset(self.encoder)
        reset(self.summary)
        uniform(self.hidden_channels, self.weight)


    def forward(self, *args, **kwargs) -> Tuple[Tensor, Tensor, Tensor]:
        pos_z = self.encoder(*args, **kwargs)

        cor = self.corruption(*args, **kwargs)
        cor = cor if isinstance(cor, tuple) else (cor, )
        cor_args = cor[:len(args)]
        cor_kwargs = copy.copy(kwargs)
        for key, value in zip(kwargs.keys(), cor[len(args):]):
            cor_kwargs[key] = value
        neg_z = self.encoder(*cor_args, **cor_kwargs)
        summary = self.summary(pos_z, *args, **kwargs)
        return pos_z, neg_z, summary


    def discriminate(self, z: Tensor, summary: Tensor, sigmoid: bool = True) -> Tensor:
        summary = summary.t() if summary.dim() > 1 else summary
        value = torch.matmul(z, torch.matmul(self.weight, summary))
        return torch.sigmoid(value) if sigmoid else value


    def loss(self, pos_z: Tensor, neg_z: Tensor, summary: Tensor) -> Tensor:
        pos_loss = -torch.log(
            self.discriminate(pos_z, summary, sigmoid=True) + EPS).mean()
        neg_loss = -torch.log(1 - self.discriminate(neg_z, summary, sigmoid=True) + EPS).mean()
        return pos_loss + neg_loss