from argparse import Namespace
import torch
import torch.nn.functional as F
import numpy as np
import random
"""Set all seeds to ensure reproducibility."""
np.random.seed(0)
torch.manual_seed(0)
random.seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPS = 1e-15

def disable(func):
    def wrapper(*args, **kwargs):
        return torch.tensor(0.).to(device)
    return wrapper

def cross_entropy(preds, targets, reduction='none'):
    log_softmax = torch.nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()

def discriminate(z, summary, model_weight, sigmoid=True):            
    summary = summary.t() if summary.dim() > 1 else summary
    value = torch.matmul(z, torch.matmul(model_weight, summary))
    return torch.sigmoid(value) if sigmoid else value

def compute_distance(a, b, dist_method='euclidean'):
    if dist_method == 'cosine':
        # Cosine Similarity
        sim = (
            torch.mm(a, torch.t(b))
            / (a.norm(dim=1).reshape(-1, 1) * b.norm(dim=1).reshape(1, -1))
        )
        diff = 1 - sim
        return sim, diff

    elif dist_method == 'euclidean':
        # Euclidean Distance
        dist = torch.cdist(a, b, p=2)
        sim = 1 / (1+dist)
        sim = 2 * sim - 1  # This scaling line is important
        sim = sim / sim.max()
        return sim, dist

_f = lambda x: float(x)
    
class LossFunctions:

    @staticmethod
    def bce_with_logits(preds, labels, pos_weight, norm):
        cost = norm * F.binary_cross_entropy_with_logits(
            preds, 
            labels, 
            pos_weight=pos_weight,
            reduction='mean')
        return cost 
    
    @staticmethod
    def bce_flat(reconstructed, data, reduction='sum'):
        reconstructed = reconstructed.clip(min=0, max=1)
        data = data.clip(min=0, max=1)
        
        return F.binary_cross_entropy(reconstructed.view(-1), data.view(-1), reduction=reduction)
    
    @staticmethod
    def spatial_loss(z1, z2, z3=None, sp_dists=None):
        """
        Pushes the closeness between embeddings to 
        not only reflect the expression similarity 
        but also their spatial proximity
        """
        sp_loss = 0
        
        iterate_over = [z1, z2]
        if z3 is not None:
            iterate_over.append(z3)
        for z in iterate_over:
            z_dists = torch.cdist(z, z, p=2)
            z_dists = torch.div(z_dists, torch.max(z_dists))
            n_items = z.size(dim=0) * z.size(dim=0)
            sp_loss += torch.div(torch.sum(torch.mul(1.0 - z_dists, sp_dists)), n_items)
        return sp_loss 
        
    @staticmethod
    def alignment_loss(emb1, emb2, P):
        """
        Pushes the embeddings for each modality 
        to be on the same latent space.
        """
        P_sum = P.sum(axis=1)
        P_sum[P_sum==0] = 1
        P = P / P_sum[:, None]
        _, cdiff = compute_distance(emb1, emb2)
        weighted_P_cdiff = cdiff * P
        alignment_loss = weighted_P_cdiff.absolute().sum() / P.absolute().sum()
        return alignment_loss

    @staticmethod
    def balance_loss(sigma):
        """Balances the contribution of each modality 
        in shaping the shared latent space."""
        sig_norm = sigma / sigma.sum()
        sigma_loss = (sig_norm - .5).square().mean()
        return sigma_loss
    
    @staticmethod
    def mean_sq_error(reconstructed, data):
        """Controls the ability of the latent space to encode information."""
        reconstruction_loss = (reconstructed - data).square().mean(axis=1).mean(axis=0)
        return reconstruction_loss
    
    def mean_sq_error_(reconstructed, data):
        """Controls the ability of the latent space to encode information."""
        reconstruction_loss = (reconstructed - data).square().mean(axis=1).mean(axis=0)
        return reconstruction_loss
        
    # @staticmethod
    # def cross_loss(comb1, comb2, W):
    #     _, comdiff1 = compute_distance(comb1, comb2)
    #     cross_loss = comdiff1 * W
    #     cross_loss = cross_loss.sum() / prod(W.shape)
    #     return cross_loss

    @staticmethod
    def kl(epoch, epochs, mu, logvar, anneal=False):
        """
        Controls the smoothness of the latent space.
        The KL divergence measures the amount of information lost 
        when using Q to approximate P.
        """
        kl_loss =  -.5 * torch.mean(
                        1
                        + logvar
                        - mu.square()
                        - logvar.exp(),
                        axis=1
                    ).mean(axis=0)
        
        c = epochs / 2  # Midpoint
        kl_anneal = 1 / ( 1 + np.exp( - 5 * (epoch - c) / c ) )
        
        if anneal:
            kl_loss = 1e-3 * kl_anneal * kl_loss
        else:
            kl_loss = 1e-3 * kl_loss
            
        
        return kl_loss
    
    @staticmethod
    def kl_sum(mu, logvar, ):
        f=torch.sum
        s=1
        kl= -(0.5 / s) * f(torch.sum(1 + 2 * logvar - mu.pow(2) - logvar.exp().pow(2), 1))
        return kl
    
    @staticmethod
    def f_recons(comb1, comb2):
        corF = torch.eye(comb1.shape[0], comb2.shape[0]).to(device)
        F_est = torch.square(
                    comb1 - torch.mm(corF, comb2)
                ).mean(axis=1).mean(axis=0)
        
        return F_est
    
    @staticmethod
    def cosine_loss(emb, comb):
        cosim, codiff = compute_distance(emb, comb)
        cosine_loss = torch.diag(codiff.square()).mean(axis=0) / emb.shape[1]
        return cosine_loss
    
    @staticmethod
    def mutual_info_loss(pos_z, neg_z, summary, model_weight):
        pos_loss = -torch.log(discriminate(pos_z, summary, model_weight, sigmoid=True) + EPS).mean()
        neg_loss = -torch.log(1 - discriminate(neg_z, summary, model_weight, sigmoid=True) + EPS).mean()
        return pos_loss + neg_loss


class Loss:
    
    _base_alpha = 0.1
    
    def __init__(self, max_epochs, use_hist, use_annealing=False):
        self.max_epochs = max_epochs  
        self.mse = torch.nn.MSELoss()
        self.use_hist = use_hist
        self.use_annealing = use_annealing
        
        self.alpha = {
            'kl_gex': self._base_alpha,
            'kl_pex': self._base_alpha,
            'recons_gex': self._base_alpha,
            'recons_pex': self._base_alpha,
            'cosine_gex': self._base_alpha,
            'cosine_pex': self._base_alpha,
            'adj': self._base_alpha,
            'spatial': self._base_alpha,
            'mutual_gex': self._base_alpha,
            'mutual_pex': self._base_alpha,
            'recons_img': self._base_alpha,
            'kl_img': self._base_alpha,  
            'cosine_img': self._base_alpha, 
            'align': self._base_alpha,
            'clip': self._base_alpha       
        }


    def __call__(self, epoch, varz):
        return self.compute(epoch, varz)
    
    def compute(self, epoch, varz) -> Namespace:
        buffer = Namespace()
        a = self.alpha
        c = self.max_epochs / 2  # Midpoint
        
        if self.use_annealing:
            anneal = 1 / ( 1 + np.exp( - 5 * (epoch - c) / c ) )
        else:
            anneal = 1
        
        buffer.kl_loss_gex = a['kl_gex'] * LossFunctions.kl(
            epoch, self.max_epochs, varz.gex_mu, varz.gex_logvar)
        buffer.kl_loss_pex = a['kl_pex'] * LossFunctions.kl(
            epoch, self.max_epochs, varz.pex_mu, varz.pex_logvar)
        # buffer.recons_loss_gex = a['recons_gex'] * LossFunctions.mean_sq_error(
        #     varz.gex_recons, varz.gex_input)
        # buffer.recons_loss_pex = a['recons_pex'] * LossFunctions.mean_sq_error(
        #     varz.pex_recons, varz.pex_input)
        
        buffer.recons_loss_gex = a['recons_gex'] * LossFunctions.bce_flat(
            varz.gex_recons, varz.gex_input)
        buffer.recons_loss_pex = a['recons_pex'] * LossFunctions.bce_flat(
            varz.pex_recons, varz.pex_input)
        
        buffer.cosine_loss_gex = a['cosine_gex'] * LossFunctions.cosine_loss(
            varz.gex_z, varz.gex_c)
        buffer.cosine_loss_pex = a['cosine_pex'] * LossFunctions.cosine_loss(
            varz.pex_z, varz.pex_c)
        buffer.adj_loss = a['adj'] * LossFunctions.bce_with_logits(
            varz.adj_recon, varz.adj_label, varz.pos_weight, varz.norm)
        buffer.mutual_info_loss = a['mutual_gex'] * LossFunctions.mutual_info_loss(
            varz.gex_pos_z, varz.gex_neg_z, varz.gex_summary, varz.gex_model_weight)
        buffer.mutual_info_loss = a['mutual_pex'] * LossFunctions.mutual_info_loss(
            varz.pex_pos_z, varz.pex_neg_z, varz.pex_summary, varz.pex_model_weight)
        
        if not self.use_hist:
            buffer.spatial_loss = a['spatial'] * LossFunctions.spatial_loss(
                varz.gex_z, varz.pex_z, z3=None, sp_dists=varz.gex_sp_dist)
            buffer.alignment_loss = a['align'] * self.mse(varz.gex_z, varz.pex_z) * anneal
            
        else:
            buffer.spatial_loss = a['spatial'] * LossFunctions.spatial_loss(
                varz.gex_z, varz.pex_z, varz.img_z, varz.gex_sp_dist)
            
            # buffer.kl_loss_img = a['kl_img'] * LossFunctions.kl_sum(varz.img_mu, varz.img_logvar)
            # buffer.recons_loss_img = a['recons_img'] * LossFunctions.bce_flat(varz.img_recons, varz.img_input, reduction='mean')
            
            buffer.cosine_loss_img = a['cosine_img'] * LossFunctions.cosine_loss(varz.img_z, varz.img_c)
            
            buffer.alignment_loss = a['align'] * (
                self.mse(varz.gex_z, varz.pex_z) + \
                self.mse(varz.gex_z, varz.img_z) + \
                self.mse(varz.img_z, varz.pex_z)
            )
            
            
        return buffer

class NonSpatialLoss(Loss):
    def compute(self, epoch, varz) -> Namespace:
        buffer = Namespace()
        a = self.alpha
        buffer.kl_loss_gex = a['kl_gex'] * LossFunctions.kl(epoch, self.max_epochs, varz.gex_mu, varz.gex_logvar)
        buffer.kl_loss_pex = a['kl_pex'] * LossFunctions.kl(epoch, self.max_epochs, varz.pex_mu, varz.pex_logvar)
        buffer.recons_loss_gex = a['recons_gex'] * LossFunctions.mean_sq_error(varz.gex_recons, varz.gex_input)
        buffer.recons_loss_pex = a['recons_pex'] * LossFunctions.mean_sq_error(varz.pex_recons, varz.pex_input)
        buffer.cosine_loss_gex = a['cosine_gex'] * LossFunctions.cosine_loss(varz.gex_z, varz.gex_c)
        buffer.cosine_loss_pex = a['cosine_pex'] * LossFunctions.cosine_loss(varz.pex_z, varz.pex_c)
        
        return buffer

    
class MultiTaskLoss(torch.nn.Module):
    """
    https://arxiv.org/abs/1705.07115
    Adapted from https://github.com/ywatanabe1989/custom_losses_pytorch/    
    """
    
    def __init__(self, is_regression, reduction='sum'):
        # super(MultiTaskLoss, self).__init__()
        super().__init__()
        
        self.is_regression = is_regression
        self.n_tasks = len(is_regression)
        self.log_vars = torch.nn.Parameter(torch.zeros(self.n_tasks))
        self.reduction = reduction


    def forward(self, losses):
        dtype = losses.dtype
        device = losses.device
        stds = (torch.exp(self.log_vars)**(1/2)).to(device).to(dtype)
        self.is_regression = self.is_regression.to(device).to(dtype)
        coeffs = 1 / ( (self.is_regression+1)*(stds**2) )
        multi_task_losses = coeffs*losses + torch.log(stds)

        if self.reduction == 'sum':
            multi_task_losses = multi_task_losses.sum()
        if self.reduction == 'mean':
            multi_task_losses = multi_task_losses.mean()

        return multi_task_losses
    
    

class Metrics:
    def __init__(self, track=False):
        self.track = track
        self.means = Namespace()
        self.counters = {}
        self.values = Namespace()
        self.losses = []

            
    def __str__(self) -> str:
        str_repr = ''
        for k, v in self.means.__dict__.items():
            str_repr+=f'{k}: {v:.3e}'
            str_repr+=' | '
            
        str_repr+=f'loss: {np.mean(self.losses):.3e}'
        return str_repr
    
    def __repr__(self) -> str:
        return self.__str__() 
    
    def custom_repr(self, keys):
        ...  
    
    def update_value(self, name, value, track=False):
        _means = self.means.__dict__
        _values = self.values.__dict__
        
        if name not in _means:
            _means[name] = _f(value)
            self.counters[name] = 0
            if track:
                _values[name] = [_f(value)]
        else:
            _means[name] = (_f(
                (_means[name]*self.counters[name]) + _f(value)
            )) / (self.counters[name] + 1)
            if track:
                _values[name].append(_f(value))
                
        self.counters[name] += 1

        
                
    def update(self, bufffer):
        loss = 0
        all_losses = []
        
        for name, value in bufffer.__dict__.items(): 
            if 'loss' not in name:
                continue 
            self.update_value(name, value, track=self.track)
            loss += value
            all_losses.append(value)
            
        self.losses.append(_f(loss))
        return torch.stack(all_losses)
    
    def __call__(self, ledger):
        return self.update(ledger)
        
    def get(self, name):
        return self.mean.__dict__.get(name, None)

    def get_values(self, name):
        return self.values.__dict__.get(name, None)