from argparse import Namespace
from matplotlib import pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
import torch
import yaml
from early_stopping import EarlyStopping
from pipeline import Pipeline
from spicess.modules.losses import Loss, Metrics
from spicess.vae_infomax import InfoMaxVAE
from utils import ImageSlicer, column_corr, column_corr_p, floatify, tocpu
from tqdm import tqdm
import pandas as pd
from datetime import datetime

from .featurize import FeaturizePipeline
import random

"""Set all seeds to ensure reproducibility."""
np.random.seed(0)
torch.manual_seed(0)
random.seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
floatify = lambda x: torch.tensor(x).to(device).float()

class TrainModelPipeline(Pipeline):
    """
    Pipeline for training a model using matched spatial gene expression and protein expression data.
    """
    
    def __init__(self, 
            tissue,
            adatas: list,
            pdatas: list,
            repeats: int = 1,
            cite_seq: list = None,
            latent_dim: int = 64, 
            dropout: float = 0.0, 
            lr: float = 2e-3, 
            wd: float = 0.0,
            patience: int = 300,
            delta: float = 1e-5,
            epochs: int = 5000,
            kl_gex: float = 1e-6, 
            kl_pex: float = 1e-6,
            kl_img: float = 1e-8, 
            recons_gex: float = 1e-3, 
            recons_pex: float = 1e-3, 
            recons_img: float = 1e-3,
            cosine_gex: float = 1e-4, 
            cosine_pex: float = 1e-4,
            align: float = 1e-4,
            cosine_img: float = 1e-4, 
            adj: float = 1e-6, 
            spatial: float = 1e-5, 
            mutual_gex: float = 1e-3, 
            mutual_pex: float = 1e-3,
            batch_size: int = 8,
            use_annealing: bool = False,
            cross_validate: bool = False,
            use_histology: bool = False,
            pretrained: str = None,
            freeze_encoder: bool = True,
            minmax_vertical=False,
            min_max=True,
            gene_log=True,
            protein_log=False,
            save: bool = False):
        super().__init__()
        self.tissue = tissue
        self.adatas = adatas
        self.pdatas = pdatas
        self.repeats = repeats
        self.cite_seq = cite_seq
        self.epochs = int(epochs)
        self.latent_dim = latent_dim
        self.dropout = dropout
        self.patience = patience
        self.delta = delta
        self.batch_size = batch_size
        self.lr = lr
        self.wd = wd
        self.save = save
        self.use_histology = use_histology
        self.pretrained = pretrained
        self.freeze_encoder = freeze_encoder
        self.gene_featurizer = FeaturizePipeline(clr=False, min_max=min_max, log=gene_log, minmax_vertical=minmax_vertical)
        self.protein_featurizer = FeaturizePipeline(clr=True, min_max=min_max, log=protein_log, minmax_vertical=minmax_vertical)
        
        self.loss_func = Loss(max_epochs=epochs, use_hist=use_histology, use_annealing=use_annealing)

        self.loss_func.alpha = {
            'kl_gex': kl_gex,
            'kl_pex': kl_pex,
            'recons_gex': recons_gex,
            'recons_pex': recons_pex,
            'cosine_gex': cosine_gex,
            'cosine_pex': cosine_pex,
            'adj': adj,
            'spatial': spatial,
            'mutual_gex': mutual_gex,
            'mutual_pex': mutual_pex,    
            'kl_img': kl_img,
            'recons_img': recons_img,
            'cosine_img': cosine_img,
            'align': align,
        }
        
        # self.metrics = Metrics(track=True)
        self.cross_validate = cross_validate
        
        
        self.artifacts = Namespace(
            tissue=self.tissue,
            epochs=self.epochs,
            latent_dim=self.latent_dim,
            dropout=self.dropout,
            patience=self.patience,
            delta=self.delta,
            lr=self.lr,
            wd=self.wd,
            kl_gex=self.loss_func.alpha['kl_gex'],
            kl_pex=self.loss_func.alpha['kl_pex'],
            recons_gex=self.loss_func.alpha['recons_gex'],
            recons_pex=self.loss_func.alpha['recons_pex'],
            cosine_gex=self.loss_func.alpha['cosine_gex'],
            cosine_pex=self.loss_func.alpha['cosine_pex'],
            adj=self.loss_func.alpha['adj'],
            spatial=self.loss_func.alpha['spatial'],
            mutual_gex=self.loss_func.alpha['mutual_gex'],
            mutual_pex=self.loss_func.alpha['mutual_pex']
        )
        
        print('Created training pipeline.')
    

    
    
    def subset_adata(self, adata, segment):
        """Return adata_train and adata_eval based on the segment"""
        a1 = adata[adata.obs.train_test!=segment, :]
        a2 = adata[adata.obs.train_test==segment, :]
        
        return a1, a2
    

    def assign_groups(self, coordinates, group_size=10):
        kmeans = KMeans(n_clusters=group_size)
        kmeans.fit(coordinates)
        return kmeans.labels_
    
    
    def plot_losses(self, width=22, height=10, dpi=160):
        
        colors = ["#b357c2",
        "#82b63c",
        "#6a69c9",
        "#5cb96c",
        "#d14485",
        "#4bbeb1",
        "#cf473f",
        "#387e4d",
        "#c77cb6",
        "#c8a94a",
        "#6b94d0",
        "#e38742",
        "#767d34",
        "#b9606c",
        "#a46737"]

        f, axs = plt.subplots(3, 4, figsize=(width, height), dpi=dpi)

        ordered_losses = ['cosine_loss_gex', 'recons_loss_gex', 'kl_loss_gex', 'spatial_loss', 
                        'cosine_loss_pex', 'recons_loss_pex', 'kl_loss_pex', 'adj_loss',
                        'alignment_loss', 'mutual_info_loss', 'oracle', 'oracle_self']

        for i, (loss_name, ax) in enumerate(zip(ordered_losses, axs.flatten())):
            ax.plot(self.metrics.values.__dict__[loss_name], color=colors[i])
            ax.set_title(loss_name, color=colors[i], fontsize=15, fontweight='bold')
            ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

        plt.tight_layout()
        plt.show()
            

    
    
    def run(self):
        output = Namespace()
        if self.cross_validate:
            
            adata = self.adatas[0].copy()
            pdata = self.pdatas[0].copy()
            groups = self.assign_groups(adata.obsm['spatial'], 2)
            adata.obs['train_test'] = groups
            adata.obs['train_test'] = adata.obs['train_test'].astype('category')
            adata.obs['idx'] = list(range(0, len(adata)))
            pdata.obs['idx'] = list(range(0, len(pdata)))
            pdata.obs['train_test'] = adata.obs['train_test']
            
            imputed_concat = np.ones((adata.shape[0], pdata.shape[1]))
            real_concat = np.ones((adata.shape[0], pdata.shape[1]))
            
                
            for segment in adata.obs.train_test.unique():                
                a_train, a_eval = self.subset_adata(adata.copy(), segment=segment)
                p_train, p_eval = self.subset_adata(pdata.copy(), segment=segment)
                out, _ = self.train(a_train, p_train, a_eval, p_eval, label=f'Patch: {segment}')
                print(segment, out.results.values)
                
                imputed_concat[a_eval.obs.idx.values, :] = out.imputed_proteins
                real_concat[a_eval.obs.idx.values, :] = out.d14[:, :].data.cpu().numpy()
                
            corr = np.mean(column_corr(imputed_concat, real_concat))
            print('Final correlation: ', corr)  
            
            output.imputed_proteins = imputed_concat
            output.real_proteins = real_concat
                      
            
        # with CleanExit():
        #     output, artifacts = self.train(self.adata, self.pdata, self.adata_eval, self.pdata_eval)
        # output, artifacts = self.train(self.adata_eval, self.pdata_eval, self.adata, self.pdata)
        else:
            for i in range(self.repeats):
                out, artifacts = self.train(self.adatas[0], self.pdatas[0], self.adatas[1], self.pdatas[1])
                out.results.columns = [f'CORR_{i}']
                output.__dict__[f'iter_{i}'] = out
                
            # output, artifacts = self.train_multi(self.adatas, self.pdatas)
            
            if self.save:
                ts = datetime.now().strftime("%Y_%m_%d_%H_%M")
                name = f'{self.tissue}_{ts}'
                torch.save(output.model.state_dict(), f'../model_zoo/model_{name}.pth')
                artifacts.model = f'model_{name}.pth'
                artifacts.proteins = list(self.pdatas[0].var_names)
                with open(f'../model_zoo/config_{name}.yaml', 'w') as f:
                    yaml.dump(vars(self.artifacts), f)
                
                print(f'Saved model-config to ../model_zoo/config_{name}.yaml')
            
        return output
    
    
    def train_multi(self, adatas, pdatas):
        for adata, pdata in zip(adatas, pdatas):
            self.gene_featurizer.run(adata)
            self.protein_featurizer.run(pdata)
            
        model = InfoMaxVAE(
            [adata.obsm['normalized'].shape[1], pdata.obsm['normalized'].shape[1]], 
            latent_dim = self.latent_dim,
            use_hist = self.use_histology,
            pretrained = self.pretrained,
            freeze_encoder = self.freeze_encoder,
            dropout = self.dropout).to(device)
        
        
        es = EarlyStopping(model, patience=self.patience, verbose=False, delta=self.delta)
        es.best_model = model
        
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.wd)
        

        losses = []
        with tqdm(total=self.epochs) as pbar:
            for e in range(self.epochs):
                model.train()
                batch_loss = 0
                for adata, pdata in zip(adatas, pdatas):
                    d11 = floatify(adata.obsm['normalized'])
                    d12 = floatify(pdata.obsm['normalized'])
                    
                    adj_label = floatify(adata.obsm['adj_label'])
                    pos_weight = floatify(adata.uns['pos_weight'])
                    sp_dists = floatify(adata.obsm['sp_dists'])
                    norm = floatify(adata.uns['norm'])
        
                    A = floatify(adata.obsm['adj_norm'])
                        
                    model.train()                    
                    optimizer.zero_grad()
                                    
                    output = model(X=[d11, d12], A=A)            
                    output.epochs = self.epochs
                    output.adj_label = adj_label
                    output.pos_weight = pos_weight
                    output.gex_sp_dist = sp_dists
                    output.norm = norm
                    
                    buffer = self.loss_func.compute(e, output)
                    loss = self.metrics(buffer).sum()
                    loss.backward()
                    batch_loss+=loss.item()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                    optimizer.step()
                    
                    test_protein = pdata.obsm['normalized']
                    
                
                losses.append(float(batch_loss))
                
                imputed_proteins = model.impute(d11, A)
                
                oracle_corr = np.mean(column_corr(test_protein, imputed_proteins))
                self.metrics.update_value('oracle', oracle_corr, track=True)
                    
                desc_str = ""
                desc_str+=f"Imputation: {self.metrics.means.oracle:.3f} | "
                desc_str+=f"Loss: {np.mean(losses):.3g} | "
                desc_str+=f"Alignment: {self.metrics.means.alignment_loss:.3g} | "
                                        
                pbar.update()        
                pbar.set_description(desc_str)  
    
        pbar.close()
    
        model.eval()
        imputed = []
        oracle = []
        for adata, pdata in zip(adatas, pdatas):
            d11 = floatify(adata.obsm['normalized'])
            d12 = floatify(pdata.obsm['normalized'])
            adj_label = floatify(adata.obsm['adj_label'])
            pos_weight = floatify(adata.uns['pos_weight'])
            sp_dists = floatify(adata.obsm['sp_dists'])
            norm = floatify(adata.uns['norm'])
            A = floatify(adata.obsm['adj_norm'])
            test_protein = pdata.obsm['normalized']
            imputed_proteins = model.impute(d11, A)
            oracle_corr = np.mean(column_corr(test_protein, imputed_proteins))
            imputed.append(imputed_proteins)
            oracle.append(oracle_corr)
        
        
        output = Namespace()
        output.model = model
        output.imputed = imputed
        output.metrics = self.metrics
        return output, self.artifacts
    
    def train(self, adata_train, pdata_train, adata_eval, pdata_eval, label='Training'):
        self.metrics = Metrics(track=True)
        
        self.gene_featurizer.run(adata_train)
        self.gene_featurizer.run(adata_eval)
        self.protein_featurizer.run(pdata_train)
        self.protein_featurizer.run(pdata_eval)
        
        d11 = floatify(adata_train.obsm['normalized'])
        
        d13 = floatify(adata_eval.obsm['normalized'])

        d12 = floatify(pdata_train.obsm['normalized'])
        d14 = floatify(pdata_eval.obsm['normalized'])    
        
        adj_label = floatify(pdata_train.obsm['adj_label'])
        pos_weight = floatify(pdata_train.uns['pos_weight'])
        sp_dists = floatify(pdata_train.obsm['sp_dists'])
        norm = floatify(pdata_train.uns['norm'])
        
        
        A = floatify(adata_train.obsm['adj_norm'])
        A2 = floatify(adata_eval.obsm['adj_norm'])
        
        if self.use_histology:
            slicer_train = ImageSlicer(adata_train, size=64, grayscale=False)
            slicer_eval = ImageSlicer(adata_eval, size=64, grayscale=False)
            Y = floatify(np.stack([slicer_train(i) for i in range(len(slicer_train))]))
            Y2 = floatify(np.stack([slicer_eval(i) for i in range(len(slicer_eval))]))
            pool = torch.nn.MaxPool2d(2, stride=2)
            
            # print(Y.shape)
            
            # Y = pool(Y.transpose(1, -1).unsqueeze(1))
            # Y2 = pool(Y2.transpose(1, -1).unsqueeze(1))
            Y = pool(Y.permute(0, 3, 1, 2))
            Y2 = pool(Y2.permute(0, 3, 1, 2))
            
        else:
            Y = None
            Y2 = None   
        
        
                
        model = InfoMaxVAE([d11.shape[1], d12.shape[1]], 
                    latent_dim = self.latent_dim,
                    use_hist = self.use_histology,
                    pretrained = self.pretrained,
                    freeze_encoder = self.freeze_encoder,
                    dropout = self.dropout,
                ).to(device)
        
        # model = torch.compile(model)
        
        es = EarlyStopping(model, patience=self.patience, verbose=False, delta=self.delta)
        es.counter = 0
        es.best_model = model
        
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.wd)
        

        losses = []
        test_protein = tocpu(d14)
        
                
        with tqdm(total=self.epochs, disable=False) as pbar:
            pbar.set_postfix_str(f'{d11.shape[1]}d -> {self.latent_dim}d')
            for e in range(self.epochs):
                
                model.train()
                if self.use_histology:
                    model.image_encoder.train()
                
                optimizer.zero_grad()
                
                output = model(X=[d11, d12], Y=Y, A=A)            
                output.epochs = self.epochs
                output.adj_label = adj_label
                output.pos_weight = pos_weight
                output.gex_sp_dist = sp_dists
                output.norm = norm
                
                                
                buffer = self.loss_func.compute(e, output)
                loss = self.metrics(buffer).sum()
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                optimizer.step()
                losses.append(float(loss))
                model.eval()

                imputed_proteins = model.impute(d13, A2)
                
                oracle_corr = np.mean(column_corr(test_protein, imputed_proteins))
                self.metrics.update_value('oracle', oracle_corr, track=True)
                
                oracle_self = np.mean(column_corr(tocpu(d12), model.impute(d11, A)))
                self.metrics.update_value('oracle_self', oracle_self, track=True)
                
                # if e % 50 == 0:
                #     output.gex_recons = torch.nan_to_num(output.gex_recons)
                #     self_recovery = np.mean(column_corr(tocpu(d11), tocpu(output.gex_recons)))
                #     self.metrics.update_value('self_recovery', self_recovery, track=True)
                                
                if self.use_histology: 
                    imputed_from_img = model.img2proteins(Y2)
                    oracle_img = np.mean(column_corr(tocpu(d14), imputed_from_img))
                    self.metrics.update_value('oracle_img', oracle_img, track=True)
                    
                    # imputed_from_img = model.img2proteins(d11, Y)
                    # oracle_img = np.mean(column_corr(d12[:, :].data.cpu().numpy(), imputed_from_img))
                    # self.metrics.update_value('oracle_img', oracle_img, track=True)
                
                es(1-self.metrics.means.oracle, model)
                if es.early_stop: 
                    model = es.best_model
                    break

                # TODO: Make pbar dynamic with show_for     
                desc_str = ""
                # desc_str+=f"{label} > " 
                # if self.use_histology:
                #     desc_str+=f"FromH&E: {self.metrics.means.oracle_img:.3f} || "
                desc_str+=f"Imput: {self.metrics.means.oracle:.3f} | "
                desc_str+=f"SelfImput: {self.metrics.means.oracle_self:.3f} | "
                desc_str+=f"Loss: {np.mean(losses):.3g} | "
                desc_str+=f"Align: {self.metrics.means.alignment_loss:.3g} | "
                # desc_str+=f"Recovery: {self.metrics.means.self_recovery:.3f}"
                
                
                pbar.update()        
                pbar.set_description(desc_str) 
                pbar.set_postfix_str(f'{d11.shape[1]}d -> {self.latent_dim}d || {es.counter}/{es.patience}')
                 

        model = es.best_model
        tmp = model(X=[d11, d12], Y=Y, A=A)            
        model.eval()
        
        imputed_proteins = model.impute(d13, A2)
        
        
        
        output = Namespace()
        output.model = model.cpu()
        output.imputed_proteins = imputed_proteins
        output.d11 = d11.cpu()
        output.d12 = d12.cpu()
        output.d13 = d13.cpu()
        output.d14 = d14.cpu()
        output.Y = Y
        output.A = A.cpu()
        output.A2 = A2.cpu()
        output.gex_z = tmp.gex_z.cpu()
        output.pex_z = tmp.pex_z.cpu()
        output.metrics = self.metrics
        output.results = pd.DataFrame(column_corr(
            imputed_proteins, tocpu(d14)), 
            columns=['CORR'], index=list(self.pdatas[0].var_names))
        output.results_p = pd.DataFrame(column_corr_p(
            imputed_proteins, tocpu(d14)), 
            columns=['CORR'], index=list(self.pdatas[0].var_names))
        
        pbar.close()
        
        
        return output, self.artifacts