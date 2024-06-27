import scipy.sparse as sp
import torch
import yaml
from pipelines.pipeline import Pipeline
from spicess.vae_infomax import InfoMaxVAE
from utils import graph_alpha, preprocess_graph, floatify
from anndata import AnnData
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

class InferencePipeline(Pipeline):
    
    def __init__(self, featurizer, config_pth=None, model = None, tissue=None, proteins=None):
        super().__init__()
        
        self.featurizer = featurizer
        
        if model is not None:
            self.model = model.cuda()
        else:
                
            with open(config_pth, 'r') as f:
                self.config = yaml.safe_load(f)
                
            self.model = InfoMaxVAE(
                [16, self.config['nproteins']], 
                latent_dim = self.config['latent_dim'], 
                dropout = self.config['dropout']
            ).cuda()
            
            self.tissue = self.config['tissue']
            self.proteins = self.config['proteins']
            
            self.model.load_state_dict(torch.load('../model_zoo/'+self.config['model']))
        
        self.model.eval()
        
        
        if tissue is not None:
            self.tissue = tissue
        if proteins is not None:
            self.proteins = proteins
        # print('Created inference pipeline.')
        
    def ingest(self, adata):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        d13 = floatify(adata.obsm['normalized'])
        adj = graph_alpha(adata.obsm['spatial'], n_neighbors=6)
        adj_label = adj + sp.eye(adj.shape[0])
        adj_label = torch.tensor(adj_label.toarray()).to(device)
        adj_norm = preprocess_graph(adj).to(device)
        A2 = adj_norm.to_dense()
        
        return d13, A2

    def run(self, adata: AnnData, normalize: bool = False, use_spatial=True):
        assert isinstance(adata, AnnData), 'adata must be an AnnData object.'
        if 'adj_norm' in adata.obsm:
            del adata.obsm['adj_norm']
        adata = adata.copy()
        self.featurizer.run(adata)
        adatax = adata
        scaler = MinMaxScaler()
        
        indices = self.model.train_idx
        
        # indices = [list(adata.var_names).index(
        #     i.replace('_1', '').replace('_2', '')) for i in self.proteins]

        if use_spatial:
            d13, A2 = self.ingest(adatax)
        else:
            d13 = floatify(adatax.obsm['normalized'])
            A2 = torch.zeros(d13.shape[0], d13.shape[0]).to(d13.device)
        imputed_proteins, z_latent, gex_recons = self.model.impute(d13, A2, indices, return_z=True)

        if normalize:
            proteins_norm = pd.DataFrame(scaler.fit_transform(imputed_proteins), 
                    columns=self.proteins)
        else:
            proteins_norm = pd.DataFrame(imputed_proteins, 
                    columns=self.proteins)
        
        pdata_eval = AnnData(proteins_norm)
        
        ## clone metadata
        pdata_eval.obs = adatax.obs
        pdata_eval.obsm['spatial'] = adatax.obsm['spatial']
        pdata_eval.uns['spatial'] = adatax.uns['spatial']
        pdata_eval.obsm['embeddings'] = z_latent
        pdata_eval.obsm['gex_recons'] = gex_recons
        
        return pdata_eval
    
    