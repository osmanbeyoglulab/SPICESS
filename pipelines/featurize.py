import numpy as np
import scanpy as sc
import scipy.sparse as sp
import squidpy as sq
import torch
from anndata import AnnData
from muon import prot as pt
from pipeline import Pipeline
from scipy.sparse import csr_matrix
from sklearn.preprocessing import MinMaxScaler
from utils import graph_alpha, preprocess_graph


class FeaturizePipeline(Pipeline):
    """
    `run()`:
        Preprocesses the input AnnData object and adds graph-related information to its `obsm` and `uns` attributes.
    
    Parameters
    ----------
    adata : AnnData
        The input AnnData object.
    min_max : bool, optional (default: True)
        Whether to apply min-max scaling to the input data.
    clr : bool, optional (default: False)
        Whether to apply center-log ratio (CLR) transformation to the input data.
    log : bool, optional (default: True)
        Whether to apply log2 transformation to the input data.
    resolution : float or None, optional (default: 0.3)
        The resolution parameter for the Leiden clustering algorithm. If None, clustering is not performed.
    layer_added : str, optional (default: 'normalized')
        The name of the layer to be added to the `layers` and `obsm` attributes of the AnnData object.
    
    Returns
    -------
    AnnData
        The preprocessed AnnData object with additional graph-related information in its `obsm` and `uns` attributes.
    """
    
    def __init__(self, clr, min_max, log, minmax_vertical=False, layer_added='normalized', neighbors=6):
        super().__init__()
        self.neighbors = neighbors
        self.clr = clr
        self.min_max = min_max
        self.log = log
        self.minmax_vertical = minmax_vertical
        self.layer_added = layer_added
        
        
    def make_graph(self, spatial_coords):
        adj = graph_alpha(spatial_coords, n_neighbors=self.neighbors)
        adj_label = adj + sp.eye(adj.shape[0])
        adj_label = adj_label.toarray()
        
        pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
        norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)
        adj_norm = preprocess_graph(adj).to_dense().numpy()
        coords = torch.tensor(spatial_coords).float()
        sp_dists = torch.cdist(coords, coords, p=2)
        
        return adj_label, adj_norm, pos_weight, norm, sp_dists
    
    
    def clr_normalize(self, X):
        """Take the logarithm of the surface protein count for each cell, 
        then center the data by subtracting the geometric mean of the counts across all cells."""
        def seurat_clr(x):
            s = np.sum(np.log1p(x[x > 0]))
            exp = np.exp(s / len(x))
            return np.log1p(x / exp)
        return np.apply_along_axis(seurat_clr, 1, X)

            
    def run(self, adata, clr=None, min_max=None, log=None, minmax_vertical=None, layer_used=None, layer_added=None) -> AnnData:
        
        if clr is None:
            clr = self.clr
        if log is None:
            log = self.log
        if min_max is None:
            min_max = self.min_max
        if layer_added is None:
            layer_added = self.layer_added
        if minmax_vertical is None:
            minmax_vertical = self.minmax_vertical
                            
        # sc.pp.normalize_total(adata, inplace=True, target_sum=1e6)
        
        if layer_used is None:
            features = csr_matrix(adata.X).toarray()
        else:
            features = adata.obsm[layer_used]
        
        if clr:
            features = pt.pp.clr(AnnData(features), inplace=False).X
            # features = self.clr_normalize(features)
            
        if log:
            features = np.log2(features+0.5)  
            # features = np.log(features+1)  
            
        if min_max:
            scaler = MinMaxScaler()
            if self.minmax_vertical:
                features = np.transpose(scaler.fit_transform(np.transpose(features)))
            features = scaler.fit_transform(features)
            
        adata.obsm[layer_added] = np.array(features)
        adata.layers[layer_added] = np.array(features)
        
        if 'spatial' in adata.obsm:
            adj_label, adj_norm, pos_weight, norm, sp_dists = self.make_graph(adata.obsm['spatial'])
        else:
            coords = torch.randn(adata.shape[0], 2)
            adj_label, adj_norm, pos_weight, norm, sp_dists = self.make_graph(coords)
            
        
        if 'leiden_colors' in adata.uns:
            adata.uns.pop('leiden_colors')
        
        adata.obsm['adj_label'] = adj_label
        adata.obsm['adj_norm'] = adj_norm
        adata.obsm['sp_dists'] = sp_dists.numpy()
        adata.uns['pos_weight'] = pos_weight
        adata.uns['norm'] = norm
        
        if 'counts' in adata.layers:
            adata.X = adata.layers['counts']
        if 'spatial' in adata.obsm:
            sq.gr.spatial_neighbors(adata)
            sq.gr.spatial_autocorr(adata, layer='normalized')
        
        return adata
    
