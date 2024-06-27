import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import torch
from anndata import AnnData
import seaborn as sns
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from scipy.stats import spearmanr
import torch.nn.functional as F
import cuml
from sklearn.preprocessing import MinMaxScaler
from matplotlib.colors import ListedColormap
from PIL import Image
from collections import OrderedDict
from scipy.stats import pearsonr
import squidpy as sq
import timm
import itertools
import anndata as ad
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import itertools
from collections import Counter
import muon as mu
import STAGATE_pyG
from pipelines.load import LoadCytAssistPipeline, LoadVisiumPipeline, LoadSeuratTonsilsPipeline
from pipelines.featurize import FeaturizePipeline
from pipelines.train import TrainModelPipeline
from pipelines.infer import InferencePipeline

from utils import cluster, align_adata, column_corr, floatify, tocpu, alpha_shape
from utils import alpha_shape

tissue = 'Tonsil'
antibody_panel = pd.read_csv('./antibody_panel.csv')
data_path = '/ix/hosmanbeyoglu/spicess_datasets/TonsilTissue'

base = '/ix/hosmanbeyoglu/kor11/CytAssist/Breast/wu_et_al/'

def load_10x_mtx(pth):
    adata = sc.read_10x_mtx(pth)
    sc.pp.filter_genes(adata, min_cells=5, inplace=True)
    adata.layers['counts'] = adata.X
    
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=1600)
    adata = adata[:, (adata.var['highly_variable'] 
                      | adata.var_names.isin(['PTPRC']+list(pdata.var_names)))]
    adata.X = adata.layers['counts']

    filename = pth[len(base):].split('/')[1].replace('raw_feature_bc_matrix', 'spatial')
    spatial_info = base+f"spatial/"+filename

    meta = base+'metadata/'+filename.replace('_spatial', '_metadata.csv')
    img = Image.open(spatial_info+"/tissue_hires_image.png")
    img = np.array(img)

    adata.uns['spatial'] = {filename: {'images': {'hires': img/255}}}
    xy = pd.read_csv(spatial_info+"/tissue_positions_list.csv", header=None).values
    adata.obs.join(pd.DataFrame(xy).set_index(0))
    adata.obsm['spatial'] = adata.obs.join(pd.DataFrame(xy).set_index(0))[[5, 4]].values

    with open(spatial_info+'/scalefactors_json.json') as f:
        adata.uns['spatial'][filename]['scalefactors'] = json.load(f)

    adata.obs = adata.obs.join(pd.read_csv(meta, index_col=0))
    
    return adata

train_loader = LoadCytAssistPipeline(
    tissue=tissue, 
    h5_file=data_path+'/GEX_PEX/filtered_feature_bc_matrix.h5',
    sample_id = 0,
    name = 'Tonsil 1',
)

eval_loader = LoadCytAssistPipeline(
    tissue=tissue, 
    h5_file=data_path+'/GEX_PEX_2/filtered_feature_bc_matrix.h5',
    sample_id = 1,
    name = 'Tonsil 2',
)

adata, pdata = train_loader.run() 
adata_eval, pdata_eval = eval_loader.run()

sc.pp.filter_genes(adata, min_cells=5, inplace=True)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, n_top_genes=1500)
adata = adata[:, (adata.var['highly_variable'] 
                  | adata.var_names.isin(['PTPRC']+list(pdata.var_names)))]
adata.X = adata.layers['counts']

adata_eval = align_adata(adata_eval, adata)

trainer = TrainModelPipeline(
    tissue = 'Tonsil',
    adatas = [adata, adata_eval],
    pdatas = [pdata, pdata_eval],
    repeats = 10,
    latent_dim = 512,
    save = True,
    lr = 2e-5,
)

spicess = trainer.run()
pd.concat([spicess.__dict__[f'iter_{i}'].results for i in range(10)], axis=1).to_csv('tonsil_results.csv')


adata, pdata = train_loader.run() 
adata_eval, pdata_eval = eval_loader.run()

sc.pp.filter_genes(adata, min_cells=5, inplace=True)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, n_top_genes=1500)
adata = adata[:, (adata.var['highly_variable'] 
                  | adata.var_names.isin(['PTPRC']+list(pdata.var_names)))]
adata.X = adata.layers['counts']

adata_eval = align_adata(adata_eval, adata)

trainer = TrainModelPipeline(
    tissue = 'Tonsil',
    adatas = [adata_eval, adata],
    pdatas = [pdata_eval, pdata],
    repeats = 10,
    latent_dim = 512,
    save = True,
    lr = 2e-5,
)

spicess = trainer.run()
pd.concat([spicess.__dict__[f'iter_{i}'].results for i in range(10)], axis=1).to_csv('tonsil_results_2.csv')