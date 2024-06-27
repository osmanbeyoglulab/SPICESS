import numpy as np
import sys
import scanpy as sc
import pandas as pd
from pipelines.load import LoadCytAssistPipeline, LoadVisiumPipeline, LoadSeuratTonsilsPipeline
from pipelines.featurize import FeaturizePipeline
from pipelines.train import TrainModelPipeline
from pipelines.infer import InferencePipeline

from utils import cluster, align_adata, column_corr, floatify, tocpu, alpha_shape
from utils import alpha_shape


sys.path.append('./JAMIE') ##https://github.com/daifengwanglab/JAMIE.git
from jamie import JAMIE

tissue = 'Tonsil'
antibody_panel = pd.read_csv('./antibody_panel.csv')
data_path = '/ix/hosmanbeyoglu/spicess_datasets/TonsilTissue'

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

gene_f = FeaturizePipeline(clr=False, min_max=True, log=True, minmax_vertical=True)
protein_f = FeaturizePipeline(clr=True, min_max=True, log=False, minmax_vertical=True)

gene_f.run(adata)
protein_f.run(adata_eval)
gene_f.run(pdata)
protein_f.run(pdata_eval);

# corr = np.eye(adata.shape[0], pdata.shape[0])
corr = np.eye(adata_eval.shape[0], pdata_eval.shape[0])

jm = JAMIE(min_epochs=500)

integrated_data = jm.fit_transform(dataset=[adata_eval.obsm['normalized'], 
                                            pdata_eval.obsm['normalized']], P=corr)

pred = jm.modal_predict(adata.obsm['normalized'], 0)

pd.DataFrame(np.array(column_corr(pdata.obsm['normalized'], pred)), 
             index=pdata.var_names, columns=['JAMIE 2']).to_csv('jamie_performance.csv')