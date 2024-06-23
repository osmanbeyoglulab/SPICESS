from collections import OrderedDict
import numpy as np
import pandas as pd
from pipeline import Pipeline
from anndata import AnnData
import glob
from utils import clean_adata
import scanpy as sc
import os
from PIL import Image
import torch
import random
import torchvision.transforms.functional as TF


UNIQUE_VAR_NAMES = [
    'CD163', 'CR2', 'PCNA', 'VIM', 'KRT5', 'CD68', 'CEACAM8', 'PTPRC_1',
    'PAX5', 'SDC1', 'PTPRC_2', 'CD8A', 'BCL2', 'CD19', 'PDCD1', 'ACTA2',
    'FCGR3A', 'ITGAX', 'CXCR5', 'EPCAM', 'MS4A1', 'CD3E', 'CD14', 'CD40',
    'PECAM1', 'CD4', 'ITGAM', 'CD27', 'CCR7', 'CD274'
    ]

VAR_NAMES = [
    'CD163', 'CR2', 'PCNA', 'VIM', 'KRT5', 'CD68', 'CEACAM8', 'PTPRC_1',
    'PAX5', 'SDC1', 'PTPRC_2', 'CD8A', 'BCL2', 'CD19', 'PDCD1', 'ACTA2',
    'FCGR3A', 'ITGAX', 'CXCR5', 'EPCAM', 'MS4A1', 'CD3E', 'CD14', 'CD40',
    'PECAM1', 'CD4', 'ITGAM', 'CD27', 'CCR7', 'CD274'
]


class ImageSlicer(torch.utils.data.Dataset):
    
    def __init__(self, adata, grayscale=False, size=224):
        self.adata = adata
        _spatial = adata.uns['spatial']
        self.key = list(_spatial.keys())[0]
        self.scale_factor = _spatial[self.key ]['scalefactors']['tissue_hires_scalef']
        self.spot_size = _spatial[self.key ]['scalefactors']['spot_diameter_fullres'] * 0.5
        self.xy = adata.obsm['spatial'] * self.scale_factor
        self.size = size
        self.img = _spatial[self.key]['images']['hires']
        self.grayscale = grayscale
        self.reduced_matrix = self.adata.obsm['normalized']
        
    def __len__(self):
        return len(self.xy)
    
    
    def transform(self, image):
        image = Image.fromarray(image)
        # Random flipping and rotations
        if random.random() > 0.5:
            image = TF.hflip(image)
        if random.random() > 0.5:
            image = TF.vflip(image)
        angle = random.choice([180, 90, 0, -90])
        image = TF.rotate(image, angle)
        return np.asarray(image)
        
    def crop_at(self, x, y, d=None):
        if d is None:
            d = int(self.size*0.5)
        image = self.img[y-d:y+d, x-d:x+d] * 255
        image = image.astype(np.uint8)
        
        # return np.dot(image[...,:3], [0.299, 0.587, 0.114])    
        return image
    
    def grab(self, idx):
        x, y = self.xy[idx].astype(int)
        image = self.crop_at(x, y)
        image = self.transform(image)
        image = torch.tensor(image).float()
        return image.numpy() / 255
    
    def __getitem__(self, idx):
        item = {}
        
        x, y = self.xy[idx].astype(int)

        image = self.crop_at(x, y)
        image = self.transform(image)
        
        item['image'] = torch.tensor(image).permute(2, 0, 1).float() #color channel first, then XY
        item['reduced_expression'] = torch.tensor(self.reduced_matrix[idx,:]).float()  #cell x features (3467)
        item['spatial_coords'] = [x, y]

        return item

class LoadSeuratTonsilsPipeline(Pipeline):
    
    def __init__(self, path: str):
        super().__init__()
        meta = """
            c28w2r_7jne4i          0.170068
            esvq52_nluss5          0.173260
            exvyh1_66caqq          0.294406
            gcyl7c_cec61b          0.294406
            p7hv1g_tjgmyj          0.174317
            qvwc8t_2vsr67          0.294406
            tarwe1_xott6q          0.171821
            zrt7gl_lhyyar          0.294406""".split()

        self.metadata = dict(zip(
                [meta[i] for i in range(0, len(meta), 2)], 
                [meta[i+1] for i in range(0, len(meta)-1, 2)]
            )
        )
        
        self.path = path
        self.colData, self.rowData, self.spatial_coords, self.counts = self.load_data()
        print(f'Created Tonsils data loader pipeline with {len(self.metadata)} STs.')
        
    
    def load_data(self):
        path = self.path
        colData = pd.read_csv(path+'colData.csv', engine='pyarrow', index_col=0)  
        rowData = pd.read_csv(path+'rowData.csv', engine='pyarrow', index_col=0)
        spatial_coords = pd.read_csv(path+'spatial_coords.csv', engine='pyarrow', index_col=0)
        counts = pd.read_csv(path+'counts.csv', engine='pyarrow', index_col=0)
        counts = counts.T
        colData.index = counts.index
        spatial_coords.index = counts.index
        
        # ((16224, 27), (26846, 8), (16224, 2), (16224, 26846))
        assert colData.shape[0] == spatial_coords.shape[0]
        assert colData.shape[0] == counts.shape[0]    
        assert rowData.shape[0] == counts.shape[1]
        
        self.sample_ids = list(colData.sample_id.unique())
        
        return colData, rowData, spatial_coords, counts
        
    def build_adata(self, ix, reload_data=False):        
        if reload_data:
            self.colData, self.rowData, self.spatial_coords, self.counts = self.load_data()
            
        sample_id = self.sample_ids[ix]
        # print(sample_id)
        
        colData = self.colData
        rowData = self.rowData
        spatial_coords = self.spatial_coords
        counts = self.counts

        target_idx = colData[colData.sample_id == sample_id].index
        img = Image.open(self.path+f'{sample_id}.png')

        xy = spatial_coords.loc[target_idx].values 

        st_adata = AnnData(
            X=counts.loc[target_idx].astype(float),
            obs=colData.loc[target_idx],
            var=rowData
        )
        st_adata.uns = OrderedDict({'spatial': {sample_id: {'images': 
                            {'hires': np.array(img)}, 
                            'scalefactors': {'spot_diameter_fullres': 1.0, 
                            'tissue_hires_scalef': float(self.metadata[sample_id])}}}})
        st_adata.obsm['spatial'] = xy.astype(float)
        
        return st_adata
    
    def run(self, i):
        return self.build_adata(i)
        

class LoadH5ADPipeline(Pipeline):
    
    def load_h5(self, h5_file, ix):
        adata3 = sc.read_h5ad(h5_file)
        adata3.obs['source'] = f'{self.tissue} Sample {ix}'
        adata3.obs['domain_id'] = 100+int(ix)
        adata3.obs['source'] = adata3.obs['source'].astype('category')
        adata3.obs['domain_id'] = adata3.obs['domain_id'].astype('category')
        return adata3
        
    
    def __init__(self, tissue: str, h5_dir: str):
        super().__init__()
        self.tissue = tissue
        self.fnames = sorted(glob.glob(h5_dir+'/*.h5ad'), key=lambda x: int(x.split('_')[-1].split('.')[0]))
        print(f'Created H5AD data loader pipeline with {len(self.fnames)} h5ad files.')
        
        
    def run(self, i) -> AnnData:
        _adata = self.load_h5(self.fnames[i], i)
        _adata.var_names = _adata.var.feature_name.values
        clean_adata(_adata)
        
        return _adata
        
        
class LoadVisiumPipeline(Pipeline):
    def __init__(self, tissue: str, visium_dir: str, sample_id: int, name: str):
        super().__init__()
        self.tissue = tissue
        self.name = name
        self.visium_dir = visium_dir
        self.sample_id = sample_id
        
        print('Created 10x Visium data loader pipeline.')
        
        
    def run(self):
        adata3 = sc.read_visium(path=self.visium_dir)
        adata3.obsm['spatial'] = adata3.obsm['spatial'].astype(float)
        adata3.layers['counts'] = adata3.X
        adata3.var_names_make_unique()
        adata3.obs['source'] = self.name
        adata3.obs['domain_id'] = self.sample_id
        adata3.obs['source'] = adata3.obs['source'].astype('category')
        adata3.obs['domain_id'] = adata3.obs['domain_id'].astype('category')
        
        clean_adata(adata3)
        
        return adata3
class LoadCytAssistPipeline(Pipeline):

    def __init__(self, 
            tissue: str, h5_file: str, 
            sample_id: int, name: str,
            geneset: str = None, 
            celltypes: list = None):
        super().__init__()
        
        self.tissue = tissue
        self.celltypes = celltypes
        self.h5_file = h5_file
        self.sample_id = sample_id
        self.name = name
        if geneset is not None:
            # self.geneset = geneset
            with open(geneset, 'r') as f:
                self.geneset = [g.strip() for g in f.readlines()]
        else:
            self.geneset = None
            
        # print('Created CytAssist data loader pipeline.')
        
        
    def load_data(self, h5_file: str):
        adata = sc.read_10x_h5(h5_file, gex_only=False)
        visium_ = sc.read_visium(path=os.path.dirname(h5_file))
        adata.uns['spatial'] = visium_.uns['spatial']  
        adata.obsm['spatial'] = visium_.obsm['spatial']
        adata.obsm['spatial'] = adata.obsm['spatial'].astype(float)        
        adata.var["mt"] = adata.var_names.str.startswith("MT-")
        sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True)
        adata = adata[adata.obs["pct_counts_mt"] < 20]
        adata = adata[:, adata.var["mt"] == False]
        pdata = adata[:, adata.var.feature_types=='Antibody Capture']
        adata = adata[:, adata.var.feature_types=='Gene Expression']
        pdata.var["isotype_control"] = (pdata.var_names.str.startswith("mouse") \
            | pdata.var_names.str.startswith("rat") \
            | pdata.var_names.str.startswith("HLA"))
        pdata = pdata[:, pdata.var.isotype_control==False]
        adata.layers['counts'] = adata.X
        pdata.layers['counts'] = pdata.X
        adata.var_names_make_unique()
        if self.geneset is not None:        
            adata = adata[:, adata.var_names.isin(self.geneset)]
        adata.obsm['spatial'] = adata.obsm['spatial'].astype(float)
        adata.obs['source'] = self.name
        adata.obs['domain_id'] = self.sample_id
        adata.obs['source'] = adata.obs['source'].astype('category')
        adata.obs['domain_id'] = adata.obs['domain_id'].astype('category')
        if self.celltypes is not None:
            adata.obs['celltypes'] = self.celltypes  
            
        if 'PTPRC' in pdata.var_names:
            pdata.var_names = UNIQUE_VAR_NAMES
        
        pdata = pdata[:, pdata.var_names.isin(UNIQUE_VAR_NAMES)]
        
        clean_adata(adata) #clean data
        clean_adata(pdata) #clean data
        
        return adata, pdata
        
    def run(self):
        return self.load_data(self.h5_file)
        
    def __call__(self):
        return self.run()
    
    
class BulkCytAssistLoaderPipeline(Pipeline):
    def __init__(self, tissue: str, data_dir: str, geneset: str = None):
        self.tissue = tissue
        self.data_dir = data_dir
        self.geneset = geneset
        self.fnames = sorted(os.listdir(data_dir))
        self.loaders = []
        for i, fname in enumerate(self.fnames):
            self.loaders.append(
                LoadCytAssistPipeline(
                    tissue=self.tissue, 
                    h5_file=data_dir+fname+'/outs/filtered_feature_bc_matrix.h5',
                    geneset=self.geneset,
                    sample_id = 200+int(i),
                    name = f'{self.tissue} Sample {i}',
                )   
            )
            
        print(f'Created new Bulk CytAssist data loader pipeline with {len(self.fnames)} samples.')
    
    def run(self, i):
        return self.loaders[i].run()
    
    def plot(self):
        for i in range(len(self.fnames)):
            adata, pdata = self.run(i)
            sc.pl.spatial(adata, title=f'{self.tissue} Sample {i}')
