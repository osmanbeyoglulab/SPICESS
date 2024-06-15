import streamlit as st
import sys
import warnings

warnings.filterwarnings('ignore')
sys.path.append('.')
sys.path.append('./spicess')
import numpy as np
from spicess.vae_infomax import InfoMaxVAE
import uniport as up
from utils import featurize, preprocess_graph, graph_alpha
from spicess.vae_infomax import InfoMaxVAE
from sklearn.preprocessing import MinMaxScaler
from streamlit_image_comparison import image_comparison
import scipy.sparse as sp

import pandas as pd
import scanpy as sc
from anndata import AnnData
import matplotlib.pyplot as plt
import seaborn as sns
from utils import graph_alpha
import torch
# from sklearn.decomposition import PCA
# from sklearn.manifold import TSNE


# UMAP = cuml.UMAP

st.set_page_config(layout="centered", page_icon='🌶️', page_title='SPICESS Web Portal')
st.set_option('deprecation.showPyplotGlobalUse', False)

st.title('Welcome to SPICESS!')
st.image('model.png', use_column_width=True)


antibody_panel = pd.read_csv('./notebooks/antibody_panel.csv')

with st.expander('Antibody Panel'):
    st.table(antibody_panel)
    
@st.cache_data
def load_data(tissue):
    adata = sc.read_10x_h5(f'./data/{tissue}Tissue/GEX_PEX/filtered_feature_bc_matrix.h5', gex_only=False)
    visium_ = sc.read_visium(path=f'./data/{tissue}Tissue/GEX_PEX/')
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
    
    return adata, pdata
        
# @st.cache_data
def project(ix, adata_ref, tissue):
    adata3 = sc.read_h5ad(f'./data/{tissue}Tissue/Kumar_et_al_2023/patient_{ix}.h5ad')
    adata3.obs['source'] = f'Breast Sample {ix}'
    adata3.obs['domain_id'] = ix
    adata3.obs['source'] = adata3.obs['source'].astype('category')
    adata3.obs['domain_id'] = adata3.obs['domain_id'].astype('category')
    
    # sc.pp.normalize_total(adata3)
    # sc.pp.log1p(adata3)
    # up.batch_scale(adata3)
    
    adata3.var_names = adata3.var.feature_name.values
    
    st.success(f'Sample {ix} has {adata3.shape[0]} spots and {adata3.shape[1]} genes.')
    
    # adata_cm = AnnData.concatenate(adata3, adata_ref, join='inner')
    # adata_new = up.Run(
    #     name=tissue, 
    #     adatas=[adata3, adata_ref], 
    #     adata_cm =adata_cm, 
    #     lambda_s=1.0, 
    #     out='project', 
    #     ref_id=1, 
    #     outdir='./notebooks/output'
    # )
    # adata3.obsm['latent'] = adata_new[adata_new.obs.domain_id==ix].obsm['project']
    # adata3.obsm['project'] = adata_new[adata_new.obs.domain_id==ix].obsm['project']
    
    adata3.obsm['latent'] = adata3.obsm['project'] = np.load(f'./data/{tissue}Tissue/preprocessed/projected_{ix}.npy')
    
    return adata3

@st.cache_data
def build_scaffold(tissue):
    
    with open(f'./notebooks/{tissue}.txt', 'r') as f:
        genes = [g.strip() for g in f.readlines()]
    
    adata, pdata = load_data(tissue)
    adata = adata[:, adata.var_names.isin(genes)]
    adata.obsm['spatial'] = adata.obsm['spatial'].astype(float)
    adata.obs['source'] = 'Breast CytAssist (Ref)'
    adata.obs['domain_id'] = 0
    adata.obs['source'] = adata.obs['source'].astype('category')
    adata.obs['domain_id'] = adata.obs['domain_id'].astype('category')

    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    
    return adata, pdata



cols = st.columns(3)
tissue = st.radio('Select Tissue', ['Breast', 'Tonsil', 'Brain'], horizontal=True)


def clean_adata(adata):
    for v in ['mt', 'n_cells_by_counts', 'mean_counts', 'log1p_mean_counts', 'pct_dropout_by_counts', 'total_counts', 'log1p_total_counts', 
        'gene_ids', 'feature_types', 'genome', 'mt-0', 'n_cells_by_counts-0',
        'mean_counts-0', 'log1p_mean_counts-0', 'pct_dropout_by_counts-0',
        'total_counts-0', 'log1p_total_counts-0', 'gene_ids-0', 'feature_types-0', 'genome-0', 'feature_is_filtered-1', 
        'feature_name-1', 'feature_reference-1', 'feature_biotype-1', 'highly_variable', 'means', 'dispersions', 'dispersions_norm']:
        if v in adata.var.columns:
            del adata.var[v]
        
    for o in ['log1p_total_counts', 'mapped_reference_assembly', 'mapped_reference_annotation', 
        'alignment_software', 'donor_id', 'self_reported_ethnicity_ontology_term_id', 'donor_living_at_sample_collection', 
        'donor_menopausal_status', 'organism_ontology_term_id', 'sample_uuid', 'sample_preservation_method', 'tissue_ontology_term_id', 
        'development_stage_ontology_term_id', 'sample_derivation_process', 'sample_source', 'donor_BMI_at_collection', 
        'tissue_section_uuid', 'tissue_section_thickness', 'library_uuid', 'assay_ontology_term_id', 'sequencing_platform', 
        'is_primary_data', 'cell_type_ontology_term_id', 'disease_ontology_term_id', 'sex_ontology_term_id', 
        'nCount_Spatial', 'nFeature_Spatial', 'nCount_SCT', 'nFeature_SCT', 
        'suspension_type', 'tissue', 
        'self_reported_ethnicity', 'development_stage', 'n_genes_by_counts', 'log1p_n_genes_by_counts', 'total_counts',
        'pct_counts_in_top_50_genes', 'pct_counts_in_top_100_genes',
        'pct_counts_in_top_200_genes', 'pct_counts_in_top_500_genes',
        'total_counts_mt', 'log1p_total_counts_mt', 'pct_counts_mt']:
        if o in adata.obs.columns:
            del adata.obs[o]
            


with st.spinner('Loading Reference Scaffold...'):
    adata, pdata = build_scaffold(tissue)
    pdata.obsm['spatial'] = adata.obsm['spatial']
    pdata.raw = pdata
    pdata.X = pdata.X.astype(float)
    columns = list(pdata.to_df().columns)
    columns[columns.index('PTPRC')] = 'PTPRC-A'
    pdata.var_names = columns   

with st.spinner('Integrating reference...'):
    adata_cyt = up.Run(name=tissue, 
        adatas=[adata], 
        adata_cm =adata, 
        lambda_s=1.0, 
        out='project', 
        ref_id=1, 
        outdir='./notebooks/output'
    )

    adata_cyt.obsm['latent'] = adata_cyt.obsm['project']
    adata = adata_cyt[adata_cyt.obs.source=='Breast CytAssist (Ref)']
    clean_adata(adata)
    
    
    st.caption('CytAssist Reference AnnData:')
    st.code(adata)


# with st.spinner('Featurizing...'):
#     gex = featurize(adata)
#     pex = featurize(pdata, clr=True)
#     d11 = adata.obsm['latent'].copy()
#     d12 = pex.features
    
d11 = adata.obsm['latent']

st.title(f'{tissue} (Human)')
st.info(f'🧬 Reference sample has {adata.obsm["latent"].shape[0]} spots, {adata.shape[1]} genes, and {pdata.shape[1]} proteins.')

with st.spinner(f'Loading pre-trained model for...'):
    model = InfoMaxVAE([d11.shape[1], pdata.shape[1]], latent_dim=16, dropout=0.1)
    model.load_state_dict(torch.load(f'./notebooks/{tissue}_model.pth', map_location=torch.device('cpu')))

st.caption('Upload data to analyze or use available examples.')
upload, examples  = st.tabs(["Upload", "Examples"])

with upload:
    st.file_uploader('Upload your own ST data', type='h5ad', help='ST AnnData objects only.')
            

# with perf:
#     st.image(f'{tissue}_predictions.png', use_column_width=True)
    
#     with st.expander('Show embeddings'):
#         st.caption('Embeddings alignment results during training')
#         st.image(f'{tissue}_alignment.png', use_column_width=True)
        
#     with st.expander('View Model Architechture'):
#         st.code(model)
        
        


import squidpy as sq

sample_shapes = [
    (1, (2103, 36503)),
    (2, (1400, 36503)),
    (3, (2364, 36503)),
    (4, (2504, 36503)),
    (5, (2694, 36503)),
    (6, (3037, 36503)),
    (7, (2086, 36503)),
    (8, (2801, 36503)),
    (9, (2694, 36503)),
    (10, (2473, 36503))
]

# @st.cache_data
def get_imputations(patient_id, tissue):     
    a = project(
            patient_id, 
            adata_ref=adata, 
            tissue=tissue
        ) 
    st.success('Projection complete.')
            
    clean_adata(a)

    
    adj = graph_alpha(a.obsm['spatial'], n_neighbors=12)
    adj_label = adj + sp.eye(adj.shape[0])
    adj_label = torch.tensor(adj_label.toarray())
    adj_norm = preprocess_graph(adj)
    
    A = adj_norm.to_dense()
    d11a = torch.tensor(a.obsm['latent']).float()
    imputed_proteins, z = model.impute(d11a, A, return_z=True)                
    proteins_df = pd.DataFrame(imputed_proteins, index=a.obs.index, columns=pdata.var_names)
    
    return proteins_df, a, z

with examples:
    st.header('Built-in examples')

    cols = st.columns(5)
    sample_ids = \
        [cols[0].checkbox(f'Sample {x} (n={sample_shapes[x-1][1][0]})') for x in range(1, 3)] +\
        [cols[1].checkbox(f'Sample {x} (n={sample_shapes[x-1][1][0]})') for x in range(3, 5)] +\
        [cols[2].checkbox(f'Sample {x} (n={sample_shapes[x-1][1][0]})') for x in range(5, 7)] +\
        [cols[3].checkbox(f'Sample {x} (n={sample_shapes[x-1][1][0]})') for x in range(7, 9)] +\
        [cols[4].checkbox(f'Sample {x} (n={sample_shapes[x-1][1][0]})') for x in range(9, 11)]
            
        
    sample_ids = [x+1 for x, y in enumerate(sample_ids) if y]

    st.divider()

    a, b, c = st.columns(3)
    resolution = a.slider('Clustering resolution', 0.1, 1.0, step=0.05, value=0.5)
    alpha = b.slider('Transparency', 0.1, 1.0, step=0.1, value=0.9)
    size = c.slider('Spot Size', 0.1, 5.0, step=0.1, value=1.5)
        
    ix = 1
    a, b, c = st.columns(3)
    image_type = a.selectbox('Left', list(pdata.var_names)+['Proteins', 'cell_type', 'author_cell_type', None], key=f'{ix}_image_1', index=33)  
    image_type2 = b.selectbox('Right', list(pdata.var_names)+['Proteins', 'cell_type', 'author_cell_type', None], key=f'{ix}_image_2', index=31)  
    
    ct = c.selectbox('Markers', antibody_panel.cell_type.unique())
    
    st.code(f'{ct}: {antibody_panel[antibody_panel.cell_type==ct].name.values}')
    
    for ix in sample_ids:            
        
        proteins, sample6, z = get_imputations(ix, tissue)
        
        
        scaler = MinMaxScaler()
        proteins_norm = pd.DataFrame(scaler.fit_transform(proteins), columns=proteins.columns)
        pdata2 = AnnData(proteins_norm)
        pdata2.uns['spatial'] = sample6.uns['spatial']
        pdata2.obsm['spatial'] = sample6.obsm['spatial']
        pdata2.obs['cell_type'] = sample6.obs['cell_type'].values
        pdata2.obs['author_cell_type'] = sample6.obs['author_cell_type'].values
                    
        with st.spinner(f'🎨 Segmenting...'):
            
            sc.pp.neighbors(pdata2)
            sc.tl.leiden(pdata2, resolution=resolution)
            a = AnnData(z)
            sc.pp.neighbors(a)
            sc.tl.leiden(a, resolution=resolution)
            pdata2.obs['latent'] = a.obs.leiden.values
            
            if 'leiden_colors' in pdata2.uns: pdata2.uns.pop('leiden_colors');
            if 'latent_colors' in pdata2.uns: pdata2.uns.pop('latent_colors');
            pdata2.obs['Proteins'] = pdata2.obs['leiden'].astype('category')
            

            sq.pl.spatial_scatter(
                pdata2, color=image_type2, ncols=1, dpi=180, size=size, alpha=alpha,
                save = f'/tmp/patient_{ix}_regions_blank.png', 
                legend_loc=False, colorbar=False,
                figsize=(5, 5), frameon=False, palette='Set1',
                title='')
            
            sq.pl.spatial_scatter(
                pdata2, color=image_type, ncols=1, dpi=180, size=size, alpha=alpha,
                save = f'/tmp/patient_{ix}_regions.png', 
                legend_loc=False,colorbar=False,
                figsize=(5, 5), frameon=False, edgecolor=None, palette='Dark2',
                title='')
            
            
            image_comparison(
                img1=f'/tmp/patient_{ix}_regions.png',
                img2=f'/tmp/patient_{ix}_regions_blank.png',
                label1=image_type or 'Tissue',
                label2=image_type2 or 'Tissue',
                # width=750,
                starting_position=50,
                show_labels=True,
                make_responsive=True,
                in_memory=True,
            )