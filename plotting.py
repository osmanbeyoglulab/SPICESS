try:
    import cuml
except: ## No GPU?
    import umap.umap_ as cuml
    
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA
from scipy.stats import spearmanr
import pandas as pd
import squidpy as sq

np.random.seed(1)

ref_colors = [
    "#4a3e38",
    "#9fbfa2",
    "#b99bbd",
    "#53366c",
    "#b2834b",
    "#00612e",
    "#00acef",
    "#00eee9",
    "#ba4850",
    "#cac84d",
    "#ce52ad",
    "#76d05c",
    "#794bc8",
    "#e85a00"
]

def plot_recons(integrated_data):
    gex_recons = integrated_data.gex_recons.data.cpu().numpy()
    gex_input = integrated_data.gex_input.data.cpu().numpy()
    pex_recons = integrated_data.pex_recons.data.cpu().numpy()
    pex_input = integrated_data.pex_input.data.cpu().numpy()

    f, axs = plt.subplots(1, 2, figsize=(12, 3), dpi=140)

    corrs = []
    for ixs in range(gex_input.shape[1]):
        corrs.append(spearmanr(gex_input[:, ixs], gex_recons[:, ixs]).statistic)  
    sns.histplot(corrs, color='red', alpha=0.5, label='TrainedModel', ax=axs[0])
    axs[0].set_title(f'Gene Reconstruction\nMean:{np.mean(corrs):.3f}')
    axs[0].set_xlabel('Spearman Correlation')
    plt.xlim(0, 1)

    corrs = []
    for ixs in range(pex_input.shape[1]):
        corrs.append(spearmanr(pex_input[:, ixs], pex_recons[:, ixs]).statistic)   
    corrs = np.array(corrs)
    sns.histplot(corrs, color='green', alpha=0.5, label='TrainedModel', ax=axs[1])
    axs[1].set_title(f'Protein Reconstruction\nMean:{np.mean(corrs):.3f}')
    axs[1].set_xlabel('Spearman Correlation')
    plt.xlim(0, 1)
    plt.show()

def plot_norm(encoded, labels):
    
    z = np.random.multivariate_normal([0]*2, np.eye(2), len(labels))
    xy = PCA(2).fit_transform(encoded)
    plt.rcParams['figure.figsize'] = [6, 6]

    sns.scatterplot(x=z[:, 0], y=z[:, 1], color='grey', alpha=0.2, s=70)
    sns.scatterplot(x=xy[:, 0], y=xy[:, 1], hue=labels, edgecolor='black', s=65, legend=False)
    sns.despine(left=True, bottom=True, trim=True, offset=10)
    plt.xlabel('PCA-1')
    plt.ylabel('PCA-2')
    plt.show()

def plot_umap_grid(
    emb, 
    imputed_proteins, 
    protein_names, 
    cmap='winter', 
    size=40, 
    save=None, 
    fmt='svg', 
    desc='antibody_panel.csv',
    colors=None):
    
    antibody_panel = pd.read_csv(desc)
    
    plt.rcParams['figure.figsize'] = (12, 12)
    
    if colors is None:
        colors = ref_colors

    f, axx = plt.subplots(5, 6, figsize=(37, 16), dpi=120, sharex=True, sharey=True)
    axs = axx.flatten()

    norm = plt.Normalize(0, int(imputed_proteins.max()))
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    f.colorbar(sm, ax=axx[:, :], shrink=0.55, location='right')

    for ix, ax in enumerate(axs):
            
        sns.scatterplot(x=emb[:, 0], y=emb[:, 1], 
            hue=imputed_proteins[:, ix], edgecolor='black', s=size, ax=ax, palette=cmap)
        
        try:
            _, name, desc = antibody_panel[antibody_panel.name==protein_names[ix].replace('-A', '')].values[0]
        except Exception as e:
            print(protein_names[ix])
            raise e
        
        ax.set_title(f'{name}\n({desc})')
        ax.get_legend().remove()
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])
        

    

    if save is not None:
        plt.savefig(save, format=fmt, dpi=180)
        
    plt.show()
    
def plot_latent(
    data,
    labels,
    names=['Gene\nEmbeddings', 'Protein\nEmbeddings'],
    legend=False,
    edgecolor='black',
    remove_outliers=False,
    n_components=2,
    separate_dim=False,
    square=False,
    method='umap',
    n_neighbors=None,
    seed=42,
    reduce_only=False,
    save=None,
    colors=None,
    size=100,
    fmt='svg',
    learning_rate=1.0,
    min_dist=0.1,
    spread=1.0,
    local_connectivity=1.0,
    linewidth = 1.0,
    repulsion_strength=1.0,
):
    method_names = {'pca': 'PC', 'umap': 'UMAP'}
    axs = []
    datax = []
    
    if colors is None:
        colors = ref_colors
    
    plt.rcParams['figure.figsize'] = (16, 8)
    
    for i, (dat, lab) in enumerate(zip(data, labels)):
        ax = plt.gcf().add_subplot(1, len(data), i+1, projection=None)
        axs.append(ax)
        if i == 0 or separate_dim:
            red = cuml.UMAP(
                n_components=n_components,
                n_neighbors=min(200, dat.shape[0] - 1) if n_neighbors is None else n_neighbors,
                learning_rate=learning_rate,
                min_dist=min_dist,
                spread=spread, 
                local_connectivity=local_connectivity,
                repulsion_strength=repulsion_strength,
                random_state=seed)
            if separate_dim:
                red.fit(dat)
            else:
                red.fit(np.concatenate(data, axis=0))
        plot_data = red.transform(dat)
        datax.append(plot_data)
        
        unique_labels = np.unique(np.concatenate(labels))
        for ix, l in enumerate(unique_labels):
            data_subset = np.transpose(plot_data[lab == l])

            scatter = ax.scatter(*data_subset, label=l, edgecolor=edgecolor, color=colors[ix], s=size, linewidth=linewidth)
        fig = plt.gcf()
        # if i == 1 and legend:
        #     fig.legend(scatter, labels=unique_labels, bbox_to_anchor=(0.5, -0.05), loc='lower center', ncols=7)
        if names is not None:
            ax.set_title(names[i])
        ax.set_xlabel(f'{method_names[method]}-1')
        ax.set_ylabel(f'{method_names[method]}-2')
        if n_components == 2 and square:
            ax.set_aspect('equal')
                    
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        
        ax.set_facecolor('black')
        
    if not separate_dim:
        axs_xlim = np.array([ax.get_xlim() for ax in axs])
        axs_ylim = np.array([ax.get_ylim() for ax in axs])
        new_xlim = (axs_xlim.min(axis=0)[0], axs_xlim.max(axis=0)[1])
        new_ylim = (axs_ylim.min(axis=0)[0], axs_ylim.max(axis=0)[1])
        for ax in axs:
            ax.set_xlim(new_xlim)
            ax.set_ylim(new_ylim)
    
    if i == 1 and legend:
        fig.legend(scatter, labels=unique_labels, bbox_to_anchor=(0.5, -0.05), loc='lower center', ncols=7)

    plt.tight_layout()
    fig.subplots_adjust(bottom=0.2)  
    
    if save is not None:
        plt.savefig(save, format=fmt, dpi=180, bbox_inches='tight')
    
    if reduce_only:
        plt.close()
    else:
        plt.show()
        
    return datax


def plot_feature_maps(adata, pdata, protein_names, titles):
    # adts = ['CD68', 'PAX5', 'CD8A', 'CD19', 'CXCR5', 'MS4A1', 'CD3E', 'ITGAM']
    adts = protein_names
    figsize = (32, 8)
    dpi = 180
    # titles = ['CD68\n(Macrophages)', 'PAX5\n(B Cells)', 'CD8A\n(Killer T Cells)', 'CD19\n(B Cells)', 'CD4\n(T Helper Cells)', 
    # 'CD20\n(B Cells)', 'CD3E\n(Regulatory T Cells)', 'CD11b\n(Neutrophils)']

    f, ax = plt.subplots(2, 8, figsize=figsize, dpi=dpi)
    ax = ax.flatten()

    sq.pl.spatial_scatter(adata, color=adts, crop_coord=tuple([0, 0, 28000, 40000]), dpi=dpi, size=1.6, 
                        ncols=len(protein_names), frameon=False, figsize=figsize, edgecolor='black', linewidth=0.15, cmap='rainbow', colorbar=False,
                        title='', ax=ax[:8], fig=f, return_ax=True)

    sq.pl.spatial_scatter(pdata, color=adts, crop_coord=tuple([0, 0, 28000, 40000]), dpi=dpi, size=1.6, 
                        ncols=len(protein_names), frameon=False, figsize=figsize, edgecolor='black', linewidth=0.15, cmap='rainbow', colorbar=False,
                        title=titles, ax=ax[8:], fig=f, return_ax=True)

    plt.tight_layout()
    plt.show()
