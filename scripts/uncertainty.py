from captum.attr import IntegratedGradients
from captum.attr import DeepLift
from captum.attr import KernelShap, GradientShap


class ProteinImputer(nn.Module):
    def __init__(self, model, adj, indices):
        super().__init__()
        self.model = model
        self.adj = adj
        self.indices = indices
        
    def forward(self, X):
        edge_index = self.adj.nonzero().t().contiguous()
        pos_z, _, _ = self.model.encoders[0](X, edge_index)
        z = self.model.fc_mus[0](pos_z)
        
        r = self.model.gene_projector(X[:, self.indices])
        decoded = self.model.decoders[1](z+r)
        
        # return decoded.mean(0)
        return decoded
    
    
adata, pdata = train_loader.run() 
adata_eval, pdata_eval = eval_loader.run()

sc.pp.filter_genes(adata, min_cells=5, inplace=True)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, n_top_genes=1600)
adata = adata[:, (adata.var['highly_variable'] | adata.var_names.isin(['PTPRC']+list(pdata.var_names)))]
adata.X = adata.layers['counts']

adata_eval = align_adata(adata_eval, adata)


dropout_predictions = np.array([sp.model.impute(
    X=spicess_2.iter_0.d13.cuda(), 
    adj=sp.A.cuda(), 
    indices=spicess_2.iter_0.model.train_idx,
    enable_dropout=True
).T for _ in range(10)])

# Calculating mean across multiple MCD forward passes 
mean = np.mean(dropout_predictions, axis=0)  # shape (n_samples, n_classes)

# Calculating variance across multiple MCD forward passes 
variance = np.var(dropout_predictions, axis=0)  # shape (n_samples, n_classes)

epsilon = sys.float_info.min
# Calculating entropy across multiple MCD forward passes 
entropy = -np.sum(mean * np.log(mean + epsilon), axis=-1)  # shape (n_samples,)

# Calculating mutual information across multiple MCD forward passes 
mutual_info = entropy - np.mean(np.sum(-dropout_predictions * np.log(dropout_predictions + epsilon),
                                       axis=-1), axis=0)  # shape (n_samples,)


plt.rcParams['lines.markersize'] = 15

f, axs = plt.subplots(1, 3, figsize=(20, 8), dpi=120)

sns.scatterplot(x=tons_corr.reshape(-1), 
                y=mutual_info.reshape(-1)/mutual_info.sum(), ax=axs[0],  
                edgecolor='black', color='salmon')

sns.scatterplot(x=tons_corr.reshape(-1), 
                y=entropy.reshape(-1)/entropy.sum(), ax=axs[1], 
                edgecolor='black', color='skyblue')

sns.scatterplot(x=mutual_info.reshape(-1)/mutual_info.sum(), 
                y=entropy.reshape(-1)/entropy.sum(), ax=axs[2],
                edgecolor='black', color='limegreen')

for i, label in enumerate(pdata.var_names):
    axs[0].annotate(label, (tons_corr.reshape(-1)[i], (mutual_info.reshape(-1)/mutual_info.sum())[i]))

for i, label in enumerate(pdata.var_names):
    axs[1].annotate(label, (tons_corr.reshape(-1)[i], (entropy.reshape(-1)/entropy.sum())[i]))

for i, label in enumerate(pdata.var_names):
    axs[2].annotate(label, ((mutual_info.reshape(-1)/mutual_info.sum())[i], (entropy.reshape(-1)/entropy.sum())[i]))

axs[0].set_title('Confidence', fontsize=15)
axs[0].set_xlabel('Spearman Correlation\nTrained: Tonsil #1 | Tested: Tonsil #2', fontsize=12)
axs[0].set_ylabel('Normalized Mutual Information', fontsize=12)

axs[1].set_title('Uncertainty', fontsize=15)
axs[1].set_xlabel('Spearman Correlation\nTrained: Tonsil #1 | Tested: Tonsil #2', fontsize=12)
axs[1].set_ylabel('Normalized Entropy', fontsize=12)

axs[2].set_title('Uncertainty-Confidence', fontsize=15)
axs[2].set_xlabel('Normalized Mutual Information', fontsize=12)
axs[2].set_ylabel('Normalized Entropy', fontsize=12)

plt.tight_layout()
plt.savefig('uncertain.svg', format='svg', dpi=120)
plt.show()