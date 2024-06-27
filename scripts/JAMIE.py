sys.path.append('./JAMIE') ##https://github.com/daifengwanglab/JAMIE.git
from jamie import JAMIE

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