```{python}
trainer = TrainModelPipeline(
    tissue = 'Tonsil',
    adatas = [adata, adata_eval],
    pdatas = [pdata, pdata_eval],
    repeats = 10,
    latent_dim = 512,
    save = False,
    lr = 2e-5,
    dropout=0.1,
    use_histology=True,
    patience = 100,
    epochs = 9000,
    recons_gex = 1e-4, 
    recons_pex = 1e-4, 
    adj = 1e-3,
    align = 1e-4,
    minmax_vertical = True,
    min_max = True,
    gene_log = True,
    protein_log = False
)

spicess = trainer.run()

```
