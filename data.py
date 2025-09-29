import scanpy as sc

def load_and_normalize_data(path):
    adata = sc.read_h5ad(path)
    adata.var_names_make_unique()

    sc.pp.filter_genes(adata, min_cells=10)
    sc.pp.normalize_total(adata, inplace=True)
    sc.pp.log1p(adata)

    return adata