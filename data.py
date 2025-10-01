import scanpy as sc

def load_and_preprocess_data(path):
    """
    Reads the h5ad into an adata object and performs some preprocessing. Filters genes, so only those
    that are present in a minimum od 10 cells are relevant. Then, normalizes the data using logarithms.

    :param path: Path to the h5ad file containing the data.
    :return: Annotated data object (adata).
    """
    adata = sc.read_h5ad(path)
    adata.var_names_make_unique()

    sc.pp.filter_genes(adata, min_cells=10)
    sc.pp.normalize_total(adata, inplace=True)
    sc.pp.log1p(adata)

    return adata