import numpy as np
import scipy.sparse as sp
from tqdm import tqdm
from skimage.filters import threshold_otsu
from data import load_and_preprocess_data
import constants

def calculate_parameters(adata):
    """
    Extracts relevant parameters from the adata object.
    x - Array of cell x coordinates.
    y - Array of cell y coordinates.
    N - Number of cells (observations).
    G - Number of genes (features).

    :param adata: Annotated data object.
    :return: Tuple of x, y, N, G data.
    """
    coords = np.asarray(adata.obsm["spatial"])
    x, y = coords[:, 0], coords[:, 1]
    N, G = adata.n_obs, adata.n_vars

    return x, y, N, G

def shannon_entropy(x, y, expr, num_bins):
    """
    Computes the gene's entropy by dividing the cell space into a grid (num_bins x num_bins).
    Uses the Shannon entropy formula: -sum(p * log(p)).

    :param x: Array of cell x coordinates.
    :param y: Array of cell y coordinates.
    :param expr: Array of gene expression throughout the cells.
    :param num_bins: Number of bins for grid creation.
    :return: Shannon entropy value.
    """

    # Creates a "bin grid" over the entire space where the cells genes are expressed.
    expression_sum_per_bin, _, _ = np.histogram2d(x, y, bins=(num_bins, num_bins), weights=expr)
    cells_per_bin, _, _ = np.histogram2d(x, y, bins=(num_bins, num_bins))
    mean_expr_per_bin = np.divide(expression_sum_per_bin, cells_per_bin, out=np.zeros_like(expression_sum_per_bin, dtype=np.float32), where=cells_per_bin > 0)

    # Each bin is normalized by calculating probability instead of gene expression count.
    # That is why the total expression of the gene (all bins) is calculated.
    # All genes with 0 probability are removed.
    total_expression = mean_expr_per_bin.sum()
    if total_expression <= 0:
        return 1.0
    spatial_probabilities = mean_expr_per_bin / total_expression
    gene_probabilities = spatial_probabilities.ravel()
    p = gene_probabilities[gene_probabilities > 0]

    # The entropy is calculated using the formula, and then it is normalized to [0, 1] using logarithms.
    entropy = -(p*np.log2(p)).sum()
    max_entropy = np.log2(len(p))
    return float(entropy / max_entropy)

if __name__ == "__main__":
    # Loads the adata objects.
    adata_embryo = load_and_preprocess_data(constants.MOUSE_EMBRYO_DATA_PATH)
    adata_brain = load_and_preprocess_data(constants.MOUSE_BRAIN_DATA_PATH)
    adata_objects = [adata_embryo, adata_brain]
    dataset_names = ["mouse_embryo", "mouse_brain"]

    num_bins = 28
    spatial_bins = (num_bins, num_bins)

    # Processing is performed for each of the adata objects.
    for i, adata in enumerate(adata_objects):
        current_dataset = dataset_names[i]
        x, y, N, G = calculate_parameters(adata)
        expression_matrix = adata.X

        # Initializes a dictionary for gene entropies (length will be equal to the number of genes).
        gene_entropy = dict()
        gene_names = adata.var_names.to_numpy().tolist()

        # Calculation for each gene
        for gene in tqdm(range(G)):
            # Extracts the gene's expression throughout the cells (column of X matrix).
            if sp.issparse(expression_matrix):
                expr = expression_matrix[:, gene].toarray().ravel()
            else:
                expr = np.asarray(expression_matrix[:, gene]).ravel()

            # Calculates the entropy for current gene.
            gene_entropy[gene_names[gene]] = shannon_entropy(x, y, expr, num_bins)

        # Writes results in a .csv file in pairs: gene name and if it's SVG or not (True/False).
        with open(f"{constants.result_directory}/shannon_entropy_results_{current_dataset}.csv", "w") as f:
            entropies = np.asarray(list(gene_entropy.values()))
            for gene, entropy in gene_entropy.items():
                # OTSU threshold is used to determine whether a gene is SVG or not based on its entropy.
                threshold = float(threshold_otsu(entropies))
                looser_threshold = min(1.0, threshold + 0.025)
                is_svg = (entropy <= looser_threshold)
                f.write(f"{gene},{is_svg}\n")