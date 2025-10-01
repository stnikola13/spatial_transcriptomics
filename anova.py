import numpy as np
import scipy.sparse as sp
from sklearn.cluster import KMeans
from tqdm import tqdm
from statsmodels.stats.multitest import multipletests
from data import load_and_preprocess_data
import constants

def make_cell_region_matrix(cell_region_labels, num_regions):
    """
    Creates a cell region matrix based of a list of cell region labels (which cell belongs to
    which region). The cell m rows, and n columns, where m is the number of cells and n is
    the number of regions. A cell region_matrix[x][y] contains a 1 if cell x belongs to region y,
    and 0 otherwise.

    :param cell_region_labels: List of regions to which the cells belong to.
    :param num_regions: The number of different cell regions.
    :return: Cell region matrix.
    """

    num_cells = len(cell_region_labels)
    cell_region_matrix = np.zeros((num_cells, num_regions), dtype=np.float32)
    cell_region_matrix[np.arange(num_cells), cell_region_labels] = 1.0

    return cell_region_matrix

def calculate_anova_parameters(region_sums, region_sums_squared, cells_per_region, sum_all, sum_squared_all, num_cells):
    """
    Calculates F and eta squared parameters using ANOVA statistical analysis.

    :param region_sums: Sum of gene expression for current region (array).
    :param region_sums_squared: Squared sum of gene expression for current region (array).
    :param cells_per_region: Number of cells per region.
    :param sum_all: Sum of all gene expression for all regions (matrix).
    :param sum_squared_all: Squared sum of all gene expression for all regions (matrix).
    :param num_cells: Total number of cells.
    :return: F and eta squared parameters.
    """

    # Calculates the variation in region and total variation.
    region_variation = (region_sums_squared - (region_sums ** 2) / cells_per_region[None, :]).sum(axis=1)
    total_variation = sum_squared_all - (sum_all ** 2) / num_cells
    between_variation = total_variation - region_variation

    # Calculates ANOVA degrees of freedom.
    num_groups = cells_per_region.size
    df_between_groups = max(num_groups - 1, 1)
    df_within_groups  = max(num_cells - num_groups, 1)

    # Calculates the variance mean.
    mean_between = between_variation / df_between_groups
    mean_within  = region_variation  / df_within_groups

    # Calculates the ANOVA F and eta squared parameters. If any unpredictable values occur, sets them to 0.
    F = np.where(mean_within > 0, mean_between / mean_within, 0.0)
    F[~np.isfinite(F)] = 0.0
    eta2 = np.where(total_variation > 0, between_variation / total_variation, 0.0)
    eta2[~np.isfinite(eta2)] = 0.0

    return F.astype(np.float32), eta2.astype(np.float32)


def anova_algorithm(adata, num_regions=None, num_permutations=200, alpha=0.05, method="fdr_bh", seed=42,
                    min_eta_squared=0.0, min_cells_per_region=3):
    """
    Determines which genes in the adata object are SVG, and which aren't using KMeans clusterization and
    ANOVA statistical analysis.

    :param adata: An annotated data object.
    :param num_regions: Number of different cell regions for the KMeans clusterization.
    :param num_permutations: Number of permutations for the calculation of p-values.
    :param alpha: Parameter alpha - maximum value of the q-value for a gene to be considered SVG.
    :param method: Multiple tests method for calculation of q-values.
    :param seed: Seed for the random number generator.
    :param min_eta_squared: Minimum value of eta squared for a gene to be considered SVG.
    :param min_cells_per_region: Minimum number of cells per region.
    :return: A dictionary of pairs: gene_name:is_svg.
    """

    # A random number generator is used with a set seed to allow reproducibility.
    rng = np.random.default_rng(seed)

    # Relevant data is extracted from the adata object.
    coordinates = np.asarray(adata.obsm["spatial"])
    num_cells = adata.n_obs
    num_genes = adata.n_vars
    genes = adata.var_names.to_numpy()
    expression_matrix = adata.X

    # If the number of regions is not parameterized, it is calculated as sqrt(N)/2, where N is the number of cells.
    # This is a commonly used formula for this technique. The number cannot be smaller than 4 regions.
    if num_regions is None:
        num_regions = max(4, int(np.ceil(np.sqrt(num_cells) / 2)))

    # Extracts the genes' expression throughout the cells (column of X matrix).
    if sp.issparse(expression_matrix):
        expression_matrix = expression_matrix.toarray()
    else:
        expression_matrix = np.asarray(expression_matrix)

    # The cells are classified and labeled into num_regions clusters using the KMeans algorithm.
    kmeans_regions = KMeans(n_clusters=num_regions, n_init=20, random_state=seed)
    cell_region_labels = kmeans_regions.fit_predict(coordinates)

    # Regions which have below the minimum number of cells are ignored.
    # Those regions and cells which belong to them are removed.
    region_ids, cell_count = np.unique(cell_region_labels, return_counts=True)
    filtered_regions = region_ids[cell_count >= int(min_cells_per_region)]
    cell_filtering_mask = np.isin(cell_region_labels, filtered_regions)
    cell_region_labels = cell_region_labels[cell_filtering_mask]
    expression_matrix = expression_matrix[cell_filtering_mask, :]
    new_region_ids = {old_id: new_id for new_id, old_id in enumerate(filtered_regions)}
    cell_region_labels = np.array([new_region_ids[l] for l in cell_region_labels], dtype=np.int32)

    # The number of cells and regions, and cell count for each group are updated to match the mentioned filtering.
    num_cells = expression_matrix.shape[0]
    num_regions = filtered_regions.size
    cells_per_group = np.bincount(cell_region_labels, minlength=num_regions).astype(np.float32)

    # A cell region matrix is made which indicated which cell belongs to which region.
    cell_region_matrix = make_cell_region_matrix(cell_region_labels, num_regions)

    # Calculates the expression sum (and squared sum) for each gene (globally).
    sum_all = expression_matrix.sum(axis=0, dtype=np.float32)
    sum_squared_all = (expression_matrix * expression_matrix).sum(axis=0, dtype=np.float32)

    # Calculates the gene expression sums and squared sums by multiplying the expression matrix
    # (has to be transposed because of its format in the adata) with the cell region matrix.
    sums = expression_matrix.T @ cell_region_matrix
    sums_squared = (expression_matrix * expression_matrix).T @ cell_region_matrix

    # Calculates the F and eta squared ANOVA parameters.
    F, eta_squared = calculate_anova_parameters(sums, sums_squared, cells_per_group, sum_all, sum_squared_all,
                                                num_cells)

    # F structureless is a matrix of F parameters. Each row is a list of F parameters for each gene for
    # one permutation of the regions.
    F_structureless = np.empty((num_permutations, num_genes), dtype=np.float32)
    index_array = np.arange(num_cells)

    # The regions are permutated num_permutations times and the F parameters for all genes are saved after
    # each permutation. Permutations are done using the index_array.
    # The F parameters are calculated in a similar manner as the original calculation.
    for i in tqdm(range(num_permutations)):
        perm = rng.permutation(index_array)
        labels_permutated = cell_region_labels[perm]
        cell_region_matrix_permutated = make_cell_region_matrix(labels_permutated, num_regions)

        sums_permutated = expression_matrix.T @ cell_region_matrix_permutated
        sums_squared_permutated = (expression_matrix * expression_matrix).T @ cell_region_matrix_permutated

        F_i, _ = calculate_anova_parameters(sums_permutated, sums_squared_permutated, cells_per_group, sum_all,
                                            sum_squared_all, num_cells)
        F_structureless[i] = F_i

    # Calculates p-values and q-values.
    larger_perm_Fs = (F_structureless >= F[None, :]).sum(axis=0).astype(np.int32)
    p_values = (larger_perm_Fs + 1.0) / (num_permutations + 1.0)
    _, q_values, _, _ = multipletests(p_values, alpha=alpha, method=method)

    # Gene is SVG if its q-value is smaller than the parameter alpha and if its variance is larger than min_eta_squared.
    is_svg = (q_values < alpha) & (eta_squared >= float(min_eta_squared))
    is_svg = is_svg.tolist()

    # A dictionary of pairs: gene_name:is_svg is created and returned.
    gene_list = genes.tolist()
    dictionary = dict()
    for i, gene in enumerate(gene_list):
        dictionary[gene] = is_svg[i]

    return dictionary

if __name__ == "__main__":
    # Loads the adata objects.
    adata_embryo = load_and_preprocess_data(constants.MOUSE_EMBRYO_DATA_PATH)
    adata_brain = load_and_preprocess_data(constants.MOUSE_BRAIN_DATA_PATH)
    adata_objects = [adata_embryo, adata_brain]
    dataset_names = ["mouse_embryo", "mouse_brain"]

    # Processing is performed for each of the adata objects.
    for i, adata in enumerate(adata_objects):
        data_type = dataset_names[i]
        print(f"Running Anova algorithm for {constants.display_names.get(data_type)} data.")

        result = anova_algorithm(adata, num_permutations=300, alpha=0.05, method="fdr_bh", min_eta_squared=0.035, min_cells_per_region=3)

        # Writes results in a .csv file in pairs: gene name and if it's SVG or not (True/False).
        with open(f"{constants.result_directory}/anova_results_{data_type}.csv", "w") as f:
            for gene, is_svg in result.items():
                f.write(f"{gene},{is_svg}\n")

        print(f"Finished Anova algorithm for {constants.display_names.get(data_type)} data.")