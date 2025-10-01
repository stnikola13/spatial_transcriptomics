import SpaGFT as spg
import constants
from data import load_and_preprocess_data


def execute_spagft(adata):
    """
    Executes the SpaGFT algorithm to determine which genes are SVG and which are not.

    :param adata: An annotated data object.
    :return: Dictionary of pairs: gene name and boolean value indicating whether the gene is SVG or not.
    """
    ratio_low,ratio_high = spg.gft.determine_frequency_ratio(adata,
                                                             ratio_neighbors=1,
                                                             spatial_info="spatial")

    gene_df = spg.detect_svg(adata,
                             spatial_info="spatial",
                             ratio_low_freq=ratio_low,
                             ratio_high_freq=ratio_high,
                             ratio_neighbors=1,
                             filter_peaks=True,
                             S=6)

    svg_genes = gene_df[gene_df.cutoff_gft_score][gene_df.fdr<0.05].index.tolist()
    all_genes = gene_df.index.tolist()

    # All genes that are in the svg_genes list are SVGs.
    gene_svg_map = dict()
    for gene in all_genes:
        if gene in svg_genes:
            gene_svg_map[gene] = True
        else:
            gene_svg_map[gene] = False

    print("Number of SVG genes: {}".format(len(svg_genes)))
    print("Total number of genes: {}".format(len(all_genes)))

    #plot_svgs = ['Actc1','Kremen1']
    #sc.pl.spatial(adata,color=plot_svgs,size=1.6,spot_size=1)

    return gene_svg_map


if __name__ == "__main__":
    # Loads and preprocesses the data.
    adata_embryo = load_and_preprocess_data(constants.MOUSE_EMBRYO_DATA_PATH)
    adata_brain = load_and_preprocess_data(constants.MOUSE_BRAIN_DATA_PATH)

    adata_list = [adata_embryo, adata_brain]
    adata_names = ["mouse_embryo", "mouse_brain"]

    # Processes data for each dataset.
    for i, adata in enumerate(adata_list):
        data_type = adata_names[i]

        print("Running SpaGFT algorithm for {} data.".format(data_type))
        gene_svg_map = execute_spagft(adata)

        # Writes pairs: gene, SVG boolean into a .csv file.
        with open("./results/spagft_results_{}.csv".format(data_type), "w") as f:
            for key, val in gene_svg_map.items():
                f.write(f"{key},{val}\n")

        print("Finished running SpaGFT algorithm for {} data.\n".format(data_type))