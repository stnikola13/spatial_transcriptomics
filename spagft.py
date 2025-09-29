import scanpy as sc
import SpaGFT as spg

import constants
from data import load_and_normalize_data


def execute_spagft(adata):
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

    gene_svg_map = dict()
    for gene in all_genes:
        if gene in svg_genes:
            gene_svg_map[gene] = True
        else:
            gene_svg_map[gene] = False

    print("SVG genes: {}".format(len(svg_genes)))
    print("Total genes: {}".format(len(all_genes)))

    #plot_svgs = ['Actc1','Kremen1']
    #sc.pl.spatial(adata,color=plot_svgs,size=1.6,spot_size=1)

    return gene_svg_map


if __name__ == "__main__":
    adata_embryo = load_and_normalize_data(constants.MOUSE_EMBRYO_DATA_PATH)
    adata_brain = load_and_normalize_data(constants.MOUSE_BRAIN_DATA_PATH)

    adata_list = [adata_embryo, adata_brain]
    adata_names = ["mouse_embryo", "mouse_brain"]

    for ind, adata in enumerate(adata_list):
        data_type = adata_names[ind]

        print("Running SpaGFT algorithm for {} data.".format(data_type))
        gene_svg_map = execute_spagft(adata)

        with open("./results/spagft_results_{}.csv".format(data_type), "w") as f:
            for key, val in gene_svg_map.items():
                f.write(f"{key},{val}\n")

        print("Finished running SpaGFT algorithm for {} data.\n".format(data_type))