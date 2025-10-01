MOUSE_EMBRYO_DATA_PATH = "./assets/E9.5_E1S1.MOSTA.h5ad"
MOUSE_BRAIN_DATA_PATH = "./assets/Mouse_brain_cell_bin.h5ad"

result_directory = "./results"
plots_directory = "./plots"
dataset_names = ["mouse_embryo", "mouse_brain"]
reference_algorithm = "spagft"
subject_algorithms = ["shannon_entropy", "anova"]

display_names = {"mouse_embryo": "Mouse embryo",
                 "mouse_brain": "Mouse brain",
                 "spagft": "SpaGFT",
                 "shannon_entropy": "Shannon entropy",
                 "anova": "Anova"}