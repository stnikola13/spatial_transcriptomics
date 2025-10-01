import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
import constants


def load_processed_data(path):
    d1 = dict()

    with open(path, "r") as f1:
        for line in f1:
            data = line.strip().split(",")
            d1[str(data[0])] = (data[1])

    return d1


def plot_confusion_matrix(y_true_dict, y_pred_dict, x_label="Predicted", y_label="Actual", title="Confusion matrix", save_path=None):
    if y_true_dict.keys() != y_pred_dict.keys():
        raise ValueError()

    keys = list(y_true_dict.keys())
    y_true = [y_true_dict[k] for k in keys]
    y_pred = [y_pred_dict[k] for k in keys]

    cm_labels = ["True", "False"]
    cm = confusion_matrix(y_true, y_pred, labels=cm_labels)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=cm_labels, yticklabels=cm_labels)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)

    if save_path:
        plt.savefig(save_path)

    # Values have to be ints for the metric calculation, so they are converted.
    y_true_ints = [1 if val == "True" else 0 for val in y_true]
    y_pred_ints = [1 if val == "True" else 0 for val in y_pred]

    accuracy = accuracy_score(y_true_ints, y_pred_ints)
    precision = precision_score(y_true_ints, y_pred_ints, zero_division=0)
    recall = recall_score(y_true_ints, y_pred_ints, zero_division=0)
    f1 = f1_score(y_true_ints, y_pred_ints, zero_division=0)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }


def plot_roc_curve(y_true_dict, y_pred_dict, title="ROC curve", save_path=None):
    if y_true_dict.keys() != y_pred_dict.keys():
        raise ValueError()

    keys = list(y_true_dict.keys())
    y_true = [y_true_dict[k] for k in keys]
    y_pred = [y_pred_dict[k] for k in keys]

    # Values have to be ints for the metric calculation, so they are converted.
    y_true_ints = [1 if val == "True" else 0 for val in y_true]
    y_pred_ints = [1 if val == "True" else 0 for val in y_pred]

    fpr, tpr, thresholds = roc_curve(y_true_ints, y_pred_ints)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, color="blue", lw=2, label="ROC curve")
    plt.plot([0, 1], [0, 1], color="gray", lw=2, linestyle="--", label="Random guess")
    plt.xlabel("False Positive rate")
    plt.ylabel("True Positive rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True, linestyle="--", alpha=0.7)

    if save_path:
        plt.savefig(save_path)

    return roc_auc


if __name__ == "__main__":
    display_names = constants.display_names

    with open(f"{constants.result_directory}/output.txt", "w") as f:
        for subject_algorithm in constants.subject_algorithms:
            for dataset_name in constants.dataset_names:
                reference_csv_path = f"{constants.result_directory}/{constants.reference_algorithm}_results_{dataset_name}.csv"
                subject_csv_path = f"{constants.result_directory}/{subject_algorithm}_results_{dataset_name}.csv"

                data1 = load_processed_data(reference_csv_path)
                data2 = load_processed_data(subject_csv_path)

                f.write(f"{display_names.get(constants.reference_algorithm)} vs {display_names.get(subject_algorithm)} on {display_names.get(dataset_name)} dataset:\n")

                cm_save_path = f"{constants.plots_directory}/cm_{subject_algorithm}_{dataset_name}.png"
                cm_title = f"Confusion matrix - {display_names.get(subject_algorithm)} ({display_names.get(dataset_name)})"

                try:
                    results = plot_confusion_matrix(data1, data2, title=cm_title, save_path=cm_save_path)
                except ValueError:
                    print("Error while plotting confusion matrix.")

                f.write(f"Accuracy:  {results['accuracy']:.4f}\n")
                f.write(f"Precision: {results['precision']:.4f}\n")
                f.write(f"Recall:    {results['recall']:.4f}\n")
                f.write(f"F1 Score:  {results['f1_score']:.4f}\n")

                roc_save_path = f"{constants.plots_directory}/roc_{subject_algorithm}_{dataset_name}.png"
                roc_title = f"ROC curve - {display_names.get(subject_algorithm)} ({display_names.get(dataset_name)})"

                try:
                    roc_area = plot_roc_curve(data1, data2, title=roc_title, save_path=roc_save_path)
                except ValueError:
                    print("Error while plotting ROC curve.")

                f.write(f"Area under ROC curve: {roc_area:.2f}\n")
                f.write("\n")

        print(f"Plots were saved to {constants.plots_directory}!")
        print(f"Results were saved to {constants.result_directory}/output.txt!")