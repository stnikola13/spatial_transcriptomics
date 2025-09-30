import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc


def load_processed_data(path):
    d1 = dict()

    with open(path, "r") as f1:
        for line in f1:
            data = line.strip().split(",")
            d1[str(data[0])] = (data[1])

    return d1


def plot_confusion_matrix(y_true_dict, y_pred_dict, x_label="Predicted", y_label="Actual", title="Confusion matrix"):
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
    plt.show()

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


def plot_roc_curve(y_true_dict, y_pred_dict, title="ROC curve"):
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
    plt.plot(fpr, tpr, color="blue", lw=2, label=title)
    plt.plot([0, 1], [0, 1], color="gray", lw=2, linestyle="--", label="Random guess")
    plt.xlabel("False Positive rate")
    plt.ylabel("True Positive rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.show()

    return roc_auc


if __name__ == "__main__":
    data1_csv_path = "./results/spagft_results_mouse_embryo.csv"
    data2_csv_path = "./results/fabricated_results_temp.csv"

    data1 = load_processed_data(data1_csv_path)
    data2 = load_processed_data(data2_csv_path)
    results = plot_confusion_matrix(data1, data2)

    print(f"Accuracy:  {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall:    {results['recall']:.4f}")
    print(f"F1 Score:  {results['f1_score']:.4f}")

    auc = plot_roc_curve(data1, data2)
    print(f"Area under ROC curve: {auc:.2f}")
