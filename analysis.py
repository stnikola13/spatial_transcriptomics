import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score


def load_processed_data(path1, path2):
    d1 = dict()
    d2 = dict()

    with open(path1, "r") as f1:
        for line in f1:
            data = line.strip().split(",")
            d1[str(data[0])] = (data[1])

    with open(path2, "r") as f2:
        for line in f2:
            data = line.strip().split(",")
            d2[str(data[0])] = (data[1])

    return d1, d2


def plot_confusion_matrix(y_true_dict, y_pred_dict, x_label="Predicted", y_label="Actual", title="Confusion matrix"):
    if y_true_dict.keys() != y_pred_dict.keys():
        raise ValueError()

    keys = list(y_true_dict.keys())
    y_true = [y_true_dict[k] for k in keys]
    y_pred = [y_pred_dict[k] for k in keys]

    cm_labels = ['True', 'False']
    cm = confusion_matrix(y_true, y_pred, labels=cm_labels)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
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


if __name__ == "__main__":
    data1_csv_path = "./results/spagft_results_mouse_embryo.csv"
    data2_csv_path = "./results/fabricated_results_temp.csv"

    data1, data2 = load_processed_data(data1_csv_path, data2_csv_path)
    results = plot_confusion_matrix(data1, data2)

    print(f"Accuracy:  {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall:    {results['recall']:.4f}")
    print(f"F1 Score:  {results['f1_score']:.4f}")