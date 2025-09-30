import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score, confusion_matrix, accuracy_score, recall_score

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path

def plot_roc_curve(y_true, y_prob, show=True, save_path=None):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    plt.figure(figsize=(6,6))
    plt.plot(fpr, tpr, label=f"AUROC={auc:.3f}")
    plt.plot([0,1],[0,1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    if save_path:
        ensure_dir(os.path.dirname(save_path))
        plt.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()

def plot_pr_curve(y_true, y_prob, show=True, save_path=None):
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    plt.figure(figsize=(6,6))
    plt.plot(recall, precision, label=f"AP={ap:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    if save_path:
        ensure_dir(os.path.dirname(save_path))
        plt.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()

def plot_confusion(y_true, y_pred, label_names, show=True, save_path=None):
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    plt.figure(figsize=(6,5))
    plt.imshow(cm, cmap="Blues")
    plt.xticks([0,1], label_names)
    plt.yticks([0,1], label_names)
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i,j]), ha="center", va="center")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    if save_path:
        ensure_dir(os.path.dirname(save_path))
        plt.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()

def plot_calibration(y_true, y_prob, bins=10, show=True, save_path=None):
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    edges = np.linspace(0,1,bins+1)
    idx = np.digitize(y_prob, edges) - 1
    acc = []
    conf = []
    for b in range(bins):
        m = idx == b
        if m.sum() == 0:
            continue
        acc.append(y_true[m].mean())
        conf.append(y_prob[m].mean())
    plt.figure(figsize=(6,6))
    plt.plot([0,1],[0,1], linestyle="--")
    plt.scatter(conf, acc)
    plt.xlabel("Predicted probability")
    plt.ylabel("Empirical accuracy")
    plt.title("Calibration Plot")
    if save_path:
        ensure_dir(os.path.dirname(save_path))
        plt.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()

def plot_threshold_sweep(y_true, y_prob, label_names, show=True, save_path=None):
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    ts = np.linspace(0.0, 1.0, 101)
    recalls = []
    accuracies = []
    for t in ts:
        y_pred = (y_prob >= t).astype(int)
        recalls.append(recall_score(y_true, y_pred, pos_label=1, zero_division=0))
        accuracies.append(accuracy_score(y_true, y_pred))
    plt.figure(figsize=(7,4))
    plt.plot(ts, recalls, label=f"Recall({label_names[1]})")
    plt.plot(ts, accuracies, label="Accuracy")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title("Threshold Sweep")
    plt.legend()
    if save_path:
        ensure_dir(os.path.dirname(save_path))
        plt.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()

def plot_loss_acc_from_history(history_csv_path, show=True, out_prefix=None):
    if not os.path.isfile(history_csv_path):
        return None
    import csv
    epochs, tr, vl, va = [], [], [], []
    with open(history_csv_path, "r") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            epochs.append(int(row[0]))
            tr.append(float(row[1]))
            vl.append(float(row[2]))
            va.append(float(row[3]))
    plt.figure(figsize=(7,4))
    plt.plot(epochs, tr, label="train_loss")
    plt.plot(epochs, vl, label="val_loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.title("Loss Curves")
    if out_prefix:
        ensure_dir(os.path.dirname(out_prefix))
        plt.savefig(out_prefix + "_loss.png", bbox_inches="tight")
    if show:
        plt.show()
    plt.close()
    plt.figure(figsize=(7,4))
    plt.plot(epochs, va, label="val_acc")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.legend()
    plt.title("Validation Accuracy")
    if out_prefix:
        plt.savefig(out_prefix + "_val_acc.png", bbox_inches="tight")
    if show:
        plt.show()
    plt.close()
    return True

def plot_confusion_mc(y_true, y_pred, label_names, normalize=False, show=True, save_path=None):
    labels_idx = list(range(len(label_names)))
    cm = confusion_matrix(y_true, y_pred, labels=labels_idx, normalize="true" if normalize else None)
    plt.figure(figsize=(8,7))
    plt.imshow(cm, cmap="Blues")
    plt.xticks(labels_idx, label_names, rotation=45, ha="right")
    plt.yticks(labels_idx, label_names)
    for i in range(len(label_names)):
        for j in range(len(label_names)):
            val = cm[i,j]
            txt = f"{val:.2f}" if normalize else str(int(val))
            plt.text(j, i, txt, ha="center", va="center")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix" + (" (normalized)" if normalize else ""))
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()

def plot_roc_multi(y_true, y_prob, label_names, show=True, save_path=None):
    y_true_arr = np.array(y_true)
    n_classes = len(label_names)
    plt.figure(figsize=(7,7))
    for c in range(n_classes):
        y_bin = (y_true_arr == c).astype(int)
        fpr, tpr, _ = roc_curve(y_bin, y_prob[:, c])
        auc = roc_auc_score(y_bin, y_prob[:, c])
        plt.plot(fpr, tpr, label=f"{label_names[c]} (AUC={auc:.3f})")
    plt.plot([0,1],[0,1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves (One-vs-Rest)")
    plt.legend(fontsize=8)
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()