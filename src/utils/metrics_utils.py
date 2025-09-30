from sklearn.metrics import accuracy_score, recall_score, roc_auc_score

def compute_core_metrics(y_true, y_pred, y_prob, pos_label_idx):
    acc = accuracy_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred, pos_label=pos_label_idx, zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except Exception:
        auc = float("nan")
    return {"accuracy": float(acc), "recall": float(rec), "auroc": float(auc)}

def compute_metrics_multiclass(y_true, y_pred, y_prob, average="macro"):
    acc = accuracy_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred, average=average, zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_prob, multi_class="ovr", average=average)
    except Exception:
        auc = float("nan")
    return {"accuracy": float(acc), "recall_macro": float(rec), "auroc_macro_ovr": float(auc)}