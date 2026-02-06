import numpy as np

def classification_metrics(y_true, y_pred, average="micro", pos_label=1):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    assert len(y_true) == len(y_pred)

    labels = np.unique(np.concatenate([y_true, y_pred]))

    # Accuracy (same for all averaging modes)
    accuracy = float(np.mean(y_true == y_pred))

    def safe_div(a, b):
        return float(a / b) if b != 0 else 0.0

    # Binary case
    if average == "binary":
        TP = np.sum((y_true == pos_label) & (y_pred == pos_label))
        FP = np.sum((y_true != pos_label) & (y_pred == pos_label))
        FN = np.sum((y_true == pos_label) & (y_pred != pos_label))

        precision = safe_div(TP, TP + FP)
        recall = safe_div(TP, TP + FN)
        f1 = safe_div(2 * precision * recall, precision + recall)

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }

    precisions = []
    recalls = []
    f1s = []
    supports = []

    TP_micro = FP_micro = FN_micro = 0

    for c in labels:
        TP = np.sum((y_true == c) & (y_pred == c))
        FP = np.sum((y_true != c) & (y_pred == c))
        FN = np.sum((y_true == c) & (y_pred != c))
        support = np.sum(y_true == c)

        p = safe_div(TP, TP + FP)
        r = safe_div(TP, TP + FN)
        f1_c = safe_div(2 * p * r, p + r)

        precisions.append(p)
        recalls.append(r)
        f1s.append(f1_c)
        supports.append(support)

        TP_micro += TP
        FP_micro += FP
        FN_micro += FN

    if average == "micro":
        precision = safe_div(TP_micro, TP_micro + FP_micro)
        recall = safe_div(TP_micro, TP_micro + FN_micro)
        f1 = safe_div(2 * precision * recall, precision + recall)

    elif average == "macro":
        precision = float(np.mean(precisions))
        recall = float(np.mean(recalls))
        f1 = float(np.mean(f1s))

    elif average == "weighted":
        total = np.sum(supports)
        precision = safe_div(np.sum(np.array(precisions) * supports), total)
        recall = safe_div(np.sum(np.array(recalls) * supports), total)
        f1 = safe_div(np.sum(np.array(f1s) * supports), total)

    else:
        raise ValueError("average must be one of: micro, macro, weighted, binary")

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }
