import numpy as np

def f1_micro(y_true, y_pred) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    TP = np.sum(y_true == y_pred)
    FP = np.sum(y_true != y_pred)
    FN = FP  # in single-label multiclass, FP == FN

    return (2 * TP) / (2 * TP + FP + FN) if len(y_true) > 0 else 0.0
