"""Metrics utilities aligned with AIDE and CNNDetection pipelines."""

import numpy as np
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score, roc_curve


def evaluate_metrics(test_labels, y_score, threshold_param: float = -1, target_fpr: float = 0.05):
    """Compute detection metrics for binary classification outputs.

    Args:
        test_labels: Iterable of ground-truth binary labels.
        y_score: Iterable of predicted scores for the positive class.
        threshold_param: Threshold to convert scores to labels. If ``-1``, the best F1 threshold
            is selected automatically.
        target_fpr: False positive rate used to report the corresponding true positive rate.

    Returns:
        Dictionary with AUROC, PR-AUC, F1, precision, recall, accuracy, average recall, positive
        and negative recall, the chosen threshold, and TPR at the requested FPR.
    """
    test_labels = np.array(test_labels)
    y_score = np.array(y_score)

    if threshold_param != -1 and not (0 <= threshold_param <= 1):
        raise ValueError("Threshold must be between 0 and 1.")

    auroc = roc_auc_score(test_labels, y_score)

    precision, recall, thresholds = precision_recall_curve(test_labels, y_score, pos_label=1)
    pr_auc = auc(recall, precision)

    epsilon = 1e-6
    f1_scores = 2 * precision * recall / (precision + recall + epsilon)

    if threshold_param == -1:
        best_index = f1_scores.argmax()
        F1 = f1_scores[best_index]
        Precision = precision[best_index]
        Recall = recall[best_index]
        threshold = thresholds[best_index] if best_index < len(thresholds) else 1.0
    else:
        threshold = threshold_param
        index = np.where(thresholds >= threshold)[0][0]
        Precision = precision[index]
        Recall = recall[index]
        F1 = f1_scores[index]

    y_pred = (y_score >= threshold).astype(int)
    acc = (y_pred == test_labels).mean()

    tp = ((y_pred == 1) & (test_labels == 1)).sum()
    fn = ((y_pred == 0) & (test_labels == 1)).sum()
    fp = ((y_pred == 1) & (test_labels == 0)).sum()
    tn = ((y_pred == 0) & (test_labels == 0)).sum()

    pos_recall = tp / (tp + fn + epsilon)  # TPR
    neg_recall = tn / (tn + fp + epsilon)  # TNR
    avg_recall = (pos_recall + neg_recall) / 2

    fpr, tpr, thds = roc_curve(test_labels, y_score)
    if len(fpr) > 0 and len(tpr) > 0:
        idx = np.argmin(np.abs(fpr - target_fpr))
        tpr_at_fpr = tpr[idx]
        tpr_at_fpr_threshold = thds[idx]
    else:
        tpr_at_fpr = 0.0
        tpr_at_fpr_threshold = 0.0

    return {
        "auroc": auroc,
        "pr_auc": pr_auc,
        "F1": F1,
        "Precision": Precision,
        "Recall": Recall,
        "threshold": threshold,
        "acc": acc,
        "avg_recall": avg_recall,
        "pos_recall": pos_recall,
        "neg_recall": neg_recall,
        "tpr_at_fpr": tpr_at_fpr,
        "tpr_at_fpr_threshold": tpr_at_fpr_threshold,
    }
