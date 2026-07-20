import numpy as np
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score, roc_curve


def evaluate_metrics(test_labels, y_score, threshold_param=-1, target_fpr=0.05):
    test_labels = np.array(test_labels)
    y_score = np.array(y_score)

    auroc = roc_auc_score(test_labels, y_score)
    precision, recall, thresholds = precision_recall_curve(test_labels, y_score, pos_label=1)
    pr_auc = auc(recall, precision)

    epsilon = 1e-6
    f1_scores = 2 * precision * recall / (precision + recall + epsilon)

    if threshold_param == -1:
        best_index = f1_scores.argmax()
        f1_value = f1_scores[best_index]
        precision_value = precision[best_index]
        recall_value = recall[best_index]
        threshold = thresholds[best_index] if best_index < len(thresholds) else 1.0
    else:
        threshold = threshold_param
        index = np.where(thresholds >= threshold)[0][0]
        precision_value = precision[index]
        recall_value = recall[index]
        f1_value = f1_scores[index]

    y_pred = (y_score >= threshold).astype(int)
    acc = (y_pred == test_labels).mean()

    tp = ((y_pred == 1) & (test_labels == 1)).sum()
    fn = ((y_pred == 0) & (test_labels == 1)).sum()
    fp = ((y_pred == 1) & (test_labels == 0)).sum()
    tn = ((y_pred == 0) & (test_labels == 0)).sum()

    pos_recall = tp / (tp + fn + epsilon)
    neg_recall = tn / (tn + fp + epsilon)
    avg_recall = (pos_recall + neg_recall) / 2

    fpr, tpr, roc_thresholds = roc_curve(test_labels, y_score)
    if len(fpr) > 0 and len(tpr) > 0:
        idx = np.argmin(np.abs(fpr - target_fpr))
        tpr_at_fpr = tpr[idx]
        tpr_at_fpr_threshold = roc_thresholds[idx]
    else:
        tpr_at_fpr = 0.0
        tpr_at_fpr_threshold = 0.0

    return {
        "auroc": auroc,
        "pr_auc": pr_auc,
        "F1": f1_value,
        "Precision": precision_value,
        "Recall": recall_value,
        "threshold": threshold,
        "acc": acc,
        "avg_recall": avg_recall,
        "pos_recall": pos_recall,
        "neg_recall": neg_recall,
        "tpr_at_fpr": tpr_at_fpr,
        "tpr_at_fpr_threshold": tpr_at_fpr_threshold,
    }
