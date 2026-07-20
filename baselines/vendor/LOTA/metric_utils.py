import numpy as np
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score, roc_curve


def evaluate_metrics(test_labels, y_score, threshold_param=-1, target_fpr=0.05):
    test_labels = np.asarray(test_labels, dtype=np.int64)
    y_score = np.asarray(y_score, dtype=np.float32)

    if test_labels.size == 0:
        raise ValueError('No predictions were collected for evaluation.')

    epsilon = 1e-6

    if len(np.unique(test_labels)) < 2:
        threshold = 0.5 if threshold_param == -1 else threshold_param
        y_pred = (y_score >= threshold).astype(int)
        acc = float((y_pred == test_labels).mean())
        return {
            'auroc': float('nan'),
            'pr_auc': float('nan'),
            'F1': float('nan'),
            'Precision': float('nan'),
            'Recall': float('nan'),
            'threshold': float(threshold),
            'acc': acc,
            'avg_recall': float('nan'),
            'pos_recall': float('nan'),
            'neg_recall': float('nan'),
            'tpr_at_fpr': float('nan'),
            'tpr_at_fpr_threshold': float('nan'),
        }

    auroc = roc_auc_score(test_labels, y_score)
    precision, recall, thresholds = precision_recall_curve(test_labels, y_score, pos_label=1)
    pr_auc = auc(recall, precision)
    f1_scores = 2 * precision * recall / (precision + recall + epsilon)

    if threshold_param == -1:
        best_index = int(f1_scores.argmax())
        threshold = float(thresholds[best_index]) if best_index < len(thresholds) else 1.0
        F1 = float(f1_scores[best_index])
        Precision = float(precision[best_index])
        Recall = float(recall[best_index])
    else:
        threshold = float(threshold_param)
        index = int(np.argmin(np.abs(thresholds - threshold))) if len(thresholds) > 0 else 0
        F1 = float(f1_scores[index])
        Precision = float(precision[index])
        Recall = float(recall[index])

    y_pred = (y_score >= threshold).astype(int)
    acc = float((y_pred == test_labels).mean())

    tp = float(((y_pred == 1) & (test_labels == 1)).sum())
    fn = float(((y_pred == 0) & (test_labels == 1)).sum())
    fp = float(((y_pred == 1) & (test_labels == 0)).sum())
    tn = float(((y_pred == 0) & (test_labels == 0)).sum())

    pos_recall = tp / (tp + fn + epsilon)
    neg_recall = tn / (tn + fp + epsilon)
    avg_recall = (pos_recall + neg_recall) / 2

    fpr, tpr, thds = roc_curve(test_labels, y_score)
    target_index = int(np.argmin(np.abs(fpr - target_fpr)))
    tpr_at_fpr = float(tpr[target_index])
    tpr_at_fpr_threshold = float(thds[target_index])

    return {
        'auroc': float(auroc),
        'pr_auc': float(pr_auc),
        'F1': F1,
        'Precision': Precision,
        'Recall': Recall,
        'threshold': threshold,
        'acc': acc,
        'avg_recall': float(avg_recall),
        'pos_recall': float(pos_recall),
        'neg_recall': float(neg_recall),
        'tpr_at_fpr': tpr_at_fpr,
        'tpr_at_fpr_threshold': tpr_at_fpr_threshold,
    }
