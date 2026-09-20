from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
import numpy as np

def calculate_metrics(y_true, y_scores, threshold=None):
    """
    Calculates AUROC, F1, Precision, and Recall.
    y_true: list of ground truth labels (0 for normal, 1 for anomaly)
    y_scores: list of predicted anomaly scores
    """
    if len(np.unique(y_true)) == 1:
        # Cannot calculate AUROC with only one class present
        auroc = float('nan')
    else:
        auroc = roc_auc_score(y_true, y_scores)
        
    if threshold is None:
        # A simple heuristic to pick a threshold if none provided (e.g., mean of scores + 1 std)
        threshold = np.mean(y_scores) + np.std(y_scores)
        
    y_pred = (np.array(y_scores) > threshold).astype(int)
    
    f1 = f1_score(y_true, y_pred, zero_division=0)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    
    return {
        "auroc": auroc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "threshold": threshold
    }
