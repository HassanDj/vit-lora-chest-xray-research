"""Small, offline-testable scientific helpers for the ViT/LoRA benchmark."""
from __future__ import annotations
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix

SEED = 42


def nested_stratified_indices(labels, seed=SEED):
    """Draw 1,200 stratified train indices, then 400 stratified within those 1,200."""
    y = np.asarray(labels).reshape(-1).astype(int)
    if len(y) < 1200 or set(np.unique(y)) != {0, 1}:
        raise ValueError('Expected at least 1,200 examples and both binary classes')
    all_idx = np.arange(len(y))
    idx1200, _ = train_test_split(all_idx, train_size=1200, stratify=y,
                                  random_state=seed)
    idx400, _ = train_test_split(idx1200, train_size=400,
                                 stratify=y[idx1200], random_state=seed)
    idx1200 = np.sort(idx1200)
    idx400 = np.sort(idx400)
    assert len(idx1200) == 1200 and len(idx400) == 400
    assert len(set(idx400).difference(idx1200)) == 0
    for idx in (idx400, idx1200):
        assert set(np.unique(y[idx])) == {0, 1}
    return {400: idx400, 1200: idx1200}


def class_weights(labels):
    y = np.asarray(labels).reshape(-1).astype(int)
    counts = np.bincount(y, minlength=2)
    if len(y) == 0 or np.any(counts == 0):
        raise ValueError('Binary class weights require two nonempty classes')
    return len(y) / (2.0 * counts)


def binary_metrics(labels, positive_scores, threshold=0.5):
    y = np.asarray(labels).reshape(-1).astype(int)
    scores = np.asarray(positive_scores).reshape(-1).astype(float)
    if not len(y) or len(y) != len(scores) or set(np.unique(y)) != {0, 1}:
        raise ValueError('Metrics require nonempty, aligned arrays containing both labels')
    if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError('Positive class probabilities must be finite and in [0, 1]')
    predictions = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, predictions, labels=[0, 1]).ravel()
    sensitivity = float(tp / (tp + fn))
    specificity = float(tn / (tn + fp))
    return {
        'auroc': float(roc_auc_score(y, scores)),
        'balanced_accuracy': 0.5 * (sensitivity + specificity),
        'sensitivity': sensitivity,
        'specificity': specificity,
        'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)
    }


def should_extend_after_three(auroc_history):
    """Pre-registered extension rule applied exclusively to validation AUROC."""
    if len(auroc_history) != 3:
        raise ValueError('Need exactly three validation AUCs')
    return (int(np.argmax(auroc_history)) == 2 and
            auroc_history[2] - auroc_history[1] >= 0.003)


def should_stop_extension(auroc_history):
    """Stop after two consecutive epochs without ≥0.002 improvement in best val AUROC."""
    if len(auroc_history) < 5:
        return False
    best = max(auroc_history[:3])
    consecutive = 0
    for auc in auroc_history[3:]:
        if auc >= best + 0.002:
            best = auc
            consecutive = 0
        else:
            consecutive += 1
        if consecutive >= 2:
            return True
    return False
