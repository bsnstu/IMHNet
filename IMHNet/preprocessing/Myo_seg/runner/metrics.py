
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, jaccard_score
from sklearn.metrics import confusion_matrix


def calculate_metrics(pred, true):
    # pred: []
    # true: []
    print(pred.shape)
    true = np.expand_dims(true.ravel(), axis=1)
    pred = np.expand_dims((pred > 0.5).astype(np.float32).ravel(), axis=1)

    """ TN, FP, FN, TP """
   # tn, fp, fn, tp = confusion_matrix(true, pred).ravel()

    """ metrics """
    print(np.max(true))
    precision = precision_score(true, pred)
    recall = recall_score(true, pred)
    f1 = f1_score(true, pred)
    acc = accuracy_score(true, pred)
    jac = jaccard_score(true, pred)

    return {
            'MET': {'attr': ['Precision', 'Recall', 'F1-Score', 'Accuracy', 'Jaccard'],
                    'value': np.hstack([precision, recall, f1, acc, jac])}
            }

