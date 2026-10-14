"""Metrics shared by baseline training and final locked test evaluation."""
import numpy as np
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_recall_fscore_support)


def evaluate_predictions(y_true, predictions, valid_mask, classes):
    """Report classifier-only and end-to-end image metrics separately.

    `predictions` contains one class prediction per row; invalid extraction rows
    are replaced by an empty string for the end-to-end/system metric.
    """
    y_true=np.asarray(y_true).astype(str)
    predictions=np.asarray(predictions).astype(str)
    valid_mask=np.asarray(valid_mask,dtype=bool)
    if not (len(y_true)==len(predictions)==len(valid_mask)):
        raise ValueError('Metric inputs must have the same number of rows')
    if not valid_mask.any():
        raise ValueError('No feature-valid examples in evaluation split')
    labels=list(map(str,classes))
    classifier_true=y_true[valid_mask]
    classifier_pred=predictions[valid_mask]
    system_pred=np.where(valid_mask,predictions,'')
    per_precision,per_recall,per_f1,per_support=precision_recall_fscore_support(
        classifier_true,classifier_pred,labels=labels,zero_division=0)
    sys_precision,sys_recall,sys_f1,sys_support=precision_recall_fscore_support(
        y_true,system_pred,labels=labels,zero_division=0)
    return {
        'total':int(len(y_true)),
        'feature_valid':int(valid_mask.sum()),
        'feature_coverage':float(valid_mask.mean()),
        'classifier_valid_only':{
            'accuracy':float(accuracy_score(classifier_true,classifier_pred)),
            'macro_f1':float(f1_score(classifier_true,classifier_pred,labels=labels,average='macro',zero_division=0)),
            'per_class':{label:{'precision':float(per_precision[i]),'recall':float(per_recall[i]),
                'f1':float(per_f1[i]),'support':int(per_support[i])} for i,label in enumerate(labels)},
            'confusion_matrix':confusion_matrix(classifier_true,classifier_pred,labels=labels).tolist(),
        },
        'system_all_images':{
            'accuracy':float(accuracy_score(y_true,system_pred)),
            'macro_f1':float(f1_score(y_true,system_pred,labels=labels,average='macro',zero_division=0)),
            'rejected_due_to_invalid_features':int((~valid_mask).sum()),
            'per_class':{label:{'precision':float(sys_precision[i]),'recall':float(sys_recall[i]),
                'f1':float(sys_f1[i]),'support':int(sys_support[i])} for i,label in enumerate(labels)},
            'confusion_matrix_with_reject_column':confusion_matrix(y_true,system_pred,labels=labels+['']).tolist(),
        },
    }
