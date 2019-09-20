def calculate_class_precision(l, preds, labels):
    '''
    `l` is a valid label that is found in `preds` and `labels`, 
    `l` will be treated as the "positive" label \\
    `preds` is numpy array of predicted labels \\
    `labels` is a numpy array of true labels

    precision defined as true positives / (true positives + false positives)
    '''
    tp = preds[(labels==l) & (preds==l)].shape[0]
    fp = preds[(labels!=l) & (preds==l)].shape[0]

    return tp / (tp + fp) if (tp + fp) != 0 else 0

def calculate_class_recall(l, preds, labels):
    '''
    `l` is a valid label that is found in `preds` and `labels`, 
    `l` will be treated as the "positive" label \\
    `preds` is numpy array of predicted labels \\
    `labels` is a numpy array of true labels

    recall defined as true positives / (true positives + false negatives)
    '''
    tp = preds[(labels==l) & (preds==l)].shape[0]
    fn = preds[(labels==l) & (preds!=l)].shape[0]

    return tp / (tp + fn) if (tp + fn) != 0 else 0

def calculate_class_fscore(prec, rec):
    '''
    `prec` is class precision
    `rec` is class recall

    assumes precision and recall calculations are compatible
    (same data, same positive class, etc.)

    F-Score defined as 2 * precision * recall / (precision * recall)
    '''
    return (2 * prec * rec) / (prec + rec) if (prec + rec) != 0 else 0

def calculate_cluster_purity(cluster_labels, true_labels):
    max_labels_per_cluster = []
    clusters = cluster_labels.unique()
    for c in clusters:
        data_in_cluster = true_labels[cluster_labels==c]
        most_common_label = data_in_cluster.mode()[0]
        max_labels_per_cluster.append(data_in_cluster[data_in_cluster == most_common_label].shape[0])
    return sum(max_labels_per_cluster) / len(clusters)