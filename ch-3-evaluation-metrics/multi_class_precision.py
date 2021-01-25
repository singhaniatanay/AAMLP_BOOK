import numpy as np
from confusion_matrix_vals_implementation import true_positive,false_positive 
from collections import Counter

def macro_precision(y_true, y_pred):
    """
    Function to calc macro-avg precision
    """

    num_classes = len(np.unique(y_true))
    precision = 0
    
    for class_ in range(num_classes):

        #all classses expect current class are considered negative
        temp_true = [1 if p == class_ else 0 for p in y_true]
        temp_pred = [1 if p == class_ else 0 for p in y_pred]

        #calc true positives for curr class
        tp = true_positive(temp_true,temp_pred)
        fp = false_positive(temp_pred, temp_pred)
        temp_precision = tp / (tp+fp)
        precision += temp_precision
    precision /= num_classes

    return precision


def micro_precision(y_true,y_pred):
    """
    Function to calc micro avg precision
    """

    num_classes = len(np.unique(y_true))

    tp = 0
    fp = 0

    for class_ in range(num_classes):

        temp_true = [1 if p == class_ else 0 for p in y_true]
        temp_pred = [1 if p == class_ else 0 for p in y_pred]

        tp += true_positive(temp_true,temp_pred)
        fp += false_positive(temp_true,temp_pred)

    precision = tp/(tp+fp)
    return precision

def weighted_precision(y_true,y_pred):

    """
    Function to calc weighted avg precision
    """

    num_classes = len(np.unique(y_true))

    class_counts = Counter(y_true)

    precision = 0

    for class_ in range(num_classes):

        temp_true = [1 if p == class_ else 0 for p in y_true]
        temp_pred = [1 if p == class_ else 0 for p in y_pred]

        tp = true_positive(temp_true,temp_pred)
        fp = false_positive(temp_true,temp_pred)

        temp_precision = tp / (tp+fp)

        weighted_precision = class_counts[class_] * temp_precision

        precision += weighted_precision

    overall_precision = precision / len(y_true)

    return overall_precision


