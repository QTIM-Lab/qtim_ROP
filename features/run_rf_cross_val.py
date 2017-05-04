from os.path import basename, join, isfile
from features.rf_cnn_codes import main as cnn_rf
from utils.common import get_subdirs, make_sub_dir, dict_reverse
from utils.metrics import plot_roc_auc
from keras.utils.np_utils import to_categorical
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from glob import glob

CLASSES = {'No': 0, 'Plus': 1, 'Pre-Plus': 2}

def run_cross_val(all_splits, out_dir):

    results_dir = make_sub_dir(out_dir, 'results')
    predictions = defaultdict(list)
    labels = defaultdict(list)

    for i, split_dir in enumerate(sorted(get_subdirs(all_splits))):

        results_dir = make_sub_dir(out_dir, basename(split_dir))

        cnn_model = join(split_dir, 'Split{}_Model'.format(i), 'Split{}_Model.yaml'.format(i))
        print cnn_model

        test_data = join(split_dir, 'test.h5')
        y_test, y_pred, cnn_features = cnn_rf(cnn_model, test_data, results_dir)
        print cnn_features['classes']
        # roc_auc, fpr, tpr = calculate_roc_auc(y_pred, to_categorical(y_test), cnn_features['classes'], None)

        # Save predictions and labels
        y_test = to_categorical(y_test)  # binarize true labels
        print y_test.shape
        print y_pred.shape

        for class_name, c in CLASSES.items():

            predictions[class_name].append(y_pred[:, c])
            labels[class_name].append(y_test[:, c])

    # Save predictions
    save_predictions(predictions, labels, CLASSES, results_dir)

    # Plot ROC curves for No and Plus classes, combined across all splits
    fig, ax = plt.subplots()
    for class_name, c in CLASSES.items():

        if class_name == 'Pre-Plus':
            continue

        J = []
        for s in range(0, 5):
            j = plot_roc_auc(predictions[class_name][s], labels[class_name][s], name=class_name)
            J.append(j)

        print class_name
        print J


    plt.savefig(join(out_dir, 'combined_roc.svg'))


def save_predictions(predictions, labels, class_dict, out_dir):

    # Save as CSV
    for class_name, c in class_dict.items():

        pred_out = join(out_dir, 'predictions_{}.csv'.format(class_name))
        labels_out = join(out_dir, 'labels_{}.csv'.format(class_name))

        pred_df = pd.DataFrame(predictions[class_name]).T
        labels_df = pd.DataFrame(labels[class_name]).T

        print pred_df
        print labels_df

        pred_df.to_csv(pred_out)
        labels_df.to_csv(labels_out)


if __name__ == '__main__':

    import sys

    run_cross_val(sys.argv[1], sys.argv[2])